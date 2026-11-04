"""O checkpoint guarda o estado REAL e o melhor ponto não é o último.

Duas afirmações testadas aqui, ambas com consequência prática:

1. `guardar`/`carregar` preservam a saída do modelo ao bit. Sem isto, um
   checkpoint é decoração: não se pode avaliar, comparar nem servir.

2. O melhor ponto de validacão não é o fim do treino. Em `sem tying`, o
   mínimo chegou entre o passo 600 e 950 e o treino continuou até 1000,
   pagando 0,0524 nats para nada. Se o código só guardasse o último passo,
   esse valor era desperdiçado.

O teste 2 lê `resultado_sobreajuste.json` (medido) em vez de re-treinar
(custaria ~14 minutos) e falha se a conclusão deixar de valer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def modelo_pequeno():
    from nucleo.modelo.transformer import Config, Transformer

    cfg = Config(vocab=64, d_model=16, n_camadas=2, n_cabecas=2, ctx=16,
                 d_ff=32, tying=False)
    return Transformer(cfg, semente=3)


def test_guardar_e_carregar_preservam_a_saida(modelo_pequeno, tmp_path: Path) -> None:
    """A saída tem de ser idêntica, não parecida."""
    from nucleo.modelo.transformer import Transformer

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 64, size=(2, 15))
    antes = modelo_pequeno.forward(ids).data.copy()

    # mexer nos pesos para garantir que não estamos a ler a inicialização
    for p in modelo_pequeno.parametros().values():
        p.data += rng.normal(0, 0.02, size=p.data.shape).astype(np.float32)
    treinado = modelo_pequeno.forward(ids).data.copy()
    assert not np.allclose(antes, treinado), "a perturbação não teve efeito: teste sem valor"

    destino = tmp_path / "ckpt"
    modelo_pequeno.guardar(destino)
    assert (destino / "config.json").exists()
    assert (destino / "pesos.npz").exists()

    recarregado = Transformer.carregar(destino)
    depois = recarregado.forward(ids).data
    np.testing.assert_array_equal(depois, treinado,
                                  err_msg="o checkpoint não devolve o mesmo modelo")


def test_caminho_de_treino_grava_checkpoint_coerente(tmp_path: Path) -> None:
    """O caminho que o treino usa — forward/backward/step — grava coerente.

    `Adam.step` chama `_matriz()` em todos os parâmetros, o que sincroniza a
    cabeça derivada. Quem treina passa por aqui, por isso o checkpoint é válido.
    """
    from nucleo import nn
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada

    cfg = Config(vocab=32, d_model=8, n_camadas=1, n_cabecas=2, ctx=8,
                 d_ff=16, tying=True)
    m = Transformer(cfg, semente=1)
    opt = nn.Adam(m.parametros(), eta=1e-3)
    ids = np.arange(9).reshape(1, 9) % 32
    opt.zero_grad()
    perda_entropia_cruzada(m.forward(ids[:, :-1]), ids[0, 1:]).backward()
    opt.step()

    destino = tmp_path / "ckpt_treino"
    m.guardar(destino)
    r = Transformer.carregar(destino)          # não pode levantar ValueError
    # e a cabeça carregada continua a ser a transposta do embedding carregado
    r.forward(ids[:, :-1])
    np.testing.assert_allclose(r.cabeca.W.data, r.emb.peso.data.T, rtol=0, atol=1e-6)


def test_mutacao_directa_sem_forward_continua_a_gravar_bem(tmp_path: Path) -> None:
    """REGRESSÃO CORRIGIDA.

    Defeito encontrado por este ficheiro: `guardar()` lia `cabeca.W.data`, que
    com tying é uma vista sincronizada só em `_matriz()` (forward / step do
    optimizador). Alterar `emb.peso.data` à mão e gravar sem forward escrevia
    a transposta ANTIGA, e `carregar` recusava o ficheiro acabado de gravar.

    Corrigido em `guardar()`: a cabeça passa a ser derivada de `emb.peso`
    no momento de gravar, em vez de lida da vista. Este teste impede o regresso.
    """
    from nucleo.modelo.transformer import Config, Transformer

    cfg = Config(vocab=32, d_model=8, n_camadas=1, n_cabecas=2, ctx=8,
                 d_ff=16, tying=True)
    m = Transformer(cfg, semente=1)
    m.forward(np.arange(7).reshape(1, 7) % 32)     # materializa a cabeça
    m.emb.peso.data += 0.01                        # mutação directa, sem forward

    destino = tmp_path / "ckpt_direto"
    m.guardar(destino)

    # o ficheiro gravado tem de ser coerente com o estado actual
    z = np.load(destino / "pesos.npz")
    np.testing.assert_allclose(z["cabeca.W"], z["emb.peso"].T, rtol=0, atol=1e-6)

    # e tem de carregar sem erro
    r = Transformer.carregar(destino)
    np.testing.assert_allclose(r.emb.peso.data, m.emb.peso.data, rtol=0, atol=1e-7)


def test_guardar_detecta_ficheiro_corrompido(tmp_path: Path) -> None:
    """A validação de `carregar` continua a servir para o que foi escrita."""
    import json as _json

    from nucleo.modelo.transformer import Config, Transformer

    cfg = Config(vocab=32, d_model=8, n_camadas=1, n_cabecas=2, ctx=8,
                 d_ff=16, tying=True)
    m = Transformer(cfg, semente=1)
    destino = tmp_path / "ckpt_mau"
    m.guardar(destino)

    dados = dict(np.load(destino / "pesos.npz"))
    dados["cabeca.W"] = dados["cabeca.W"] + 0.5     # corromper a cabeça
    np.savez_compressed(destino / "pesos.npz", **dados)

    with pytest.raises(ValueError, match="não é a transposta"):
        Transformer.carregar(destino)


def test_o_melhor_ponto_nao_e_o_ultimo() -> None:
    f = RAIZ / "nucleo" / "experimentos" / "resultado_sobreajuste.json"
    if not f.exists():
        pytest.skip("correr nucleo.experimentos.sobreajuste")
    d = json.loads(f.read_text())

    sem = d["sem_tying"]
    assert sem["perda_ao_parar_media"] > 0, \
        "sem tying, parar no fim tem de custar (o mínimo chega antes)"
    assert any(p < 1000 for p in sem["passo_do_minimo"]), \
        "se todas as seeds minimizarem no último passo, não há nada a guardar"

    # com tying o modelo ainda desce: é falta de capacidade, não virtude
    assert all(p == 1000 for p in d["tying"]["passo_do_minimo"])


def test_ha_pelo_menos_um_checkpoint_em_disco() -> None:
    pasta = RAIZ / "nucleo" / "experimentos" / "checkpoints"
    if not pasta.exists():
        pytest.skip("checkpoints não persistidos (fora do git por *.npz)")
    pesos = list(pasta.glob("*/pesos.npz"))
    assert pesos, "pasta de checkpoints existe mas está vazia"
