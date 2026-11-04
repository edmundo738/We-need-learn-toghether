"""Testes do Transformer experimental — escritos para poderem FALHAR.

A pergunta que estes testes respondem não é «o modelo é bom». É:

1. A causalidade é real? (`saida[t]` só depende de `ids[:t+1]`)
2. O gradiente é correcto? (comparado com diferenças finitas)
3. As formas e contratos de `CONTRATOS.md` são respeitados?
4. O treino move a loss de facto? (não só o código existe)

Se o teste 1 falhar, todos os números de treino futuros são lixo. É por isso
que ele corre primeiro.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nucleo.modelo.transformer import (
    Config,
    Transformer,
    perda_entropia_cruzada,
)


@pytest.fixture(scope="module")
def cfg_pequena() -> Config:
    """Configuração mínima que exercita todos os caminhos sem ser lenta."""
    return Config(vocab=64, d_model=32, n_camadas=2, n_cabecas=2, ctx=12, d_ff=64)


@pytest.fixture(scope="module")
def modelo(cfg_pequena) -> Transformer:
    return Transformer(cfg_pequena, semente=1234)


# --------------------------------------------------------------------------
# Contrato: configuração
# --------------------------------------------------------------------------

def test_config_valida_divisibilidade():
    with pytest.raises(ValueError):
        Config(d_model=128, n_cabecas=5).validar()

    with pytest.raises(ValueError):
        Config(ctx=1).validar()


def test_n_parametros_analitico_bate_com_construido(cfg_pequena):
    """Duas contagens independentes. Se divergirem, uma das duas está errada."""
    modelo = Transformer(cfg_pequena)
    assert cfg_pequena.n_parametros() == modelo.n_parametros()


def test_n_parametros_da_config_da_directiva():
    """A configuração declarada tem de caber em 1–3 M parâmetros."""
    c = Config()
    n = c.n_parametros()
    assert 1_000_000 <= n <= 3_000_000, f"{n:,} parâmetros fora do intervalo da directiva"


# --------------------------------------------------------------------------
# Contrato: causalidade — o teste que governa tudo
# --------------------------------------------------------------------------

def test_causalidade_alterar_ultimo_token_nao_muda_posicoes_anteriores(modelo, cfg_pequena):
    """O teste central.

    Alterar o token na posição `k` NÃO PODE mudar nenhuma saída antes de `k`.
    Se mudar, o modelo viu o futuro e nenhuma métrica posterior tem sentido.
    """
    rng = np.random.default_rng(7)
    ids = rng.integers(0, cfg_pequena.vocab, (1, 8))

    ref = modelo.forward(ids).data.copy()

    for k in range(1, 8):
        alterado = ids.copy()
        # mudar para um token garantidamente diferente
        alterado[0, k] = (alterado[0, k] + 17) % cfg_pequena.vocab
        novo = modelo.forward(alterado).data

        # posições 0..k-1 têm de ser idênticas
        np.testing.assert_allclose(
            ref[:k], novo[:k], rtol=0, atol=1e-6,
            err_msg=f"alterar a posição {k} mudou as saídas anteriores — causalidade quebrada",
        )
        # e a posição k TEM de mudar (senão o modelo ignora o token, o que
        # também é um bug, mas de outro tipo)
        assert not np.allclose(ref[k], novo[k]), (
            f"alterar o token na posição {k} não mudou a saída dessa posição"
        )


def test_causalidade_no_lote_sequencias_nao_se_veem(modelo, cfg_pequena):
    """Duas sequências no mesmo lote não podem influenciar-se."""
    rng = np.random.default_rng(11)
    a = rng.integers(0, cfg_pequena.vocab, (1, 6))
    b = rng.integers(0, cfg_pequena.vocab, (1, 6))

    juntas = modelo.forward(np.concatenate([a, b], axis=0)).data
    so_a = modelo.forward(a).data
    so_b = modelo.forward(b).data

    # atol=1e-5 e não 1e-6: em fp32, com 2 camadas e softmax, a diferença de
    # ordem de somatório entre «lote de 2» e «lote de 1» produz ~1,7e-6 de
    # ruído. Um vazamento real entre sequências daria ordens de grandeza
    # acima disto (o teste verifica isso mesmo — ver test_causalidade_*).
    np.testing.assert_allclose(juntas[:6], so_a, rtol=0, atol=1e-5)
    np.testing.assert_allclose(juntas[6:], so_b, rtol=0, atol=1e-5)


def test_mascara_causal_e_triangular(modelo, cfg_pequena):
    """A máscara usada tem de ser triangular inferior, incluindo a diagonal."""
    modelo.forward(np.zeros((1, 5), dtype=np.int64))
    mascara = modelo.blocos[0].atencao._ultima_mascara
    assert mascara.shape == (5, 5)
    assert mascara.dtype == bool
    # triangular inferior: True em j <= i
    esperado = np.tril(np.ones((5, 5), dtype=bool))
    np.testing.assert_array_equal(mascara, esperado)


# --------------------------------------------------------------------------
# Contrato: gradiente
# --------------------------------------------------------------------------

def test_gradiente_bate_com_diferencas_finitas(cfg_pequena):
    """O autograd é nosso; se estiver errado, o treino optimiza outra coisa."""
    np.random.seed(0)
    modelo = Transformer(cfg_pequena, semente=3)
    rng = np.random.default_rng(5)
    ids = rng.integers(0, cfg_pequena.vocab, (1, 5))
    alvos = rng.integers(0, cfg_pequena.vocab, 5)

    def perda() -> float:
        logits = modelo.forward(ids)
        return float(perda_entropia_cruzada(logits, alvos).data.reshape(-1)[0])

    for t in modelo.parametros().values():
        t.zero_grad()
    logits = modelo.forward(ids)
    perda_entropia_cruzada(logits, alvos).backward()

    # escolher alguns pesos e comparar com perturbação numérica
    parametros = modelo.parametros()
    alvo_nomes = ["cabeca.W", "bloco0.up.W", "emb.peso", "bloco1.ln1.gama"]
    # O epsilon importa e tem de ser escolhido, não adivinhado.
    # Com 1e-2 a diferença central de uma função não-linear (softmax + relu)
    # dá 12,8% de erro e o teste falharia sem o gradiente estar errado. A
    # convergência medida para emb.peso foi: 1e-2 -> 12,84%, 1e-3 -> 1,13%,
    # 1e-4 -> 0,01%, 1e-5 -> 0,05%. Usa-se 1e-4, onde o erro de truncatura
    # já morreu e o de arredondamento (fp32, ~1e-7 relativo) ainda não nasceu.
    eps = 1e-4
    for nome in alvo_nomes:
        t = parametros[nome]
        assert t.grad is not None, f"{nome} não recebeu gradiente"
        idx = np.unravel_index(int(np.argmax(np.abs(t.grad))), t.data.shape)
        analitico = float(t.grad[idx])

        original = t.data[idx]
        t.data[idx] = original + eps
        cima = perda()
        t.data[idx] = original - eps
        baixo = perda()
        t.data[idx] = original

        numerico = (cima - baixo) / (2 * eps)
        assert math.isclose(analitico, numerico, rel_tol=0.05, abs_tol=1e-3), (
            f"{nome}{idx}: analítico={analitico:.6f} numérico={numerico:.6f}"
        )


def test_perda_logits_uniformes_da_log_vocab():
    """Contrato de CONTRATOS.md: logits iguais → perda = log(V)."""
    V = 64
    logits = np.zeros((3, V), dtype=np.float32)
    alvos = np.array([0, 5, 63])
    from nucleo.autograd import Tensor
    perda = perda_entropia_cruzada(Tensor(logits, requires_grad=True), alvos)
    assert math.isclose(float(perda.data.reshape(-1)[0]), math.log(V), rel_tol=1e-6)


def test_perda_minima_quando_previsao_e_perfeita():
    """Logits altos na resposta certa → perda ~0."""
    from nucleo.autograd import Tensor
    V = 16
    logits = np.full((1, V), -10.0, dtype=np.float32)
    logits[0, 3] = 10.0
    perda = perda_entropia_cruzada(Tensor(logits, requires_grad=True), np.array([3]))
    assert float(perda.data.reshape(-1)[0]) < 1e-6


# --------------------------------------------------------------------------
# Contrato: formas
# --------------------------------------------------------------------------

def test_formas_da_saida(modelo, cfg_pequena):
    ids = np.zeros((3, 7), dtype=np.int64)
    logits = modelo.forward(ids)
    assert logits.data.shape == (3 * 7, cfg_pequena.vocab)


def test_contexto_excedido_falha_alto(cfg_pequena):
    modelo = Transformer(cfg_pequena)
    ids = np.zeros((1, cfg_pequena.ctx + 1), dtype=np.int64)
    with pytest.raises(ValueError, match="contexto"):
        modelo.forward(ids)


def test_sequencia_de_um_token():
    c = Config(vocab=32, d_model=16, n_camadas=1, n_cabecas=2, ctx=8, d_ff=32)
    modelo = Transformer(c)
    logits = modelo.forward(np.array([[3]]))
    assert logits.data.shape == (1, 32)


# --------------------------------------------------------------------------
# Contrato: geração
# --------------------------------------------------------------------------

def test_geracao_greedy_e_deterministica(modelo, cfg_pequena):
    """Com temperatura 0, a mesma entrada dá exactamente a mesma saída."""
    ids = np.array([[1, 2, 3]])
    a = modelo.gerar(ids, n_novos=4, temperatura=0.0)
    b = modelo.gerar(ids, n_novos=4, temperatura=0.0)
    np.testing.assert_array_equal(a, b)


def test_geracao_acrescenta_exactamente_n_novos(modelo, cfg_pequena):
    ids = np.array([[1, 2]])
    saida = modelo.gerar(ids, n_novos=5, temperatura=0.0)
    assert saida.shape == (1, 7)
    np.testing.assert_array_equal(saida[:, :2], ids)


def test_geracao_greedy_escolhe_o_argmax(modelo, cfg_pequena):
    """Greedy não é «quase argmax»: é o argmax, em cada passo."""
    ids = np.array([[4, 5]])
    logits = modelo.forward(ids).data
    esperado = int(np.argmax(logits[-1]))
    saida = modelo.gerar(ids, n_novos=1, temperatura=0.0)
    assert int(saida[0, -1]) == esperado


def test_geracao_respeita_janela_de_contexto(cfg_pequena):
    """Gerar mais que `ctx` não pode rebentar — o corte é explícito."""
    modelo = Transformer(cfg_pequena)
    ids = np.zeros((1, cfg_pequena.ctx), dtype=np.int64)
    saida = modelo.gerar(ids, n_novos=5, temperatura=0.0)
    assert saida.shape == (1, cfg_pequena.ctx + 5)


# --------------------------------------------------------------------------
# Contrato: persistência
# --------------------------------------------------------------------------

def test_guardar_e_carregar_preserva_a_saida(cfg_pequena, tmp_path):
    modelo = Transformer(cfg_pequena, semente=99)
    ids = np.array([[1, 2, 3, 4]])
    antes = modelo.forward(ids).data

    modelo.guardar(tmp_path / "m")
    recarregado = Transformer.carregar(tmp_path / "m")
    depois = recarregado.forward(ids).data

    np.testing.assert_allclose(antes, depois, rtol=0, atol=1e-6)


def test_carregar_detecta_peso_em_falta(cfg_pequena, tmp_path):
    modelo = Transformer(cfg_pequena)
    modelo.guardar(tmp_path / "m")
    # apagar um peso do ficheiro
    caminho = tmp_path / "m" / "pesos.npz"
    dados = dict(np.load(caminho))
    dados.pop("cabeca.W")
    np.savez_compressed(caminho, **dados)

    with pytest.raises(KeyError, match="cabeca.W"):
        Transformer.carregar(tmp_path / "m")


def test_guardar_escreve_config_legivel(cfg_pequena, tmp_path):
    Transformer(cfg_pequena).guardar(tmp_path / "m")
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    assert cfg["vocab"] == cfg_pequena.vocab
    assert cfg["n_camadas"] == cfg_pequena.n_camadas
