"""O painel mostra números reais, e recusa inventar os que faltam.

O risco de um painel é este: como é visual, um número errado passa melhor
despercebido do que num JSON. Estes testes garantem três coisas:

1. o que o painel mostra vem dos ficheiros medidos, não de constantes no HTML;
2. um resultado em falta aparece como `None` (a interface mostra «—»), nunca
   como um número plausível;
3. a geração ao vivo carrega um checkpoint real e devolve texto.

O ponto 1 é o que distingue um painel de um panfleto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "painel"))


@pytest.fixture(scope="module")
def servidor():
    import servidor as s
    return s


def test_estado_le_os_ficheiros_e_nao_constantes(servidor) -> None:
    e = servidor.estado()
    # estes valores têm de coincidir com os JSON, não com o HTML
    melhor = json.loads((RAIZ / "nucleo" / "experimentos" /
                         "resultado_melhor_ponto.json").read_text())
    assert e["baselines"]["transformer_bpb"] == melhor["melhor_observado"]["bits_por_byte_media"]
    # o trigrama vem agora da comparação feita com o MESMO texto e a MESMA
    # tokenização dos dois lados, não de um número passado
    cmp = json.loads((RAIZ / "nucleo" / "experimentos" /
                      "resultado_comparacao_final.json").read_text())
    assert e["baselines"]["trigrama_bpb"] == cmp["trigrama"]["bits_por_byte"]

    gram = json.loads((RAIZ / "nucleo" / "experimentos" /
                       "resultado_gramatical.json").read_text())
    assert e["gramatica"]["transformer"] == gram["transformer_prosa_800_passos"]["taxa"]


def test_campo_em_falta_e_none_e_nao_numero(servidor, tmp_path) -> None:
    """O contrato mais importante: não inventar."""
    e = servidor.estado()
    # todos os campos numéricos são None ou número — nunca string com número
    for bloco in ("corpus", "baselines", "gramatica"):
        for k, v in e[bloco].items():
            if k == "vencedor":       # campo textual por desenho
                assert v in (None, "trigrama", "transformer"), f"{bloco}.{k} = {v!r}"
            else:
                assert v is None or isinstance(v, (int, float)), f"{bloco}.{k} = {v!r}"


def test_o_painel_reconhece_que_o_trigrama_ganha(servidor) -> None:
    """Se o painel dissesse «ganha» quando a medição diz «perde», seria propaganda."""
    e = servidor.estado()
    assert e["baselines"]["ganha"] is False
    assert e["baselines"]["margem"] < 0


def test_a_curva_do_sobreajuste_esta_completa(servidor) -> None:
    e = servidor.estado()
    assert e["curva"] is not None
    assert len(e["curva"]) >= 10
    passos = [c["passo"] for c in e["curva"]]
    assert passos == sorted(passos)
    # a validação do último ponto tem de ser pior que a do melhor
    melhor = min(c["perda_val"] for c in e["curva"])
    assert e["curva"][-1]["perda_val"] > melhor + 3


def test_checkpoints_disponiveis_para_gerar(servidor) -> None:
    e = servidor.estado()
    assert e["checkpoints"], "sem checkpoints o painel não pode gerar"
    assert any(c.startswith("publico_") for c in e["checkpoints"])


def test_gerar_produz_texto(servidor) -> None:
    nome = servidor.estado()["checkpoints"]
    alvo = next((c for c in nome if c.startswith("publico_")), None)
    if alvo is None:
        pytest.skip("sem checkpoint publico_*")
    if not (servidor.CHECKPOINTS / alvo / "pesos.npz").exists():
        pytest.skip("checkpoint sem pesos (*.npz fora do git)")
    r = servidor.gerar(alvo, "Capitu olhou", n=12, temp=0.0)
    assert r["checkpoint"] == alvo
    assert isinstance(r["texto"], str) and len(r["texto"]) > 0


def test_o_html_nao_contem_numeros_medidos(servidor) -> None:
    """Um número escrito à mão no HTML nunca se actualiza."""
    html = servidor.HTML
    for suspeito in ("2.6796", "2.4766", "0.4615", "381981", "938807"):
        assert suspeito not in html, (
            f"«{suspeito}» está escrito à mão no HTML: tem de vir do /api/estado")


# ---------------------------------------------------------------------------
# Regressão: os parâmetros do pedido têm de chegar ao gerador.
#
# DEFEITO APANHADO: o handler só passava `checkpoint`, `prompt`, `n` e `temp`.
# Enviar `repeticao: 1.2` devolvia `repeticao: 0.0` na resposta — o parâmetro
# era silenciosamente ignorado. Um utilizador a mexer no slider não veria
# efeito nenhum e concluiria, com razão, que a ferramenta não funciona.
# ---------------------------------------------------------------------------


def test_repeticao_chega_ao_gerador(monkeypatch) -> None:
    """O valor de `repeticao` no pedido tem de aparecer na resposta."""
    import numpy as np

    from painel import servidor

    class _FalsoModelo:
        cfg = type("C", (), {"ctx": 128})()

        def gerar(self, ids, n_novos, temperatura=0.0, semente=None, repeticao=0.0):
            _FalsoModelo.recebido = repeticao
            return np.concatenate([ids, np.zeros((1, n_novos), dtype=np.int64)], axis=1)

    class _FalsoTok:
        def codificar(self, t):
            return [1, 2, 3]

        def descodificar(self, ids):
            return "texto"

    monkeypatch.setattr(servidor, "carregar_modelo", lambda _n: (_FalsoModelo(), _FalsoTok()))
    r = servidor.gerar("x", "abc", n=5, temp=0.7, repeticao=1.2)
    assert _FalsoModelo.recebido == 1.2, "o gerador não recebeu a penalidade de repetição"
    assert r["repeticao"] == 1.2


def test_checkpoint_por_omissao_e_o_do_melhor_ponto() -> None:
    """O painel não pode servir por omissão o checkpoint do passo 1500.

    MEDIDO: `publico_seed*` foi gravado no passo 1500 (val 6,7498), que é o
    PIOR ponto da curva. `machado_melhor` está no passo 550 (val 5,9231).
    Servir o primeiro foi o defeito que fez o utilizador ver palavras
    inventadas e concluir que nada funcionava.
    """
    from painel import servidor

    assert servidor.CHECKPOINT_OMISSAO == "machado_melhor"


def test_html_tem_controlos_e_ligacoes() -> None:
    """Cada slider tem de existir, enviar o valor e mostrar o rótulo.

    DEFEITO APANHADO: o HTML não tinha controlo de `repeticao` nem de `n`, e
    o `fetch` só mandava três campos. Os sliders novos existirem no HTML não
    chega — se não estiverem declarados e ligados no JS, ficam decorativos.
    """
    from painel import servidor

    h = servidor.HTML
    for ident in ("temp", "rep", "nn", "ck", "prompt", "btn", "out"):
        assert f'id="{ident}"' in h, f"falta o controlo {ident} no HTML"
    for valor in ("rval", "nval", "tval"):
        assert f'id="{valor}"' in h, f"falta o rótulo {valor}"

    # o corpo do pedido leva os três parâmetros
    corpo = h[h.index("JSON.stringify({"):h.index("JSON.stringify({") + 400]
    for campo in ("temperatura", "repeticao", "n"):
        assert campo in corpo, f"o pedido não envia {campo}"

    # e cada slider está ligado ao seu rótulo
    for slider, rotulo in (("temp", "tval"), ("rep", "rval"), ("nn", "nval")):
        assert f"getElementById('{slider}')" in h, f"o slider {slider} não é lido"
        assert f"{slider}.oninput" in h, f"o slider {slider} não tem oninput"
        assert f"getElementById('{rotulo}')" in h, f"falta o rótulo {rotulo}"


def test_html_nao_tem_numeros_medidos_a_mao() -> None:
    """Nenhum resultado medido pode estar literal no HTML.

    DEFEITO APANHADO DUAS VEZES: o HTML tinha `2,77 M parâmetros`, `10,9 → 5,4
    amostras` e `trigrama 2,4766` escritos directamente. Se um experimento for
    refeito, o painel passa a mentir e ninguém dá por isso.

    Ficam permitidos apenas números de INTERFACE: mínimo/máximo/passo de um
    slider e os valores iniciais dos controlos. Tudo o resto tem de vir do
    JSON através de `estado()`.
    """
    import re

    from painel import servidor

    permitidos = {
        "1.5", "1.55", "1.7", "0.6", "6.0",     # CSS
        "0.1", "0.2", "0.8", "1.2",             # passos de sliders
        "0,8", "1,2",                            # valores iniciais mostrados
    }
    achados = set(re.findall(r"\b\d+[.,]\d+\b", servidor.HTML))
    sobra = achados - permitidos
    assert not sobra, (
        f"números medidos escritos à mão no HTML: {sorted(sobra)}. "
        "Têm de vir de estado() / dos JSON."
    )


def test_checkpoint_servido_esta_versionado() -> None:
    """Quem clonar o repositório tem de conseguir gerar texto sem treinar.

    DEFEITO REAL: `nucleo/experimentos/checkpoints/` estava inteiro no
    .gitignore, incluindo o checkpoint que o painel serve. Um clone limpo
    dava HTTP 500 em /api/gerar. O painel funcionava só na máquina onde os
    pesos tinham sido treinados.
    """
    import subprocess

    from painel import servidor

    alvo = f"nucleo/experimentos/checkpoints/{servidor.CHECKPOINT_OMISSAO}/pesos.npz"
    r = subprocess.run(["git", "check-ignore", "-q", alvo], cwd=RAIZ)
    assert r.returncode == 1, f"{alvo} está no .gitignore e não seria distribuído"

    r = subprocess.run(["git", "ls-files", "--error-unmatch", alvo],
                       cwd=RAIZ, capture_output=True)
    # ou já está versionado, ou o git aceita adicioná-lo
    if r.returncode != 0:
        d = subprocess.run(["git", "add", "--dry-run", alvo], cwd=RAIZ, capture_output=True)
        assert d.returncode == 0 and d.stdout, f"{alvo} não pode ser versionado"
