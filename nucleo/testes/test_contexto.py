"""Mais contexto PIOROU. O mecanismo está identificado e medido.

`contexto_maior.py` comparou em orçamento igual (800 passos, 1 024 tokens por
passo nos dois braços, lote compensado 8 vs 4). A única variável era o ctx:

    ctx 128  ->  2,6796 bits/byte   (melhor @ 500)
    ctx 256  ->  2,7018 bits/byte   (melhor @ 700)
    diferença: -0,0222 bpb  (256 é PIOR)

O MECANISMO (medido, `resultado_diagnostico_contexto.json`)
    `emb_pos = Embedding(cfg.ctx, d_model)` — uma tabela com um vector POR
    POSIÇÃO. Não é RoPE nem sinusóide. Cada posição tem 128 parâmetros que só
    são treinados com os tokens que caem naquela posição:

        ctx 128: 16 384 parâmetros de posição, 1 395 janelas -> 10,9 amostras/par
        ctx 256: 32 768 parâmetros de posição,   697 janelas ->  5,4 amostras/par

    Dobrar o contexto duplica a tabela e mantém o corpus. **Metade dos dados
    por parâmetro de posição.** É o único sítio do modelo onde o argumento
    dados/parâmetro se aplica: em `modelo_menor.py`, reduzir o modelo inteiro
    não ajudou precisamente porque a posicional ficava do mesmo tamanho.

A PERDA POR POSIÇÃO (treino, ctx 128)
    inicio (0-31)   5,0396
    meio  (60-91)   4,8814
    fim   (96-127)  5,0046
    Uniforme. **Não** memorizou os inícios de janela. A hipótese de «decorou o
    início dos capítulos» está descartada por medição.

O QUE ISTO MUDA
    A posicional aprendida é o único componente cujo custo cresce linearmente
    com ctx sem ganho de dados. É o suspeito certo — e o teste que o confirmaria
    é substituí-la por RoPE/sinusóide, que não têm parâmetros por posição.
    Isso NÃO foi feito: mudar a arquitectura está congelado pela directiva e
    exige regressão. Fica como próximo passo declarado.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]


def _c(nome: str) -> dict:
    f = RAIZ / "nucleo" / "experimentos" / nome
    if not f.exists():
        pytest.skip(f"correr o experimento que produz {nome}")
    return json.loads(f.read_text())


@pytest.fixture(scope="module")
def ctx_res() -> dict:
    return _c("resultado_contexto_maior.json")


@pytest.fixture(scope="module")
def diag() -> dict:
    return _c("resultado_diagnostico_contexto.json")


def test_o_desenho_igualou_o_orcamento(ctx_res: dict) -> None:
    """Sem isto, o ctx 256 ganhava só por ver mais tokens."""
    a, b = ctx_res["resultado"]
    assert a["tokens_vistos"] == b["tokens_vistos"], "orçamentos diferentes: teste inválido"
    assert a["tokens_por_passo"] == b["tokens_por_passo"]
    assert a["ctx"] == 128 and b["ctx"] == 256


def test_o_contexto_maior_e_pior(ctx_res: dict) -> None:
    a, b = ctx_res["resultado"]
    assert b["bits_por_byte"] > a["bits_por_byte"], (
        "o ctx 256 passou a ser melhor: o mecanismo mudou e a conclusão tem de ser revista")
    assert ctx_res["melhoria_do_ctx_256_bpb"] < 0


def test_nenhum_dos_dois_bate_o_trigrama(ctx_res: dict) -> None:
    assert not ctx_res["ctx_256_bate_o_trigrama"]
    for r in ctx_res["resultado"]:
        assert r["bits_por_byte"] > ctx_res["trigrama_bits_por_byte"]


def test_a_aritmetica_da_posicional_esta_certa(diag: dict) -> None:
    """O número que explica tudo: amostras por parâmetro de posição."""
    a128, a256 = diag["aritmetica"]
    assert a128["parametros_de_posicao"] == 128 * 128
    assert a256["parametros_de_posicao"] == 256 * 128
    assert a256["parametros_de_posicao"] == 2 * a128["parametros_de_posicao"], \
        "dobrar ctx tem de duplicar a tabela de posição"
    r = a256["amostras_por_parametro_de_posicao"] / a128["amostras_por_parametro_de_posicao"]
    assert 0.4 < r < 0.6, f"razao {r:.2f}: esperava-se ~0,5 (metade dos dados por parâmetro)"


def test_a_perda_e_uniforme_por_posicao(diag: dict) -> None:
    """Descarta «decorou o início das janelas»."""
    f = {x["faixa"].split(" ")[0]: x for x in diag["perda_por_faixa_de_posicao"]}
    inicio = f["inicio"]["perda_treino"]
    fim = f["fim"]["perda_treino"]
    meio = f["meio"]["perda_treino"]
    assert abs(inicio - fim) < 0.5, (
        f"início {inicio} vs fim {fim}: se o início ficar muito melhor, há "
        "memorização posicional e a leitura muda")
    assert abs(meio - inicio) < 0.5
