"""A explicação «faltam dados por parâmetro» foi TESTADA e REFUTADA.

Eu tinha escrito na Adenda D9 que o Transformer perdia para o trigrama por
causa da razão dados/parâmetros (0,064 tokens/parâmetro), e classifiquei isso
como INFERRED. A directiva diz que uma explicação que não se mede não é um
resultado. Medi-a, e está errada.

CRITÉRIO DECLARADO ANTES DE CORRER
    «Se um modelo menor chegar a <= 2,4766 bpb, a explicação confirma-se.»

MEDIÇÃO (mesmo corpus, mesma divisão por obra, 800 passos, lote 8)
    d_model 128, d_ff 384   2 771 200 par   2,6796 bpb   margem -0,2030
    d_model 128, d_ff 256   2 640 128 par   2,6976 bpb   margem -0,2210
    d_model 128, d_ff 128   2 509 056 par   2,7014 bpb   margem -0,2248
    d_model  64, d_ff 192   1 221 760 par   2,7066 bpb   margem -0,2300
    d_model  64, corpus antigo (66k)        2,7594 bpb   margem -0,2828

O QUE ISTO DIZ
    1. NENHUM tamanho bate o trigrama, e o maior é o MELHOR. Reduzir o modelo
       piora-o monotonicamente — o contrário do que a explicação previa.
    2. O controlo separa as duas variáveis: o mesmo modelo pequeno no corpus
       novo (2,7066) é melhor que no corpus antigo (2,7594). Os dados ajudam
       (0,05 bpb), o tamanho não é o problema.
    3. Logo a explicação «faltam dados por parâmetro» é INSUFICIENTE. O que
       resta é estrutural: um Transformer com posicionais aprendidas e
       contexto de 128, treinado em 178k tokens, não captura regularidades
       locais tão bem como contar trigramas. Isso é uma hipótese, não um
       resultado — e o próximo passo para a testar é diferente: contexto
       maior, ou comparar com um n-grama de ordem superior.

Este ficheiro impede que a conclusão antiga (INFERRED, errada) seja citada
como se fosse medida.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def res() -> dict:
    f = RAIZ / "nucleo" / "experimentos" / "resultado_modelo_menor.json"
    if not f.exists():
        pytest.skip("correr nucleo.experimentos.modelo_menor")
    return json.loads(f.read_text())


def test_nenhum_tamanho_bate_o_trigrama(res: dict) -> None:
    assert res["algum_bate_o_trigrama"] is False, (
        "algum modelo passou a bater o trigrama: resultado importante, "
        "reescrever a conclusão em vez de relaxar o teste")


def test_os_modelos_publicos_ficam_todos_acima(res: dict) -> None:
    """Só as 4 variantes com o corpus novo (a 5.ª é o controlo)."""
    publicos = [r for r in res["resultados"] if r.get("corpus") != "antigo_66k"]
    assert len(publicos) == 4
    for r in publicos:
        assert r["bits_por_byte"] > res["trigrama_bits_por_byte"], \
            f"{r['rotulo']} com {r['bits_por_byte']} bpb bateu o trigrama"


def test_menos_parametros_pioram(res: dict) -> None:
    """A refutação da explicação: o maior modelo é o melhor, não o pior."""
    publicos = sorted([r for r in res["resultados"] if r.get("corpus") != "antigo_66k"],
                      key=lambda r: -r["parametros"])
    bpb = [r["bits_por_byte"] for r in publicos]
    assert bpb[0] == min(bpb), (
        f"o modelo maior deixou de ser o melhor: {bpb}. Se a relação inverter, "
        "a explicação dados/parâmetro volta a ser plausível e isto deve ser revisto")
    # mas as diferenças são pequenas — não é uma relação forte
    assert max(bpb) - min(bpb) < 0.15, "a variação com o tamanho é grande demais"


def test_o_controlo_com_corpus_antigo_e_pior(res: dict) -> None:
    """Separa «modelo menor» de «mais dados»."""
    ctrl = next(r for r in res["resultados"] if r.get("corpus") == "antigo_66k")
    pequeno_novo = next(r for r in res["resultados"]
                        if r["d_model"] == 64 and r.get("corpus") != "antigo_66k")
    assert ctrl["parametros"] == pequeno_novo["parametros"], \
        "o controlo tem de ser o MESMO modelo, só com outros dados"
    assert ctrl["bits_por_byte"] > pequeno_novo["bits_por_byte"], \
        "o corpus novo deixou de ajudar: rever a Adenda D"


def test_o_criterio_foi_declarado_antes(res: dict) -> None:
    assert "criterio_declarado_antes" in res
    assert "2,4766" in res["criterio_declarado_antes"] or "2.4766" in res["criterio_declarado_antes"]
