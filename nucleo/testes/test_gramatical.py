"""Dimensão 2 da avaliação: o modelo julga gramaticalidade? NÃO.

Medido em `resultado_gramatical.json` com 26 pares correcto/violado em 14
fenómenos. O baseline é a moeda ao ar (50%), porque sem ele «acertou 46%» não
significa nada.

    moeda ao ar          13,0/26   0,500   —
    Transformer (prosa)  12,0/26   0,462   p = 1,0000
    trigrama             18,5/26   0,712   p = 0,0755

O Transformer fica ABAIXO do acaso. Não é «um pouco mau»: é indistinguível de
atirar uma moeda, e o ponto estimado está do lado errado. O trigrama, com os
MESMOS dados, acerta 71% — logo os dados contêm o sinal gramatical e o
Transformer não o extrai.

ONDE FALHA, E PORQUÊ (por fenómeno)
    colocação pronominal      0/2   — «Me deu o livro» e «Não disse-me nada»
    haver impessoal           0/2   — «Haviam muitas pessoas», «Houveram problemas»
    concordância verbal       1/3
    regência verbal           1/3

Estes são exactamente os fenómenos em que a frase violada tem uma forma que
o modelo pode preferir por razões de frequência: «me deu» é raro no corpus e
«haviam» também, mas o modelo não está a comparar as duas frases — está a
atribuir probabilidade a cada uma em separado, e o corpus de 66 mil tokens de
um autor não contém evidência suficiente para estes contrastes.

O QUE ISTO AUTORIZA DIZER
    «O modelo julga gramática portuguesa» — NÃO. Recusado por medição.
    «O trigrama é melhor que o Transformer neste teste» — sim, com a ressalva
    de que p=0,0755 não é significativo a 5%: é uma tendência, não um facto
    estabelecido. Com 26 pares o poder estatístico é baixo; para detectar um
    efeito de 0,2 precisaria de ~100 pares.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
PAUTA = RAIZ / "nucleo" / "avaliacao" / "gramatica" / "pares_v1.json"


@pytest.fixture(scope="module")
def res() -> dict:
    f = RAIZ / "nucleo" / "experimentos" / "resultado_gramatical.json"
    if not f.exists():
        pytest.skip("correr nucleo.avaliacao.gramatical")
    return json.loads(f.read_text())


def test_a_base_de_pares_e_valida() -> None:
    d = json.loads(PAUTA.read_text(encoding="utf-8"))
    pares = d["pares"]
    assert len(pares) >= 20, "menos de 20 pares não dá poder estatístico mínimo"
    for p in pares:
        assert p["correcto"] != p["violado"], "par idêntico não mede nada"
        assert p["fenomeno"], "par sem fenómeno declarado"
    assert d["aviso_de_licenca"], "tem de declarar que o texto do Bechara não é copiado"


def test_cada_par_isola_um_fenomeno() -> None:
    """Regra do instrumento: a violação é só do fenómeno declarado.

    Compara em CARACTERES, não em palavras separadas por espaço. A primeira
    versão deste teste usava `.split()` e rejeitou o par
    «Deu-me o livro» / «Me deu o livro» — que está correcto: o hífen muda a
    tokenização mas não muda as palavras. O teste era demasiado rígido, não
    a pauta.
    """
    import difflib

    d = json.loads(PAUTA.read_text(encoding="utf-8"))
    for p in d["pares"]:
        a, b = p["correcto"], p["violado"]
        razao = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
        assert razao > 0.72, (
            f"par com alterações demais ({p['fenomeno']}): semelhança {razao:.2f} "
            f"— «{a}» vs «{b}». Só o fenómeno declarado pode mudar.")


def test_o_transformer_nao_bate_o_acaso(res: dict) -> None:
    t = res["transformer_prosa_800_passos"]
    assert t["taxa"] <= 0.65, (
        f"taxa {t['taxa']:.3f}: se o modelo passou a distinguir gramática, a "
        "conclusão «não julga gramaticalidade» deixa de valer e deve ser revista")
    assert t["p_valor_vs_moeda"] > 0.05, \
        "significativo contra a moeda: a leitura muda"


def test_o_trigrama_e_melhor_mas_nao_significativo(res: dict) -> None:
    """Resistir à tentação de declarar vitória: p=0,0755 não é 5%."""
    n = res["trigrama_mesmos_dados"]
    t = res["transformer_prosa_800_passos"]
    assert n["taxa"] > t["taxa"] + 0.15, "a diferença trigrama-Transformer encolheu"
    assert n["p_valor_vs_moeda"] > 0.05, (
        "o trigrama passou a ser significativo contra a moeda — se for o caso, "
        "actualizar a conclusão de «tendência» para «efeito»")


def test_o_baseline_esta_declarado(res: dict) -> None:
    assert res["baseline_moeda"] == 0.5
    assert "p_valor_vs_moeda" in res["transformer_prosa_800_passos"]


def test_fenomenos_onde_falha_estao_registados(res: dict) -> None:
    pf = res["por_fenomeno"]
    # estes dois são 0/2 e são o núcleo do diagnóstico
    assert pf["colocacao_pronominal"]["acertos_transformer"] == 0
    assert pf["sujeito_inexistente_haver"]["acertos_transformer"] == 0
