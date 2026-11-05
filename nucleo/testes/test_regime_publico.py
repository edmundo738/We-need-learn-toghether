"""4,5x mais dados mudam o regime de memorização — mas não bastam para ganhar.

Machado de Assis (domínio público) deu 294 107 tokens contra 65 999. Isto
permitiu responder a três perguntas que ficaram abertas durante toda a Fase 2.

PERGUNTA 1 — mais dados reduzem a decoração? SIM, E MUITO
    folga treino-validação: 8-9 nats (66k tokens) -> 3,49 nats (294k)
    A validação deixou de ser «acaso ou pior»: 6,75 nats contra log(8192)=9,01.

PERGUNTA 2 — mais passos ajudam? NÃO, PIORAM
    Curva de 4000 passos (`resultado_underfitting.json`):
        passo  500 -> val 5,9340  (2,6796 bpb)  <- melhor
        passo 1500 -> val 6,7745  (3,0591 bpb)
        passo 4000 -> val 10,6986 (4,8311 bpb)
    Não era underfitting: era sobreajuste desde o passo ~600. Com 1 395 janelas
    de treino e lote 8, o modelo esgota o sinal aos ~3 épocas e começa a decorar.

PERGUNTA 3 — bate o trigrama? NÃO
    3 seeds, melhor ponto de cada (`resultado_melhor_ponto.json`):
        trigrama          2,4766 bits/byte
        Transformer       2,6821 ± 0,0133 bpb  (margem -0,2055)
    No passo fixo 500, escolhido a priori por uma corrida separada:
        Transformer       2,6990 bpb  (margem -0,2224)
    O optimismo de escolher o melhor ponto vale 0,017 bpb — pequeno, e mesmo
    assim insuficiente. **O trigrama ganha de forma inequívoca.**

A LEITURA HONESTA
    2,77 M parâmetros com 178 560 tokens de treino = **0,064 tokens/parâmetro**.
    Os modelos reais usam ~4 000 — são 62 000x mais. (Eu escrevi primeiro
    «64 tokens/parâmetro»: confundi 0,064 com 64.) O factor não invalida nada
    do que está acima, e torna a leitura mais severa do que eu a tinha feito:
    não é «estamos 60x abaixo», é «estamos 4-5 ordens de grandeza abaixo».
    Mais 4,5x de dados melhorou a decoração e não chegou para bater um
    contador de trigramas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]


def _carregar(nome: str) -> dict:
    f = RAIZ / "nucleo" / "experimentos" / nome
    if not f.exists():
        pytest.skip(f"correr o experimento que produz {nome}")
    return json.loads(f.read_text())


@pytest.fixture(scope="module")
def regime() -> dict:
    return _carregar("resultado_regime_publico.json")


@pytest.fixture(scope="module")
def curva() -> dict:
    return _carregar("resultado_underfitting.json")


@pytest.fixture(scope="module")
def ponto() -> dict:
    return _carregar("resultado_melhor_ponto.json")


def test_a_divisao_e_por_obra(regime: dict) -> None:
    assert "por obra" in regime["divisao"]
    assert "dom_casmurro" in regime["divisao"]


def test_a_folga_caiu_com_mais_dados(regime: dict) -> None:
    """O ganho real: 8-9 nats de decoração -> 3,49."""
    assert regime["folga_media"] < 5.0, (
        f"folga {regime['folga_media']}: se subiu, o corpus novo não ajudou")
    assert regime["folga_media"] > 2.0, "folga quase nula seria generalização real"


def test_o_modelo_ja_nao_esta_no_acaso(regime: dict) -> None:
    import math
    acaso = math.log(8192)
    assert regime["transformer_perda_val_media"] < acaso * 0.85, \
        "validação perto do acaso: o corpus novo não ensinou nada"


def test_mais_passos_pioram(curva: dict) -> None:
    """A curva que matou a hipótese de underfitting."""
    assert curva["diagnostico"].startswith("a piorar"), curva["diagnostico"]
    melhor = curva["melhor"]
    assert melhor["passo"] <= 500, f"o óptimo mudou de sítio: {melhor['passo']}"
    ultimo = curva["curva"][-1]
    assert ultimo["perda_val"] > melhor["perda_val"] + 3.0, \
        "a validação não piorou o suficiente para sustentar «sobreajuste»"


def test_o_trigrama_ganha(ponto: dict) -> None:
    """A conclusão negativa, fixada para não ser arredondada a favor do modelo."""
    assert ponto["trigrama_bits_por_byte"] == 2.4766
    assert not ponto["ganha_do_trigrama_melhor"], (
        "o Transformer passou a bater o trigrama: resultado importante, "
        "reescrever a conclusão em vez de relaxar o teste")
    assert not ponto["ganha_do_trigrama_fixo"]
    assert ponto["margem_fixo"] < -0.05


def test_o_optimismo_do_melhor_ponto_e_pequeno(ponto: dict) -> None:
    """Se escolher o melhor ponto valesse muito, a medição seria enganosa."""
    m = ponto["melhor_observado"]["bits_por_byte_media"]
    f = ponto["passo_fixo_500"]["bits_por_byte_media"]
    assert 0 <= m <= f, "o melhor observado tem de ser <= o ponto fixo"
    assert (f - m) < 0.10, f"optimismo de {f-m:.3f} bpb é grande demais"


def test_as_tres_seeds_concordam(ponto: dict) -> None:
    desv = ponto["melhor_observado"]["perda_val_desvio"]
    assert desv < 0.05, f"desvio entre seeds {desv:.4f} é grande: resultado frágil"
    passos = ponto["melhor_observado"]["passos"]
    assert all(400 <= p <= 800 for p in passos), f"óptimos dispersos: {passos}"


def test_tokens_por_parametro_esta_registado(regime: dict) -> None:
    """A explicação estrutural tem de viajar com o resultado.

    Correção de um erro meu: escrevi primeiro «64 tokens/parâmetro» e o teste
    exigia 40 < x < 100. O valor real é 178 560 / 2 771 200 = **0,064** — eu
    confundi 0,064 com 64, um factor de 1000. Comparado com os ~4 000 dos
    modelos reais, este modelo tem ~62 000x menos dados por parâmetro.
    A conclusão não muda; a escala muda.
    """
    tpp = regime["tokens_por_parametro"]
    assert 0.04 < tpp < 0.10, f"{tpp} tokens/parâmetro de treino"
    assert regime["tokens_treino"] < 200_000, "a validação não pode entrar na conta"
