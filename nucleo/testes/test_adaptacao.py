"""Adicionar dados de diálogo não faz o modelo conversar — mede-se.

O utilizador entregou 50 conversas (512 turnos, 2 900 palavras). É o primeiro
dado que o projecto tem para a finalidade «conversation», que estava a zero.
A conclusão fácil seria «a perda desceu, logo aprendemos diálogo». A medição
diz outra coisa, e é isso que este ficheiro fixa.

MEDIÇÕES (`resultado_adaptacao.json`, `resultado_memorizacao_dialogo.json`)
--------------------------------------------------------------------------
    braço A (pesos aleatórios -> diálogo)   10,1973 -> 9,2549
    braço B (prosa -> diálogo)               9,7561 -> 8,8613
    perda de B nas conversas de TREINO       0,0510
    perda de B nas de VALIDAÇÃO              8,8613   (folga +8,81 nats)
    validação com nomes trocados por nomes do treino  8,3490

A folga de 8,81 nats explica-se em 5,8% pelos nomes: trocar os 13 falantes
desconhecidos por conhecidos recupera só 0,51 nats. **O modelo decorou o
texto das conversas de treino, não aprendeu a estrutura do diálogo.**

COM 1,7 M PARÂMETROS E 9 653 TOKENS, DECORAR É O QUE ACONTECE. Não é um bug:
é a razão de dados por parâmetro (ponto 1 do DEZ-PONTOS.md) outra vez, agora
em diálogo. Um corpus de 2 900 palavras não pode ensinar conversação; pode
ser decorado, e foi.

O QUE ESTE FICHEIRO IMPEDE
--------------------------
Que alguém cite «9,25 -> 8,86» como progresso em diálogo. A perda de VALIDAÇÃO
por conversa é o único número que mede generalização e esse continua em 8,86 —
pior que o log(8192)=9,01 de um modelo uniforme por muito pouco. Ou seja: em
conversas que nunca viu, o modelo está perto do acaso.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def adapt() -> dict:
    f = RAIZ / "nucleo" / "experimentos" / "resultado_adaptacao.json"
    if not f.exists():
        pytest.skip("correr nucleo.experimentos.adaptacao")
    return json.loads(f.read_text())


@pytest.fixture(scope="module")
def memor() -> dict:
    f = RAIZ / "nucleo" / "experimentos" / "resultado_memorizacao_dialogo.json"
    if not f.exists():
        pytest.skip("correr nucleo.experimentos.memorizacao_dialogo")
    return json.loads(f.read_text())


def test_a_divisao_e_por_conversa_e_nao_por_janela(adapt: dict) -> None:
    """Sem isto, turnos da mesma conversa caem nos dois lados."""
    d = adapt["divisao"]
    assert d["metodo"] == "por conversa inteira"
    assert d["conversas_treino"] + d["conversas_val"] == 50
    assert d["conversas_val"] == 10


def test_prosa_ajuda_mas_pouco(adapt: dict) -> None:
    """B parte abaixo de A (a prosa transfere algo) mas ambos ficam altos."""
    a = adapt["A_so_dialogo"]["perda_val_final"]
    b = adapt["B_prosa_para_dialogo"]["perda_val_final"]
    assert b < a, "a prosa deixou de ajudar — rever"
    # ambos perto do acaso para um vocabulário de 8192: log(8192) = 9,011
    assert b > 8.0, f"validação de {b:.2f} é boa demais para 2 900 palavras: suspeitar"


def test_o_modelo_decorou_o_treino(memor: dict) -> None:
    """A folga treino-vs-validação é enorme: é memorização, não aprendizagem."""
    assert memor["folga_memorizacao"] > 5.0, \
        "se a folga desapareceu, o regime de dados mudou e isto deve ser revisto"
    assert memor["perda_treino_dialogo"] < 0.5, \
        "perda de treino quase zero é decoração, não generalização"


def test_a_memorizacao_nao_se_explica_pelos_nomes(memor: dict) -> None:
    """A ablação que descarta a explicação fácil."""
    fraccao = memor["fraccao_explicada_pelo_nome"]
    assert fraccao < 0.20, (
        f"os nomes explicam {fraccao*100:.0f}% da folga; se passar de 20% a "
        "interpretação muda e este teste tem de ser reescrito, não relaxado")


def test_a_validacao_continua_perto_do_acaso(memor: dict) -> None:
    """O número honesto: em conversas novas, o modelo quase não bate o uniforme."""
    import math
    acaso = math.log(8192)
    p = memor["perda_val_dialogo"]
    assert p > acaso * 0.90, (
        f"validação {p:.2f} vs acaso {acaso:.2f}: se o modelo começou a "
        "generalizar, isto deixa de ser verdade e merece ser dito")
