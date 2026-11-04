"""O modelo treinado distingue o que sabe do que não sabe?

Pré-requisito para qualquer política de factualidade: se o modelo dá a mesma
perda a texto do domínio e a ruído, não pode dizer «não sei» — não tem sinal
nenhum para isso. Se separa, tem.

Medido (`nucleo/avaliacao/incerteza.py`, `resultado_incerteza.json`):

    Transformer aleatório   9,9297 / 10,0359  ->  1,01x   (controlo)
    Transformer treinado    5,7034 / 13,6268  ->  2,39x
    trigrama                5,0484 / 10,8589  ->  2,15x

O controlo negativo é obrigatório: sem ele, um rácio > 1 podia ser só o
formato do texto (token uniforme nunca é o mais provável). O aleatório dá
1,01x, o que mostra que a métrica não separa por si — separa quem aprendeu.

Nota importante: o Transformer treinado tem PIOR perda absoluta que o
trigrama (5,70 vs 5,05 dentro), mas MELHOR separação (2,39x vs 2,15x).
São duas propriedades diferentes; perder numa não implica perder na outra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
MEDICAO = RAIZ / "nucleo" / "experimentos" / "resultado_incerteza.json"


@pytest.fixture(scope="module")
def medicao() -> dict:
    if not MEDICAO.exists():
        pytest.skip("correr nucleo.avaliacao.incerteza")
    return json.loads(MEDICAO.read_text())


def test_controlo_negativo_nao_separa(medicao: dict) -> None:
    """Se o modelo ALEATÓRIO separasse, a métrica estaria a medir o formato."""
    razao = medicao["transformer_aleatorio"]["razao"]
    assert razao < 1.10, (
        f"o controlo negativo separa {razao:.2f}x — a métrica mede formato de "
        "texto, não conhecimento, e o resultado do modelo treinado não vale")


def test_modelo_treinado_separa(medicao: dict) -> None:
    treinado = medicao["transformer_treinado"]
    assert treinado is not None, "sem checkpoint: medir o treinado é o ponto todo"
    assert treinado["razao"] > 1.5, \
        f"separação de apenas {treinado['razao']:.2f}x — não sustenta «sabe que não sabe»"


def test_separacao_do_treinado_supera_a_do_aleatorio(medicao: dict) -> None:
    """A comparação que dá sentido ao número: treinado vs aleatório."""
    t = medicao["transformer_treinado"]["razao"]
    a = medicao["transformer_aleatorio"]["razao"]
    assert t > a * 2, f"treinado {t:.2f}x não é o dobro do aleatório {a:.2f}x"


def test_treinado_perde_em_perda_absoluta_mas_ganha_em_separacao(medicao: dict) -> None:
    """A distinção que impede a leitura errada do resultado.

    Seria fácil escrever «o Transformer bate o trigrama» olhando só para a
    separação. Olhando para a perda dentro do domínio, o trigrama é melhor.
    As duas frases são verdadeiras ao mesmo tempo; só uma é útil conforme a
    pergunta. O teste fixa ambas para que nenhuma seja citada sozinha.
    """
    t = medicao["transformer_treinado"]
    n = medicao["trigrama"]

    assert t["dentro"] > n["dentro"], \
        "afirmação mudou: o Transformer passou a ter melhor perda dentro do domínio"
    assert t["razao"] > n["razao"], \
        "afirmação mudou: o trigrama passou a separar melhor"
