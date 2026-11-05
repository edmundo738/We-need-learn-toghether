"""Testes da comparação Transformer vs trigrama.

O que está a ser protegido aqui não é código: é a possibilidade de o resultado
ser auto-enganador. Um baseline mal implementado (salto para uniforme em vez
de backoff) deu 3,9094 bpb em vez de 2,4788 e fez o Transformer "ganhar" por
1,23 bits/byte. Isso aconteceu de facto nesta sessão e foi apanhado por
inspecção manual, não por teste. Agora é apanhado por teste.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "nucleo" / "experimentos" / "resultado_comparacao_final.json"


def test_baseline_usa_desconto_absoluto_nao_laplace() -> None:
    """A comparação tem de usar `Interpolado`, não uma suavização inventada.

    Sobrevive a este teste qualquer implementação que não salte para uniforme
    quando o contexto é novo. A verificação concreta: num contexto nunca
    visto, a probabilidade atribuída a um token raro tem de ser MAIOR do que
    1/V (o backoff redistribui massa para o que o unigrama conhece), enquanto
    com salto para uniforme seria exactamente 1/V.
    """
    from nucleo.experimentos.baselines import Interpolado

    m = Interpolado(3, 100, 0.75)
    # treino minúsculo: "1 2 3" repetido, e um unigrama raro noutro contexto
    m.treinar(np.array([1, 2, 3, 1, 2, 3, 5, 6], dtype=np.int64))

    # contexto que NUNCA apareceu
    p = m.logp((90, 91), 1)
    assert p < 0, "logp tem de ser negativo"
    # com uniforme daria log(1/100) = -4.605
    assert p > np.log(1 / 100), \
        "contexto novo caiu para uniforme: é o baseline sabotado, não backoff"


def test_resultado_comparacao_existe_e_e_coerente() -> None:
    if not RES.exists():
        pytest.skip("comparação ainda não corrida")
    d = json.loads(RES.read_text())

    assert d["classe"] == "MEASURED"
    assert "desconto" in d["trigrama"]["suavizacao"].lower(), \
        "o baseline registado não é desconto absoluto"

    t, r = d["trigrama"]["bits_por_byte"], d["transformer"]["bits_por_byte"]
    assert d["diferenca_bpb"] == pytest.approx(r - t, abs=1e-3)
    assert d["vencedor"] == ("trigrama" if r > t else "transformer")

    # coerência com o número histórico: se o trigrama passar a dar ~3,9, algo
    # partiu no baseline outra vez
    assert 2.3 < t < 2.7, f"trigrama fora do intervalo esperado: {t}"


def test_bits_por_byte_e_conversao_de_nats() -> None:
    """Uma perda em nats só vira bits/byte pela função medida, nunca à mão."""
    from nucleo.experimentos.metricas import (nats_para_bits_por_byte,
                                               tokens_por_byte)

    tpb = tokens_por_byte()
    assert 0.30 < tpb < 0.33, f"tokens/byte implausível: {tpb}"

    # Acaso com vocabulário 8192: perda = log(8192) nats POR TOKEN
    #   = log2(8192) = 13 bits por token
    # Um token vale 1/tpb bytes = 1/0,313277 = 3,192 bytes
    #   => 13 bits / 3,192 bytes = 4,0726 bits/byte
    # Confundir bits/token com bits/byte dá 13, que é 3,19x demasiado alto —
    # esse foi o erro que este teste apanhou na primeira versão.
    acaso = nats_para_bits_por_byte(float(np.log(8192)))
    esperado = 13.0 * tpb
    assert acaso == pytest.approx(esperado, rel=1e-9), \
        f"acaso convertido errado: {acaso} vs {esperado}"
    assert 4.0 < acaso < 4.15, f"acaso fora do intervalo: {acaso}"
