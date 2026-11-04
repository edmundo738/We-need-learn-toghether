"""Testes dos baselines n-grama.

A auditoria externa mostrou que a versão anterior devolvia `1/V` quando o
contexto não existia, chamando a isso "backoff". Não era: era um salto para
uniforme, que penaliza o baseline e faz o Transformer parecer melhor.
Estes testes existem para que isso não volte a passar sem se notar.
"""

from __future__ import annotations

import math

import numpy as np

from nucleo.experimentos.baselines import Interpolado, avaliar


def test_backoff_nao_e_uniforme():
    """Um contexto nunca visto tem de recuar ao unigrama, não dar 1/V.

    É o teste central. Com um corpus em que o token 4 é raríssimo, a
    probabilidade de 4 depois de um contexto inexistente tem de ser a
    probabilidade unigrama de 4 — não 1/V, que seria muito maior.
    """
    tr = np.array([1, 2, 3] * 60 + [4], dtype=np.int64)
    m = Interpolado(2, vocab=50)
    m.treinar(tr)
    p = math.exp(m.logp((9,), 4))        # contexto (9,) nunca ocorreu
    unigrama = (1 + 0.5) / (len(tr) + 0.5 * 50)
    assert abs(p - unigrama) < 1e-9, f"{p} != {unigrama}"
    assert abs(p - 1 / 50) > 1e-3, "caiu no uniforme"


def test_backoff_recursivo_ordem_superior():
    """Com trigrama, o recuo passa por bigrama antes de chegar ao unigrama."""
    tr = np.array([1, 2, 3] * 40, dtype=np.int64)
    m = Interpolado(3, vocab=20)
    m.treinar(tr)
    # (1,2,?) ocorreu sempre; (2,1) nunca ocorreu como contexto
    assert m.logp((1, 2), 3) > m.logp((5, 2), 3)


def test_desconto_absoluto_preserva_probabilidade():
    """A massa descontada vai toda para o backoff: a soma não explode."""
    tr = np.array([1, 2, 1, 3] * 30, dtype=np.int64)
    m = Interpolado(2, vocab=10)
    m.treinar(tr)
    total = sum(math.exp(m.logp((1,), w)) for w in range(10))
    assert 0.9 < total < 1.1, f"soma das probabilidades = {total}"


def test_bigrama_aprende_transicao():
    """A cadeia 1->2 tem de ficar quase determinística depois de treinada."""
    tr = np.array([1, 2] * 100, dtype=np.int64)
    m = Interpolado(2, vocab=5)
    m.treinar(tr)
    assert math.exp(m.logp((1,), 2)) > 0.9
    assert math.exp(m.logp((1,), 3)) < 0.01


def test_ordem_superior_nao_e_pior_que_unigrama_em_dados_repetitivos():
    """Numa sequência perfeitamente previsível, mais ordem = menos perda."""
    tr = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 100)
    va = tr.reshape(1, -1)
    l1 = avaliar(1, tr.reshape(1, -1), va, 10)["perda_nats"]
    l2 = avaliar(2, tr.reshape(1, -1), va, 10)["perda_nats"]
    l3 = avaliar(3, tr.reshape(1, -1), va, 10)["perda_nats"]
    assert l2 < l1, (l2, l1)
    assert l3 < l2, (l3, l2)


def test_desconto_invalido_rejeitado():
    """Um desconto >= 1 tornaria a massa negativa em contextos raros."""
    m = Interpolado(2, vocab=10, desconto=0.75)
    assert 0 <= m.d < 1
    tr = np.array([1, 2, 3, 1, 2], dtype=np.int64)
    m.treinar(tr)
    assert all(math.isfinite(m.logp((1,), w)) for w in range(10))
