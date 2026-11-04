"""Testes da medição de memorização."""

from __future__ import annotations

from nucleo.avaliacao.memorizacao import (
    distribuicao,
    maior_copia_exata,
    sobreposicao_aproximada,
)


def test_copia_exata_encontra_trecho_literal():
    corpus = list(range(100))
    gerado = [999, 998] + list(range(10, 30)) + [997, 996]
    n, trecho = maior_copia_exata(gerado, corpus)
    assert n == 20
    assert trecho == list(range(10, 30))


def test_copia_exata_devolve_zero_quando_nao_ha():
    corpus = list(range(100))
    gerado = [500, 501, 502, 503, 504, 505]
    n, trecho = maior_copia_exata(gerado, corpus)
    assert n == 0


def test_copia_exata_detecta_uma_so_palavra_trocada_como_copia_curta():
    """Uma sequência com uma troca NÃO é cópia exacta longa — e isso é o
    ponto: é por isso que só a cópia exacta não basta."""
    corpus = list(range(100))
    gerado = list(range(10, 30))
    gerado[5] = 777          # uma troca no meio
    n, _ = maior_copia_exata(gerado, corpus)
    assert n < 20            # perde a janela toda por causa de um token


def test_sobreposicao_aproximada_apanha_o_que_a_exata_perde():
    """O mesmo caso, com a métrica aproximada: tem de dar 10."""
    corpus = list(range(100))
    gerado = list(range(10, 30))
    gerado[12] = 777
    assert maior_copia_exata(gerado, corpus)[0] < 20
    assert sobreposicao_aproximada(gerado, corpus, n=10, tolerancia=2) == 10


def test_distribuicao_por_ordem():
    corpus = list(range(50))
    gerado = list(range(10, 40))
    r = distribuicao(gerado, corpus, ordens=(5, 10, 15, 20))
    assert [x.ordem for x in r] == [5, 10, 15, 20]
    assert all(x.fraccao == 1.0 for x in r)


def test_distribuicao_com_geracao_alheia():
    corpus = list(range(50))
    gerado = [900, 901, 902, 903, 904, 905, 906, 907, 908, 909]
    r = distribuicao(gerado, corpus, ordens=(5, 10))
    assert all(x.fraccao == 0.0 for x in r)


def test_ordens_diferentes_dao_respostas_diferentes():
    """A curva importa: uma geração pode ter 5-gramas mas não 20-gramas.

    É exactamente o erro do relatório anterior — reportar só a ordem 20.
    """
    corpus = list(range(100))
    gerado = list(range(0, 8)) + [888] + list(range(20, 60))
    r = {x.ordem: x.fraccao for x in distribuicao(gerado, corpus, ordens=(5, 10, 20))}
    assert r[5] > r[20], f"esperava mais coincidência a 5 do que a 20: {r}"
