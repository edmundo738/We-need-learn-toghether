"""Memorização: o que o modelo copia, medido a várias granularidades.

PORQUE ISSO EXISTE
------------------
O relatório anterior afirmou «0 % de 20-gramas coincidentes ⇒ o modelo não
memoriza». A crítica rejeitou-o com razão: **zero 20-gramas não prova
ausência de memorização**. Um modelo pode reproduzir uma frase inteira e
falhar um 20-grama por uma palavra trocada no meio; e nada diz sobre
5-gramas, 10-gramas ou cópia aproximada.

Aqui a memorização mede-se como deve ser — a várias ordens — e reporta-se
cada uma separadamente, sem as somar numa conclusão única.

MÉTRICAS
--------
1. `copiar_exato(texto, corpus, n)`: a maior sequência de n tokens
   consecutivos da geração que existe literalmente no corpus de treino.
2. `sobreposicao_aproximada(texto, corpus, n)`: a maior sequência cujo
   conjunto de tokens existe no corpus a menos de `tolerancia` diferenças.
   Apanha o caso que a cópia exacta perde — uma palavra trocada no meio.
3. `distribuicao_de_n(task)`: para n em {5,10,15,20}, a fracção de n-gramas
   da geração que ocorrem no corpus. É a curva, não um ponto.

O que isto NÃO mede: memorização em forma semântica (paráfrase). Para isso
seria preciso um modelo de embedding, que não temos e não é honesto fingir.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResultadoMemorizacao:
    ordem: int
    ngramas_gerados: int
    ngramas_no_corpus: int

    @property
    def fraccao(self) -> float:
        return self.ngramas_no_corpus / self.ngramas_gerados if self.ngramas_gerados else 0.0


def distribuicao(tokens_gerados: list[int], tokens_corpus: list[int],
                 ordens: tuple[int, ...] = (5, 10, 15, 20)) -> list[ResultadoMemorizacao]:
    """Para cada ordem: quantos n-gramas da geração existem no corpus."""
    resultados = []
    for o in ordens:
        gramas_corpus = {tuple(tokens_corpus[i:i + o])
                         for i in range(max(0, len(tokens_corpus) - o + 1))}
        gerados = [tuple(tokens_gerados[i:i + o])
                   for i in range(max(0, len(tokens_gerados) - o + 1))]
        no_corpus = sum(1 for g in gerados if g in gramas_corpus)
        resultados.append(ResultadoMemorizacao(o, len(gerados), no_corpus))
    return resultados


def maior_copia_exata(tokens_gerados: list[int], tokens_corpus: list[int],
                      ordem_min: int = 4) -> tuple[int, list[int]]:
    """A maior sequência contígua da geração que existe no corpus.

    Devolve (comprimento, sequência). Procura por ordem decrescente e pára
    na primeira que encontrar.

    Implementado com TUPLOS e não com strings: a primeira versão juntava os
    tokens numa string separada por vírgulas e a comparação de substring
    dava falso negativo em tokens de dois dígitos («29,30» casa dentro de
    «129,30»), que é uma forma silenciosa de o teste mentir.
    """
    for n in range(min(len(tokens_gerados), 512), ordem_min - 1, -1):
        gramas = {tuple(tokens_corpus[i:i + n])
                  for i in range(len(tokens_corpus) - n + 1)}
        for i in range(len(tokens_gerados) - n + 1):
            trecho = tuple(tokens_gerados[i:i + n])
            if trecho in gramas:
                return n, list(trecho)
    return 0, []


def sobreposicao_aproximada(tokens_gerados: list[int], tokens_corpus: list[int],
                            n: int = 10, tolerancia: int = 2) -> int:
    """Maior janela de n tokens que coincide com o corpus excepto <= tolerancia.

    Mede o caso que a cópia exacta não apanha: «o modelo lembra-se da frase
    mas trocou uma palavra». Compara por distância de Hamming sobre n-gramas.
    """
    if len(tokens_gerados) < n or len(tokens_corpus) < n:
        return 0
    corpus_set = {tuple(tokens_corpus[i:i + n])
                  for i in range(len(tokens_corpus) - n + 1)}
    if not corpus_set:
        return 0
    for i in range(len(tokens_gerados) - n + 1):
        janela = tuple(tokens_gerados[i:i + n])
        if janela in corpus_set:
            return n
    # busca aproximada: quantas posições diferem de algum n-grama do corpus
    melhor = 0
    for i in range(len(tokens_gerados) - n + 1):
        janela = tokens_gerados[i:i + n]
        for c in corpus_set:
            dif = sum(1 for a, b in zip(janela, c) if a != b)
            if dif <= tolerancia:
                melhor = n
                return melhor
        if melhor:
            break
    return melhor
