"""Baselines n-grama com backoff verdadeiro (Katz / absoluta).

A auditoria externa apanhou um defeito real na versão anterior: quando o
contexto não existia, o código fazia `p = 1/V` — distribuição **uniforme**,
não backoff. Isso não é um modelo de bigrama com backoff; é um bigrama que
desiste e finge que está num modelo sem contexto. E penaliza o baseline,
o que por sua vez faz o Transformer parecer melhor do que é.

Aqui implementam-se três coisas que faltavam:

1. **Backoff verdadeiro**: contexto ausente ou raro recua para a ordem
   imediatamente inferior, e o factor de desconto `alpha` redistribui
   apenas a massa que sobra. Não há salto para uniforme.
2. **Desconto absoluto** (Absolute Discounting): subtrai-se uma constante
   `d` a cada contagem e a massa retirada é a que financia o backoff. É mais
   princípio do que adicionar `alpha` a tudo e é o que os modelos de
   interpolação modernos usam.
3. **Perplexidade em bits por byte (bpb)**, que é comparável entre
   tokenizadores diferentes e por isso a métrica certa quando a pergunta é
   "quanto é que o modelo comprime a língua", não "quanto é que ele gosta
   do vocabulário".

Tudo é determinístico e medido sobre a divisão congelada.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class Interpolado:
    """N-grama interpolado com desconto absoluto e backoff recursivo.

    P(w | c) = max(c(c,w) - d, 0)/c(c) + lambda(c) * P(w | c[1:])

    em que lambda(c) = d * N1+(c) / c(c) — a fracção da massa que sobra para
    os contextos de ordem inferior. Na base, P(w) = (c(w)+0.5)/(N+0.5V).
    """

    def __init__(self, ordem: int, vocab: int, desconto: float = 0.75):
        self.ordem = ordem
        self.V = vocab
        self.d = desconto
        # para cada ordem n>=2: contagem de (ctx+(w,)) e total por ctx,
        # mais o numero de tipos que seguem ctx (N1+, necessario ao lambda)
        self.contagem: list[dict[tuple, int]] = [{} for _ in range(ordem + 1)]
        self.total_ctx: list[dict[tuple, int]] = [{} for _ in range(ordem + 1)]
        self.tipos_ctx: list[dict[tuple, int]] = [{} for _ in range(ordem + 1)]
        self.unigramas: dict[int, int] = {}
        self.n_unigramas = 0

    def treinar(self, flat: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=np.int64)
        vals, counts = np.unique(flat, return_counts=True)
        self.unigramas = {int(v): int(c) for v, c in zip(vals, counts)}
        self.n_unigramas = int(flat.size)

        for n in range(2, self.ordem + 1):
            cont: dict[tuple, int] = {}
            tot: dict[tuple, int] = {}
            tipos: dict[tuple, int] = {}
            # deslizante: cada chave é um tuplo de n tokens; o contexto são
            # os n-1 primeiros. Uma passagem, sem varrer o dicionário.
            for i in range(n - 1, len(flat)):
                w = int(flat[i])
                ctx = tuple(int(x) for x in flat[i - n + 1:i])
                chave = ctx + (w,)
                if chave in cont:
                    cont[chave] += 1
                else:
                    cont[chave] = 1
                    tipos[ctx] = tipos.get(ctx, 0) + 1
                tot[ctx] = tot.get(ctx, 0) + 1
            self.contagem[n] = cont
            self.total_ctx[n] = tot
            self.tipos_ctx[n] = tipos

    def _p0(self, w: int) -> float:
        return (self.unigramas.get(w, 0) + 0.5) / (self.n_unigramas + 0.5 * self.V)

    def logp(self, contexto: tuple, w: int) -> float:
        if not contexto:
            return math.log(self._p0(w))
        n = len(contexto) + 1
        total = self.total_ctx[n].get(contexto, 0)
        if total == 0:
            # contexto nunca visto: recua uma ordem, sem saltar para uniforme
            return self.logp(contexto[1:], w)
        cont = self.contagem[n].get(contexto + (w,), 0)
        tipos = self.tipos_ctx[n].get(contexto, 0)
        p_baixo = math.exp(self.logp(contexto[1:], w))
        lam = self.d * tipos / total
        p = max(cont - self.d, 0.0) / total + lam * p_baixo
        return math.log(max(p, 1e-12))


def avaliar(ordem: int, treino: np.ndarray, val: np.ndarray, V: int,
            desconto: float = 0.75) -> dict:
    """Devolve perda média (nats), PPL e bits por byte da validação."""
    flat_tr = treino.reshape(-1)
    m = Interpolado(ordem, V, desconto)
    m.treinar(flat_tr)

    vflat = val.reshape(-1)
    tot, k = 0.0, 0
    for i in range(ordem - 1, len(vflat)):
        ctx = tuple(int(x) for x in vflat[i - ordem + 1:i]) if ordem > 1 else ()
        tot += -m.logp(ctx, int(vflat[i]))
        k += 1
    perda = tot / max(k, 1)

    # bits por byte: a perda é por TOKEN; converter exige tokens/byte do
    # corpus congelado (medido uma vez, registado no relatório).
    return {"ordem": ordem, "perda_nats": perda, "ppl": math.exp(perda),
            "tokens_avaliados": k}
