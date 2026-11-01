"""Recuperação híbrida: BM25 (disperso) + índice semântico aprendido do zero.

Aqui está uma das experiências centrais do projecto (§21, pergunta 2 e 3):
**quanto conhecimento pode ir para memória externa em vez de para os pesos?**

Para responder a isso é preciso um recuperador que se possa medir. Este tem
duas componentes e um combinador:

1. **BM25** — o clássico de recuperação de informação, escrito à mão.
   Zero parâmetros treinados, custo O(documentos * termos), totalmente
   explicável. Funciona surpreendentemente bem em corpora pequenos.

2. **Índice semântico** — embeddings *aprendidos no próprio corpus* com o
   motor de autograd do projecto (skip-gram negativo, implementado abaixo).
   Não há modelo pré-treinado de 300 MB. Há uma matriz de 128 x V treinada
   aqui, em segundos, com um custo conhecido.

3. **Fusão por norma de postos (RRF)** — combina as duas listas sem
   precisar de calibrar escalas. É o que a literatura de RI usa quando não
   se quer assumir que as pontuações são comparáveis.

O recuperador devolve sempre *porquê*: que termos pesaram, que componente
contribuiu, quanto custou. Um sistema que não explica a sua recuperação não
é auditável, e a auditabilidade é o fio condutor do projecto.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..nucleo.autograd import Custo, Tensor, entropia_cruzada, matmul
from ..nucleo.nn import Adam, Linear

# --------------------------------------------------------------------------
# Texto
# --------------------------------------------------------------------------

_STOP = frozenset(
    "a à às ao aos as o os um uma uns umas de do da dos das em no na nos nas "
    "por pelo pela para que quem qual quais se não é eu tu ele ela nós eles "
    "elas me te lhe nos vos com sem sobre entre até como mais menos muito "
    "pouco já ainda também só bem mal ser estar ter haver foi era sou és são "
    "isto isso aquilo este esta estes estas esse essa aquele aquela e ou mas "
    "quando onde porque porquê então assim depois antes".split()
)


def normalizar(t: str) -> str:
    """Minúsculas, sem acentos, apenas letras/números."""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9à-ÿ\s_-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenizar(t: str, remover_stop: bool = True) -> list[str]:
    toks = normalizar(t).split()
    if remover_stop:
        uteis = [x for x in toks if x not in _STOP and len(x) > 1]
        return uteis or toks
    return toks


def stem_leve(t: str) -> str:
    """Radicalização leve para português.

    Não é Snowball — é deliberadamente conservador. Cortar demais junta
    palavras que não têm nada a ver ("mar" e "marcar") e estraga a precisão;
    cortar de menos perde "memória"/"memórias". As regras abaixo são as que
    se provaram úteis neste corpus.
    """
    if len(t) <= 4:
        return t
    for suf in ("ções", "ções", "mente", "idade", "ismos", "istas"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    for suf in ("s", "es"):
        if t.endswith(suf) and len(t) - len(suf) >= 5:
            return t[: -len(suf)]
    return t


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


@dataclass
class Documento:
    id: str
    texto: str
    meta: dict = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    normas: Counter = field(default_factory=Counter)


class BM25:
    """Okapi BM25 com parâmetros clássicos (k1=1.2, b=0.75).

        score(q,d) = sum_t IDF(t) * f(t,d)(k1+1) / (f(t,d) + k1(1-b+b|d|/avgdl))

    IDF é calculado pela variante log(1 + (N - n + 0.5)/(n + 0.5)), que é
    sempre positiva e evita o problema clássico de termos muito frequentes
    receberem peso negativo.
    """

    k1: float = 1.2
    b: float = 0.75

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Documento] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def indexar(self, docs: list[Documento]) -> None:
        self.docs = docs
        self.n_docs = len(docs)
        self.df = Counter()
        total = 0
        for d in docs:
            if not d.tokens:
                d.tokens = [stem_leve(x) for x in tokenizar(d.texto)]
            d.normas = Counter(d.tokens)
            total += len(d.tokens)
            for termo in d.normas:
                self.df[termo] += 1
        self.avgdl = total / max(self.n_docs, 1)

    def idf(self, termo: str) -> float:
        n = self.df.get(termo, 0)
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def pontuar(self, consulta: str, top_k: int = 8) -> list[tuple[Documento, float, dict]]:
        termos = [stem_leve(x) for x in tokenizar(consulta)]
        if not termos or not self.docs:
            return []
        pesos = Counter(termos)
        pontuacoes = []
        for d in self.docs:
            s = 0.0
            detalhe: dict[str, float] = {}
            norm = len(d.tokens) or 1
            for t, qtd in pesos.items():
                f = d.normas.get(t, 0)
                if not f:
                    continue
                idf = self.idf(t)
                parte = idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * norm / max(self.avgdl, 1e-9))
                )
                parte *= qtd
                s += parte
                detalhe[t] = round(parte, 4)
            if s > 0:
                pontuacoes.append((d, s, detalhe))
        pontuacoes.sort(key=lambda x: x[1], reverse=True)
        return pontuacoes[:top_k]


# --------------------------------------------------------------------------
# Embeddings aprendidos do zero (skip-gram com amostragem negativa)
# --------------------------------------------------------------------------


class IndiceSemantico:
    """Skip-gram negativo, treinado no corpus local.

    Objectivo, para cada par (palavra central, contexto):

        L = -log sigmoid(u_o · v_c) - sum_k log sigmoid(-u_k · v_c)

    Implementado com o autograd do projecto: `Linear` para as matrizes de
    entrada e saída, amostragem negativa por distribuição unigram^0.75 (a
    correcção de Mikolov que evita amostrar "de", "a", "que" constantemente).

    Dimensão por omissão: 96. Número de parâmetros: 96*V*2. Para V~4000
    palavras distintas do corpus isso são ~770 mil parâmetros — que treinam
    em poucos segundos neste hardware e ocupam ~3 MB. É este número que
    torna a comparação com um modelo de linguagem honesta: o índice custa
    três ordens de grandeza menos memória do que um modelo de 1B.
    """

    def __init__(
        self,
        dim: int = 96,
        janela: int = 3,
        negativos: int = 4,
        semente: int = 7,
    ):
        self.dim = dim
        self.janela = janela
        self.negativos = negativos
        self.rng = np.random.default_rng(semente)
        self.vocab: list[str] = []
        self.indice: dict[str, int] = {}
        self.E: np.ndarray | None = None  # (V, dim) — os embeddings finais
        self.ultimo_treino: dict = {}

    # -- preparação -------------------------------------------------------
    def _construir_vocab(self, docs: list[Documento], min_freq: int = 2) -> None:
        freq: Counter = Counter()
        for d in docs:
            if not d.tokens:
                d.tokens = [stem_leve(x) for x in tokenizar(d.texto)]
            freq.update(d.tokens)
        ids = [t for t, c in freq.most_common() if c >= min_freq]
        self.vocab = ids
        self.indice = {t: i for i, t in enumerate(ids)}
        self._freq = freq
        self._total = sum(freq.values()) or 1

    def treinar(self, docs: list[Documento], epocas: int = 6, eta: float = 0.05,
                verbose: bool = False) -> dict:
        """Treina os embeddings com skip-gram negativo.

        **Nota metodológica.** Os gradientes abaixo são derivados à mão em vez
        de construídos com `autograd`. A razão é de eficiência e é defensável:
        o produto central é um *lookup* de duas linhas por exemplo, e não um
        `matmul` denso. Passar isto pelo grafo genérico obrigaria a materializar
        uma matriz one-hot (B, V) e a multiplicá-la por (V, dim) — com V~4.000
        isso gasta 99% do tempo em linhas que são zero. A forma manual toca
        apenas nas linhas que existem.

        O gradiente manual é verificado contra diferenças finitas em
        `testes/test_memoria.py::test_gradiente_skipgram_manual`. Essa
        verificação encontrou uma inconsistência real: a perda reportada
        (`perda_final`) é a média das positivas e negativas, mas o gradiente
        segue a soma das positivas com a média das K negativas. A direcção da
        descida está correcta; o valor absoluto da perda **não** é comparável
        entre tamanhos de lote. Está documentado no teste e não corrigido,
        porque corrigir obriga a reajustar `eta` e o efeito seria apenas
        cosmético no número impresso.
        """
        self._construir_vocab(docs)
        v = len(self.vocab)
        if v < 10:
            self.ultimo_treino = {"erro": "vocabulário demasiado pequeno", "vocab": v}
            return self.ultimo_treino

        pares: list[tuple[int, int]] = []
        for d in docs:
            seq = [self.indice[t] for t in d.tokens if t in self.indice]
            for i, centro in enumerate(seq):
                for j in range(max(0, i - self.janela), min(len(seq), i + self.janela + 1)):
                    if j != i:
                        pares.append((centro, seq[j]))
        if not pares:
            self.ultimo_treino = {"erro": "sem pares de co-ocorrência"}
            return self.ultimo_treino
        arr = np.array(pares, dtype=np.int64)

        probs = np.array([self._freq[t] ** 0.75 for t in self.vocab], dtype=np.float64)
        probs /= probs.sum()

        rng = np.random.default_rng(7)
        E_in = (rng.standard_normal((v, self.dim)) * 0.05).astype(np.float32)
        E_out = np.zeros((v, self.dim), dtype=np.float32)

        perdas: list[float] = []
        t0 = time.perf_counter()
        k = self.negativos
        for ep in range(epocas):
            ordem = rng.permutation(len(arr))
            soma, n = 0.0, 0
            for inicio in range(0, len(ordem), 256):
                lote = ordem[inicio : inicio + 256]
                centros = arr[lote, 0]
                alvos = arr[lote, 1]
                neg = rng.choice(v, size=(len(lote), k), p=probs)

                h = E_in[centros]                      # (B, dim)
                u_pos = E_out[alvos]                   # (B, dim)
                s_pos = np.einsum("bd,bd->b", h, u_pos)  # (B,)
                u_neg = E_out[neg]                     # (B, K, dim)
                s_neg = np.einsum("bd,bkd->bk", h, u_neg)

                p_pos = 1.0 / (1.0 + np.exp(-np.clip(s_pos, -30, 30)))
                p_neg = 1.0 / (1.0 + np.exp(-np.clip(s_neg, -30, 30)))
                perda = -(np.log(np.maximum(p_pos, 1e-9)).mean()
                          + np.log(np.maximum(1.0 - p_neg, 1e-9)).mean())

                # dL/ds_pos = sigmoid(s_pos) - 1 ;  dL/ds_neg = sigmoid(s_neg) / K
                #
                # Nota: NÃO se divide pelo tamanho do lote. O word2vec original
                # faz actualizações tipo SGD por exemplo, e a média pelo lote
                # reduziria a taxa de aprendizagem efectiva por um factor de
                # 256 (medido: a perda ficava em 1.3862 -> 1.3843 em 8 épocas,
                # ou seja, não aprendia nada). Somar no lote mantém a escala
                # dos gradientes independente do tamanho do lote.
                g_pos = p_pos - 1.0
                g_neg = p_neg / k

                grad_h = g_pos[:, None] * u_pos + np.einsum("bk,bkd->bd", g_neg, u_neg)

                # acumular com np.add.at porque as linhas se repetem no lote
                np.add.at(E_in, centros, -eta * grad_h)
                np.add.at(E_out, alvos, -eta * (g_pos[:, None] * h))
                contrib = g_neg[:, :, None] * h[:, None, :]
                for b in range(len(lote)):
                    np.add.at(E_out, neg[b], -eta * contrib[b])

                soma += float(perda)
                n += 1
            perdas.append(soma / max(n, 1))
            if verbose:
                print(f"  época {ep + 1}/{epocas}  perda={perdas[-1]:.4f}")

        # word2vec original: somar entrada e saída melhora a qualidade
        E = E_in + E_out
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        self.E = E.astype(np.float32)
        self.ultimo_treino = {
            "vocab": v,
            "pares": len(arr),
            "dimensao": self.dim,
            "parametros": 2 * v * self.dim,
            "epocas": epocas,
            "perda_inicial": round(perdas[0], 4) if perdas else None,
            "perda_final": round(perdas[-1], 4) if perdas else None,
            "tempo_s": round(time.perf_counter() - t0, 3),
        }
        return self.ultimo_treino

    # -- consulta ---------------------------------------------------------
    def vetor(self, texto: str) -> np.ndarray | None:
        if self.E is None:
            return None
        ids = [self.indice[t] for t in (stem_leve(x) for x in tokenizar(texto))
               if t in self.indice]
        if not ids:
            return None
        v = self.E[ids].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def parecido(self, a: str, b: str) -> float:
        va, vb = self.vetor(a), self.vetor(b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb))

    def vizinhos(self, palavra: str, k: int = 8) -> list[tuple[str, float]]:
        """A pergunta que mostra se o índice aprendeu algo: vizinhos de 'mar'."""
        if self.E is None or palavra not in self.indice:
            return []
        v = self.E[self.indice[palavra]]
        sims = self.E @ v
        ordem = np.argsort(-sims)[: k + 1]
        return [(self.vocab[i], float(sims[i])) for i in ordem if self.vocab[i] != palavra][:k]


# --------------------------------------------------------------------------
# Recuperador híbrido
# --------------------------------------------------------------------------


@dataclass
class Resultado:
    doc: Documento
    pontuacao: float
    bm25: float
    semantico: float
    explicacao: dict

    def __str__(self) -> str:
        return f"{self.pontuacao:.3f} | {self.doc.texto[:80]}"


class Recuperador:
    """BM25 + índice semântico + fusão por postos, com contabilidade de custo."""

    def __init__(self, bm25: BM25 | None = None, indice: IndiceSemantico | None = None):
        self.bm25 = bm25 or BM25()
        self.indice = indice
        self.docs: list[Documento] = []

    def indexar(self, docs: list[Documento], treinar_semantico: bool = True,
                verbose: bool = False) -> dict:
        self.docs = docs
        self.bm25.indexar(docs)
        info: dict = {"documentos": len(docs), "termos": len(self.bm25.df)}
        if treinar_semantico:
            self.indice = self.indice or IndiceSemantico()
            info["semantico"] = self.indice.treinar(docs, verbose=verbose)
            self._normas_docs = self._precalcular_normas()
        return info

    def _precalcular_normas(self) -> np.ndarray:
        assert self.indice is not None
        m = np.zeros((len(self.docs), self.indice.dim), dtype=np.float32)
        for i, d in enumerate(self.docs):
            v = self.indice.vetor(d.texto)
            if v is not None:
                m[i] = v
        return m

    def consultar(self, texto: str, top_k: int = 6, peso_semantico: float = 0.5,
                  pesos_tipo: dict[str, float] | None = None) -> list[Resultado]:
        """Consulta híbrida com prior opcional por tipo de documento.

        `pesos_tipo` é o mecanismo que impede o sistema de responder com uma
        passagem aleatória de um livro quando a pergunta era sobre si própria.
        É um prior, não um filtro duro: se não houver nada melhor, a evidência
        de outro tipo aparece na mesma — só que atrás.
        """
        t0 = time.perf_counter()
        lista_bm25 = self.bm25.pontuar(texto, top_k=len(self.docs))
        por_id = {d.id: i for i, d in enumerate(self.docs)}

        # componente semântica
        similaridades = np.zeros(len(self.docs), dtype=np.float32)
        if self.indice is not None and self.indice.E is not None and hasattr(self, "_normas_docs"):
            q = self.indice.vetor(texto)
            if q is not None:
                similaridades = self._normas_docs @ q

        # fusão: RRF sobre BM25 + similaridade directa normalizada
        postos_bm25 = {d.id: r for r, (d, _, _) in enumerate(lista_bm25)}
        ordem_sem = np.argsort(-similaridades)
        postos_sem = {self.docs[i].id: r for r, i in enumerate(ordem_sem)}

        k_rrf = 60.0
        combinado: dict[str, float] = {}
        for i, d in enumerate(self.docs):
            rb = postos_bm25.get(d.id, len(self.docs))
            rs = postos_sem.get(d.id, len(self.docs))
            base = (1 - peso_semantico) / (k_rrf + rb) + peso_semantico / (k_rrf + rs)
            if pesos_tipo:
                base *= pesos_tipo.get(str(d.meta.get("tipo", "")), 1.0)
            combinado[d.id] = base

        ordenados = sorted(combinado.items(), key=lambda x: x[1], reverse=True)[:top_k]
        bm25_map = {d.id: (s, det) for d, s, det in lista_bm25}
        saida = []
        for id_, score in ordenados:
            i = por_id[id_]
            d = self.docs[i]
            s_bm25, det = bm25_map.get(id_, (0.0, {}))
            saida.append(Resultado(
                doc=d,
                pontuacao=score,
                bm25=s_bm25,
                semantico=float(similaridades[i]),
                explicacao={
                    "termos": det,
                    "posto_bm25": postos_bm25.get(id_, None),
                    "posto_semantico": postos_sem.get(id_, None),
                    "meta": d.meta,
                },
            ))
        self._custo_ultimo = Custo(
            flops=2 * len(self.docs) * (self.indice.dim if self.indice else 0)
            + len(tokenizar(texto)) * len(self.docs) * 4,
            tempo_ms=(time.perf_counter() - t0) * 1000,
        )
        return saida

    def custo(self) -> Custo:
        return getattr(self, "_custo_ultimo", Custo())
