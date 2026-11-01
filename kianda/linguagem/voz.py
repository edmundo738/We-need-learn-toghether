"""A voz da Kianda: gerador de linguagem treinado do zero.

Duas camadas, e a distinção importa:

1. **`GeradorNeuronal`** — um modelo de linguagem de caracteres, treinado no
   corpus do próprio projecto com o autograd de `kianda.nucleo`. Poucos
   milhões de parâmetros, treina em minutos, corre em CPU. É o que garante
   que a Kianda fala mesmo sem internet e sem um modelo gigante. A sua
   qualidade é limitada de propósito — e essa limitação é medida.

2. **`Voz`** — a camada que transforma intenção em texto. Tem dois modos:
   - `modo="local"`: usa o gerador neuronal e templates de estrutura.
   - `modo="remoto"`: usa uma API de linguagem, se houver chave configurada.

A regra do projecto (§18) é separar claramente **o que é nosso** do **que é
externo**. Por isso cada resposta carrega a etiqueta de quem a produziu, e o
benchmark mede as duas em separado. Nunca se apresenta a qualidade de um
modelo externo como se fosse resultado da investigação local.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..nucleo.autograd import Custo, Tensor, entropia_cruzada, matmul
from ..nucleo.nn import Adam


# --------------------------------------------------------------------------
# Modelo de linguagem de caracteres
# --------------------------------------------------------------------------


@dataclass
class ConfigLM:
    contexto: int = 24        # caracteres de contexto
    oculto: int = 128         # dimensão do embedding e da camada oculta
    camadas: int = 2
    vocab_max: int = 96
    eta: float = 3e-3
    lote: int = 48


class GeradorNeuronal:
    """MLP autoregressivo sobre caracteres.

    arquitectura (do zero, com `kianda.nucleo`):

        entrada: C caracteres anteriores -> one-hot (C*V)
        h1 = relu(W1 x + b1)
        h2 = relu(W2 h1 + b2)
        logits = W3 h2 + b3      (V classes)

    Não é um transformer — e isso é uma escolha informada, não uma limitação
    escondida. Um modelo de caracteres com contexto de 24 é suficiente para
    aprender a *forma* da língua (maiúsculas, acentuação, comprimento de
    palavra, pontuação) e insuficiente para coerência longa. O benchmark
    mede exactamente essa diferença em vez de a disfarçar.
    """

    def __init__(self, cfg: ConfigLM | None = None, semente: int = 4):
        self.cfg = cfg or ConfigLM()
        self.rng = np.random.default_rng(semente)
        self.vocab: list[str] = []
        self.indice: dict[str, int] = {}
        self.W1: Tensor | None = None
        self.b1: Tensor | None = None
        self.W2: Tensor | None = None
        self.b2: Tensor | None = None
        self.W3: Tensor | None = None
        self.b3: Tensor | None = None
        self.treinado = False
        self.historico: list[tuple[int, float, float]] = []

    # -- vocabulário ------------------------------------------------------
    def construir_vocab(self, texto: str) -> None:
        contagem: dict[str, int] = {}
        for c in texto:
            contagem[c] = contagem.get(c, 0) + 1
        chars = sorted(contagem, key=lambda c: -contagem[c])[: self.cfg.vocab_max]
        self.vocab = chars
        self.indice = {c: i for i, c in enumerate(chars)}
        self.unc = "\x00"  # unknown
        self.indice[self.unc] = len(self.vocab)
        self.vocab = self.vocab + [self.unc]

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    def _codificar(self, texto: str) -> np.ndarray:
        u = self.indice[self.unc]
        return np.array([self.indice.get(c, u) for c in texto], dtype=np.int64)

    def _lotes(self, ids: np.ndarray, lote: int | None = None):
        C = self.cfg.contexto
        n = len(ids) - C - 1
        if n <= 0:
            return
        ordem = np.arange(n)
        self.rng.shuffle(ordem)
        lote = lote or self.cfg.lote
        for i in range(0, n, lote):
            idx = ordem[i : i + lote]
            X = np.stack([ids[j : j + C] for j in idx])
            y = np.array([ids[j + C] for j in idx], dtype=np.int64)
            yield X, y

    def _um_quente(self, X: np.ndarray) -> np.ndarray:
        B, C = X.shape
        V = self.n_vocab
        out = np.zeros((B, C * V), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                out[b, c * V + X[b, c]] = 1.0
        return out

    # -- treino -----------------------------------------------------------
    def treinar(self, texto: str, epocas: int = 6, verbose: bool = False) -> dict:
        if not self.vocab:
            self.construir_vocab(texto)
        ids = self._codificar(texto)
        d_in = self.cfg.contexto * self.n_vocab
        H, V = self.cfg.oculto, self.n_vocab

        self.W1 = Tensor(self.rng.standard_normal((d_in, H)).astype(np.float32) * math.sqrt(2.0 / d_in), True)
        self.b1 = Tensor(np.zeros((1, H), dtype=np.float32), True)
        self.W2 = Tensor(self.rng.standard_normal((H, H)).astype(np.float32) * math.sqrt(2.0 / H), True)
        self.b2 = Tensor(np.zeros((1, H), dtype=np.float32), True)
        self.W3 = Tensor(self.rng.standard_normal((H, V)).astype(np.float32) * math.sqrt(1.0 / H), True)
        self.b3 = Tensor(np.zeros((1, V), dtype=np.float32), True)

        params = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                  "W3": self.W3, "b3": self.b3}
        opt = Adam(params, eta=self.cfg.eta)
        t0 = time.perf_counter()

        for ep in range(epocas):
            perdas, acertos, total = [], 0, 0
            for X, y in self._lotes(ids):
                xb = Tensor(self._um_quente(X), False)
                h = (matmul(xb, self.W1) + self.b1).relu()
                h = (matmul(h, self.W2) + self.b2).relu()
                logits = matmul(h, self.W3) + self.b3
                perda = entropia_cruzada(logits, y)
                perda.backward()
                opt.step()
                perdas.append(float(perda.data.item()))
                acertos += int(perda.meta.get("acertos", 0))
                total += len(y)
            media = sum(perdas) / max(len(perdas), 1)
            acc = acertos / max(total, 1)
            self.historico.append((ep, media, acc))
            if verbose:
                print(f"  época {ep+1}/{epocas}  perda={media:.3f}  "
                      f"ppl={math.exp(min(media, 20)):.1f}  acc={acc:.3f}")
        self.treinado = True
        return {
            "epocas": epocas,
            "perda_final": round(self.historico[-1][1], 4),
            "perplexidade": round(math.exp(min(self.historico[-1][1], 20)), 2),
            "acc_char": round(self.historico[-1][2], 4),
            "parametros": sum(p.n_elementos for p in params.values()),
            "tempo_s": round(time.perf_counter() - t0, 2),
            "vocab": self.n_vocab,
        }

    # -- geração ----------------------------------------------------------
    def gerar(self, semente: str = "\n", n: int = 160, temperatura: float = 0.85,
              top_k: int = 12, paragem: str | None = "\n\n") -> tuple[str, Custo]:
        assert self.treinado, "modelo não treinado"
        C = self.cfg.contexto
        contexto = (semente or "\n")[-C:].rjust(C, " ")
        ids = self._codificar(contexto).tolist()
        saida: list[str] = []
        custo = Custo()
        for _ in range(n):
            x = np.array(ids[-C:], dtype=np.int64).reshape(1, -1)
            xb = Tensor(self._um_quente(x), False)
            h = (matmul(xb, self.W1) + self.b1).relu()
            h = (matmul(h, self.W2) + self.b2).relu()
            logits = (matmul(h, self.W3) + self.b3)
            custo = custo + logits.custo_graph()
            z = logits.data[0] / max(temperatura, 1e-3)
            if top_k:
                limiar = np.sort(z)[-top_k]
                z = np.where(z >= limiar, z, -1e9)
            e = np.exp(z - z.max())
            p = e / e.sum()
            escolha = int(self.rng.choice(len(p), p=p))
            ch = self.vocab[escolha]
            ids.append(escolha)
            saida.append(ch)
            if paragem and "".join(saida).endswith(paragem):
                break
        return "".join(saida).strip(), custo


# --------------------------------------------------------------------------
# Voz — a camada que produz texto com intenção
# --------------------------------------------------------------------------


@dataclass
class Resposta:
    texto: str
    autor: str                # "kianda-local" | "kianda-remoto" | "estrutura"
    custo: Custo = field(default_factory=Custo)
    plano: dict = field(default_factory=dict)
    fontes: list[str] = field(default_factory=list)
    confianca: float = 0.6

    def __str__(self) -> str:
        return self.texto


# marcas de personalidade: NÃO são uma lista de adjectivos (§8) —
# são operadores sobre a estrutura da frase. Cada um muda algo observável.
MARCA = {
    "incerteza": ("Não sei.", "Talvez.", "Posso estar errada, mas"),
    "discordancia": ("Não concordo.", "Discordo.", "Aqui não te acompanho."),
    "curiosidade": ("E tu?", "Porquê?", "O que te fez pensar nisso?"),
    "correccao": ("Estava errada.", "Corrijo-me."),
    "limite": ("Isso não faço.", "Aqui paro."),
}


def _limpar(texto: str) -> str:
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


class Voz:
    """Produz respostas. Recebe plano de comportamento + contexto de memória."""

    def __init__(self, gerador: GeradorNeuronal | None = None, modo: str = "local"):
        self.gerador = gerador
        self.modo = modo
        self.ultima: Resposta | None = None

    # -- modo remoto (opcional) ------------------------------------------
    def _remoto(self, sistema: str, historico: list[dict], max_tokens: int,
                temperatura: float) -> tuple[str, Custo] | None:
        """Chamada opcional a uma API de linguagem.

        Só corre se `KIANDA_LLM=1` e existir uma chave. Nunca é o caminho por
        omissão: a Kianda tem de funcionar inteira sem depender de terceiros,
        e qualquer resultado remoto aparece etiquetado como tal.
        """
        if self.modo != "remoto" or os.environ.get("KIANDA_LLM") != "1":
            return None
        chave = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not chave:
            return None
        try:
            import json
            import urllib.request

            if os.environ.get("OPENAI_API_KEY"):
                url = "https://api.openai.com/v1/chat/completions"
                corpo = {
                    "model": os.environ.get("KIANDA_MODELO", "gpt-4o-mini"),
                    "messages": [{"role": "system", "content": sistema}] + historico,
                    "max_tokens": max_tokens,
                    "temperature": temperatura,
                }
                req = urllib.request.Request(
                    url, data=json.dumps(corpo).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {chave}"},
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    d = json.loads(r.read())
                texto = d["choices"][0]["message"]["content"]
                usados = d.get("usage", {}).get("completion_tokens", 0)
                return _limpar(texto), Custo(flops=usados * 2 * 7e9)  # estimativa declarada
        except Exception:
            return None
        return None

    # -- modo local -------------------------------------------------------
    def responder(
        self,
        intencao: str,
        plano: dict,
        *,
        contexto: list[str] | None = None,
        memorias: list[str] | None = None,
        historico: list[dict] | None = None,
        sistema: str = "",
        pergunta: str = "",
    ) -> Resposta:
        """Constrói a resposta a partir da intenção, do plano e do contexto."""
        t0 = time.perf_counter()
        contexto = contexto or []
        memorias = memorias or []

        remoto = self._remoto(sistema, historico or [], plano.get("max_tokens", 200),
                              plano.get("temperatura_amostragem", 0.8))
        if remoto:
            texto, custo = remoto
            r = Resposta(texto, "kianda-remoto", custo, plano, contexto, 0.8)
            self.ultima = r
            return r

        texto, custo = self._estruturar(intencao, plano, contexto, memorias, pergunta)
        if self.gerador is not None and self.gerador.treinado and intencao == "criativo":
            s, c = self.gerador.gerar(
                semente="\n", n=plano.get("max_tokens", 120),
                temperatura=plano.get("temperatura_amostragem", 0.85),
            )
            custo = custo + c
            if s:
                texto = s
                autor = "kianda-local"
            else:
                autor = "estrutura"
        else:
            autor = "estrutura"
        custo.tempo_ms = (time.perf_counter() - t0) * 1000
        r = Resposta(_limpar(texto), autor, custo, plano, contexto, 0.55)
        self.ultima = r
        return r

    def _estruturar(self, intencao: str, plano: dict, contexto: list[str],
                    memorias: list[str], pergunta: str) -> tuple[str, Custo]:
        """Esqueleto discursivo: a parte que tokens e temperatura não dão.

        O plano de comportamento decide **o que fazer** (discordar, admitir
        incerteza, perguntar de volta). Esta função decide **como dizer**.
        Separar as duas é o que permite testar uma sem a outra.
        """
        partes: list[str] = []
        if memorias:
            partes.append(_naturalizar_memoria(memorias[0]))
        if contexto:
            partes.append(contexto[0])

        if plano.get("pode_discordar") and "discorda" in intencao:
            partes.append(MARCA["discordancia"][1])
        if plano.get("admite_incerteza") and not contexto and not memorias:
            partes.append(MARCA["incerteza"][0])
        if plano.get("devolve_pergunta") and pergunta:
            partes.append(MARCA["curiosidade"][0])

        if not partes:
            partes.append(
                "Estou aqui. Não sei ainda o que queres de mim nesta conversa, "
                "mas estou a ouvir."
            )
        texto = " ".join(partes)
        if plano.get("comprimento") == "curta":
            texto = texto.split(". ")[0] + "."
        custo = Custo(flops=len(texto) * 4, bytes_escritos=len(texto))
        return texto, custo


def _naturalizar_memoria(m: str) -> str:
    """Converte uma linha de memória em fala. Sem isto, ela recita a base de dados."""
    m = re.sub(r"^Sobre \w+: ", "", m).strip()
    if m and not m[0].isupper():
        m = m[0].upper() + m[1:]
    return m if m.endswith((".", "!", "?")) else m + "."
