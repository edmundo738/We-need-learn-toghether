"""Transformer decoder-only — construído sobre o autograd próprio.

Este módulo existe para provar uma coisa só, e essa coisa não é «o modelo é
bom». É: **cada peso deste grafo é nosso, o gradiente vem do nosso `autograd`,
e a causalidade é verificável.**

A asserção que governa tudo
---------------------------
`saida[t]` só pode depender de `ids[:t+1]`. Se isto falhar, o modelo viu a
resposta durante o treino, a loss desce, a validação desce, e nenhum dos dois
números significa nada. É por isso que o teste de causalidade existe antes do
treino e não depois: um teste que passa depois de uma loss baixa já é tarde.

Formas
------
Como o `Tensor` deste autograd só tem duas dimensões, todas as sequências são
achatadas: `(B, T)` → `(B*T, d_model)` e voltam a `(B, T, ...)` onde é preciso.
O truque que faz isto funcionar sem índice de posições: a atenção causal dentro
de uma sequência usa uma **máscara triangular construída com a posição de cada
linha**, e essa posição calcula-se com um `arange` sobre `B*T`.

Convenção de eixo
-----------------
A dimensão de sequência é a **última** em tudo o que é por-token
(`(B*T, d)`, `(B*T, vocab)`), e as cabeças vivem em `(B*T, n_cabecas, d_cabeca)`.
Isto evita transposições de eixo intermédio, que este autograd não tem.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nucleo import nn
from nucleo.autograd import BYTES_F32, Custo, Tensor, matmul


# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Hiperparâmetros do modelo. Congelados num ficheiro ao guardar."""

    vocab: int = 8192
    d_model: int = 128
    n_camadas: int = 4
    n_cabecas: int = 4
    ctx: int = 256
    d_ff: int = 384          # 3x d_model, o valor que a directiva fixa
    dropout: float = 0.0

    def validar(self) -> None:
        if self.d_model % self.n_cabecas != 0:
            raise ValueError(
                f"d_model={self.d_model} não é divisível por n_cabecas={self.n_cabecas}"
            )
        if self.d_cabeca * self.n_cabecas != self.d_model:
            raise ValueError("n_cabecas * d_cabeca != d_model")
        if self.ctx < 2:
            raise ValueError("ctx tem de ser >= 2")

    @property
    def d_cabeca(self) -> int:
        return self.d_model // self.n_cabecas

    def n_parametros(self) -> int:
        """Contagem analítica, independente da construção dos tensores."""
        emb = self.vocab * self.d_model
        pos = self.ctx * self.d_model
        por_camada = (
            self.d_model * self.d_model * 4        # q, k, v, o (sem bias)
            + self.d_model * self.d_ff             # up: d_model -> d_ff
            + self.d_ff * self.d_model             # down: d_ff -> d_model
            + 4 * self.d_model                     # 2 LayerNorms × (gama + beta)
        )
        final = self.d_model * 2                   # LayerNorm final
        saida = self.d_model * self.vocab          # projecção para o vocabulário
        return emb + pos + self.n_camadas * por_camada + final + saida


# --------------------------------------------------------------------------
# Componentes
# --------------------------------------------------------------------------

class Embedding(nn.Modulo):
    """Tabela de embeddings com gradiente esparso por linha."""

    def __init__(self, n: int, d: int, rng: np.random.Generator):
        self.peso = Tensor(
            (rng.standard_normal((n, d)) * 0.02).astype(np.float32),
            requires_grad=True,
        )

    def parametros(self) -> dict[str, Tensor]:
        return {"peso": self.peso}

    def forward(self, ids: np.ndarray) -> Tensor:
        """`ids` tem forma `(B*T,)`. Devolve `(B*T, d)`."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        out = Tensor(self.peso.data[ids], self.peso.requires_grad, (self.peso,), "embedding")
        out.meta["custo"] = Custo(
            flops=0,
            bytes_lidos=int(ids.size * self.peso.data.shape[1] * BYTES_F32),
            bytes_escritos=int(ids.size * self.peso.data.shape[1] * BYTES_F32),
        )

        def _bw(g: np.ndarray) -> None:
            grad = np.zeros_like(self.peso.data)
            np.add.at(grad, ids, g)   # duplicados somam, como deve ser
            self.peso.acumular_grad(grad)

        out._backward = _bw
        return out


class LayerNorm(nn.Modulo):
    """LayerNorm sobre a última dimensão, com gama e beta aprendíveis."""

    def __init__(self, d: int, eps: float = 1e-5):
        self.gama = Tensor(np.ones((1, d), dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros((1, d), dtype=np.float32), requires_grad=True)
        self.eps = eps

    def parametros(self) -> dict[str, Tensor]:
        return {"gama": self.gama, "beta": self.beta}

    def forward(self, x: Tensor) -> Tensor:
        media = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x.data - media) * inv
        y = xhat * self.gama.data + self.beta.data
        out = Tensor(y, x.requires_grad, (x, self.gama, self.beta), "layernorm")
        d = x.data.shape[-1]

        def _bw(g: np.ndarray) -> None:
            self.gama.acumular_grad((g * xhat).sum(axis=0, keepdims=True))
            self.beta.acumular_grad(g.sum(axis=0, keepdims=True))
            gx = g * self.gama.data
            # forma fechada do backward do LayerNorm
            dx = inv * (gx - gx.mean(axis=-1, keepdims=True)
                        - xhat * (gx * xhat).mean(axis=-1, keepdims=True))
            x.acumular_grad(dx)

        out._backward = _bw
        return out


class AtencaoCausal(nn.Modulo):
    """Atenção multi-cabeça com máscara triangular estrita.

    A máscara é literalmente `pos_consulta >= pos_chave`: uma posição vê-se a
    si própria e a todas as anteriores, e nenhuma futura. Construída dentro do
    `forward`, a partir das posições, e não passada de fora — assim não há
    forma de a esquecer num sítio e não noutro.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        d = cfg.d_model
        self.wq = nn.Linear(d, d, rng, viés=False)
        self.wk = nn.Linear(d, d, rng, viés=False)
        self.wv = nn.Linear(d, d, rng, viés=False)
        self.wo = nn.Linear(d, d, rng, viés=False)
        self._ultima_mascara: np.ndarray | None = None

    def parametros(self) -> dict[str, Tensor]:
        p = {}
        for nome, mod in (("q", self.wq), ("k", self.wk), ("v", self.wv), ("o", self.wo)):
            for k, t in mod.parametros().items():
                p[f"{nome}.{k}"] = t
        return p

    def forward(self, x: Tensor, pos: np.ndarray, n_seq: int) -> Tensor:
        """
        x:   `(B*T, d_model)`
        pos: `(B*T,)` posição de cada linha dentro da sua sequência (0..T-1)

        Tudo aqui passa por `Tensor`. A tentação é calcular os scores em numpy
        puro e embrulhar o resultado num `Tensor` novo — é 3x mais rápido e
        está errado: os pesos de q/k/v ficariam fora do grafo e nunca
        receberiam gradiente. Já aconteceu neste ficheiro.
        """
        cfg = self.cfg
        # ATENÇÃO à forma: x é (B*T, d_model). A sequência já está achatada
        # na PRIMEIRA dimensão; a segunda é sempre d_model.
        n, d = x.data.shape
        assert d == cfg.d_model, f"esperava d_model={cfg.d_model}, recebi {d}"
        h, dh = cfg.n_cabecas, cfg.d_cabeca

        q = self.wq(x)          # (B*T, d)
        k = self.wk(x)
        v = self.wv(x)

        # máscara causal: consulta i só vê chaves j com pos[j] <= pos[i] e da
        # mesma sequência. Construída aqui, não passada de fora.
        # índice de sequência: cada linha corresponde a uma posição achatada
        seq_i = np.arange(n) // max(n_seq, 1)
        mesmo = seq_i[:, None] == seq_i[None, :]
        mascara = (pos[:, None] >= pos[None, :]) & mesmo     # (B*T, B*T)
        self._ultima_mascara = mascara

        # scores (B*T, B*T) — para cada par (i, j), o dot de q_i com k_j
        scores = _scores(q, k, h, dh, n)                    # (h*B*T, B*T)
        # a máscara repete-se por cabeça
        scores = _mascara_negativa(scores, np.tile(mascara, (h, 1)))
        scores = scores * (1.0 / math.sqrt(dh))

        att = _softmax_ultima(scores)                       # (h*B*T, B*T)
        contexto = _aplicar_por_cabeca(att, v, h, n, dh)     # (B*T, d)
        return self.wo(contexto)



class BlocoTransformer(nn.Modulo):
    """Pre-norm: x + atencao(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.ln1 = LayerNorm(cfg.d_model)
        self.atencao = AtencaoCausal(cfg, rng)
        self.ln2 = LayerNorm(cfg.d_model)
        d, d_ff = cfg.d_model, cfg.d_ff
        self.up = nn.Linear(d, d_ff, rng, viés=False)
        self.down = nn.Linear(d_ff, d, rng, viés=False)

    def parametros(self) -> dict[str, Tensor]:
        p = {f"ln1.{k}": t for k, t in self.ln1.parametros().items()}
        p.update({f"atencao.{k}": t for k, t in self.atencao.parametros().items()})
        p.update({f"ln2.{k}": t for k, t in self.ln2.parametros().items()})
        p.update({f"up.{k}": t for k, t in self.up.parametros().items()})
        p.update({f"down.{k}": t for k, t in self.down.parametros().items()})
        return p

    def forward(self, x: Tensor, pos: np.ndarray, n_seq: int) -> Tensor:
        h = x + self.atencao.forward(self.ln1(x), pos, n_seq)
        m = self.up(self.ln2(h)).relu()
        return h + self.down(m)


class Transformer(nn.Modulo):
    """Decoder-only completo: embedding + posição + blocos + norm + cabeça."""

    def __init__(self, cfg: Config, semente: int = 0):
        cfg.validar()
        self.cfg = cfg
        rng = np.random.default_rng(semente)
        self.emb = Embedding(cfg.vocab, cfg.d_model, rng)
        self.emb_pos = Embedding(cfg.ctx, cfg.d_model, rng)
        self.blocos = [BlocoTransformer(cfg, rng) for _ in range(cfg.n_camadas)]
        self.ln_final = LayerNorm(cfg.d_model)
        self.cabeca = nn.Linear(cfg.d_model, cfg.vocab, rng, viés=False)

    # -- parametros ------------------------------------------------------
    def parametros(self) -> dict[str, Tensor]:
        p = {f"emb.{k}": t for k, t in self.emb.parametros().items()}
        p.update({f"emb_pos.{k}": t for k, t in self.emb_pos.parametros().items()})
        for i, b in enumerate(self.blocos):
            p.update({f"bloco{i}.{k}": t for k, t in b.parametros().items()})
        p.update({f"ln_final.{k}": t for k, t in self.ln_final.parametros().items()})
        p.update({f"cabeca.{k}": t for k, t in self.cabeca.parametros().items()})
        return p

    def n_parametros(self) -> int:
        return sum(t.n_elementos for t in self.parametros().values())

    # -- forward ---------------------------------------------------------
    def forward(self, ids: np.ndarray) -> Tensor:
        """
        ids: `(B, T)` inteiros.
        Devolve logits `(B*T, vocab)`.

        A ORDEM DAS LINHAS É a ordem de (batch, posição) achatada: a linha
        `i*T + t` corresponde à sequência `i`, posição `t`.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        Bn, Tn = ids.shape
        if Tn > self.cfg.ctx:
            raise ValueError(f"sequência de {Tn} excede o contexto {self.cfg.ctx}")

        pos = np.tile(np.arange(Tn, dtype=np.int64), Bn)      # (B*T,)
        x = self.emb(ids.reshape(-1)) + self.emb_pos(pos)
        for bloco in self.blocos:
            x = bloco.forward(x, pos, Tn)
        x = self.ln_final(x)
        return self.cabeca(x)

    # -- geração ---------------------------------------------------------
    def gerar(
        self,
        ids: np.ndarray,
        n_novos: int,
        temperatura: float = 0.0,
        semente: int | None = None,
    ) -> np.ndarray:
        """Geração autoregressiva. `temperatura=0` é greedy e determinística.

        Corta o contexto à janela: se a sequência crescer além de `ctx`,
        as posições mudam de sentido, por isso o corte é explícito e não
        silencioso.
        """
        ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
        rng = np.random.default_rng(semente)
        for _ in range(n_novos):
            janela = ids[:, -self.cfg.ctx:]
            logits = self.forward(janela).data[-1]          # última posição
            if temperatura <= 0:
                proximo = int(np.argmax(logits))
            else:
                z = logits / temperatura
                z = z - z.max()
                p = np.exp(z)
                p = p / p.sum()
                proximo = int(rng.choice(len(p), p=p))
            ids = np.concatenate([ids, np.array([[proximo]], dtype=np.int64)], axis=1)
        return ids

    # -- persistência ----------------------------------------------------
    def guardar(self, pasta: str | Path) -> Path:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        tensores = {k: t.data for k, t in self.parametros().items()}
        np.savez_compressed(pasta / "pesos.npz", **tensores)
        (pasta / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False)
        )
        return pasta

    @classmethod
    def carregar(cls, pasta: str | Path) -> "Transformer":
        pasta = Path(pasta)
        cfg = Config(**json.loads((pasta / "config.json").read_text()))
        modelo = cls(cfg)
        dados = np.load(pasta / "pesos.npz")
        for nome, t in modelo.parametros().items():
            if nome not in dados:
                raise KeyError(f"peso ausente no ficheiro: {nome}")
            if dados[nome].shape != t.data.shape:
                raise ValueError(
                    f"{nome}: forma {dados[nome].shape} != esperada {t.data.shape}"
                )
            t.data = dados[nome].astype(np.float32)
        return modelo


# --------------------------------------------------------------------------
# Perda
# --------------------------------------------------------------------------

def perda_entropia_cruzada(logits: Tensor, alvos: np.ndarray) -> Tensor:
    """Entropia cruzada média sobre as posições.

    Teste que este contrato tem de passar: com logits todos iguais, o valor
    tem de ser exactamente `log(V)`.
    """
    alvos = np.asarray(alvos, dtype=np.int64).reshape(-1)
    assert logits.data.shape[0] == alvos.size, (
        f"logits {logits.data.shape} não casam com {alvos.size} alvos"
    )
    n = alvos.size
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - logsumexp
    perda_dados = -logp[np.arange(n), alvos].mean()
    out = Tensor(np.array([[perda_dados]], dtype=np.float32), logits.requires_grad,
                 (logits,), "entropia_cruzada")

    def _bw(g: np.ndarray) -> None:
        p = np.exp(logp)
        p[np.arange(n), alvos] -= 1.0
        logits.acumular_grad(p * (float(g.reshape(-1)[0]) / n))

    out._backward = _bw
    return out


def _aplicar_por_cabeca(att: Tensor, v: Tensor, h: int, n: int, dh: int) -> Tensor:
    """`contexto[i] = concat_cabeca( att[cabeca, i, :] @ v_cabeca )`.

    att: `(h*B*T, B*T)` · v: `(B*T, d)` · devolve `(B*T, h*dh)`.
    """
    a3 = att.data.reshape(h, n, n)
    v3 = v.data.reshape(n, h, dh)
    ctx = np.einsum("hij,jhd->ihd", a3, v3).reshape(n, h * dh)
    out = Tensor(ctx, att.requires_grad or v.requires_grad, (att, v), "contexto")

    def _bw(g: np.ndarray) -> None:
        g3 = g.reshape(n, h, dh)
        datt = np.einsum("ihd,jhd->hij", g3, v3).reshape(h * n, n)
        dv = np.einsum("hij,ihd->jhd", a3, g3).reshape(n, h * dh)
        att.acumular_grad(datt)
        v.acumular_grad(dv)

    out._backward = _bw
    return out


# --------------------------------------------------------------------------
# Primitivas de atenção (com gradiente próprio)
# --------------------------------------------------------------------------
#
# Estas três funções existem porque a atenção precisa de formas que o
# `autograd` base não cobre: um produto tensorial com cabeças, uma máscara
# aditiva e um softmax por linha de matriz quadrada. A regra é a mesma de
# todas as outras operações do autograd: escrever a regra da cadeia uma vez,
# e testá-la contra diferenças finitas.

def _scores(q: Tensor, k: Tensor, h: int, dh: int, n: int) -> Tensor:
    """`scores[i, j] = sum_cabeca q[i, cabeca] · k[j, cabeca]`.

    Não há dimensão de lote além do achatamento: cada linha `i` é um token
    de uma sequência concreta, e `j` percorre todos os tokens do lote. A
    máscara trata de impedir que uma sequência olhe para outra.
    """
    q3 = q.data.reshape(n, h, dh)
    k3 = k.data.reshape(n, h, dh)
    # POR CABEÇA: (h, n, n). Somar as cabeças aqui (->ij) colapsa-as numa
    # só e o modelo perde a capacidade de olhar para sítios diferentes ao
    # mesmo tempo. Foi um bug real neste ficheiro.
    s = np.einsum("ihd,jhd->hij", q3, k3)
    out = Tensor(s.reshape(h * n, n), q.requires_grad or k.requires_grad, (q, k), "scores")
    out.meta["forma_cabecas"] = (h, n)

    def _bw(gg: np.ndarray) -> None:
        g = gg.reshape(h, n, n)
        dq = np.einsum("hij,jhd->ihd", g, k3).reshape(n, h * dh)
        dk = np.einsum("hij,ihd->jhd", g, q3).reshape(n, h * dh)
        q.acumular_grad(dq)
        k.acumular_grad(dk)

    out._backward = _bw
    return out


def _mascara_negativa(t: Tensor, mascara: np.ndarray, valor: float = -1e9) -> Tensor:
    """Soma `valor` onde a máscara é falsa. Gradiente passa só onde é verdadeira."""
    m = np.where(mascara, 0.0, valor).astype(np.float32)
    out = Tensor(t.data + m, t.requires_grad, (t,), "mascara_negativa")

    def _bw(g: np.ndarray) -> None:
        t.acumular_grad(g)

    out._backward = _bw
    return out


def _softmax_ultima(t: Tensor) -> Tensor:
    """Softmax por linha, numericamente estável."""
    z = t.data - t.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=-1, keepdims=True)
    out = Tensor(y, t.requires_grad, (t,), "softmax_linha")

    def _bw(g: np.ndarray) -> None:
        t.acumular_grad(y * (g - (g * y).sum(axis=-1, keepdims=True)))

    out._backward = _bw
    return out
