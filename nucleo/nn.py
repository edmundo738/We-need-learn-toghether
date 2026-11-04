"""Camadas neuronais, optimizadores e técnicas de eficiência — do zero.

Tudo construído em cima de `kianda.nucleo.autograd`. Cada classe expõe
`parametros()`, `custo(dtype=...)` e `bytes(...)`, o que permite ao benchmark
comparar variantes da mesma arquitectura em FLOPs e memória.

Técnicas implementadas (cada uma testável isoladamente):

- `Linear`            — camada densa base (Wx + b)
- `Adam`              — optimizador adaptativo com correcção de viés
- `LoRA`              — adaptação de baixo posto, congela W e aprende A·B
- `podar_estruturado` — zera neurónios inteiros por magnitude (sparsity real)
- `destilar`          — knowledge distillation professor -> aluno
- `calibrar_escala`   — quantização pós-treino int8 com métrica de erro
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .autograd import (
    BYTES_F32,
    BYTES_I8,
    Custo,
    Tensor,
    entropia_cruzada,
    matmul,
    matmul_int8,
    quantizar_int8,
)


# --------------------------------------------------------------------------
# Inicialização
# --------------------------------------------------------------------------


def inicializar_he(rng: np.random.Generator, forma: tuple[int, int]) -> np.ndarray:
    """Inicialização He para ReLU: std = sqrt(2/fan_in)."""
    fan_in = forma[0]
    return (rng.standard_normal(forma) * math.sqrt(2.0 / fan_in)).astype(np.float32)


def inicializar_xavier(rng: np.random.Generator, forma: tuple[int, int]) -> np.ndarray:
    fan_in, fan_out = forma
    limite = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limite, limite, size=forma).astype(np.float32)


# --------------------------------------------------------------------------
# Camadas
# --------------------------------------------------------------------------


class Modulo:
    """Base de todos os módulos: parâmetros nomeados e custo declarado."""

    nome: str = "modulo"

    def parametros(self) -> dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def custo(self, batch: int = 1, **kw) -> Custo:
        return Custo()

    def n_parametros(self) -> int:
        return sum(t.n_elementos for t in self.parametros().values())

    def bytes_pesos(self, bytes_por_param: int = BYTES_F32) -> int:
        return self.n_parametros() * bytes_por_param


class Linear(Modulo):
    """Camada densa. Duas variantes de execução: float32 e int8 quantizado."""

    nome = "linear"

    def __init__(
        self,
        entrada: int,
        saida: int,
        rng: np.random.Generator | None = None,
        viés: bool = True,
        escala_init: float = 1.0,
        activacao: str | None = None,
        peso: "Tensor | None" = None,
        transpor_peso: bool = False,
    ):
        """`peso` permite PARTILHAR a matriz com outro módulo (weight tying).

        Quando é dado, não se cria um W novo: usa-se esse tensor. O
        gradiente das duas utilizações acumula-se no mesmo `Tensor`, que é
        exactamente o comportamento correcto — e a razão de `parametros()`
        do Transformer não listar a cabeça duas vezes com tying ligado.
        """
        rng = rng or np.random.default_rng(0)
        self.entrada = entrada
        self.saida = saida
        self.activacao = activacao
        # Weight tying precisa da TRANSPOSTA: a tabela de embeddings é
        # (vocab, d_model) e a cabeça precisa de (d_model, vocab). O truque
        # do nanoGPT (`wte.weight = lm_head.weight`) funciona porque lá as
        # convenções coincidem; aqui não, e um reshape estaria errado — é a
        # mesma matriz lida ao contrário.
        if peso is not None:
            if transpor_peso:
                if peso.data.T.shape != (entrada, saida):
                    raise ValueError(
                        f"peso transposto tem forma {peso.data.T.shape}, "
                        f"esperava {(entrada, saida)}")
                # NÃO guardar uma CÓPIA da transposta: `Tensor(peso.data.T)`
                # faz np.asarray e desliga-se do pai. Quando `carregar()`
                # substitui `peso.data`, a cabeça continuaria a usar os pesos
                # antigos — medido: o modelo carregado dava saída diferente
                # em 100% dos elementos. Guarda-se o pai e a transposta é
                # calculada a cada forward.
                # Inicializar já com a transposta do dono, não com zeros:
                # um `W` a zeros significa que o PRIMEIRO forward do modelo
                # usa uma cabeça nula, e isso não é o modelo que se quer
                # medir (a perda inicial mudava em função do tying, o que
                # falseava comparações e o gradcheck).
                self.W = Tensor(peso.data.T.copy(), True, (peso,), "transposta")
                self.W._backward = lambda g: peso.acumular_grad(g.T)
                self._partilhado = peso
            else:
                if peso.data.shape != (entrada, saida):
                    raise ValueError(
                        f"peso partilhado tem forma {peso.data.shape}, "
                        f"esperava {(entrada, saida)}")
                self.W = peso
                self._partilhado = peso
        else:
            self.W = Tensor(inicializar_he(rng, (entrada, saida)) * escala_init, True, _op="W")
            self._partilhado = None
        self.b = Tensor(np.zeros((1, saida), dtype=np.float32), True, _op="b") if viés else None

        # cache de quantização (invalidada ao treinar)
        self._w_int8: np.ndarray | None = None
        self._escala: float = 0.0
        self._w_ref: np.ndarray | None = None

    def parametros(self) -> dict[str, Tensor]:
        p = {"W": self.W}
        if self.b is not None:
            p["b"] = self.b
        return p

    def _matriz(self) -> Tensor:
        """A matriz de pesos efectiva.

        Para uma camada normal é `self.W`. Para uma camada com peso
        partilhado transposto, é a transposta VIVA do tensor dono — assim
        carregar um checkpoint ou perturbar o dono num gradcheck propaga-se
        imediatamente à cabeça.
        """
        if self._partilhado is not None:
            self.W.data = self._partilhado.data.T
        return self.W

    def forward(self, x: Tensor, quantizado: bool = False) -> Tensor:
        self._matriz()
        if quantizado:
            if self._w_int8 is None or self._w_ref is None or not np.array_equal(self._w_ref, self.W.data):
                self._w_ref = self.W.data.copy()
                self._w_int8, self._escala = quantizar_int8(self.W.data)
            z = matmul_int8(x, self._w_int8, self._escala, self._w_ref)
        else:
            z = matmul(x, self.W)
        if self.b is not None:
            z = z + self.b
        return z

    def custo(self, batch: int = 1, quantizado: bool = False) -> Custo:
        bpp = BYTES_I8 if quantizado else BYTES_F32
        return Custo(
            flops=2 * batch * self.entrada * self.saida
            + (0 if quantizado else 0),
            bytes_lidos=self.entrada * self.saida * bpp + (self.saida * 4 if self.b is not None else 0),
            bytes_escritos=batch * self.saida * BYTES_F32,
        )

    def bytes_pesos(self, bytes_por_param: int = BYTES_F32) -> int:
        return self.n_parametros() * bytes_por_param

    # -- poda -----------------------------------------------------------
    def podar(self, fracao: float) -> int:
        """Zera as `fracao` colunas de menor norma L2 (poda estruturada).

        Devolve o número de parâmetros efectivamente desactivados. A poda
        estruturada é mais interessante do que a poda de pesos isolados
        porque permite *saltar* a multiplicação inteira na inferência.
        """
        normas = np.linalg.norm(self.W.data, axis=0)
        n = int(round(fracao * self.saida))
        if n <= 0:
            return 0
        idx = np.argsort(normas)[:n]
        self.W.data[:, idx] = 0.0
        self._w_int8 = None
        return int(n)

    def mascara_esparsa(self) -> float:
        return float((self.W.data == 0).mean())


class MLP(Modulo):
    """Perceptrão multicamadas com GELU aproximada e norma de saída opcional."""

    nome = "mlp"

    def __init__(
        self,
        entrada: int,
        oculto: int,
        saida: int | None = None,
        rng: np.random.Generator | None = None,
        camadas: int = 2,
    ):
        rng = rng or np.random.default_rng(0)
        dims = [entrada] + [oculto] * (camadas - 1) + [saida or entrada]
        self.camadas = [Linear(dims[i], dims[i + 1], rng) for i in range(len(dims) - 1)]

    def parametros(self) -> dict[str, Tensor]:
        p: dict[str, Tensor] = {}
        for i, c in enumerate(self.camadas):
            for k, v in c.parametros().items():
                p[f"{i}.{k}"] = v
        return p

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, c in enumerate(self.camadas):
            h = c(h)
            if i < len(self.camadas) - 1:
                h = _gelu(h)
        return h

    def custo(self, batch: int = 1, **kw) -> Custo:
        total = Custo()
        for c in self.camadas:
            total = total + c.custo(batch)
        return total


def _gelu(x: Tensor) -> Tensor:
    """GELU tanh-aproximada: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))."""
    c = math.sqrt(2.0 / math.pi)
    interno = (x + ((x ** 3.0) * 0.044715)) * c
    return x * (interno.tanh() * 0.5 + 0.5)


# --------------------------------------------------------------------------
# Optimizador
# --------------------------------------------------------------------------


class Adam:
    """Adam (Kingma & Ba, 2015) implementado passo a passo.

        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        m^ <- m/(1-b1^t),  v^ <- v/(1-b2^t)
        W <- W - eta * m^/(sqrt(v^)+eps)

    Inclui *weight decay desacoplado* (AdamW), que é o que se usa hoje.
    """

    def __init__(
        self,
        parametros: dict[str, Tensor],
        eta: float = 3e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = parametros
        self.eta = eta
        self.b1, self.b2, self.eps = b1, b2, eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v.data) for k, v in parametros.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in parametros.items()}
        self.t = 0

    def step(self, grad_clip: float | None = 1.0) -> float:
        self.t += 1
        norma_total = 0.0
        if grad_clip is not None:
            for p in self.params.values():
                if p.grad is not None:
                    norma_total += float(np.sum(p.grad**2))
            norma_total = math.sqrt(norma_total)
            factor = min(1.0, grad_clip / (norma_total + 1e-12)) if norma_total > 0 else 1.0
        else:
            factor = 1.0

        for k, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad * factor
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            p.data -= self.eta * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p.data)
            p.grad = None
        return norma_total

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = None


# --------------------------------------------------------------------------
# LoRA — adaptação de baixo posto
# --------------------------------------------------------------------------


class LoRA(Modulo):
    """Low-Rank Adaptation (Hu et al., 2021), escrita do zero.

    Congela a matriz original `W (d_in, d_out)` e aprende `A (d_in, r)` e
    `B (r, d_out)` com r << d_in, d_out, de modo que

        W_efectivo = W + alpha/r * A @ B

    O número de parâmetros treináveis cai de `d_in*d_out` para `r*(d_in+d_out)`.
    É a técnica que permite adaptar a Kianda a um novo utilizador com uma
    fracção do custo de re-treinar.
    """

    nome = "lora"

    def __init__(self, base: Linear, rank: int = 4, alpha: float | None = None,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.base = base
        self.rank = max(1, min(rank, min(base.entrada, base.saida)))
        self.alpha = alpha or float(self.rank)
        self.A = Tensor(
            rng.standard_normal((base.entrada, self.rank)).astype(np.float32) * 0.01, True, _op="loraA"
        )
        self.B = Tensor(np.zeros((self.rank, base.saida), dtype=np.float32), True, _op="loraB")
        self.activo = True

    def parametros(self) -> dict[str, Tensor]:
        return {"A": self.A, "B": self.B}

    def forward(self, x: Tensor) -> Tensor:
        z = matmul(x, self.base.W)
        if self.activo:
            z = z + matmul(matmul(x, self.A), self.B) * (self.alpha / self.rank)
        if self.base.b is not None:
            z = z + self.base.b
        return z

    def custo(self, batch: int = 1, **kw) -> Custo:
        c = Custo(flops=2 * batch * self.base.entrada * self.base.saida)
        if self.activo:
            c = c + Custo(flops=2 * batch * self.base.entrada * self.rank
                          + 2 * batch * self.rank * self.base.saida)
        return c

    def n_parametros_treinaveis(self) -> int:
        return self.A.n_elementos + self.B.n_elementos

    def fusao_delta(self) -> np.ndarray:
        """Delta de pesos aprendido, para explicar ou auditar a adaptação."""
        return (self.A.data @ self.B.data) * (self.alpha / self.rank)


# --------------------------------------------------------------------------
# Podar / destilar / calibrar
# --------------------------------------------------------------------------


def podar_estruturado(modulos: list[Modulo], fracao: float) -> dict:
    """Aplica poda estruturada a todas as camadas dadas e mede o resultado.

    Devolve a fracção de multiplicações que seria poupada se o *runtime*
    soubesse saltar as colunas zeradas. A poupança só e real quando a
    arquitectura a sabe explorar - e exactamente isso que o benchmark mede.
    """
    antes = sum(m.n_parametros() for m in modulos)
    removidos = 0
    flops_antes = 0
    flops_depois = 0
    for m in modulos:
        if isinstance(m, Linear):
            flops_antes += 2 * m.entrada * m.saida
            colunas = m.podar(fracao)
            flops_depois += 2 * m.entrada * (m.saida - colunas)
            removidos += colunas * m.entrada
    total = sum(m.n_parametros() for m in modulos)
    return {
        "fracao_alvo": fracao,
        "parametros_antes": antes,
        "parametros_zerados": removidos,
        "esparsidade_global": round(removidos / max(total - removidos, 1), 4),
        "flops_antes": flops_antes,
        "flops_com_salto": flops_depois,
        "poupanca_flops_pct": round(100 * (1 - flops_depois / max(flops_antes, 1)), 2),
    }


def destilar(
    professor: Modulo,
    aluno: Modulo,
    x: Tensor,
    alvos: np.ndarray,
    *,
    temperatura: float = 2.0,
    peso_duro: float = 0.3,
    passos: int = 300,
    eta: float = 3e-3,
    verbose: bool = False,
) -> dict:
    """Knowledge distillation (Hinton et al., 2015), sem framework.

    O aluno treina em duas perdas: os rótulos reais e a distribuição *suave*
    do professor (logits / T). A segunda transporta informação sobre
    semelhanças entre classes que o rótulo rígido não contém — é por isso
    que modelos pequenos destilados batem modelos pequenos treinados do
    zero a partir dos mesmos dados.
    """
    prof = professor(x).data
    z_p = prof / temperatura
    p_p = np.exp(z_p - z_p.max(axis=1, keepdims=True))
    p_p /= p_p.sum(axis=1, keepdims=True)

    opt = Adam(aluno.parametros(), eta=eta)
    hist = []
    for passo in range(passos):
        logits = aluno(x)
        # perda dura
        l_duro = entropia_cruzada(logits, alvos)
        # perda suave, em numpy puro sobre a mesma saída autograd
        z_s = logits.data / temperatura
        p_s = np.exp(z_s - z_s.max(axis=1, keepdims=True))
        p_s /= p_s.sum(axis=1, keepdims=True)
        l_suave = float(-np.mean(np.sum(p_p * np.log(np.maximum(p_s, 1e-12)), axis=1)))
        logits.acumular_grad((p_s - p_p) / p_s.shape[0] * temperatura)
        (l_duro * peso_duro).backward()
        opt.step()
        if passo % 50 == 0 or passo == passos - 1:
            hist.append((passo, l_duro.data.item(), l_suave))
            if verbose:
                print(f"  passo {passo:3d}  duro={l_duro.data.item():.4f}  suave={l_suave:.4f}")
    return {"hist": hist, "perda_final": float(l_duro.data.item())}


def calibrar_escala(modelo: list[Linear], x: np.ndarray) -> dict:
    """Compara float32 vs int8 na mesma entrada e mede o dano real.

    Devolve a divergência KL média entre as distribuições de saída dos dois
    modos. Um KL pequeno é a definição operacional de "a quantização é
    aceitável para esta tarefa" — não uma afirmação genérica.
    """
    h = x.astype(np.float32)
    for m in modelo[:-1]:
        h = np.maximum(m.forward(Tensor(h)).data, 0.0)
    ult = modelo[-1]
    z_f = ult.forward(Tensor(h)).data
    z_q = ult.forward(Tensor(h), quantizado=True).data

    def _p(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    p, q = _p(z_f), _p(z_q)
    kl = float(np.mean(np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12)), axis=1)))
    acertos_f = float(np.mean(p.argmax(1) == q.argmax(1)))
    return {
        "kl_medio": kl,
        "acordo_top1": acertos_f,
        "bytes_float": ult.bytes_pesos(BYTES_F32),
        "bytes_int8": ult.bytes_pesos(BYTES_I8),
        "reducao_memoria_pct": round(100 * (1 - BYTES_I8 / BYTES_F32), 1),
    }
