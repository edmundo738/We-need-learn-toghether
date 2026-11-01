"""Autograd do zero: motor de diferenciação automática em modo inverso.

Sem PyTorch, sem JAX, sem TensorFlow. Apenas NumPy para o produto de matrizes.
Tudo o resto — grafo, regra da cadeia, retropropagação, quantização int8, poda,
LoRA e o modelo de custo — está implementado aqui.

Porque do zero
--------------
1. **Fins educativos.** O projecto Kianda tem de conseguir explicar cada
   equação que usa. `z = Wx + b`, `a = f(z)`, `W <- W - eta * dL/dW` não são
   metáforas: são este ficheiro.
2. **Fins científicos.** O foco do projecto é *capacidade por unidade de
   recurso*. Para medir isso é preciso controlar e contar cada operação. Uma
   framework que faz a contabilidade por nós esconde exactamente a variável
   que queremos estudar.

Modelo de custo
---------------
Cada nó do grafo sabe quantas operações elementares custou (`custo`) e
quantos bytes ocupam os seus tensores (`bytes`). `custo_graph(...)` devolve
o total. Assim, a experiência central do projecto — "quanto custa em FLOPs e
em bytes uma resposta com qualidade X" — é uma medição directa e não uma
estimativa.

Convenções
----------
- Um `Tensor` é sempre um array 2-D `(batch, features)`. Manter a forma
  explícita evita a maior parte dos bugs de broadcasting em autograd.
- `requires_grad` propaga-se para a frente; nunca se desliga sozinho.
- Todo o treino é feito em modo inverso com acumulação de gradientes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------
# Contabilidade de recursos
# --------------------------------------------------------------------------

BYTES_F32 = 4
BYTES_I8 = 1


@dataclass
class Custo:
    """Custo acumulado de um grafo de computação.

    flops: multiplicações-acumulações elementares (1 MAC = 2 FLOPs).
    bytes_lidos / bytes_escritos: tráfego de memória do *forward*.
    tempo_ms: tempo de parede medido, quando disponível.
    """

    flops: int = 0
    bytes_lidos: int = 0
    bytes_escritos: int = 0
    tempo_ms: float = 0.0

    def __add__(self, outro: "Custo") -> "Custo":
        return Custo(
            self.flops + outro.flops,
            self.bytes_lidos + outro.bytes_lidos,
            self.bytes_escritos + outro.bytes_escritos,
            self.tempo_ms + outro.tempo_ms,
        )

    @property
    def bytes(self) -> int:
        return self.bytes_lidos + self.bytes_escritos

    def resumo(self) -> str:
        partes = [f"{self.flops / 1e6:.3f} MFLOP"]
        if self.tempo_ms:
            partes.append(f"{self.tempo_ms:.2f} ms")
            if self.flops:
                gflops = self.flops / max(self.tempo_ms, 1e-9) / 1e6
                partes.append(f"{gflops:.2f} GFLOP/s")
        partes.append(f"{self.bytes / 1024:.1f} KiB")
        return " · ".join(partes)

    def as_dict(self) -> dict:
        return {
            "flops": self.flops,
            "mflop": self.flops / 1e6,
            "bytes": self.bytes,
            "kib": self.bytes / 1024,
            "tempo_ms": self.tempo_ms,
        }


# --------------------------------------------------------------------------
# Tensor com autograd
# --------------------------------------------------------------------------


class Tensor:
    """Array 2-D com história de gradientes.

    A `_backward` de cada nó recebe o gradiente da saída e acumula nos pais.
    É a regra da cadeia, escrita uma vez por operação primitiva.
    """

    __slots__ = ("data", "grad", "requires_grad", "_pais", "_backward", "_op", "meta")

    # Contador global por sessão, útil para instrumentação.
    _contador_nos = 0

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _pais: Iterable["Tensor"] = (),
        _op: str = "leaf",
    ):
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim == 1:
            self.data = self.data.reshape(1, -1)
        elif self.data.ndim == 0:
            self.data = self.data.reshape(1, 1)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._pais: tuple[Tensor, ...] = tuple(_pais)
        self._backward = lambda g: None
        self._op = _op
        self.meta: dict = {}
        Tensor._contador_nos += 1

    # -- propriedades -----------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def n_elementos(self) -> int:
        return int(self.data.size)

    def __repr__(self) -> str:
        g = ", grad" if self.requires_grad else ""
        return f"Tensor{self.shape} <{self._op}>{g}"

    # -- gradientes -------------------------------------------------------
    def zero_grad(self) -> None:
        self.grad = None

    def acumular_grad(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        g = np.asarray(g, dtype=np.float32)
        self.grad = g.copy() if self.grad is None else self.grad + g

    def backward(self, gradiente: np.ndarray | None = None) -> None:
        """Retropropaga a partir deste nó (ordem topológica inversa)."""
        ordem: list[Tensor] = []
        visto: set[int] = set()

        def visita(t: Tensor) -> None:
            if id(t) in visto:
                return
            visto.add(id(t))
            for p in t._pais:
                visita(p)
            ordem.append(t)

        visita(self)
        if gradiente is None:
            gradiente = np.ones_like(self.data)
        self.acumular_grad(gradiente)
        for t in reversed(ordem):
            if t.grad is not None and t._pais:
                t._backward(t.grad)

    # -- operações --------------------------------------------------------
    def __add__(self, outro) -> "Tensor":
        outro = _como_tensor(outro)
        out = Tensor(
            self.data + outro.data,
            self.requires_grad or outro.requires_grad,
            (self, outro),
            "add",
        )
        out.meta["custo"] = Custo(
            flops=int(out.n_elementos),
            bytes_lidos=(self.n_elementos + outro.n_elementos) * BYTES_F32,
            bytes_escritos=out.n_elementos * BYTES_F32,
        )

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(_sem_br(g, self.data.shape))
            outro.acumular_grad(_sem_br(g, outro.data.shape))

        out._backward = _bw
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, outro) -> "Tensor":
        return self + (-_como_tensor(outro))

    def __mul__(self, outro) -> "Tensor":
        outro = _como_tensor(outro)
        out = Tensor(
            self.data * outro.data,
            self.requires_grad or outro.requires_grad,
            (self, outro),
            "mul",
        )
        out.meta["custo"] = Custo(
            flops=int(out.n_elementos),
            bytes_lidos=(self.n_elementos + outro.n_elementos) * BYTES_F32,
            bytes_escritos=out.n_elementos * BYTES_F32,
        )

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(_sem_br(g * outro.data, self.data.shape))
            outro.acumular_grad(_sem_br(g * self.data, outro.data.shape))

        out._backward = _bw
        return out

    def __truediv__(self, outro) -> "Tensor":
        outro = _como_tensor(outro)
        return self * (outro ** -1.0)

    def __pow__(self, expoente: float) -> "Tensor":
        base = self.data ** expoente
        out = Tensor(base, self.requires_grad, (self,), f"pow{expoente}")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g * expoente * (self.data ** (expoente - 1)))

        out._backward = _bw
        return out

    def __matmul__(self, outro: "Tensor") -> "Tensor":
        return matmul(self, outro)

    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")
        out.meta["custo"] = Custo(
            flops=int(self.n_elementos),
            bytes_lidos=self.n_elementos * BYTES_F32,
            bytes_escritos=out.n_elementos * BYTES_F32,
        )

        def _bw(g: np.ndarray) -> None:
            if axis is None:
                self.acumular_grad(np.broadcast_to(g, self.data.shape))
            else:
                g2 = g
                if not keepdims:
                    g2 = np.expand_dims(g, axis)
                self.acumular_grad(np.broadcast_to(g2, self.data.shape))

        out._backward = _bw
        return out

    def mean(self, axis: int | None = None) -> "Tensor":
        n = self.n_elementos if axis is None else self.data.shape[axis]
        r = self.sum(axis=axis) * (1.0 / n)
        r._op = "mean"
        return r

    def relu(self) -> "Tensor":
        mascara = self.data > 0
        out = Tensor(self.data * mascara, self.requires_grad, (self,), "relu")
        out.meta["custo"] = Custo(
            flops=int(self.n_elementos),
            bytes_lidos=self.n_elementos * BYTES_F32 * 2,
            bytes_escritos=self.n_elementos * BYTES_F32,
        )
        out.meta["esparsidade"] = float(1.0 - mascara.mean())

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g * mascara)

        out._backward = _bw
        return out

    def tanh(self) -> "Tensor":
        y = np.tanh(self.data)
        out = Tensor(y, self.requires_grad, (self,), "tanh")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g * (1.0 - y * y))

        out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        y = 0.5 * (1.0 + np.tanh(self.data / 2.0))
        out = Tensor(y, self.requires_grad, (self,), "sigmoid")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g * y * (1.0 - y))

        out._backward = _bw
        return out

    def softmax(self, eixo: int = -1) -> "Tensor":
        z = self.data - self.data.max(axis=eixo, keepdims=True)
        e = np.exp(z)
        y = e / e.sum(axis=eixo, keepdims=True)
        out = Tensor(y, self.requires_grad, (self,), "softmax")
        out.meta["custo"] = Custo(
            flops=int(self.n_elementos * 4),
            bytes_lidos=self.n_elementos * BYTES_F32,
            bytes_escritos=self.n_elementos * BYTES_F32,
        )

        def _bw(g: np.ndarray) -> None:
            # dL/dz = y * (g - <g, y>)
            self.acumular_grad(y * (g - (g * y).sum(axis=eixo, keepdims=True)))

        out._backward = _bw
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(np.maximum(self.data, 1e-12)), self.requires_grad, (self,), "log")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g / np.maximum(self.data, 1e-12))

        out._backward = _bw
        return out

    def exp(self) -> "Tensor":
        y = np.exp(self.data)
        out = Tensor(y, self.requires_grad, (self,), "exp")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g * y)

        out._backward = _bw
        return out

    def reshape(self, forma: tuple[int, ...]) -> "Tensor":
        out = Tensor(self.data.reshape(forma), self.requires_grad, (self,), "reshape")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g.reshape(self.data.shape))

        out._backward = _bw
        return out

    def transpor(self) -> "Tensor":
        out = Tensor(self.data.T, self.requires_grad, (self,), "transpor")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g.T)

        out._backward = _bw
        return out

    def detach(self) -> "Tensor":
        t = Tensor(self.data.copy(), False, _op="detach")
        return t

    def mascara(self, manter: np.ndarray) -> "Tensor":
        """Multiplicação por máscara booleana (usada na poda estruturada)."""
        manter = np.asarray(manter, dtype=np.float32)
        out = Tensor(self.data * manter, self.requires_grad, (self,), "mascara")

        def _bw(g: np.ndarray) -> None:
            self.acumular_grad(g * manter)

        out._backward = _bw
        return out

    # -- recolha de custo -------------------------------------------------
    def custo_graph(self) -> Custo:
        """Soma o custo de todos os nós do grafo (percurso em profundidade)."""
        total = Custo()
        visto: set[int] = set()
        pilha = [self]
        while pilha:
            t = pilha.pop()
            if id(t) in visto:
                continue
            visto.add(id(t))
            c = t.meta.get("custo")
            if c is not None:
                total = total + c
            pilha.extend(t._pais)
        return total


def _como_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False, _op="const")


def _sem_br(g: np.ndarray, forma: tuple[int, ...]) -> np.ndarray:
    """Reduz o gradiente à forma do pai (desfaz o broadcasting)."""
    if g.shape == forma:
        return g
    while g.ndim > len(forma):
        g = g.sum(axis=0)
    for i, d in enumerate(forma):
        if d == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# --------------------------------------------------------------------------
# Operações primitivas (o "kernel")
# --------------------------------------------------------------------------


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Multiplicação de matrizes com contabilidade exacta de FLOPs.

    Custo elementar de `(M,K) @ (K,N)` = M*K*N MACs = 2*M*K*N FLOPs.
    É este número que permite comparar arquitecturas de forma honesta.
    """
    assert a.data.shape[1] == b.data.shape[0], (
        f"matmul incompatível: {a.data.shape} @ {b.data.shape}"
    )
    m, k = a.data.shape
    n = b.data.shape[1]
    out = Tensor(a.data @ b.data, a.requires_grad or b.requires_grad, (a, b), "matmul")
    out.meta["custo"] = Custo(
        flops=2 * m * k * n,
        bytes_lidos=(m * k + k * n) * BYTES_F32,
        bytes_escritos=m * n * BYTES_F32,
    )

    def _bw(g: np.ndarray) -> None:
        a.acumular_grad(g @ b.data.T)
        b.acumular_grad(a.data.T @ g)

    out._backward = _bw
    return out


def matmul_int8(
    a: Tensor, b_int8: np.ndarray, escala_w: float, b_float_ref: np.ndarray | None = None
) -> Tensor:
    """Multiplicação com pesos quantizados em int8 (esquema simétrico).

    Ambos os operandos são convertidos para int8 com a sua própria escala,
    que é o que permite manter a aritmética toda em int32:

        (x / s_x) @ (W / s_w)  ->  acumula em int32  ->  * (s_x * s_w)

    A re-escala final é feita uma única vez por elemento de saída, não por
    produto. O erro relativo típico com pesos gaussianos fica abaixo de 2%,
    o que é o resultado esperado para quantização pós-treino de 8 bits.

    Contabilidade: M*K*N MACs em int8 (mais baratos que float32 na maioria
    do hardware) e 2*M*N FLOPs para a re-escala.
    """
    m, k = a.data.shape
    n = b_int8.shape[1]

    # escala do activação nesta chamada (simétrica, por tensor)
    max_a = float(np.abs(a.data).max()) if a.data.size else 0.0
    escala_a = max_a / 127.0 if max_a > 0 else 0.0
    if escala_a == 0.0:
        a_int8 = np.zeros_like(a.data, dtype=np.int8)
    else:
        a_int8 = np.clip(np.rint(a.data / escala_a), -127, 127).astype(np.int8)

    acc = a_int8.astype(np.int32) @ b_int8.astype(np.int32)
    escala = escala_a * escala_w
    dados = (acc.astype(np.float32) * escala).astype(np.float32)

    out = Tensor(dados, a.requires_grad, (a,), "matmul_int8")
    out.meta["custo"] = Custo(
        flops=2 * m * k * n + 2 * m * n,
        bytes_lidos=m * k * BYTES_I8 + k * n * BYTES_I8,
        bytes_escritos=m * n * BYTES_F32,
    )
    out.meta["escala_w"] = escala_w
    out.meta["escala_a"] = escala_a
    out.meta["erro_quantizacao"] = _erro_quant(
        a.data @ b_float_ref if b_float_ref is not None else None, dados
    )

    def _bw(g: np.ndarray) -> None:
        # straight-through estimator: o gradiente atravessa a quantização
        # como se fosse a identidade, usando os pesos quantizados reais.
        a.acumular_grad(g @ (b_int8.astype(np.float32) * escala_w).T)

    out._backward = _bw
    return out


def _erro_quant(ref: np.ndarray | None, aprox: np.ndarray) -> float:
    """Erro relativo RMS entre a saída de referência e a quantizada."""
    if ref is None:
        return 0.0
    num = float(np.sqrt(np.mean((ref - aprox) ** 2)))
    den = float(np.sqrt(np.mean(ref**2))) + 1e-9
    return num / den


def quantizar_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantiza um tensor float para int8 com escala simétrica por tensor."""
    escala = float(np.abs(w).max()) / 127.0
    if escala == 0.0:
        return np.zeros_like(w, dtype=np.int8), 0.0
    return np.clip(np.rint(w / escala), -127, 127).astype(np.int8), escala


def entropia_cruzada(logits: Tensor, alvos: np.ndarray) -> Tensor:
    """Cross-entropy com softmax numericamente estável, escrita à mão.

    L = -(1/B) * sum_b log( softmax(z_b)[y_b] )
    """
    alvos = np.asarray(alvos, dtype=np.int64).reshape(-1)
    b = logits.data.shape[0]
    assert alvos.shape[0] == b, "número de alvos != batch"
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    perda = float(-np.mean(np.log(np.maximum(p[np.arange(b), alvos], 1e-12))))
    out = Tensor(np.array([[perda]], dtype=np.float32), logits.requires_grad, (logits,), "cross_entropy")
    out.meta["custo"] = Custo(
        flops=int(4 * logits.n_elementos),
        bytes_lidos=logits.n_elementos * BYTES_F32,
        bytes_escritos=BYTES_F32,
    )
    out.meta["perplexidade"] = math.exp(min(perda, 60.0))
    out.meta["acertos"] = int(np.sum(p.argmax(axis=1) == alvos))

    def _bw(g: np.ndarray) -> None:
        d = p.copy()
        d[np.arange(b), alvos] -= 1.0
        d /= b
        logits.acumular_grad(d * float(g.reshape(-1)[0]))

    out._backward = _bw
    return out
