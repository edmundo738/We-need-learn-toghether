"""Testes do motor de autograd: verificação numérica de gradientes.

A regra é simples e não negociável: **nenhuma camada entra no projecto sem
passar um teste de gradiente numérico**. Um erro de sinal na regra da cadeia
produz um modelo que aprende ao contrário e o sintoma parece "o modelo é mau".

Correr com:  python -m pytest testes/ -q
"""

from __future__ import annotations

import math

import numpy as np

from kianda.nucleo.autograd import (
    Tensor,
    entropia_cruzada,
    matmul,
    matmul_int8,
    quantizar_int8,
)
from kianda.nucleo.nn import Adam, Linear, LoRA, MLP, calibrar_escala, podar_estruturado


def _numerico(f, x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Gradiente por diferenças centrais, elemento a elemento.

    Nota empírica deste projecto: em float32, `eps=1e-4` produz um erro
    *maior* do que `eps=1e-2`. A diferença `f(x+e) - f(x-e)` perde precisão
    por cancelamento catastrófico (a perda varia ~1e-6 e o float32 só tem
    ~7 dígitos). Medido: erro relativo L2 de 6.8e-3 com eps=1e-4 contra
    6.7e-5 com eps=1e-2. Quem faz verificação de gradientes em float32
    deve usar passos maiores, não menores.
    """
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        fp = f()
        x[i] = orig - eps
        fm = f()
        x[i] = orig
        g[i] = (fp - fm) / (2 * eps)
    return g


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a) + np.abs(b)) + 1e-9))


# --------------------------------------------------------------------------
# O autograd contra diferenças finitas
# --------------------------------------------------------------------------


def test_matmul_gradiente():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    ta, tb = Tensor(a.copy(), True), Tensor(b.copy(), True)
    out = matmul(ta, tb)
    out.sum().backward()
    # d(sum(A@B))/dA = ones @ B^T ; d/dB = A^T @ ones
    ga_esperado = np.ones((3, 5), dtype=np.float32) @ b.T
    gb_esperado = a.T @ np.ones((3, 5), dtype=np.float32)
    assert _rel(ta.grad, ga_esperado) < 1e-4
    assert _rel(tb.grad, gb_esperado) < 1e-4


def test_cadeia_completa_contra_diferencas_finitas():
    """Uma rede pequena inteira: o gradiente tem de coincidir ao 1e-3 relativo."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 3)).astype(np.float32)
    y = np.array([0, 2, 1, 0, 2, 1])
    W1 = rng.standard_normal((3, 8)).astype(np.float32) * 0.3
    W2 = rng.standard_normal((8, 3)).astype(np.float32) * 0.3

    def perda(W2x):
        h = Tensor(X) @ Tensor(W1)
        h = h.relu()
        z = h @ Tensor(W2x)
        return entropia_cruzada(z, y).data.item()

    t2 = Tensor(W2.copy(), True)
    h = (Tensor(X) @ Tensor(W1)).relu()
    entropia_cruzada(h @ t2, y).backward()
    # atenção: passar o MESMO array que a função usa, não uma cópia,
    # senão a perturbação nunca chega à perda.
    g_num = _numerico(lambda: perda(W2), W2)
    # erro relativo L2 (a métrica robusta para gradientes com entradas perto de zero)
    den = np.linalg.norm(g_num) + 1e-12
    assert np.linalg.norm(t2.grad - g_num) / den < 1e-3, "gradiente não coincide"


def test_softmax_estabilidade():
    """Logits muito grandes não devem produzir NaN."""
    z = Tensor(np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32), True)
    s = z.softmax()
    assert np.all(np.isfinite(s.data))
    assert abs(float(s.data.sum()) - 1.0) < 1e-5


def test_broadcasting_no_bias():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    b = rng.standard_normal((1, 4)).astype(np.float32)
    tb = Tensor(b.copy(), True)
    (Tensor(x) + tb).sum().backward()
    assert tb.grad.shape == (1, 4)
    assert np.allclose(tb.grad, 5.0, atol=1e-6)


# --------------------------------------------------------------------------
# Contabilidade de custo
# --------------------------------------------------------------------------


def test_custo_de_matmul_exacto():
    a, b = Tensor(np.zeros((2, 3))), Tensor(np.zeros((3, 4)))
    out = matmul(a, b)
    assert out.meta["custo"].flops == 2 * 2 * 3 * 4


def test_custo_da_rede_soma():
    rng = np.random.default_rng(3)
    cam = Linear(10, 5, rng)
    x = Tensor(rng.standard_normal((7, 10)).astype(np.float32))
    out = cam(x)
    # matmul (2*7*10*5) + soma do bias (7*5)
    assert out.custo_graph().flops == 2 * 7 * 10 * 5 + 7 * 5


# --------------------------------------------------------------------------
# Técnicas de eficiência
# --------------------------------------------------------------------------


def test_quantizacao_mantem_ordem_de_grandeza():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    q, escala = quantizar_int8(w)
    assert q.dtype == np.int8
    assert q.max() <= 127 and q.min() >= -127
    w2 = q.astype(np.float32) * escala
    erro = float(np.abs(w - w2).max())
    assert erro < 0.02 * max(float(np.abs(w).max()), 1e-6)


def test_matmul_int8_proximo_do_float():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((8, 6)).astype(np.float32)
    q, esc = quantizar_int8(w)
    from kianda.nucleo.autograd import matmul_int8

    ref = x @ w
    aprox = matmul_int8(Tensor(x), q, esc, w).data
    # erro relativo RMS abaixo de 2% e erro absoluto pequeno
    err_rel = np.sqrt(np.mean((ref - aprox) ** 2)) / np.sqrt(np.mean(ref**2))
    assert err_rel < 0.02, f"erro relativo {err_rel:.4f}"
    assert np.abs(ref - aprox).max() < 1.0


def test_poda_estruturada_zera_colunas():
    cam = Linear(6, 8, np.random.default_rng(6))
    r = podar_estruturado([cam], 0.5)
    z = (np.abs(cam.W.data).sum(axis=0) == 0).sum()
    assert z == 4
    assert r["poupanca_flops_pct"] > 40


def test_lora_tem_menos_parametros_que_a_base():
    base = Linear(64, 64, np.random.default_rng(7))
    lora = LoRA(base, rank=4)
    assert lora.n_parametros_treinaveis() < base.n_parametros() / 4
    rng = np.random.default_rng(8)
    x = Tensor(rng.standard_normal((3, 64)).astype(np.float32))
    # LoRA inicializada em zeros => saída idêntica à base
    assert np.allclose(lora(x).data, base(x).data, atol=1e-6)


def test_adam_desce_uma_quadratica():
    rng = np.random.default_rng(9)
    alvo = np.array([[3.0, -2.0]], dtype=np.float32)
    p = Tensor(np.zeros((1, 2), dtype=np.float32), True)
    opt = Adam({"p": p}, eta=0.1)
    for _ in range(200):
        perda = ((p - Tensor(alvo)) ** 2.0).sum()
        perda.backward()
        opt.step()
    assert np.allclose(p.data, alvo, atol=0.05)


def test_calibracao_reporta_erro():
    rng = np.random.default_rng(10)
    modelo = [Linear(8, 16, rng), Linear(16, 4, rng)]
    x = rng.standard_normal((20, 8)).astype(np.float32)
    r = calibrar_escala(modelo, x)
    assert r["reducao_memoria_pct"] == 75.0
    assert 0.0 <= r["kl_medio"] < 5.0


def test_gelu_e_finito():
    rng = np.random.default_rng(11)
    m = MLP(8, 32, 4, rng)
    out = m(Tensor(rng.standard_normal((5, 8)).astype(np.float32)))
    assert np.all(np.isfinite(out.data))
    assert out.data.shape == (5, 4)


if __name__ == "__main__":
    import sys

    falhas = 0
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {nome}")
            except AssertionError as e:
                falhas += 1
                print(f"  FALHA {nome}: {e}")
    print("tudo verde" if not falhas else f"{falhas} falhas")
    sys.exit(1 if falhas else 0)
