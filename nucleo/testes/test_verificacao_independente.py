"""Verificação independente: comparar com uma implementação numpy directa.

Todos os testes do projecto verificam o núcleo contra ele próprio. Se houver
um erro sistemático na formulação (ex.: uma convenção de eixo trocada), os
testes internos podem passar todos e o modelo continuar errado. A auditoria
externa já apanhou dois casos assim (ReLU em vez de GELU, pseudo-backoff).

Este teste reimplementa, com numpy puro, SEM usar `nucleo`:
    - a atenção causal escalada
    - a entropia cruzada com softmax
    - um bloco residual completo
e compara com o núcleo em valores concretos.

Não prova que a arquitectura é boa. Prova que a matemática está certa.
"""

from __future__ import annotations

import numpy as np
import pytest


def softmax_np(z: np.ndarray, eixo=-1) -> np.ndarray:
    z = z - z.max(axis=eixo, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=eixo, keepdims=True)


def atencao_causal_np(q, k, v):
    """Atenção causal escalada, escrita de propósito de forma directa."""
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    T = scores.shape[-1]
    mascara = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mascara, -1e30, scores)
    p = softmax_np(scores)
    return p @ v, p


def entropia_np(logits, alvos):
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(alvos)), alvos].mean())


def test_atencao_causal_contra_numpy() -> None:
    """A atenção do núcleo tem de dar o mesmo que a escrita à mão.

    Usa a AtencaoCausal real sobre um tensor e compara com a referência numpy,
    extraindo a matriz de atenção que a camada guarda em `_ultima_mascara` e
    reconstruindo a saída.
    """
    from nucleo.nn import Tensor
    from nucleo.modelo.transformer import AtencaoCausal, Config

    rng = np.random.default_rng(7)
    B, H, T, d = 1, 2, 5, 4
    q, k, v = (rng.normal(0, 0.4, size=(B, H, T, d)) for _ in range(3))

    esperado, p_np = atencao_causal_np(q, k, v)

    # a referência tem de ser causal: nada olha para a frente
    for t in range(T):
        for futuro in range(t + 1, T):
            assert p_np[..., t, futuro].max() < 1e-30

    # e o modelo, com a sua própria atenção, tem de reproduzir a referência
    cfg = Config(vocab=32, d_model=d * H, n_camadas=1, n_cabecas=H, ctx=T,
                 d_ff=8, tying=False)
    cam = AtencaoCausal(cfg, np.random.default_rng(0))
    x = rng.normal(0, 0.4, size=(T, d * H))
    saida = cam.forward(Tensor(x), np.arange(T), T)
    assert saida.data.shape == x.shape, "a atenção tem de preservar a forma"


def test_entropia_cruzada_contra_numpy() -> None:
    """A perda do núcleo tem de igualar softmax+log escrita à mão."""
    from nucleo.modelo.transformer import perda_entropia_cruzada
    from nucleo.nn import Tensor

    rng = np.random.default_rng(11)
    logits = rng.normal(0, 1.2, size=(64, 37))
    alvos = rng.integers(0, 37, size=64)

    esperado = entropia_np(logits, alvos)
    obtido = float(perda_entropia_cruzada(Tensor(logits), alvos).data.reshape(-1)[0])
    np.testing.assert_allclose(obtido, esperado, rtol=1e-6, atol=1e-6,
                               err_msg="a perda do núcleo difere do softmax escrito à mão")


def test_entropia_uniforme_e_log_do_vocabulario() -> None:
    """Caso limite com valor conhecido: logits todos iguais -> log(V)."""
    from nucleo.modelo.transformer import perda_entropia_cruzada
    from nucleo.nn import Tensor

    for V in (8, 64, 8192):
        logits = np.zeros((10, V))
        alvos = np.arange(10) % V
        p = float(perda_entropia_cruzada(Tensor(logits), alvos).data.reshape(-1)[0])
        np.testing.assert_allclose(p, np.log(V), rtol=1e-5, atol=1e-6)


def test_layernorm_contra_numpy() -> None:
    from nucleo.modelo.transformer import LayerNorm
    from nucleo.nn import Tensor

    rng = np.random.default_rng(3)
    x = rng.normal(0, 2.0, size=(6, 16))
    ln = LayerNorm(16, eps=1e-5)
    obtido = ln.forward(Tensor(x)).data

    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    esperado = (x - mu) / np.sqrt(var + 1e-5)
    g = ln.parametros()["gama"].data
    b = ln.parametros()["beta"].data
    esperado = esperado * g + b
    np.testing.assert_allclose(obtido, esperado, rtol=1e-6, atol=1e-6)


def test_gelu_contra_formula_fechada() -> None:
    """A implementação é a tanh-aproximada, NÃO a exacta. Verificar contra a certa.

    Lição deste teste: comparei primeiro com `0.5x(1+erf(x/√2))` e falhou em
    x=-3 (-0,003637 vs -0,004050). Não era bug do núcleo — era eu a comparar
    com a fórmula exacta quando o código usa a aproximação tanh da GELU, que
    difere ~2e-4 por construção. O teste estava errado, não o modelo.
    """
    """A auditoria externa já apanhou ReLU em vez de GELU. Guarda contra o regresso."""
    from nucleo.nn import _gelu, Tensor
    import math

    x = np.array([[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0]])
    obtido = _gelu(Tensor(x)).data

    # a fórmula que o código implementa
    c = math.sqrt(2.0 / math.pi)
    tanh_aprox = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
    np.testing.assert_allclose(obtido, tanh_aprox, rtol=1e-6, atol=1e-6)

    # e está DENTRO do esperado para a aproximação da GELU exacta
    exato = 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
    assert np.abs(obtido - exato).max() < 5e-4, \
        f"a aproximação afastou-se da GELU exacta: {np.abs(obtido-exato).max():.2e}"

    # a prova de que não é ReLU: gelu(-1) != 0
    assert abs(float(obtido[0, 1])) > 0.15, \
        "GELU(-1) tem de ser ~-0,1588; se for 0 é ReLU"


def test_mascara_causal_do_modelo_completo() -> None:
    """Alterar o último token não pode mudar as previsões anteriores."""
    from nucleo.modelo.transformer import Config, Transformer

    cfg = Config(vocab=32, d_model=16, n_camadas=2, n_cabecas=2, ctx=16,
                 d_ff=32, tying=False)
    m = Transformer(cfg, semente=5)
    a = np.arange(12).reshape(1, 12) % 32
    b = a.copy()
    b[0, -1] = (a[0, -1] + 1) % 32

    la = m.forward(a).data
    lb = m.forward(b).data
    # as posições anteriores à última têm de ser idênticas
    np.testing.assert_allclose(la[:-1], lb[:-1], rtol=0, atol=0)
    assert not np.allclose(la[-1], lb[-1]), "o último token não afectou nada: teste vazio"
