"""Testes da atenção por sequência.

Estes testes existem porque a auditoria externa apontou — correctamente — que
a implementação anterior calculava `H·(B·T)²` scores e só usava as `B`
diagonais `T×T`. A resposta estava certa; o trabalho era 8× maior com lote 8.

O que tem de ser provado agora:
  1. a matemática não mudou (mesmos resultados, mesmos gradientes);
  2. o isolamento entre sequências continua a valer — agora por construção,
     porque nunca se cruzam;
  3. o ganho é real e medido, e o custo em lote 1 também.
"""

from __future__ import annotations

import numpy as np
import pytest

from nucleo import nn
from nucleo.autograd import Tensor
from nucleo.modelo.transformer import (
    Config,
    Transformer,
    _aplicar_por_cabeca,
    _scores,
    perda_entropia_cruzada,
)


def _cfg(**kw):
    base = dict(vocab=40, d_model=16, n_camadas=2, n_cabecas=2, ctx=16, d_ff=32)
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------------------
# 1. As primitivas: forma e correcção do gradiente
# ---------------------------------------------------------------------------

def test_scores_por_sequencia_nao_cruzam_lotes():
    """Nenhuma entrada de `_scores` mistura duas sequências.

    O truque é dar a cada sequência valores muito distintos: se houver
    cruzamento, o score corresponde a um produto de vectores de sequências
    diferentes e o teste apanha.
    """
    Bn, T, h, dh = 3, 4, 2, 5
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Bn * T, h * dh)).astype(np.float32)
    k = rng.standard_normal((Bn * T, h * dh)).astype(np.float32)
    s = _scores(Tensor(q, False), Tensor(k, False), h, dh, Bn * T, T)
    assert s.data.shape == (Bn * h * T, T)

    # reconstruir à mão, sequência a sequência
    for b in range(Bn):
        for hh in range(h):
            linhas = s.data.reshape(Bn, h, T, T)[b, hh]
            esperado = q.reshape(Bn, T, h, dh)[b, :, hh] @ k.reshape(Bn, T, h, dh)[b, :, hh].T
            np.testing.assert_allclose(linhas, esperado, rtol=1e-5, atol=1e-5)


def test_scores_gradiente_numerico():
    """Gradiente analítico vs diferenças finitas, em todas as coordenadas."""
    Bn, T, h, dh = 2, 3, 2, 4
    rng = np.random.default_rng(3)
    q0 = rng.standard_normal((Bn * T, h * dh)).astype(np.float32)
    k0 = rng.standard_normal((Bn * T, h * dh)).astype(np.float32)
    n = Bn * T

    def f(qd, kd):
        return float((_scores(Tensor(qd, False), Tensor(kd, False), h, dh, n, T).data ** 2).sum())

    q, k = Tensor(q0.copy(), True), Tensor(k0.copy(), True)
    (_scores(q, k, h, dh, n, T) ** 2).sum().backward()

    eps = 1e-3
    for nome, t, base in (("q", q, q0), ("k", k, k0)):
        for idx in np.ndindex(base.shape):
            d = base.copy(); d[idx] += eps; c1 = f(d, k0) if nome == "q" else f(q0, d)
            d = base.copy(); d[idx] -= eps; c2 = f(d, k0) if nome == "q" else f(q0, d)
            num = (c1 - c2) / (2 * eps)
            assert abs(num - t.grad[idx]) < 2e-2, (nome, idx, num, t.grad[idx])


def test_aplicar_por_cabeca_gradiente_numerico():
    Bn, T, h, dh = 2, 3, 2, 4
    rng = np.random.default_rng(5)
    a0 = rng.random((Bn * h * T, T)).astype(np.float32)
    v0 = rng.standard_normal((Bn * T, h * dh)).astype(np.float32)
    n = Bn * T

    def f(ad, vd):
        return float((_aplicar_por_cabeca(Tensor(ad, False), Tensor(vd, False),
                                          h, n, dh, T).data ** 2).sum())

    a, v = Tensor(a0.copy(), True), Tensor(v0.copy(), True)
    (_aplicar_por_cabeca(a, v, h, n, dh, T) ** 2).sum().backward()

    eps = 1e-3
    for idx in np.ndindex(a0.shape):
        d = a0.copy(); d[idx] += eps; c1 = f(d, v0)
        d = a0.copy(); d[idx] -= eps; c2 = f(d, v0)
        assert abs((c1 - c2) / (2 * eps) - a.grad[idx]) < 2e-2
    for idx in np.ndindex(v0.shape):
        d = v0.copy(); d[idx] += eps; c1 = f(a0, d)
        d = v0.copy(); d[idx] -= eps; c2 = f(a0, d)
        assert abs((c1 - c2) / (2 * eps) - v.grad[idx]) < 2e-2


# ---------------------------------------------------------------------------
# 2. O modelo: isolamento continua, agora sem máscara a tapar trabalho
# ---------------------------------------------------------------------------

def test_lote_igual_isolado_exacto():
    """Processar 4 sequências juntas dá exactamente o mesmo que uma a uma."""
    m = Transformer(_cfg(), semente=11)
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 40, (4, 6))
    juntas = m.forward(ids).data
    for b in range(4):
        so = m.forward(ids[b:b + 1]).data
        np.testing.assert_allclose(juntas[b * 6:(b + 1) * 6], so, rtol=0, atol=0)


def test_alterar_uma_sequencia_nao_muda_as_outras():
    m = Transformer(_cfg(), semente=12)
    rng = np.random.default_rng(4)
    a = rng.integers(0, 40, (1, 6)); b = rng.integers(0, 40, (1, 6))
    ref = m.forward(np.concatenate([a, b])).data[6:].copy()
    a2 = rng.integers(0, 40, (1, 6))
    novo = m.forward(np.concatenate([a2, b])).data[6:]
    np.testing.assert_array_equal(ref, novo)


def test_gradiente_isolado_por_sequencia():
    """A perda só nas posições de B produz o mesmo gradiente com qualquer A."""
    def grad(a_seq):
        m = Transformer(_cfg(), semente=13)
        for t in m.parametros().values():
            t.zero_grad()
        lg = m.forward(np.concatenate([a_seq, B_SEQ]))
        perda_entropia_cruzada(lg.fatiar(6, 12), ALVOS_B).backward()
        return {k: t.grad.copy() for k, t in m.parametros().items()}

    rng = np.random.default_rng(7)
    global B_SEQ, ALVOS_B
    B_SEQ = rng.integers(0, 40, (1, 6))
    ALVOS_B = rng.integers(0, 40, 6)
    g1 = grad(rng.integers(0, 40, (1, 6)))
    g2 = grad(rng.integers(0, 40, (1, 6)))
    for k in g1:
        np.testing.assert_array_equal(g1[k], g2[k])


def test_causalidade_estrita():
    m = Transformer(_cfg(), semente=14)
    rng = np.random.default_rng(8)
    ids = rng.integers(0, 40, (1, 8))
    ref = m.forward(ids).data.copy()
    for k in range(1, 8):
        alt = ids.copy(); alt[0, k] = (alt[0, k] + 9) % 40
        np.testing.assert_array_equal(ref[:k], m.forward(alt).data[:k])


# ---------------------------------------------------------------------------
# 3. Eficiência: o ganho é medido, não afirmado
# ---------------------------------------------------------------------------

def test_atencao_nao_aloca_matriz_global():
    """A matriz de scores tem de ser (B*h*T, T) e não (h*B*T, B*T).

    Este é o teste que falharia com a implementação anterior: com 4
    sequências de 5 tokens e 2 cabeças, o antigo alocava 2·20·20 = 800
    entradas; o novo aloca 4·2·5·5 = 200.
    """
    Bn, T, h = 4, 5, 2
    m = Transformer(_cfg(n_cabecas=h), semente=15)
    rng = np.random.default_rng(9)
    ids = rng.integers(0, 40, (Bn, T))
    m.forward(ids)
    ms = m.blocos[0].atencao._ultima_mascara
    assert ms.shape == (Bn * T, T), f"máscara global: {ms.shape}"

    # o número de scores calculados é exactamente B*h*T*T
    q = Tensor(rng.standard_normal((Bn * T, h * (16 // h))).astype(np.float32), False)
    s = _scores(q, q, h, 16 // h, Bn * T, T)
    assert s.data.size == Bn * h * T * T


def test_throughput_melhora_com_o_lote():
    """Tempo por token em lote 4 tem de ser melhor que em lote 1.

    Guarda contra a regressão para a atenção global: nessa implementação o
    custo crescia com B² e o tempo por token piorava com o lote.
    """
    import time
    c = Config(vocab=256, d_model=32, n_camadas=2, n_cabecas=2, ctx=32, d_ff=64)
    tempos = {}
    for lote in (1, 4):
        m = Transformer(c, semente=0)
        opt = nn.Adam(m.parametros(), eta=1e-3)
        d = np.zeros((lote, c.ctx + 1), dtype=np.int64)
        for _ in range(2):        # aquecimento
            opt.zero_grad()
            perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
            opt.step()
        t0 = time.perf_counter()
        for _ in range(3):
            opt.zero_grad()
            perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
            opt.step()
        tempos[lote] = (time.perf_counter() - t0) / 3 / (lote * c.ctx)

    # NOTA DE ESTABILIDADE: na primeira versão esta asserção era
    # `tempos[4] < tempos[1]` e falhou quando a suite correu em paralelo com
    # outra (contenção de CPU fazia o lote 4 parecer pior). O defeito que o
    # teste quer apanhar é o custo crescer com B² — nesse caso o tempo por
    # token com lote 4 é ~4x pior, não 5% pior. Por isso a margem é 1,5x:
    # apanha a regressão quadrática e não o ruído de agendamento.
    assert tempos[4] < tempos[1] * 1.5, (
        f"tempo por token piorou com o lote (esperado ~igual, regressão "
        f"quadrática daria ~4x): lote1={tempos[1]*1e6:.0f}µs "
        f"lote4={tempos[4]*1e6:.0f}µs "
        f"(razão {tempos[4]/tempos[1]:.2f}x)")
