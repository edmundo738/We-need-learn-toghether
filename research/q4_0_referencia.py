"""Testa quantização Q4_0 (técnica do llama.cpp) em matrizes do tamanho do KIANDA."""
import sys, time, numpy as np
sys.path.insert(0,'.')

QK = 32  # QK4_0 do ggml

def quant_q4_0(x: np.ndarray):
    """Replica quantize_row_q4_0_ref: 1 float16 de escala + 16 bytes por 32 valores."""
    plano = x.reshape(-1, QK).astype(np.float32)
    nb = plano.shape[0]
    escala = np.zeros(nb, dtype=np.float16)
    q = np.zeros((nb, QK//2), dtype=np.uint8)
    for i in range(nb):
        blk = plano[i]
        idx = np.argmax(np.abs(blk))
        d = blk[idx] / -8.0
        escala[i] = np.float16(d)
        id_ = 1.0/d if d else 0.0
        v = np.clip(blk*id_ + 8.5, 0, 15).astype(np.uint8)
        q[i] = (v[:QK//2] | (v[QK//2:] << 4)).astype(np.uint8)
    return escala, q

def dequant_q4_0(escala, q, forma):
    nb = escala.shape[0]
    d = escala.astype(np.float32)
    out = np.zeros((nb, QK), dtype=np.float32)
    lo = (q & 0x0F).astype(np.int8) - 8
    hi = ((q >> 4) & 0x0F).astype(np.int8) - 8
    out[:, :QK//2] = lo.astype(np.float32) * d[:, None]
    out[:, QK//2:] = hi.astype(np.float32) * d[:, None]
    return out.reshape(forma)

rng = np.random.default_rng(42)
print(f"{'forma':<18} {'bytes fp32':>11} {'bytes q4_0':>11} {'ratio':>7} {'erro rel':>9} {'tempo':>8}")
print("-"*70)
casos = [("embedding 8192x128", (8192,128)), ("bloco 256x128", (256,128)),
         ("qkv 128x384", (128,384)), ("ffn 384x128", (384,128)),
         ("todo o modelo ~1.2M", (1200000,))]
for nome, forma in casos:
    # gerar com distribuição realista de pesos (normal, pequena variância)
    x = rng.normal(0, 0.02, forma).astype(np.float32)
    t0=time.perf_counter(); e,q = quant_q4_0(x); xr = dequant_q4_0(e,q,forma); dt=time.perf_counter()-t0
    fp32 = x.size*4; q4 = e.nbytes + q.nbytes
    err = np.linalg.norm(x-xr)/np.linalg.norm(x)
    print(f"{nome:<18} {fp32:>11,} {q4:>11,} {fp32/q4:>6.2f}x {err:>8.4f} {dt*1000:>7.1f}ms")
