"""Benchmark da atenção: global com máscara vs por sequência.

Existe porque a auditoria externa mostrou que a implementação global
calculava H·(B·T)² scores e só usava as B diagonais T×T — trabalho B vezes
maior do que o necessário, escondido por uma máscara que deitava fora o
resultado.

Corre o mesmo passo (forward + backward + optimizer) em vários tamanhos de
lote e reporta ms/passo e tokens/s. Reproduzível:

    python -m nucleo.experimentos.bench_atencao

Nota sobre o que se mede: é o passo COMPLETO, não só a atenção. A atenção é
uma fracção do tempo (logits e MLP pesam), por isso o ganho de 3,96x em lote
8 não é 8x — é o que a Amdahl permite.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nucleo import nn
from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada


def medir(lote: int, repeticoes: int = 3) -> dict:
    c = Config(vocab=8192, d_model=128, n_camadas=4, n_cabecas=4, ctx=128, d_ff=384)
    m = Transformer(c, semente=0)
    opt = nn.Adam(m.parametros(), eta=1e-3)
    d = np.zeros((lote, c.ctx + 1), dtype=np.int64)

    # aquecimento
    opt.zero_grad()
    perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
    opt.step()

    t0 = time.perf_counter()
    for _ in range(repeticoes):
        opt.zero_grad()
        perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
        opt.step()
    dt = (time.perf_counter() - t0) / repeticoes
    tokens = lote * c.ctx
    return {"lote": lote, "ms_passo": round(dt * 1000, 1),
            "tokens_s": round(tokens / dt), "tokens": tokens}


def main() -> None:
    print(f"{'lote':>5} {'ms/passo':>10} {'tokens/s':>10}")
    linhas = []
    for lote in (1, 2, 4, 8):
        r = medir(lote)
        linhas.append(r)
        print(f"{r['lote']:>5} {r['ms_passo']:>10.1f} {r['tokens_s']:>10}")
    Path("nucleo/experimentos/resultado_bench_atencao.json").write_text(
        json.dumps(linhas, indent=2))
    print("\nEsperado: tokens/s a CRESCER com o lote até ao limite de memória.")
    print("Se piorar, a atenção voltou a calcular pares entre sequências.")


if __name__ == "__main__":
    main()
