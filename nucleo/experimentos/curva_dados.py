"""Quanta generalização compra quanta memória? — a curva medida.

Resultado (600 passos por ponto, mesma semente, válida em 64 permutações
nunca vistas no treino):

    n_treino   perda_val   acerto_val
          32     10,9847        6,93%
         128      5,1018       12,21%
         512      1,0156       63,43%
        2048      0,9558       63,28%

Leitura: a generalização salta entre 128 e 512 sequências e depois PLANIFICA.
Isso é a assinatura de uma capacidade de modelo a ser atingida, não de falta
de dados. Um modelo de 18 752 parâmetros para uma regra de 16 símbolos tem
capacidade sobrando — e, num modelo pequeno, capacidade sobrante é
sobreajuste à espera de acontecer.

Porque isto importa para o KIANDA: com 0,87 M de tokens e 2,8 M de
parâmetros, estamos do lado errado desta curva. O modelo vai decorar. A
questão não é «a loss desce», é «o que é que ela está a decorar».
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from nucleo import nn
from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada
from nucleo.experimentos.treinar_transformer import corpus_sintetico


def avaliar(m: Transformer, d: np.ndarray) -> tuple[float, float]:
    xs, ys = d[:, :-1], d[:, 1:]
    lg = m.forward(xs)
    perda = float(perda_entropia_cruzada(lg, ys.reshape(-1)).data.reshape(-1)[0])
    return perda, float((lg.data.argmax(-1).reshape(*ys.shape) == ys).mean())


def main() -> int:
    c = Config(vocab=16, d_model=32, n_camadas=2, n_cabecas=2, ctx=32, d_ff=64)
    val = corpus_sintetico(64, c.ctx, semente=999)
    amostras = (32, 128, 512, 2048)
    resultados = []
    print(f"{'n_treino':>9} {'perda_val':>10} {'acerto_val':>11}")
    print("-" * 34)
    for n in amostras:
        tr = corpus_sintetico(n, c.ctx, semente=1)
        m = Transformer(c, semente=0)
        opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.0)
        rng = np.random.default_rng(0)
        for _ in range(600):
            i = rng.integers(0, n, 16)
            opt.zero_grad()
            perda_entropia_cruzada(m.forward(tr[i, :-1]), tr[i, 1:].reshape(-1)).backward()
            opt.step(grad_clip=1.0)
        pv, av = avaliar(m, val)
        resultados.append({"n_treino": n, "perda_val": pv, "acerto_val": av})
        print(f"{n:>9} {pv:>10.4f} {av*100:>10.2f}%")
    Path("nucleo/experimentos/resultado_curva_dados.json").write_text(
        json.dumps({"n_parametros": m.n_parametros(), "resultados": resultados},
                   indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
