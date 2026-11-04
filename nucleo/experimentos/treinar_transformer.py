"""Primeiro treino real do Transformer, num corpus sintético controlado.

Porque um corpus sintético e não o corpus v0.1: este experimento responde a
uma pergunta de engenharia («o nosso autograd treina?), não a uma pergunta de
linguagem («o modelo aprendeu português?»). Com dados sintéticos a resposta
certa é conhecida, por isso um resultado bom é interpretável. Com o corpus
real, uma loss que desce pode ser qualquer coisa.

A regra que este script tem de respeitar: **se o modelo decorar o corpus, o
teste tem de o mostrar**, e não esconder isso atrás de uma loss bonita.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nucleo import nn
from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada


def corpus_sintetico(n: int, comprimento: int, semente: int = 0):
    """Repetição com deslocamento: prever o próximo token exige memória.

    Cada sequência é uma permutação de 0..V-1 repetida. Um modelo sem atenção
    causal acerta ~1/V; um modelo que aprendeu o padrão acerta quase tudo.
    """
    rng = np.random.default_rng(semente)
    # Cada sequência é uma permutação DISTINTA de 0..15 repetida dentro de si.
    # A regra a aprender é condicional ao prefixo: dado o token inicial, o
    # próximo é o que o segue nesta permutação. Isto distingue «aprendeu a
    # regra» de «decorou as 64 sequências de treino»: a validação usa
    # permutações que o treino nunca viu.
    #
    # Porque não uma só permutação para todas: com uma só, a tabela de
    # posições que o modelo tem não cobre 33 posições e o modelo fica
    # obrigado a memorizar cada offset — validação e treino divergem por
    # construção, e o resultado não diz nada sobre o autograd.
    dados = []
    for _ in range(n):
        base = rng.permutation(16)
        dados.append(np.resize(base, comprimento + 1))
    return np.array(dados, dtype=np.int64)


def avaliar(modelo: Transformer, sequencias: np.ndarray) -> tuple[float, float]:
    """Devolve (perda, acerto top-1) na previsão do próximo token."""
    xs = sequencias[:, :-1]
    ys = sequencias[:, 1:]
    B, T = xs.shape
    logits = modelo.forward(xs)
    perda = float(perda_entropia_cruzada(logits, ys.reshape(-1)).data.reshape(-1)[0])
    previsao = logits.data.argmax(axis=-1).reshape(B, T)
    acerto = float((previsao == ys).mean())
    return perda, acerto


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passos", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", type=str, default="nucleo/experimentos/resultado_treino.json")
    args = ap.parse_args()

    cfg = Config(vocab=16, d_model=32, n_camadas=2, n_cabecas=2, ctx=32, d_ff=64)
    cfg.validar()
    modelo = Transformer(cfg, semente=args.semente)

    # O que o modelo vê: as primeiras 17 posições de uma sequência dão a
    # permutação inteira. O que tem de prever: os tokens seguintes. Treino e
    # validação usam permutações disjuntas.
    treino = corpus_sintetico(96, cfg.ctx, semente=1)
    validacao = corpus_sintetico(24, cfg.ctx, semente=999)

    opt = nn.Adam(modelo.parametros(), eta=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.semente)

    p0, a0 = avaliar(modelo, validacao)
    print(f"modelo: {modelo.n_parametros():,} params | vocab {cfg.vocab} | ctx {cfg.ctx}")
    print(f"baseline aleatorio: perda={p0:.4f} (log V = {np.log(cfg.vocab):.4f}) | acerto={a0*100:.2f}%")
    print()
    print(f"{'passo':>6} {'perda_treino':>13} {'perda_val':>10} {'acerto_val':>11} {'ms':>7}")
    print("-" * 54)

    historico = []
    t0 = time.perf_counter()
    for passo in range(1, args.passos + 1):
        idx = rng.integers(0, len(treino), 8)
        lote = treino[idx]
        xs, ys = lote[:, :-1], lote[:, 1:]

        opt.zero_grad()
        logits = modelo.forward(xs)
        perda = perda_entropia_cruzada(logits, ys.reshape(-1))
        perda.backward()
        opt.step(grad_clip=1.0)

        if passo % 25 == 0 or passo == 1:
            pv, av = avaliar(modelo, validacao)
            pt = float(perda.data.reshape(-1)[0])
            dt = (time.perf_counter() - t0) / passo * 1000
            print(f"{passo:>6} {pt:>13.4f} {pv:>10.4f} {av*100:>10.2f}% {dt:>6.0f}")
            historico.append({"passo": passo, "perda_treino": pt, "perda_val": pv, "acerto_val": av})

    p_final, a_final = avaliar(modelo, validacao)
    dur = time.perf_counter() - t0
    resultado = {
        "config": cfg.__dict__,
        "n_parametros": modelo.n_parametros(),
        "passos": args.passos,
        "lr": args.lr,
        "segundos": round(dur, 1),
        "baseline": {"perda": p0, "acerto": a0},
        "final": {"perda": p_final, "acerto": a_final},
        "historico": historico,
        "nota": "Corpus sintetico. Mede a cadeia treino->pesos->geracao, nao a qualidade linguistica.",
    }
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False))

    print()
    print(f"final: perda={p_final:.4f} acerto={a_final*100:.2f}% | {dur:.1f}s")
    print(f"gravado em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
