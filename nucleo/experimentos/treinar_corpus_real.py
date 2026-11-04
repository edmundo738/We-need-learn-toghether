"""Primeiro treino no corpus REAL, com o tokenizador v1 a fechar o ciclo.

A pergunta deste script não é «o modelo é bom». É a única que ainda não tinha
resposta com dados reais:

    A cadeia completa funciona de ponta a ponta?

    texto -> tokenizador v1 -> ids -> Transformer -> logits -> ids -> texto

Se o texto que sai do outro lado for legível — mesmo mau — então a máquina
existe. Se sair lixo binário, falta qualquer coisa. Se sair cópia literal do
treino, o modelo decorou — e isso também é uma resposta, só que diferente.

Regra que este script respeita: guarda a amostra gerada tal como saiu, sem
escolher a melhor de várias tentativas.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nucleo import nn
from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada
from nucleo.tokenizador.bpe import TokenizadorBPE

RAIZ = Path(__file__).resolve().parents[2]
OBRAS = RAIZ / "dados" / "obras"


def carregar_corpus() -> str:
    partes = []
    for f in sorted(OBRAS.glob("*.txt")):
        partes.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(partes)


def empacotar(ids: list[int], ctx: int, n: int | None = None) -> np.ndarray:
    """Corta o fluxo de ids em janelas de `ctx` tokens.

    Sem sobreposição, sem padding, sem tokens especiais: o corpus é um fluxo
    contínuo e cada janela é um pedaço dele. O que fica de fora é o resto que
    não enche uma janela — declarado no resultado, não silenciado.
    """
    total = len(ids) // (ctx + 1)
    if n is not None:
        total = min(total, n)
    arr = np.empty((total, ctx + 1), dtype=np.int64)
    for i in range(total):
        arr[i] = ids[i * (ctx + 1):(i + 1) * (ctx + 1)]
    return arr


def avaliar(modelo: Transformer, dados: np.ndarray, max_lote: int = 8) -> tuple[float, float]:
    """Avalia em lotes pequenos, e a razão é medida, não estética.

    A memória do nosso autograd cresce com o QUADRADO do número de tokens no
    lote, não linearmente: a matriz de atenção é (B*T, B*T). Medido neste
    ambiente (4 camadas, d_model 128, ctx 64):

        8 janelas  -> 0.58 s | 278 MB
       16 janelas  -> 2.02 s | 751 MB
       32 janelas  -> 8.29 s | 2 471 MB   (e a 51 o processo é morto)

    Com 3 GB de RAM isto obriga a avaliar em lotes de 8. É um limite real do
    autograd experimental, não um detalhe de configuração.
    """
    perdas, acertos = [], []
    for i in range(0, len(dados), max_lote):
        lote = dados[i:i + max_lote]
        xs, ys = lote[:, :-1], lote[:, 1:]
        logits = modelo.forward(xs)
        perdas.append(float(perda_entropia_cruzada(logits, ys.reshape(-1)).data.reshape(-1)[0]))
        acertos.append(float((logits.data.argmax(-1).reshape(*ys.shape) == ys).mean()))
    return float(np.mean(perdas)), float(np.mean(acertos))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passos", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--camadas", type=int, default=4)
    ap.add_argument("--cabecas", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=384)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--n_janelas", type=int, default=4000)
    ap.add_argument("--saida", type=str, default="nucleo/experimentos/resultado_corpus_real.json")
    args = ap.parse_args()

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    texto = carregar_corpus()
    print(f"corpus: {len(texto):,} caracteres de {len(list(OBRAS.glob('*.txt')))} obras")

    # Cache de ids em disco. A tokenização é determinística; repeti-la a cada
    # arranque custa segundos e, mais importante, faz uma sessão de treino
    # lenta parecer um problema do modelo quando é do pipeline.
    cache = Path("/tmp/kianda_ids_corpus.json")
    if cache.exists():
        ids = json.loads(cache.read_text())
        print(f"ids carregados da cache: {len(ids):,} | vocab {tok.tamanho_vocab}")
    else:
        t0 = time.perf_counter()
        ids = tok.codificar(texto)
        cache.write_text(json.dumps(ids))
        print(f"tokenizado: {len(ids):,} ids em {time.perf_counter()-t0:.1f}s "
              f"| vocab {tok.tamanho_vocab}")
    assert tok.descodificar(ids[:200]) == texto[:len(tok.descodificar(ids[:200]))], "round-trip"

    dados = empacotar(ids, args.ctx, args.n_janelas)
    resto = len(ids) - dados.size
    print(f"janelas de {args.ctx}: {len(dados):,} | descartados {resto} ids no fim")

    # divisão 90/10 por janela — não por ficheiro, para o modelo ver as duas
    corte = int(len(dados) * 0.9)
    treino, validacao = dados[:corte], dados[corte:]
    print(f"treino {len(treino):,} janelas | validacao {len(validacao):,}")

    cfg = Config(vocab=tok.tamanho_vocab, d_model=args.d_model,
                 n_camadas=args.camadas, n_cabecas=args.cabecas,
                 ctx=args.ctx, d_ff=args.d_ff)
    cfg.validar()
    modelo = Transformer(cfg, semente=args.semente)
    print(f"modelo: {modelo.n_parametros():,} parametros | {modelo.n_parametros()*4/1048576:.1f} MB fp32")

    opt = nn.Adam(modelo.parametros(), eta=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.semente)

    p0, a0 = avaliar(modelo, validacao)
    print(f"baseline aleatorio: perda {p0:.4f} (log V = {np.log(cfg.vocab):.3f}) "
          f"| acerto {a0*100:.2f}%")
    print()
    print(f"{'passo':>6} {'perda_tr':>9} {'perda_val':>10} {'acerto_val':>11} {'s/passo':>8}")
    print("-" * 50)

    historico = []
    t0 = time.perf_counter()
    for passo in range(1, args.passos + 1):
        idx = rng.integers(0, len(treino), args.lote)
        lote = treino[idx]
        opt.zero_grad()
        logits = modelo.forward(lote[:, :-1])
        perda = perda_entropia_cruzada(logits, lote[:, 1:].reshape(-1))
        perda.backward()
        opt.step(grad_clip=1.0)

        if passo % 25 == 0 or passo == 1:
            pv, av = avaliar(modelo, validacao)
            pt = float(perda.data.reshape(-1)[0])
            dt = (time.perf_counter() - t0) / passo
            print(f"{passo:>6} {pt:>9.4f} {pv:>10.4f} {av*100:>10.2f}% {dt:>7.2f}s")
            historico.append({"passo": passo, "perda_treino": pt,
                              "perda_val": pv, "acerto_val": av})

    # ---- geração: o teste que se lê, não o que se mede ----
    print("\n" + "=" * 60)
    print("AMOSTRAS GERADAS (greedy, temperatura 0)")
    print("=" * 60)
    inicios = [tok.codificar("Eu não tenho filtro"),
               tok.codificar("Zuleika olhou"),
               tok.codificar("Há coisas que")]
    amostras = []
    for prompt_ids in inicios:
        saida = modelo.gerar(np.array([prompt_ids]), n_novos=60, temperatura=0.0)
        texto_gerado = tok.descodificar(saida[0].tolist())
        amostras.append({"prompt": tok.descodificar(prompt_ids), "gerado": texto_gerado})
        print(f"\nprompt: {tok.descodificar(prompt_ids)!r}")
        print(f"saida:  {texto_gerado!r}")

    # ---- quanto do gerado é cópia literal do treino? ----
    print("\n" + "=" * 60)
    copia = []
    for a in amostras:
        trecho = a["gerado"][len(a["prompt"]):]
        n_gramas = [trecho[i:i + 20] for i in range(0, max(1, len(trecho) - 20), 10)]
        encontrados = sum(1 for g in n_gramas if g and g in texto)
        frac = encontrados / len(n_gramas) if n_gramas else 0.0
        copia.append(frac)
        print(f"cópia literal (20-gramas do gerado que existem no corpus): {frac*100:.0f}%")

    p_final, a_final = avaliar(modelo, validacao)
    dur = time.perf_counter() - t0
    resultado = {
        "tokenizador": "v1 (8192)",
        "corpus": {"caracteres": len(texto), "tokens": len(ids),
                   "janelas": len(dados), "ctx": args.ctx,
                   "treino": len(treino), "validacao": len(validacao),
                   "descartados_no_fim": resto},
        "config": cfg.__dict__,
        "n_parametros": modelo.n_parametros(),
        "passos": args.passos,
        "lote": args.lote,
        "lr": args.lr,
        "segundos": round(dur, 1),
        "baseline": {"perda": p0, "acerto": a0},
        "final": {"perda": p_final, "acerto": a_final},
        "historico": historico,
        "amostras": amostras,
        "copia_literal": copia,
        "nota": ("Treino no corpus real (5 obras do autor, 253k caracteres). "
                 "Nao e avaliacao linguistica: e a prova de que a cadeia "
                 "texto->ids->modelo->ids->texto funciona ponta a ponta."),
    }
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False))

    print(f"\nfinal: perda {p_final:.4f} | acerto {a_final*100:.2f}% | {dur:.0f}s")
    print(f"gravado em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
