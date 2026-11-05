"""O teste que o projecto precisava: 4,5x mais dados mudam o regime?

Todas as medições anteriores tinham a mesma limitação: 65 999 tokens para
2,77 M parâmetros = 24 tokens/parâmetro. O diagnóstico era sempre o mesmo —
decora, folga de 8-9 nats entre treino e validação, perde para o trigrama.

Agora há 294 107 tokens de Machado de Assis (domínio público). Isso dá
106 tokens/parâmetro — ainda longe dos ~4 000 de um modelo real, mas 4,5x
mais. A pergunta: **muda o regime ou é mais do mesmo?**

DESENHO, com as três coisas que faltaram nas medições anteriores
----------------------------------------------------------------
1. DIVISÃO POR OBRA, não por janelas. Dom Casmurro e Brás Cubas nunca
   aparecem nos dois lados. Isto resolve o defeito de contaminação que a
   auditoria registou (7 de 20-gramas atravessavam a fronteira).
2. BASELINE TRIGRAMA no MESMO conjunto de validação, com bits-por-byte.
3. TRÊS SEEDS, para saber se a diferença é maior que o ruído.

O QUE SE MEDE
    perda_val        — nats por token, tokenizador v1
    bits_por_byte    — comparável entre tokenizadores (exigido pela directiva)
    folga            — perda_val - perda_treino; mede decoração
    vs trigrama      — o baseline que o modelo nunca bateu
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128
TOKENS_POR_BYTE = 0.313   # medido: 294 107 tokens / 938 807 caracteres


def carregar() -> dict[str, list[int]]:
    from nucleo.tokenizador.bpe import TokenizadorBPE
    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    out = {}
    for f in sorted((RAIZ / "dados" / "publico_limpo").glob("*.txt")):
        out[f.stem] = tok.codificar(f.read_text(encoding="utf-8"))
    return out


def janelas(ids, ctx=CTX):
    n = len(ids) // ctx
    return np.array([ids[i * ctx:(i + 1) * ctx] for i in range(n)], dtype=np.int64)


def avaliar(modelo, dados) -> float:
    from nucleo.modelo.transformer import perda_entropia_cruzada
    if len(dados) == 0:
        return float("nan")
    ps = []
    for i in range(0, len(dados), 8):
        w = dados[i:i + 8]
        ps.append(float(perda_entropia_cruzada(
            modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    return float(np.mean(ps))


def treinar(modelo, dados, passos, lr, semente, lote=8):
    from nucleo import nn
    from nucleo.modelo.transformer import perda_entropia_cruzada
    opt = nn.Adam(modelo.parametros(), eta=lr, weight_decay=0.01)
    rng = np.random.default_rng(semente)
    for p in range(1, passos + 1):
        idx = rng.integers(0, len(dados), lote)
        w = dados[idx]
        opt.zero_grad()
        perda_entropia_cruzada(modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)
        # cosine decay: o OLMo e o nanoGPT usam-no e este projecto nunca o testou
        if p % 200 == 0:
            for g in opt.parametros if hasattr(opt, "parametros") else []:
                pass


def trigrama_pbb(ids_treino, ids_val, vocab, ordem=3, desconto=0.75) -> float:
    """Perda do n-grama em bits por BYTE (a métrica comparável)."""
    from nucleo.experimentos.baselines import Interpolado
    m = Interpolado(ordem, vocab, desconto)
    m.treinar(np.array(ids_treino, dtype=np.int64))
    total, n = 0.0, 0
    for i in range(ordem - 1, len(ids_val)):
        ctx = tuple(int(x) for x in ids_val[i - ordem + 1:i])
        total += -m.logp(ctx, int(ids_val[i]))
        n += 1
    nats_por_token = total / max(n, 1)
    # nats/token -> bits/byte: bits/token / (bytes por token)
    bytes_por_token = 1.0 / TOKENS_POR_BYTE if TOKENS_POR_BYTE else 1.0
    return (nats_por_token / math.log(2)) / bytes_por_token


def main() -> None:
    from nucleo.modelo.transformer import Config, Transformer

    obras = carregar()
    # divisão POR OBRA: validação = Dom Casmurro (a maior), treino = as outras
    VAL = "dom_casmurro"
    treino_ids, val_ids = [], []
    for nome, ids in obras.items():
        if nome == VAL:
            val_ids.extend(ids)
        else:
            treino_ids.extend(ids)
    d_tr, d_va = janelas(treino_ids), janelas(val_ids)
    print(f"treino: {len(treino_ids):,} tokens ({len(d_tr)} janelas) de "
          f"{[n for n in obras if n != VAL]}")
    print(f"validação: {len(val_ids):,} tokens ({len(d_va)} janelas) de {VAL}")
    print(f"divisão POR OBRA — sem contaminação entre treino e validação\n")

    vocab = max(max(i) for i in obras.values()) + 1
    from nucleo.tokenizador.bpe import TokenizadorBPE
    vocab = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1").tamanho_vocab

    print("trigrama (baseline) a calcular...", flush=True)
    t0 = time.time()
    k3 = trigrama_pbb(treino_ids, val_ids, vocab, 3)
    print(f"  3-grama: {k3:.4f} bits/byte ({time.time()-t0:.0f}s)\n")

    cfg = Config(vocab=vocab, d_model=128, n_camadas=4, n_cabecas=4,
                 ctx=CTX, d_ff=384, tying=False)

    resultados = []
    for semente in (0, 1, 2):
        m = Transformer(cfg, semente=semente)
        t0 = time.time()
        treinar(m, d_tr, passos=1500, lr=3e-3, semente=semente)
        p_tr, p_va = avaliar(m, d_tr), avaliar(m, d_va)
        bpb = (p_va / math.log(2)) / (1.0 / TOKENS_POR_BYTE)
        r = {"semente": semente, "perda_treino": round(p_tr, 4),
             "perda_val": round(p_va, 4), "folga": round(p_va - p_tr, 4),
             "bits_por_byte": round(bpb, 4), "segundos": round(time.time() - t0, 1)}
        resultados.append(r)
        print(f"seed {semente}: treino {p_tr:>8.4f} | val {p_va:>7.4f} | "
              f"folga {r['folga']:>7.4f} | {bpb:.4f} bpb | {r['segundos']:.0f}s")
        m.guardar(RAIZ / "nucleo" / "experimentos" / "checkpoints" / f"publico_seed{semente}")

    vals = [r["perda_val"] for r in resultados]
    bpbs = [r["bits_por_byte"] for r in resultados]
    folgas = [r["folga"] for r in resultados]
    med = float(np.mean(bpbs))
    res = {
        "classe": "MEASURED",
        "corpus": "Machado de Assis, domínio público, 294 107 tokens",
        "divisao": f"por obra: treino={[n for n in obras if n != VAL]}, validação={VAL}",
        "passos": 1500, "lote": 8, "lr": 3e-3, "ctx": CTX, "seeds": [0, 1, 2],
        "trigrama_bits_por_byte": round(k3, 4),
        "transformer_bits_por_byte_media": round(med, 4),
        "transformer_bits_por_byte_desvio": round(float(np.std(bpbs, ddof=1)), 4),
        "transformer_perda_val_media": round(float(np.mean(vals)), 4),
        "transformer_perda_val_desvio": round(float(np.std(vals, ddof=1)), 4),
        "folga_media": round(float(np.mean(folgas)), 4),
        "ganha_do_trigrama": bool(med < k3),
        "margem_bits_por_byte": round(k3 - med, 4),
        "tokens_por_parametro": round(len(treino_ids) / 2771200, 1),
        "tokens_treino": len(treino_ids), "parametros": 2771200,
        "nota_tpp": "tokens de TREINO por parâmetro. Dividir pelos 294 107 totais "
                    "(incluindo validação) daria 0,1 e estaria errado: a validação "
                    "não treina nada.",
        "resultados": resultados,
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_regime_publico.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n{'':<12} {'nats':>8} {'bits/byte':>11}")
    print("-" * 34)
    print(f"{'trigrama':<12} {'—':>8} {k3:>11.4f}")
    print(f"{'Transformer':<12} {np.mean(vals):>8.4f} {med:>11.4f} ±{np.std(bpbs, ddof=1):.4f}")
    print()
    if med < k3:
        print(f"O TRANSFORMER BATE O TRIGRAMA pela primeira vez: "
              f"{k3-med:.4f} bits/byte ({med/k3*100:.1f}% do valor do trigrama).")
    else:
        print(f"O trigrama continua a ganhar por {med-k3:.4f} bits/byte.")
    print(f"folga média treino-val: {np.mean(folgas):.4f} nats "
          f"(antes, com 66k tokens: 8-9 nats)")


if __name__ == "__main__":
    main()
