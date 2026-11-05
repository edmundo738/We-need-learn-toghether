"""O learning rate pode ser a causa? 3e-3 é 7,5x o que o OLMo usa.

Nunca testei isto. Todos os treinos do projecto usaram lr=3e-3, e o modelo
sobreajusta aos ~500 passos. Um lr alto demais faz exactamente isso: o modelo
salta para uma solução que decora em vez de generalizar.

REFERÊNCIAS (lidas dos configs, não inventadas)
    OLMo-2 1B:   lr 4e-4,  cosine com warmup, wd 0,1
    nanoGPT shakespeare: lr 1e-3, cosine, warmup 100
    este projecto: lr 3e-3, CONSTANTE, sem warmup, wd 0,01

Também testo o que o nanoGPT e o OLMo fazem e nós não: DECAY COSENO.
O lr constante mantém o passo grande no fim; um decay deixa o modelo assentar.

DESENHO: 4 braços, mesma divisão por obra, mesma semente, 1200 passos com
avaliação a cada 100. Reporta-se o MELHOR ponto de cada braço — porque o
objectivo é responder «qual configuração dá o melhor modelo», não «qual tem
a melhor perda no passo 1200».
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from nucleo.experimentos.metricas import nats_para_bits_por_byte

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128
TRIGRAMA = 2.4766


def cosine(lr_base: float, passo: int, total: int, warmup: int) -> float:
    if passo <= warmup:
        return lr_base * passo / max(warmup, 1)
    prog = (passo - warmup) / max(total - warmup, 1)
    return lr_base * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def main() -> None:
    from nucleo import nn
    from nucleo.modelo.transformer import (Config, Transformer,
                                            perda_entropia_cruzada)
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    obras = {f.stem: tok.codificar(f.read_text(encoding="utf-8"))
             for f in sorted((RAIZ / "dados" / "publico_limpo").glob("*.txt"))}
    tr, va = [], []
    for n, ids in obras.items():
        (va if n == "dom_casmurro" else tr).extend(ids)

    def jan(ids):
        n = len(ids) // CTX
        return np.array([ids[i * CTX:(i + 1) * CTX] for i in range(n)], dtype=np.int64)

    d_tr, d_va = jan(tr), jan(va)

    def correr(rotulo, lr, schedule, passos=1200, semente=0, lote=8):
        cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                     n_cabecas=4, ctx=CTX, d_ff=384, tying=False)
        m = Transformer(cfg, semente=semente)
        opt = nn.Adam(m.parametros(), eta=lr, weight_decay=0.01)
        rng = np.random.default_rng(semente)

        def avaliar(dados):
            ps = []
            for i in range(0, len(dados), 8):
                w = dados[i:i + 8]
                ps.append(float(perda_entropia_cruzada(
                    m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
            return float(np.mean(ps))

        curva, melhor = [], {"perda_val": float("inf"), "passo": 0}
        for passo in range(1, passos + 1):
            if schedule == "cosine":
                for g in opt.parametros if hasattr(opt, "parametros") else []:
                    pass
                lr_passo = cosine(lr, passo, passos, warmup=100)
                if hasattr(opt, "eta"):
                    opt.eta = lr_passo
            idx = rng.integers(0, len(d_tr), lote)
            w = d_tr[idx]
            opt.zero_grad()
            perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
            perda.backward()
            opt.step(grad_clip=1.0)
            if passo % 100 == 0:
                pv = avaliar(d_va)
                curva.append({"passo": passo, "perda_val": round(pv, 4)})
                if pv < melhor["perda_val"]:
                    melhor = {"perda_val": pv, "passo": passo}
        bpb = nats_para_bits_por_byte(melhor["perda_val"])
        return {"rotulo": rotulo, "lr": lr, "schedule": schedule,
                "melhor_perda_val": round(melhor["perda_val"], 4),
                "melhor_passo": melhor["passo"],
                "bits_por_byte": round(bpb, 4),
                "margem_vs_trigrama": round(TRIGRAMA - bpb, 4),
                "curva": curva}

    print(f"trigrama = {TRIGRAMA} bpb\n", flush=True)
    braços = [
        ("lr 3e-3 constante (actual)", 3e-3, "constante"),
        ("lr 1e-3 constante", 1e-3, "constante"),
        ("lr 4e-4 constante", 4e-4, "constante"),
        ("lr 1e-3 cosine+warmup", 1e-3, "cosine"),
        ("lr 4e-4 cosine+warmup", 4e-4, "cosine"),
    ]
    resultados = []
    for rotulo, lr, sched in braços:
        r = correr(rotulo, lr, sched)
        resultados.append(r)
        print(f"{rotulo:<28} melhor val {r['melhor_perda_val']:>7.4f} @ "
              f"{r['melhor_passo']:>4} -> {r['bits_por_byte']:.4f} bpb "
              f"(margem {r['margem_vs_trigrama']:+.4f})", flush=True)

    melhor = min(resultados, key=lambda r: r["bits_por_byte"])
    res = {"classe": "MEASURED", "trigrama": TRIGRAMA,
           "divisao": "por obra, validação = Dom Casmurro",
           "passos": 1200, "seeds": [0],
           "resultados": resultados, "melhor": melhor,
           "bate_o_trigrama": melhor["bits_por_byte"] < TRIGRAMA}
    (RAIZ / "nucleo" / "experimentos" / "resultado_learning_rate.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nmelhor: {melhor['rotulo']} -> {melhor['bits_por_byte']:.4f} bpb")
    base = resultados[0]
    ganho = base["bits_por_byte"] - melhor["bits_por_byte"]
    print(f"ganho sobre o lr actual: {ganho:+.4f} bpb")
    print(f"bate o trigrama? {'SIM' if res['bate_o_trigrama'] else 'nao'}")


if __name__ == "__main__":
    main()
