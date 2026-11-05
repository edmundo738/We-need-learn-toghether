"""O melhor ponto com 3 seeds: a medição honesta que substitui «1500 passos».

A curva de `underfitting.py` mostrou que a validação melhora até ao passo ~500
e depois piora monotonamente até ao fim. Treinar 1500 passos, como fiz em
`regime_publico.py`, foi reportar um ponto já 0,57 bpb pior que o óptimo.

Este ficheiro faz a coisa certa:
  - treina até 1000 passos, avaliando a CADA 100
  - 3 seeds
  - reporta o MELHOR ponto de cada seed, com o passo em que ocorreu
  - compara com o trigrama no mesmo conjunto

E resolve um problema de método: escolher o melhor ponto de validação e depois
reportar a perda DESSE ponto é optimismo selectivo. Para não enganar, reporta-se
as duas coisas:
    a) melhor validação observada (optimista por construção)
    b) validação no passo fixo 500 (honesta: escolhido a priori pela curva da seed 0)
Se (a) e (b) forem próximos, o optimismo é pequeno.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128
from nucleo.experimentos.metricas import tokens_por_byte as _tpb
TOKENS_POR_BYTE = _tpb()
PASSO_FIXO = 500          # escolhido a priori, da curva independente da seed 0


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

    def avaliar(m, dados):
        ps = []
        for i in range(0, len(dados), 8):
            w = dados[i:i + 8]
            ps.append(float(perda_entropia_cruzada(
                m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
        return float(np.mean(ps))

    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=CTX, d_ff=384, tying=False)
    resultados = []
    for semente in (0, 1, 2):
        m = Transformer(cfg, semente=semente)
        opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.01)
        rng = np.random.default_rng(semente)
        curva, no_fixo = [], None
        for passo in range(1, 1001):
            idx = rng.integers(0, len(d_tr), 8)
            w = d_tr[idx]
            opt.zero_grad()
            perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
            perda.backward()
            opt.step(grad_clip=1.0)
            if passo % 100 == 0:
                pv = avaliar(m, d_va)
                curva.append({"passo": passo, "perda_val": round(pv, 4)})
                if passo == PASSO_FIXO:
                    no_fixo = pv
        melhor = min(curva, key=lambda c: c["perda_val"])
        resultados.append({"semente": semente, "curva": curva,
                           "melhor_passo": melhor["passo"],
                           "melhor_val": melhor["perda_val"],
                           "val_no_passo_fixo": round(no_fixo, 4) if no_fixo else None})
        print(f"seed {semente}: melhor {melhor['perda_val']:.4f} @ {melhor['passo']} | "
              f"no passo {PASSO_FIXO}: {no_fixo:.4f}", flush=True)

    def bpb(nats):
        return (nats / math.log(2)) / (1.0 / TOKENS_POR_BYTE)

    melhores = [r["melhor_val"] for r in resultados]
    fixes = [r["val_no_passo_fixo"] for r in resultados if r["val_no_passo_fixo"]]
    k3 = 2.4766
    res = {
        "classe": "MEASURED",
        "divisao": "por obra (validação = Dom Casmurro)",
        "passo_fixo_escolhido_a_priori": PASSO_FIXO,
        "trigrama_bits_por_byte": k3,
        "melhor_observado": {
            "perda_val_media": round(float(np.mean(melhores)), 4),
            "perda_val_desvio": round(float(np.std(melhores, ddof=1)), 4),
            "bits_por_byte_media": round(bpb(float(np.mean(melhores))), 4),
            "passos": [r["melhor_passo"] for r in resultados],
            "aviso": "optimismo selectivo: escolhido olhando a validação",
        },
        "passo_fixo_500": {
            "perda_val_media": round(float(np.mean(fixes)), 4),
            "perda_val_desvio": round(float(np.std(fixes, ddof=1)), 4),
            "bits_por_byte_media": round(bpb(float(np.mean(fixes))), 4),
            "aviso": "honesto: o passo 500 foi escolhido numa corrida separada",
        },
        "resultados": resultados,
    }
    res["ganha_do_trigrama_melhor"] = res["melhor_observado"]["bits_por_byte_media"] < k3
    res["ganha_do_trigrama_fixo"] = res["passo_fixo_500"]["bits_por_byte_media"] < k3
    res["margem_melhor"] = round(k3 - res["melhor_observado"]["bits_por_byte_media"], 4)
    res["margem_fixo"] = round(k3 - res["passo_fixo_500"]["bits_por_byte_media"], 4)
    (RAIZ / "nucleo" / "experimentos" / "resultado_melhor_ponto.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\ntrigrama: {k3:.4f} bpb")
    print(f"melhor observado: {res['melhor_observado']['bits_por_byte_media']:.4f} bpb "
          f"(±{res['melhor_observado']['perda_val_desvio']:.4f} nats) -> margem "
          f"{res['margem_melhor']:+.4f} bpb")
    print(f"passo 500 fixo  : {res['passo_fixo_500']['bits_por_byte_media']:.4f} bpb "
          f"-> margem {res['margem_fixo']:+.4f} bpb")
    print()
    if res["ganha_do_trigrama_melhor"]:
        print("BATE o trigrama no melhor ponto observado.")
    else:
        print("NÃO bate o trigrama, nem no melhor ponto observado.")
    print(f"(o triggerama ganha por {res['margem_fixo']:.4f} bpb no passo honesto)")


if __name__ == "__main__":
    main()
