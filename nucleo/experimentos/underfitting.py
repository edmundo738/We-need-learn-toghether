"""O Transformer perde para o trigrama por falta de treino ou por capacidade?

`regime_publico.py` mediu 3,0480 bits/byte contra 2,4766 do trigrama — perde
por 0,57. Antes de aceitar «o modelo é pior», há uma explicação alternativa
que tem de ser descartada: UNDERFITTING. Foram só 1500 passos × lote 8 =
12 000 janelas vistas; com 1 395 janelas de treino, isso é ~8,6 épocas.
Pode simplesmente não ter treinado o suficiente.

COMO SE DISTINGUE
    Se é underfitting: a perda de TREINO continua a descer e a de validação
    acompanha. Mais passos -> melhor validação.
    Se é capacidade: a perda de treino desce e a de validação estagna ou sobe.
    Se é o corpus: ambas estagnam altas — o modelo esgotou o que os dados dão.

Mede-se a curva de validação a cada 250 passos, até 4000, com 1 seed.
4 000 x lote 8 = 32 000 janelas = 23 épocas. Custa ~25 min.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128
from nucleo.experimentos.metricas import tokens_por_byte as _tpb
TOKENS_POR_BYTE = _tpb()


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
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=CTX, d_ff=384, tying=False)
    m = Transformer(cfg, semente=0)
    opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)

    def avaliar(dados):
        ps = []
        for i in range(0, len(dados), 8):
            w = dados[i:i + 8]
            ps.append(float(perda_entropia_cruzada(
                m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
        return float(np.mean(ps))

    curva = []
    t0 = time.time()
    lote = 8
    for passo in range(1, 4001):
        idx = rng.integers(0, len(d_tr), lote)
        w = d_tr[idx]
        opt.zero_grad()
        perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
        perda.backward()
        opt.step(grad_clip=1.0)
        if passo % 250 == 0:
            p_tr = float(perda.data.reshape(-1)[0])
            p_va = avaliar(d_va)
            bpb = (p_va / math.log(2)) / (1.0 / TOKENS_POR_BYTE)
            curva.append({"passo": passo, "perda_treino": round(p_tr, 4),
                          "perda_val": round(p_va, 4),
                          "bits_por_byte": round(bpb, 4),
                          "epocas": round(passo * lote / len(d_tr), 1)})
            print(f"{passo:>5} ({passo*lote/len(d_tr):>5.1f} ép) treino {p_tr:>7.3f} | "
                  f"val {p_va:>7.4f} | {bpb:.4f} bpb | {time.time()-t0:>5.0f}s",
                  flush=True)

    melhor = min(curva, key=lambda c: c["perda_val"])
    primeira = curva[0]
    ultima = curva[-1]
    res = {
        "classe": "MEASURED", "seeds": [0], "passos": 4000, "lote": lote,
        "trigrama_bpb": 2.4766,
        "curva": curva,
        "melhor": melhor,
        "melhoria_de_1500_a_4000": round(
            next(c for c in curva if c["passo"] == 1500)["perda_val"]
            - ultima["perda_val"], 4),
        "ainda_a_descer": ultima["perda_val"] <= melhor["perda_val"],
        "diagnostico": ("underfitting: validação ainda a descer" if
                        ultima["perda_val"] < ultima["perda_val"] + 1 else
                        "ver curva"),
    }

    # diagnóstico explícito
    d_val = ultima["perda_val"] - melhor["perda_val"]
    d_tr = ultima["perda_treino"] - primeira["perda_treino"]
    if abs(d_val) < 0.02:
        diag = "estagnou: mais passos não ajudam — o limite é de dados ou capacidade"
    elif d_val < -0.02:
        diag = "ainda a descer: underfitting, mais treino ajuda"
    else:
        diag = "a piorar: sobreajuste"
    res["diagnostico"] = diag
    (RAIZ / "nucleo" / "experimentos" / "resultado_underfitting.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nmelhor: passo {melhor['passo']} val {melhor['perda_val']} "
          f"({melhor['bits_por_byte']} bpb) | trigrama 2,4766")
    print(f"diagnóstico: {diag}")


if __name__ == "__main__":
    main()
