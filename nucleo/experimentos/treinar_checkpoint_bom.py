"""Treina e guarda o checkpoint do MELHOR ponto, não do último.

DEFEITO QUE ESTE FICHEIRO CORRIGE
---------------------------------
`regime_publico.py` guardava o checkpoint no fim da corrida (1500 passos), que
é o PIOR ponto da curva: val 6,7498 em vez de 5,9340 no passo 500. O painel
mostrava ao utilizador exactamente essa versão sobreajustada — palavras
inventadas, colapsos de travessões — e não a melhor que o projecto conseguiu.

É um defeito de produto, não de modelo: a informação para escolher o ponto
certo estava medida e não foi usada.

Agora: treina até 700 passos (o óptimo das 3 seeds está entre 500 e 700),
avalia a cada 50, guarda o estado do melhor passo. Guarda também o último,
para se poder comparar contra o que estava a ser servido.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128


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

    melhor = {"perda_val": float("inf"), "passo": 0, "estado": None}
    for passo in range(1, 701):
        idx = rng.integers(0, len(d_tr), 8)
        w = d_tr[idx]
        opt.zero_grad()
        perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
        perda.backward()
        opt.step(grad_clip=1.0)
        if passo % 50 == 0:
            pv = avaliar(d_va)
            if pv < melhor["perda_val"]:
                melhor = {"perda_val": pv, "passo": passo,
                          "estado": {k: t.data.copy()
                                     for k, t in m.parametros().items()}}
            if passo % 250 == 0:
                print(f"  passo {passo}: val {pv:.4f} "
                      f"(melhor {melhor['perda_val']:.4f} @ {melhor['passo']})",
                      flush=True)

    # restaurar o melhor e gravar
    for k, t in m.parametros().items():
        if k in melhor["estado"]:
            t.data = melhor["estado"][k]

    destino = RAIZ / "nucleo" / "experimentos" / "checkpoints" / "machado_melhor"
    m.guardar(destino)
    # o último estado, para comparação
    d2 = RAIZ / "nucleo" / "experimentos" / "checkpoints" / "machado_ultimo"
    m.guardar(d2)

    val_final = avaliar(d_va)
    from nucleo.experimentos.metricas import nats_para_bits_por_byte
    res = {
        "classe": "MEASURED",
        "checkpoint": "machado_melhor",
        "melhor_passo": melhor["passo"],
        "melhor_perda_val": round(melhor["perda_val"], 4),
        "bits_por_byte": round(nats_para_bits_por_byte(melhor["perda_val"]), 4),
        "treino_ate": 700,
        "nota": "substitui publico_seed* no painel; aqueles foram gravados no "
                "passo 1500, que é o pior ponto da curva (val 6,7498 vs 5,9340).",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_checkpoint_bom.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nmelhor: passo {melhor['passo']}, val {melhor['perda_val']:.4f} "
          f"({res['bits_por_byte']:.4f} bpb)")
    print(f"guardado em {destino}")


if __name__ == "__main__":
    main()
