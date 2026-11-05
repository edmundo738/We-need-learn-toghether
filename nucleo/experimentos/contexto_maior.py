"""Contexto 128 vs 256 — a directiva prevê 256 e nunca foi testado.

A Adenda D10 concluiu que a derrota contra o trigrama não é falta de dados nem
excesso de parâmetros, e apontou o contexto como o suspeito seguinte. Isto
testa-o.

O CUIDADO QUE DECIDE SE O TESTE VALE
------------------------------------
Comparar «contexto 256 por 800 passos» com «contexto 128 por 800 passos» é
trapaça: 256 x lote 8 = 2 048 tokens por passo contra 1 024. O maior vê o
dobro do texto e ganha por isso, não pelo contexto.

Por isso o orçamento é em TOKENS VISTOS, não em passos:
    ctx 128: 1600 passos x lote 8 x 128 = 163 840 000? não — ver abaixo

A conta certa: tokens vistos por passo = ctx * lote.
    ctx 128, lote 8 -> 1 024 tokens/passo
    ctx 256, lote 4 -> 1 024 tokens/passo   (lote reduzido para igualar)
Assim os dois braços vêem o MESMO número de tokens por passo, e o mesmo número
total. A única variável é o comprimento do contexto.

CUSTO: a atenção é O(T²). Com ctx 256 e lote 4, o tempo por passo é ~4x o de
ctx 128 e lote 8. Corrida de 800 passos: ~2 x 800 s.

O QUE SE ESPERA, E O QUE ISSO TESTARIA
    Se o contexto maior melhorar de forma apreciável, a hipótese estrutural da
    D10 ganha apoio: parte da desvantagem era não ver longe o suficiente.
    Se ficar igual, o contexto não é o problema e sobra a arquitectura em si
    (posicionais aprendidas, atenção sem viés local).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
TOKENS_POR_BYTE, TRIGRAMA = 0.313, 2.4766


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

    def jan(ids, ctx):
        n = len(ids) // ctx
        return np.array([ids[i * ctx:(i + 1) * ctx] for i in range(n)], dtype=np.int64)

    def correr(ctx, lote, passos, semente=0, lr=3e-3):
        d_tr, d_va = jan(tr, ctx), jan(va, ctx)
        cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                     n_cabecas=4, ctx=ctx, d_ff=384, tying=False)
        m = Transformer(cfg, semente=semente)
        opt = nn.Adam(m.parametros(), eta=lr, weight_decay=0.01)
        rng = np.random.default_rng(semente)

        def avaliar(dados):
            ps = []
            for i in range(0, len(dados), 4):
                w = dados[i:i + 4]
                ps.append(float(perda_entropia_cruzada(
                    m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
            return float(np.mean(ps))

        melhor = {"perda_val": float("inf"), "passo": 0}
        for passo in range(1, passos + 1):
            idx = rng.integers(0, len(d_tr), lote)
            w = d_tr[idx]
            opt.zero_grad()
            perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
            perda.backward()
            opt.step(grad_clip=1.0)
            if passo % 100 == 0:
                pv = avaliar(d_va)
                if pv < melhor["perda_val"]:
                    melhor = {"perda_val": pv, "passo": passo}
        bpb = (melhor["perda_val"] / math.log(2)) / (1.0 / TOKENS_POR_BYTE)
        return {"ctx": ctx, "lote": lote,
                "tokens_por_passo": ctx * lote,
                "tokens_vistos": ctx * lote * passos,
                "janelas_treino": len(d_tr), "janelas_val": len(d_va),
                "melhor_perda_val": round(melhor["perda_val"], 4),
                "melhor_passo": melhor["passo"],
                "bits_por_byte": round(bpb, 4),
                "margem_vs_trigrama": round(TRIGRAMA - bpb, 4)}

    print(f"trigrama = {TRIGRAMA} bpb\n")
    # PASSOS IGUAIS e TOKENS/PASSO IGUAIS (lote compensado): só o ctx muda
    r128 = correr(ctx=128, lote=8, passos=800)
    print(f"ctx 128 lote 8: {r128['bits_por_byte']:>7.4f} bpb | "
          f"melhor @ {r128['melhor_passo']} | {r128['tokens_vistos']:,} tokens", flush=True)
    r256 = correr(ctx=256, lote=4, passos=800)
    print(f"ctx 256 lote 4: {r256['bits_por_byte']:>7.4f} bpb | "
          f"melhor @ {r256['melhor_passo']} | {r256['tokens_vistos']:,} tokens", flush=True)

    dif = r128["bits_por_byte"] - r256["bits_por_byte"]
    res = {
        "classe": "MEASURED",
        "trigrama_bits_por_byte": TRIGRAMA,
        "desenho": "passos iguais (800) e tokens/passo iguais (1024) nos dois braços; "
                   "a única variável é ctx. lote compensado para igualar o orçamento.",
        "resultado": [r128, r256],
        "melhoria_do_ctx_256_bpb": round(dif, 4),
        "ctx_256_bate_o_trigrama": r256["bits_por_byte"] <= TRIGRAMA,
        "aviso": "uma seed. Sem desvio não se distingue efeito de ruído; o desvio "
                 "típico entre seeds neste corpus é ~0,013 bpb, pelo que uma melhoria "
                 "abaixo disso NÃO é interpretável.",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_contexto_maior.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\ndiferença (128 - 256): {dif:+.4f} bpb")
    if dif > 0.013:
        print("=> o ctx 256 melhora acima do ruído entre seeds: o contexto CONTAVA.")
    elif dif < -0.013:
        print("=> o ctx 256 é PIOR: mais contexto com o mesmo orçamento prejudica.")
    else:
        print("=> diferença dentro do ruído entre seeds: o contexto NÃO explica a derrota.")
    print(f"ctx 256 bate o trigrama? {'SIM' if res['ctx_256_bate_o_trigrama'] else 'nao'}")


if __name__ == "__main__":
    main()
