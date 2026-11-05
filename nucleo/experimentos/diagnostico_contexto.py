"""Porque é que mais contexto piora? A posicional tem parâmetros POR POSIÇÃO.

`contexto_maior.py` mediu: ctx 128 -> 2,6796 bpb, ctx 256 -> 2,7018. Mais
contexto piorou. Isso é contra-intuitivo, e há um mecanismo concreto.

`self.emb_pos = Embedding(cfg.ctx, cfg.d_model)` — uma tabela com um vector
POR POSIÇÃO. Não é RoPE nem sinusóide: cada uma das `ctx` posições tem
`d_model` parâmetros próprios, treinados apenas com os tokens que aparecem
naquela posição.

CONSEQUÊNCIA ARITMÉTICA
    ctx 128: 128 x 128 = 16 384 parâmetros de posição
             x 1395 janelas de treino = 178 560 amostras -> 10,9 por parâmetro
    ctx 256: 256 x 128 = 32 768 parâmetros de posição
             x  697 janelas de treino = 178 432 amostras -> 5,4 por parâmetro

**Não são os parâmetros totais que crescem — são os parâmetros POR POSIÇÃO.**
O dobro do contexto duplica a tabela e mantém o corpus, logo metade dos dados
por parâmetro de posição. É um caso em que o argumento «dados/parâmetro»
SE APLICA, ao contrário do que aconteceu em `modelo_menor.py` (onde reduzir
o modelo inteiro não ajudou porque a posicional ficava igual).

O QUE ESTE FICHEIRO MEDE
    a) a contagem exacta: parâmetros de posição e amostras por parâmetro
    b) a perda SEPARADA por posição inicial (1-32), média (60-90) e final
       (95-127), com ctx 128 — se as posições iniciais tiverem perda muito
       menor, é sinal de que memorizou o início dos capítulos e que o
       contexto longo não está a ser usado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]


def main() -> None:
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    obras = {f.stem: tok.codificar(f.read_text(encoding="utf-8"))
             for f in sorted((RAIZ / "dados" / "publico_limpo").glob("*.txt"))}
    tr = []
    for n, ids in obras.items():
        if n != "dom_casmurro":
            tr.extend(ids)

    aritmetica = []
    for ctx in (128, 256):
        n_jan = len(tr) // ctx
        pos_par = ctx * 128
        aritmetica.append({
            "ctx": ctx, "janelas": n_jan,
            "parametros_de_posicao": pos_par,
            "parametros_por_posicao": 128,
            "amostras_por_parametro_de_posicao": round(n_jan * ctx / pos_par, 2),
            "tokens_por_parametro_de_posicao": round(n_jan * ctx / pos_par, 2),
        })
        print(f"ctx {ctx}: {pos_par:,} parâmetros de posição, {n_jan} janelas "
              f"-> {n_jan*ctx/pos_par:.1f} amostras por parâmetro de posição")

    # perda por faixa de posição, ctx 128, novamente treinado (rápido: 500 passos)
    from nucleo import nn
    ctx = 128
    d_tr = np.array([tr[i * ctx:(i + 1) * ctx] for i in range(len(tr) // ctx)], dtype=np.int64)
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=ctx, d_ff=384, tying=False)
    m = Transformer(cfg, semente=0)
    opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for _ in range(500):
        idx = rng.integers(0, len(d_tr), 8)
        w = d_tr[idx]
        opt.zero_grad()
        perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)

    # perda por posição nos PRÓPRIOS dados de treino e em validação
    def perda_por_posicao(dados):
        tot = np.zeros(ctx - 1)
        n = 0
        for w in dados[:200]:
            lg = m.forward(w[None, :-1]).data
            if lg.ndim == 3:
                lg = lg[0]
            z = lg
            zmax = z.max(axis=-1, keepdims=True)
            logp = z - zmax - np.log(np.exp(z - zmax).sum(axis=-1, keepdims=True))
            alvo = w[1:]
            tot += -logp[np.arange(len(alvo)), alvo]
            n += 1
        return tot / max(n, 1)

    val = []
    for n, ids in obras.items():
        if n == "dom_casmurro":
            val.extend(ids)
    d_va = np.array([val[i * ctx:(i + 1) * ctx] for i in range(len(val) // ctx)], dtype=np.int64)

    ptr = perda_por_posicao(d_tr)
    pva = perda_por_posicao(d_va)

    faixas = []
    for nome, a, b in (("inicio (0-31)", 0, 32), ("meio (60-91)", 60, 92),
                       ("fim (96-127)", 96, ctx - 1)):
        faixas.append({"faixa": nome,
                       "perda_treino": round(float(ptr[a:b].mean()), 4),
                       "perda_val": round(float(pva[a:b].mean()), 4)})
        print(f"{nome:<16} treino {ptr[a:b].mean():>7.4f} | val {pva[a:b].mean():>7.4f}")

    res = {
        "classe": "MEASURED",
        "mecanismo": "emb_pos = Embedding(ctx, d_model): um vector POR POSIÇÃO, "
                     "não RoPE nem sinusóide. Dobrar ctx duplica a tabela sem "
                     "aumentar o corpus, logo metade das amostras por parâmetro "
                     "de posição — o argumento dados/parâmetro APLICA-SE aqui, "
                     "ao contrário de modelo_menor.py onde reduzir o modelo "
                     "inteiro deixava a posicional igual.",
        "aritmetica": aritmetica,
        "perda_por_faixa_de_posicao": faixas,
        "leitura": "se a perda no início for muito menor que no fim, o modelo "
                   "memorizou os inícios de janela (todos os capítulos começam "
                   "de forma parecida) e não está a usar o contexto longo.",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_diagnostico_contexto.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print()
    print("posições iniciais vs finais (treino): "
          f"{faixas[0]['perda_treino']:.3f} vs {faixas[2]['perda_treino']:.3f}")


if __name__ == "__main__":
    main()
