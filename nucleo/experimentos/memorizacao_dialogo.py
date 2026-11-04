"""Memorização em diálogo: 8,81 nats de folga. Porquê?

O braço B (prosa -> diálogo) atinge 0,0510 nats nas conversas de TREINO e
8,8613 nas de VALIDAÇÃO. Uma folga de 8,81 nats é memorização quase perfeita:
com 9 653 tokens e 1,7 M parâmetros, o modelo decora 75 janelas em 800 passos.

A pergunta que decide o que isto significa:
    a folga vem de os NOMES dos falantes serem outros, ou da ESTRUTURA
    do diálogo ser outra?

ABLAÇÃO: trocar em cada conversa de validação o nome de cada falante por um
nome que EXISTE no treino, mantendo o resto do texto igual. Se a perda cair
muito, o modelo aprendeu a associar nomes a vozes e não a conversar — e a
métrica de validação estava a punir nomes novos, não falta de diálogo.

Isto não «conserta» o modelo. Mede o que a perda de validação está a medir.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128


def carregar_conversas():
    txt = (RAIZ / "dados" / "dialogos" / "conversas_v1.md").read_text(encoding="utf-8")
    corpo = re.split(r"^# Conversas simuladas para treino de diálogo\s*$", txt, flags=re.M)
    texto = corpo[1] if len(corpo) > 1 else txt
    secs = [s.strip() for s in re.split(r"^## ", texto, flags=re.M) if s.strip()]
    rng = np.random.default_rng(20261104)
    ordem = rng.permutation(len(secs))
    val_idx = set(ordem[:10].tolist())
    treino = [secs[i] for i in range(len(secs)) if i not in val_idx]
    val = [secs[i] for i in range(len(secs)) if i in val_idx]
    return treino, val


def falantes(secs):
    return {m for s in secs for m in re.findall(r"^\*\*(.+?):\*\*", s, re.M)}


def trocar_nomes(secs, nomes_treino):
    """Substitui cada falante por um nome do treino, de forma estável."""
    mapa = {}
    for n in sorted(falentes_ordenados(secs)):
        mapa[n] = sorted(nomes_treino)[hash(n) % len(nomes_treino)]
    saida = []
    for s in secs:
        for antigo, novo in mapa.items():
            s = re.sub(rf"^\*\*{re.escape(antigo)}:\*\*", f"**{novo}:**", s, flags=re.M)
        saida.append(s)
    return saida


def falentes_ordenados(secs):
    return falantes(secs)


def janelas(ids, ctx=CTX):
    n = len(ids) // ctx
    return np.array([ids[i * ctx:(i + 1) * ctx] for i in range(n)], dtype=np.int64) if n else np.zeros((0, ctx), dtype=np.int64)


def avaliar(modelo, dados):
    from nucleo.modelo.transformer import perda_entropia_cruzada
    if len(dados) == 0:
        return float("nan")
    ps = []
    for i in range(0, len(dados), 8):
        w = dados[i:i + 8]
        ps.append(float(perda_entropia_cruzada(
            modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    return float(np.mean(ps))


def main():
    from nucleo.modelo.transformer import Config, Transformer
    from nucleo.tokenizador.bpe import TokenizadorBPE
    from nucleo import nn
    from nucleo.modelo.transformer import perda_entropia_cruzada

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    treino_txt, val_txt = carregar_conversas()

    nomes_tr = falantes(treino_txt)
    nomes_va = falantes(val_txt)
    novos = nomes_va - nomes_tr
    print(f"falantes: {len(nomes_tr)} no treino | {len(nomes_va)} na validação | "
          f"{len(novos)} SÓ na validação: {sorted(novos)}")

    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=CTX, d_ff=384, tying=False)

    # reproduzir o braço B: prosa -> diálogo
    ids_prosa = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
    m = Transformer(cfg, semente=0)
    opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)
    d_prosa = janelas(list(ids_prosa))
    for _ in range(800):
        idx = rng.integers(0, len(d_prosa), 4)
        w = d_prosa[idx]
        opt.zero_grad()
        perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)

    d_tr = janelas(tok.codificar("\n".join(treino_txt)))
    opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for _ in range(800):
        idx = rng.integers(0, len(d_tr), 4)
        w = d_tr[idx]
        opt.zero_grad()
        perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)

    p_tr = avaliar(m, d_tr)
    p_va = avaliar(m, janelas(tok.codificar("\n".join(val_txt))))

    val_trocada = trocar_nomes(val_txt, nomes_tr)
    p_trocada = avaliar(m, janelas(tok.codificar("\n".join(val_trocada))))

    res = {
        "classe": "MEASURED",
        "perda_treino_dialogo": round(p_tr, 4),
        "perda_val_dialogo": round(p_va, 4),
        "perda_val_com_nomes_do_treino": round(p_trocada, 4),
        "folga_memorizacao": round(p_va - p_tr, 4),
        "quanto_da_folga_era_nome": round(p_va - p_trocada, 4),
        "fraccao_explicada_pelo_nome": round((p_va - p_trocada) / (p_va - p_tr), 4),
        "falantes_so_na_val": sorted(novos),
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_memorizacao_dialogo.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\ntreino    : {p_tr:.4f}")
    print(f"validação : {p_va:.4f}   (folga {p_va-p_tr:+.4f})")
    print(f"validação com nomes do treino: {p_trocada:.4f}")
    print(f"\nnomes explicam {res['quanto_da_folga_era_nome']:+.4f} nats = "
          f"{res['fraccao_explicada_pelo_nome']*100:.1f}% da folga")
    if res['fraccao_explicada_pelo_nome'] < 0.10:
        print("-> a folga NÃO é dos nomes: o modelo memorizou o texto, não os falantes.")


if __name__ == "__main__":
    main()
