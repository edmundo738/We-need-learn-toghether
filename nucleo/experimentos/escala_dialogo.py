"""Duplicar os dados de diálogo reduz a memorização? — a pergunta testável.

A adenda B mediu, com 50 conversas (9 653 tokens) e 1,7 M parâmetros:
    perda no treino 0,0510  |  perda na validação 8,8613  |  folga 8,81 nats
Conclusão dessa medição: o modelo DECOROU. A hipótese implícita era que mais
dados resolvem. Esta experiência testa essa hipótese em vez de a assumir.

TRÊS CONDIÇÕES, MESMO PROTOCOLO (800 passos, lote 4, lr 3e-3, semente 0)
    v1     50 conversas pessoa-pessoa, formato "**Nome:**"
    v2     30 conversas utilizador-IA, formato "Utilizador:/IA:"
    ambas  as duas juntas

O QUE MEDE A HIPÓTESE
    folga = perda_val - perda_treino
Uma folga que ENCOLHE quando os dados dobram apoia a hipótese «faltam dados».
Uma folga que se MANTÉM diz que 20 mil tokens continuam a ser triviais para
1,7 M parâmetros, e que o próximo passo tem de ser ordens de grandeza, não 2x.

CONTROLO OBRIGATÓRIO
    `acaso` = log(vocab) = 9,011. Um modelo que não aprendeu nada dá isto.
    Se a validação de uma condição ficar ACIMA do acaso, a condição é pior
    do que não treinar — e isso tem de ser dito, não escondido.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128
PASSOS = 800
LOTE = 4
LR = 3e-3


def seccoes_v1() -> list[str]:
    txt = (RAIZ / "dados" / "dialogos" / "conversas_v1.md").read_text(encoding="utf-8")
    corpo = re.split(r"^# Conversas simuladas para treino de diálogo\s*$", txt, flags=re.M)
    texto = corpo[1] if len(corpo) > 1 else txt
    return [s.strip() for s in re.split(r"^## ", texto, flags=re.M) if s.strip()]


def seccoes_v2() -> list[str]:
    txt = (RAIZ / "dados" / "dialogos" / "conversas_v2.md").read_text(encoding="utf-8")
    corpo = re.split(r"^# Conversas sintéticas utilizador/IA — v2\s*$", txt, flags=re.M)
    texto = corpo[1] if len(corpo) > 1 else txt
    texto = re.split(r"^NOTA PARA O TREINO", texto, flags=re.M)[0]
    return [s.strip() for s in re.split(r"^CONVERSA \d+", texto, flags=re.M) if s.strip()]


def dividir(secs: list[str], n_val: int, semente: int = 20261104):
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(secs))
    vi = set(ordem[:n_val].tolist())
    return ([s for i, s in enumerate(secs) if i not in vi],
            [s for i, s in enumerate(secs) if i in vi])


def janelas(ids, ctx=CTX):
    n = len(ids) // ctx
    return (np.array([ids[i * ctx:(i + 1) * ctx] for i in range(n)], dtype=np.int64)
            if n else np.zeros((0, ctx), dtype=np.int64))


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


def treinar(modelo, dados, passos: int = PASSOS, semente: int = 0):
    from nucleo import nn
    from nucleo.modelo.transformer import perda_entropia_cruzada

    opt = nn.Adam(modelo.parametros(), eta=LR, weight_decay=0.01)
    rng = np.random.default_rng(semente)
    for _ in range(passos):
        idx = rng.integers(0, len(dados), LOTE)
        w = dados[idx]
        opt.zero_grad()
        perda_entropia_cruzada(modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)


def condicao(nome, treino_txt, val_txt, tok, cfg, com_prosa: bool):
    from nucleo.modelo.transformer import Transformer

    m = Transformer(cfg, semente=0)
    if com_prosa:
        ids_p = np.array(json.loads(
            (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
        treinar(m, janelas(list(ids_p)))
    d_tr = janelas(tok.codificar("\n".join(treino_txt)))
    d_va = janelas(tok.codificar("\n".join(val_txt)))
    treinar(m, d_tr)
    p_tr, p_va = avaliar(m, d_tr), avaliar(m, d_va)
    return {"nome": nome, "conversas_treino": len(treino_txt),
            "conversas_val": len(val_txt), "janelas_treino": len(d_tr),
            "perda_treino": round(p_tr, 4), "perda_val": round(p_va, 4),
            "folga": round(p_va - p_tr, 4)}


def main() -> None:
    from nucleo.modelo.transformer import Config
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=CTX, d_ff=384, tying=False)
    acaso = math.log(tok.tamanho_vocab)

    v1, v2 = seccoes_v1(), seccoes_v2()
    print(f"v1: {len(v1)} conversas | v2: {len(v2)} conversas | acaso = {acaso:.3f}\n")

    resultados = []
    for nome, secs, n_val in (("v1", v1, 10), ("v2", v2, 6), ("ambas", v1 + v2, 12)):
        tr, va = dividir(secs, n_val)
        t0 = time.time()
        r = condicao(nome, tr, va, tok, cfg, com_prosa=True)
        r["segundos"] = round(time.time() - t0, 1)
        r["acima_do_acaso"] = r["perda_val"] > acaso
        print(f"{nome:<6} {r['conversas_treino']:>2} conv | "
              f"treino {r['perda_treino']:>8.4f} | val {r['perda_val']:>7.4f} | "
              f"folga {r['folga']:>7.4f} | {r['segundos']:.0f}s")
        resultados.append(r)

    base = next(r for r in resultados if r["nome"] == "v1")
    ambs = next(r for r in resultados if r["nome"] == "ambas")
    res = {
        "classe": "MEASURED",
        "protocolo": {"passos": PASSOS, "lote": LOTE, "lr": LR, "ctx": CTX,
                      "pre_treino_prosa": True, "seeds": [0],
                      "divisao": "por conversa inteira, semente 20261104"},
        "acaso": round(acaso, 4),
        "condicoes": resultados,
        "comentario": "folga = perda de validacao menos perda de treino. "
                       "Encolher com mais dados apoia a hipotese de falta de dados.",
    }
    # a hipótese testada, com o número que a apoia ou refuta
    res["reducao_da_folga_de_v1_para_ambas"] = round(base["folga"] - ambs["folga"], 4)
    res["razao_de_folga_ambas_sobre_v1"] = round(ambs["folga"] / base["folga"], 4)
    (RAIZ / "nucleo" / "experimentos" / "resultado_escala_dialogo.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nacaso (log vocab): {acaso:.3f}")
    print(f"folga v1    : {base['folga']:.4f}")
    print(f"folga ambas : {ambs['folga']:.4f}")
    print(f"redução     : {res['reducao_da_folga_de_v1_para_ambas']:+.4f} nats "
          f"({res['razao_de_folga_ambas_sobre_v1']:.2f}x)")
    print()
    if res["razao_de_folga_ambas_sobre_v1"] > 0.85:
        print("=> a folga MANTEVE-SE com o dobro dos dados: 2x não resolve.")
        print("   O próximo passo tem de ser ordens de grandeza, não duplicar.")
    else:
        print("=> a folga ENCOLHEU com mais dados: apoia «faltam dados».")


if __name__ == "__main__":
    main()
