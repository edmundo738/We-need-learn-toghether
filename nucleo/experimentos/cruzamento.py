"""ONDE É QUE O TRANSFORMER CRUZA O TRIGRAMA?

Esta é a experiência que a auditoria identificou como a que falta. A pergunta
não é «o modelo é bom» — é:

    a partir de que volume de dados é que um Transformer de 2,77 M
    parâmetros bate um trigrama com backoff?

Porque a resposta anterior foi: **não bate**, com 66 mil tokens. E a hipótese,
formulada para ser falsa, é que o problema é volume e não arquitectura.

COMO SE MEDE SEM INVENTAR DADOS
-------------------------------
Não há mais corpus. O que há é a possiblidade de variar a quantidade de
treino MANTENDO o conjunto de validação fixo. Desenho:

  - validação: as últimas 52 janelas de 128 tokens do corpus real (fixas,
    sempre as mesmas, nunca treinadas em nenhuma condição);
  - treino: uma fracção das restantes 459 janelas — 10%, 25%, 50%, 100%;
  - para cada fracção: trigrama treinado NESSA fracção, Transformer
    treinado NESSA fracção com o MESMO número de passos por token visto;
  - métrica: perda de validação, em nats e bits/byte.

O ponto de cruzamento é a fracção onde a curva do Transformer passa por
baixo da do trigrama.

AVISO HONESTO SOBRE O QUE ESTA EXPERIÊNCIA NÃO PROVA
----------------------------------------------------
Nas fracções pequenas (<100%), o Transformer pode ver o mesmo token várias
vezes (as épocas), mas o trigrama só conta — não tem épocas. Está-se a
comparar «aprender por gradiente com repetição» contra «contar uma vez».
Isso é uma assimetria a favor do Transformer, e fica declarada. O que se
mede é o que se pode medir com o corpus que existe.

O número de passos é fixado em posições processadas, não em passos, para que
as condições sejam comparáveis: cada condição vê o mesmo número de tokens.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from nucleo import nn
from nucleo.experimentos.baselines import avaliar as avaliar_ngrama
from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada
from nucleo.tokenizador.bpe import TokenizadorBPE

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128
CORTE_VALIDACAO = 0.9


def carregar_janelas() -> tuple[np.ndarray, np.ndarray, int]:
    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    ids = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()),
        dtype=np.int64)
    total = len(ids) // (CTX + 1)
    dados = np.array([ids[i * (CTX + 1):(i + 1) * (CTX + 1)] for i in range(total)])
    corte = int(total * CORTE_VALIDACAO)
    return dados[:corte], dados[corte:], tok.tamanho_vocab


def treinar_transformer(treino: np.ndarray, val: np.ndarray, vocab: int,
                        posicoes_alvo: int, lote: int = 4,
                        semente: int = 0) -> dict:
    """Treina até processar `posicoes_alvo` tokens. Devolve a perda de validação."""
    cfg = Config(vocab=vocab, d_model=128, n_camadas=4, n_cabecas=4,
                 ctx=CTX, d_ff=384)
    m = Transformer(cfg, semente=semente)
    opt = nn.Adam(m.parametros(), eta=3e-3)
    rng = np.random.default_rng(semente)
    por_passo = lote * CTX
    passos = max(1, posicoes_alvo // por_passo)

    t0 = time.perf_counter()
    for p in range(passos):
        idx = rng.integers(0, len(treino), lote)
        w = treino[idx]
        opt.zero_grad()
        perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
        perda.backward()
        opt.step(grad_clip=1.0)
    tempo = time.perf_counter() - t0

    # avaliação em lotes de 8 (a memória cresce com o quadrado; ver auditoria)
    perdas = []
    for i in range(0, len(val), 8):
        w = val[i:i + 8]
        lg = m.forward(w[:, :-1])
        perdas.append(float(perda_entropia_cruzada(lg, w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    return {"perda_val": float(np.mean(perdas)), "passos": passos,
            "segundos": round(tempo, 1), "posicoes": passos * por_passo}


def main() -> None:
    treino_completo, val, vocab = carregar_janelas()
    print(f"treino disponível: {len(treino_completo)} janelas · "
          f"validação: {len(val)} janelas (fixas) · vocab {vocab}\n")

    fracs = [0.10, 0.25, 0.50, 1.00]
    # posições processadas = constantes por condição. Escolhido para caber no
    # orçamento: 245 760 posições = 480 passos a lote 4, ~2-3 min por condição.
    POSICOES = 4 * CTX * 480

    resultados = []
    print(f"{'fracção':>8} {'janelas':>8} {'trigrama':>10} {'transformer':>12} "
          f"{'diferença':>11} {'seg':>6}")
    for f in fracs:
        n = max(2, int(len(treino_completo) * f))
        tr = treino_completo[:n]
        # trigrama na MESMA fracção
        r3 = avaliar_ngrama(3, tr, val, vocab, desconto=0.75)
        # transformer na mesma fracção
        rt = treinar_transformer(tr, val, vocab, POSICOES)
        dif = rt["perda_val"] - r3["perda_nats"]
        resultados.append({
            "fraccao": f, "janelas_treino": n,
            "tokens_treino": int(tr.size), "tokens_unicos": int(n * CTX),
            "trigrama": round(r3["perda_nats"], 4),
            "transformer": round(rt["perda_val"], 4),
            "diferenca": round(dif, 4),
            "transformer_melhor": bool(dif < 0),
            "passos": rt["passos"], "segundos": rt["segundos"],
            "posicoes_processadas": rt["posicoes"],
        })
        print(f"{f:>7.0%} {n:>8} {r3['perda_nats']:>10.4f} "
              f"{rt['perda_val']:>12.4f} {dif:>+11.4f} {rt['segundos']:>6.0f}")

    (RAIZ / "nucleo" / "experimentos" / "resultado_cruzamento.json").write_text(
        json.dumps({"ctx": CTX, "validacao_janelas": len(val),
                    "posicoes_por_condicao": POSICOES,
                    "condicoes": resultados}, indent=2, ensure_ascii=False))

    ganhos = [r for r in resultados if r["transformer_melhor"]]
    print()
    if not ganhos:
        print("EM NENHUMA FRACÇÃO o Transformer bate o trigrama.")
        print("Com o corpus que existe, o ponto de cruzamento está ACIMA de 100%.")
        print("Conclusão: não se sabe se é arquitectura ou volume — mas sabe-se")
        print("que 66 mil tokens não chegam, e isso é replicável.")
    else:
        primeira = min(ganhos, key=lambda r: r["fraccao"])
        print(f"O Transformer bate o trigrama a partir de {primeira['fraccao']:.0%} "
              f"do corpus ({primeira['tokens_unicos']:,} tokens únicos).")


if __name__ == "__main__":
    main()
