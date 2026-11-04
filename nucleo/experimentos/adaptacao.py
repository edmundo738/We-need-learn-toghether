"""Adaptação a diálogo: o que muda e o que se está a medir realmente.

O projecto tem 65 999 tokens de prosa de um autor e passa a ter 512 turnos de
conversa (2 900 tokens). A tentação é «treinar no total e ver a perda descer».
Isso mede pouco: a perda em validação de diálogo desce por três razões
diferentes, e só uma é aprendizagem.

AS TRÊS RAZÕES, E COMO SE SEPARAM
---------------------------------
1. O modelo aprendeu o formato («**Nome:**» seguido de fala).
2. O modelo memorizou as conversas de treino.
3. O modelo aprendeu padrões de diálogo que TRANSFEREM para conversas que
   nunca viu.

Só a 3 generaliza. Para a isolar: divisão POR CONVERSA (uma conversa inteira
fica num lado), nunca por janelas. Se dividir por janelas, turnos da mesma
conversa aparecem nos dois lados e a perda de validação é memorização.

DOIS BRAÇOS
    A) só diálogo            — o que se aprende com 2 900 tokens sozinhos
    B) prosa -> diálogo      — pré-treino nas 5 obras, depois afinação

BRAÇO DE CONTROLO
    B com o diálogo todo no treino e uma conversa ALEATÓRIA como validação
    não serve. O controlo é: avaliar o modelo SÓ-prosa nas conversas de
    validação. Se der o mesmo que o modelo afinado, a afinação não fez nada.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128


def carregar_conversas() -> tuple[list[str], list[str]]:
    """Devolve (treino, validacao) — divisão POR CONVERSA, semente fixa."""
    txt = (RAIZ / "dados" / "dialogos" / "conversas_v1.md").read_text(encoding="utf-8")
    corpo = re.split(r"^# Conversas simuladas para treino de diálogo\s*$", txt, flags=re.M)
    texto = corpo[1] if len(corpo) > 1 else txt
    secs = [s.strip() for s in re.split(r"^## ", texto, flags=re.M) if s.strip()]

    rng = np.random.default_rng(20261104)     # congelada, declarada
    ordem = rng.permutation(len(secs))
    n_val = 10                                 # 10 de 50 conversas
    val_idx = set(ordem[:n_val].tolist())
    treino = [secs[i] for i in range(len(secs)) if i not in val_idx]
    val = [secs[i] for i in range(len(secs)) if i in val_idx]
    return treino, val


def janelas(ids: list[int], ctx: int = CTX) -> np.ndarray:
    n = len(ids) // ctx
    if n == 0:
        return np.zeros((0, ctx), dtype=np.int64)
    return np.array([ids[i * ctx:(i + 1) * ctx] for i in range(n)], dtype=np.int64)


def avaliar(modelo, dados: np.ndarray) -> float:
    from nucleo.modelo.transformer import perda_entropia_cruzada
    if len(dados) == 0:
        return float("nan")
    perdas = []
    for i in range(0, len(dados), 8):
        w = dados[i:i + 8]
        perdas.append(float(perda_entropia_cruzada(
            modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    return float(np.mean(perdas))


def treinar(modelo, dados: np.ndarray, passos: int, lr: float, semente: int,
            modelo_base=None) -> dict:
    from nucleo import nn
    from nucleo.modelo.transformer import perda_entropia_cruzada

    opt = nn.Adam(modelo.parametros(), eta=lr, weight_decay=0.01)
    rng = np.random.default_rng(semente)
    lote = 4
    hist = []
    for passo in range(1, passos + 1):
        idx = rng.integers(0, len(dados), lote)
        w = dados[idx]
        opt.zero_grad()
        perda = perda_entropia_cruzada(modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1))
        perda.backward()
        opt.step(grad_clip=1.0)
        if passo % 100 == 0 or passo == 1:
            hist.append({"passo": passo,
                         "perda_treino": round(float(perda.data.reshape(-1)[0]), 4)})
    return {"historico": hist}


def main() -> None:
    from nucleo.modelo.transformer import Config, Transformer
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    treino_txt, val_txt = carregar_conversas()

    # codificar com BOS de conversa para marcar fronteiras reais
    ids_tr = tok.codificar("\n".join(treino_txt))
    ids_va = tok.codificar("\n".join(val_txt))
    d_tr = janelas(ids_tr)
    d_va = janelas(ids_va)

    print(f"diálogo: treino {len(ids_tr):,} tokens ({len(d_tr)} janelas) | "
          f"validação {len(ids_va):,} tokens ({len(d_va)} janelas)")
    print(f"divisão POR CONVERSA: {len(treino_txt)} / {len(val_txt)} de 50\n")

    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=CTX, d_ff=384, tying=False)

    resultados = {}

    # ---------- braço A: só diálogo ----------
    mA = Transformer(cfg, semente=0)
    perda_ini_A = avaliar(mA, d_va)
    t0 = time.time()
    infoA = treinar(mA, d_tr, passos=800, lr=3e-3, semente=0)
    segA = time.time() - t0
    perda_fim_A = avaliar(mA, d_va)
    resultados["A_so_dialogo"] = {
        "perda_val_inicial": round(perda_ini_A, 4),
        "perda_val_final": round(perda_fim_A, 4),
        "segundos": round(segA, 1), "passos": 800, "lr": 3e-3,
        "historico": infoA["historico"],
    }
    print(f"A) só diálogo    : {perda_ini_A:.4f} -> {perda_fim_A:.4f} "
          f"({segA:.0f}s)")

    # ---------- braço B: prosa -> diálogo ----------
    # O checkpoint em disco NÃO é fiável: está em .gitignore (*.npz) e o
    # snapshot do workspace não o preserva — verificado, a pasta desapareceu.
    # Um resultado que depende de um ficheiro invisível não é reproduzível.
    # Por isso a prosa é re-treinada AQUI, na mesma execução, com o mesmo
    # protocolo do braço A. Custa ~9 min e torna o resultado verificável.
    ids_prosa = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
    d_prosa = janelas(list(ids_prosa))
    mB = Transformer(cfg, semente=0)
    t0 = time.time()
    treinar(mB, d_prosa, passos=800, lr=3e-3, semente=0)
    segB_pre = time.time() - t0
    print(f"   (pré-treino em prosa: {segB_pre:.0f}s, {len(d_prosa)} janelas)")
    perda_ini_B = avaliar(mB, d_va)      # CONTROLO: prosa pura sobre diálogo
    t0 = time.time()
    infoB = treinar(mB, d_tr, passos=800, lr=3e-3, semente=0)
    segB = time.time() - t0
    perda_fim_B = avaliar(mB, d_va)
    resultados["B_prosa_para_dialogo"] = {
        "perda_val_inicial_prosa_pura": round(perda_ini_B, 4),
        "perda_val_final": round(perda_fim_B, 4),
        "segundos_afinacao": round(segB, 1),
        "segundos_pretreino_prosa": round(segB_pre, 1),
        "passos": 800, "lr": 3e-3, "historico": infoB["historico"],
    }
    print(f"B) prosa->diálogo: {perda_ini_B:.4f} -> {perda_fim_B:.4f} "
          f"({segB:.0f}s)")

    # ---------- custo de memorização: perda no TREINO vs VALIDAÇÃO ----------
    # Se a perda de treino descer muito abaixo da de validação, é memorização.
    p_tr_B = avaliar(mB, d_tr)
    p_va_B = avaliar(mB, d_va)
    resultados["memorizacao_B"] = {
        "perda_treino_dialogo": round(p_tr_B, 4),
        "perda_val_dialogo": round(p_va_B, 4),
        "folga": round(p_va_B - p_tr_B, 4),
        "nota": "folga grande = memorizou as conversas de treino; "
                "só a perda de VALIDAÇÃO (conversas nunca vistas) mede generalização",
    }
    print(f"\nB no treino de diálogo: {p_tr_B:.4f} | validação: {p_va_B:.4f} "
          f"| folga {p_va_B - p_tr_B:+.4f}")

    resultados["classe"] = "MEASURED"
    resultados["divisao"] = {"metodo": "por conversa inteira", "semente": 20261104,
                              "conversas_treino": len(treino_txt),
                              "conversas_val": len(val_txt)}
    resultados["nota"] = ("A e B com os mesmos 800 passos, lote 4, lr 3e-3, semente 0. "
                           "A perda inicial de B é o CONTROL: prosa pura avaliada em diálogo.")
    (RAIZ / "nucleo" / "experimentos" / "resultado_adaptacao.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False))

    print("\n--- LEITURA ---")
    print(f"A parte de pesos aleatórios: {perda_ini_A:.2f} -> {perda_fim_A:.2f}")
    print(f"B parte de prosa treinada:  {perda_ini_B:.2f} -> {perda_fim_B:.2f}")
    print("Se B começa MUITO abaixo de A, a prosa transferiu formato e sintaxe.")
    print("Se B acaba igual a A, a afinação não acrescentou nada à prosa.")


if __name__ == "__main__":
    main()
