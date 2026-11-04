"""Avaliação de diálogo: fronteiras, padding e máscara. E um erro meu, corrigido.

HISTÓRIA DESTE FICHEIRO (fica escrita porque o erro é instrutivo)
------------------------------------------------------------------
1. `escala_dialogo.py` mediu o v2 isolado com 9,4385 de validação, ACIMA do
   acaso 9,011 — treinar nas conversas utilizador/IA parecia pior que não
   treinar.
2. Eu levantei a hipótese «efeito de fronteira»: janelas cegas de 128 começam a
   meio de conversas de ~292 tokens.
3. Escrevi um teste B que cortava uma janela por conversa e enchia as sobras
   com `pad_id = 0`. Resultado: 11,06 — pior que o corte cego. Isso "confirmava"
   a hipótese contrária.
4. **O teste B estava errado.** O tokenizador não tem token de padding; inventei
   `pad = 0`, que é o byte `\\x00` — um byte de controlo que o modelo nunca vê
   em treino. O teste pedia ao modelo que previsse um byte nulo em ~24% das
   posições, e naturalmente dava perda enorme. Também a perda de treino era
   medida com corte cego enquanto a de validação era por conversa: **contas em
   bases diferentes**, e a folga não significava nada.

A LIÇÃO: um teste que mede duas coisas diferentes e as subtrai não mede nada.

O QUE ESTE FICHEIRO FAZ AGORA
-----------------------------
Avalia as MESMAS conversas de validação de três formas, todas com máscara
(ignorar posições que não são texto real):
    A) corte cego: concatena e corta em janelas de 128
    B) por conversa: cada conversa começada numa janela nova, sobras MASCARADAS
Isola o que é do modelo e o que é do instrumento.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128


def seccoes_v2() -> list[str]:
    txt = (RAIZ / "dados" / "dialogos" / "conversas_v2.md").read_text(encoding="utf-8")
    corpo = re.split(r"^# Conversas sintéticas utilizador/IA — v2\s*$", txt, flags=re.M)
    texto = corpo[1] if len(corpo) > 1 else txt
    texto = re.split(r"^NOTA PARA O TREINO", texto, flags=re.M)[0]
    return [s.strip() for s in re.split(r"^CONVERSA \d+", texto, flags=re.M) if s.strip()]


def dividir(secs, n_val=6, semente=20261104):
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(secs))
    vi = set(ordem[:n_val].tolist())
    return ([s for i, s in enumerate(secs) if i not in vi],
            [s for i, s in enumerate(secs) if i in vi])


def janelas_cegas(ids, ctx=CTX):
    n = len(ids) // ctx
    return (np.array([ids[i * ctx:(i + 1) * ctx] for i in range(n)], dtype=np.int64)
            if n else np.zeros((0, ctx), dtype=np.int64))


def janelas_mascaradas(secs, tok, ctx=CTX):
    """Uma janela por bloco; devolve (janelas, máscara) com True só em texto real."""
    js, ms = [], []
    for s in secs:
        ids = tok.codificar(s)
        n = max(1, (len(ids) + ctx - 1) // ctx)
        m = [1] * len(ids) + [0] * (n * ctx - len(ids))
        ids = ids + [0] * (n * ctx - len(ids))     # o valor é irrelevante: mascarado
        for i in range(n):
            js.append(ids[i * ctx:(i + 1) * ctx])
            ms.append(m[i * ctx:(i + 1) * ctx])
    return np.array(js, dtype=np.int64), np.array(ms, dtype=np.int64)


def perda_mascarada(modelo, janelas: np.ndarray, mascara: np.ndarray) -> float:
    """Entropia cruzada média SÓ nas posições reais."""
    from nucleo.modelo.transformer import perda_entropia_cruzada

    total, n = 0.0, 0
    for i in range(len(janelas)):
        w = janelas[i:i + 1]
        alvo = w[0, 1:]
        m = mascara[i, 1:].astype(bool)
        if not m.any():
            continue
        lg = modelo.forward(w[:, :-1])
        z = lg.data
        if z.ndim == 3:          # (1, T, V) -> (T, V)
            z = z[0]
        # perda por posição, sem reduzir, para poder mascarar
        zmax = z.max(axis=-1, keepdims=True)
        logp = z - zmax - np.log(np.exp(z - zmax).sum(axis=-1, keepdims=True))
        lpos = -logp[np.arange(len(alvo)), alvo]
        total += float(lpos[m].sum())
        n += int(m.sum())
    return total / max(n, 1)


def avaliar_cego(modelo, dados: np.ndarray) -> float:
    from nucleo.modelo.transformer import perda_entropia_cruzada
    ps = []
    for i in range(0, len(dados), 8):
        w = dados[i:i + 8]
        ps.append(float(perda_entropia_cruzada(
            modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    return float(np.mean(ps))


def treinar(modelo, dados, passos=800, lote=4, lr=3e-3, semente=0):
    from nucleo import nn
    from nucleo.modelo.transformer import perda_entropia_cruzada
    opt = nn.Adam(modelo.parametros(), eta=lr, weight_decay=0.01)
    rng = np.random.default_rng(semente)
    for _ in range(passos):
        idx = rng.integers(0, len(dados), lote)
        w = dados[idx]
        opt.zero_grad()
        perda_entropia_cruzada(modelo.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)


def main() -> None:
    from nucleo.modelo.transformer import Config, Transformer
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=CTX, d_ff=384, tying=False)
    acaso = math.log(tok.tamanho_vocab)

    v2 = seccoes_v2()
    tr, va = dividir(v2, 6)

    m = Transformer(cfg, semente=0)
    ids_p = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
    treinar(m, janelas_cegas(list(ids_p)))
    d_tr = janelas_cegas(tok.codificar("\n".join(tr)))
    treinar(m, d_tr)

    d_va_cego = janelas_cegas(tok.codificar("\n".join(va)))
    j_conv, m_conv = janelas_mascaradas(va, tok)
    j_tr_conv, m_tr_conv = janelas_mascaradas(tr, tok)

    p_tr_cego = avaliar_cego(m, d_tr)
    p_va_cego = avaliar_cego(m, d_va_cego)
    p_va_conv = perda_mascarada(m, j_conv, m_conv)
    p_tr_conv = perda_mascarada(m, j_tr_conv, m_tr_conv)

    n_borda = sum(1 for s in va if len(tok.codificar(s)) % CTX != 0)
    res = {
        "classe": "MEASURED",
        "acaso_log_vocab": round(acaso, 4),
        "corte_cego": {"treino": round(p_tr_cego, 4), "val": round(p_va_cego, 4),
                        "folga": round(p_va_cego - p_tr_cego, 4)},
        "por_conversa_mascarado": {"treino": round(p_tr_conv, 4),
                                     "val": round(p_va_conv, 4),
                                     "folga": round(p_va_conv - p_tr_conv, 4)},
        "efeito_de_mascarar_na_val": round(p_va_conv - p_va_cego, 4),
        "conversas_val_que_nao_terminam_em_janela_cheia": n_borda,
        "conversas_val": len(va),
        "acima_do_acaso_cego": p_va_cego > acaso,
        "acima_do_acaso_mascarado": p_va_conv > acaso,
        "correcao_de_erro_anterior": "a versão anterior usava pad_id=0 (byte \\x00, "
            "nunca visto em treino) SEM máscara, e comparava treino em corte cego "
            "com validação por conversa. Valores obsoletos: val_por_conversa 11,0625. "
            "Não usar.",
        "nota": "as duas condições usam AGORA a mesma base: máscara nas posições "
                "que não são texto real. A folga é comparável.",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_fronteiras.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"acaso (log vocab)          : {acaso:.4f}\n")
    print(f"{'corte':<18} {'treino':>9} {'validação':>11} {'folga':>9}")
    print("-" * 50)
    print(f"{'cego':<18} {p_tr_cego:>9.4f} {p_va_cego:>11.4f} "
          f"{p_va_cego-p_tr_cego:>9.4f}")
    print(f"{'por conversa+mask':<18} {p_tr_conv:>9.4f} {p_va_conv:>11.4f} "
          f"{p_va_conv-p_tr_conv:>9.4f}")
    print()
    print(f"{n_borda}/{len(va)} conversas de validação não terminam em janela cheia")
    print(f"acima do acaso? cego={'SIM' if p_va_cego>acaso else 'nao'} | "
          f"mascarado={'SIM' if p_va_conv>acaso else 'nao'}")


if __name__ == "__main__":
    main()
