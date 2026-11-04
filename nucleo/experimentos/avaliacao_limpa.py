"""A avaliação honesta de diálogo: sem enchimento, sem atravessar conversas.

TRÊS TENTATIVAS, DUAS ERRADAS — a história fica escrita porque o erro ensina.

1.ª  janelas cegas de 128 sobre o texto todo. Simples, mas uma janela começa
     a meio de uma conversa e o modelo nunca sabe que mudou de assunto.

2.ª  uma janela por conversa, sobras preenchidas com o token 0 e perda
     mascarada. **ERRADO DUAS VEZES:**
       (a) o token 0 é o byte `\\x00`, que o modelo nunca vê em treino;
       (b) mais grave: mascarar a PERDA não mascara a ATENÇÃO. Os `\\x00`
           continuam no contexto e corrompem as previsões das posições reais.
           Medido: treino "0,0557" pela via cega e "6,0247" pela via mascarada
           — 100× de diferença, e a segunda via estava ainda contaminada.

3.ª  (esta) sem enchimento nenhum: dentro de cada conversa, janelas de 128
     consecutivas, descartando a cauda incompleta. Nenhuma janela atravessa
     uma fronteira e nenhuma posição é inventada. Descarta texto — e isso é
     honesto, porque o descarte é reportado.

O QUE SE COMPARA
    treino e validação, ambos pela via 3.ª -> a folga é comparável.
    E o número que importa: a validação está acima ou abaixo de log(8192)?
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128


def seccoes(ficheiro: str, titulo: str, marcador: str) -> list[str]:
    txt = (RAIZ / "dados" / "dialogos" / ficheiro).read_text(encoding="utf-8")
    corpo = re.split(titulo, txt, flags=re.M)
    texto = corpo[1] if len(corpo) > 1 else txt
    texto = re.split(r"^NOTA PARA O TREINO", texto, flags=re.M)[0]
    return [s.strip() for s in re.split(marcador, texto, flags=re.M) if s.strip()]


def dividir(secs, n_val, semente=20261104):
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(secs))
    vi = set(ordem[:n_val].tolist())
    return ([s for i, s in enumerate(secs) if i not in vi],
            [s for i, s in enumerate(secs) if i in vi])


def janelas_intra(secs, tok, ctx=CTX):
    """Janelas DENTRO de cada conversa, sem atravessar fronteiras, sem enchimento."""
    js, usados, descartados = [], 0, 0
    for s in secs:
        ids = tok.codificar(s)
        n = len(ids) // ctx
        for i in range(n):
            js.append(ids[i * ctx:(i + 1) * ctx])
            usados += ctx
        descartados += len(ids) - n * ctx
    arr = (np.array(js, dtype=np.int64) if js else np.zeros((0, ctx), dtype=np.int64))
    return arr, usados, descartados


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

    v1 = seccoes("conversas_v1.md", r"^# Conversas simuladas para treino de diálogo\s*$", r"^## ")
    v2 = seccoes("conversas_v2.md", r"^# Conversas sintéticas utilizador/IA — v2\s*$", r"^CONVERSA \d+")
    print(f"v1 {len(v1)} conversas | v2 {len(v2)} conversas | acaso {acaso:.4f}\n")

    resultados = []
    for nome, secs, n_val in (("v1", v1, 10), ("v2", v2, 6), ("ambas", v1 + v2, 12)):
        tr, va = dividir(secs, n_val)
        d_tr, u_tr, x_tr = janelas_intra(tr, tok)
        d_va, u_va, x_va = janelas_intra(va, tok)

        m = Transformer(cfg, semente=0)
        ids_p = np.array(json.loads(
            (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
        d_p = np.array([ids_p[i * CTX:(i + 1) * CTX]
                        for i in range(len(ids_p) // CTX)], dtype=np.int64)
        treinar(m, d_p)
        treinar(m, d_tr)

        p_tr, p_va = avaliar(m, d_tr), avaliar(m, d_va)
        r = {"nome": nome, "conversas_treino": len(tr), "conversas_val": len(va),
             "janelas_treino": len(d_tr), "janelas_val": len(d_va),
             "tokens_usados_treino": u_tr, "tokens_descartados_treino": x_tr,
             "tokens_usados_val": u_va, "tokens_descartados_val": x_va,
             "perda_treino": round(p_tr, 4), "perda_val": round(p_va, 4),
             "folga": round(p_va - p_tr, 4),
             "val_acima_do_acaso": p_va > acaso}
        resultados.append(r)
        print(f"{nome:<6} {len(d_tr):>3} jan | treino {p_tr:>8.4f} | "
              f"val {p_va:>7.4f} | folga {r['folga']:>7.4f} | "
              f"descarte {x_tr} tok treino / {x_va} val")

    base = next(r for r in resultados if r["nome"] == "v1")
    ambs = next(r for r in resultados if r["nome"] == "ambas")
    res = {
        "classe": "MEASURED",
        "metodo": "janelas intra-conversa, sem enchimento, sem atravessar fronteiras",
        "acaso_log_vocab": round(acaso, 4),
        "condicoes": resultados,
        "folga_v1": base["folga"], "folga_ambas": ambs["folga"],
        "razao_folga_ambas_sobre_v1": round(ambs["folga"] / base["folga"], 3),
        "todas_acima_do_acaso": all(r["val_acima_do_acaso"] for r in resultados),
        "substitui": "resultado_fronteiras.json (método 2, contaminado por padding "
                     "não mascarado na atenção) e a parte de validação de "
                     "resultado_escala_dialogo.json",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_avaliacao_limpa.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print()
    print(f"todas as condições ACIMA do acaso? "
          f"{'SIM' if res['todas_acima_do_acaso'] else 'nao'}")
    print(f"folga v1 {base['folga']:.4f} -> ambas {ambs['folga']:.4f} "
          f"({res['razao_folga_ambas_sobre_v1']:.2f}x): "
          + ("duplicar os dados NÃO resolve." if res['razao_folga_ambas_sobre_v1'] > 0.85
             else "duplicar os dados ajuda."))


if __name__ == "__main__":
    main()
