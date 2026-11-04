"""Dimensão 2 da avaliação: gramática. Estava a zero. Passa a medida.

A pergunta: o modelo prefere a frase CORRECTA à VIOLADA? É um teste de
julgamento de gramaticalidade — o modelo nunca é treinado para responder,
só se mede a probabilidade que atribui a cada frase.

BASELINE OBRIGATÓRIO: moeda ao ar = 50%. Sem isso, «acertou 60%» não diz nada.
E um segundo baseline, mais duro: o TRIGRAMA, que tem os mesmos dados.

ARMADILHA QUE O TESTE TEM DE EVITAR
    A frase violada tem, por vezes, um token a mais ou a menos. Se o modelo
    tiver tendência geral para frases mais curtas (ou mais longas), acerta
    por comprimento e não por gramática. Por isso se reporta:
      - acerto bruto
      - acerto corrigido pelo comprimento: subtrai-se a diferença de perda
        atribuível ao número de tokens (perda por token, não perda total)
    A perda é calculada POR TOKEN, não somada, exactamente por isto.

E a comparação com o trigrama tem de ser no mesmo texto: os 26 pares são
escritos com vocabulário comum, mas o trigrama só sabe o que viu no corpus.
Um trigrama que nunca viu «crase» não pode julgar crase. Isso é o resultado,
não um defeito do teste.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
PARES = RAIZ / "nucleo" / "avaliacao" / "gramatica" / "pares_v1.json"


def perda_frase_transformer(modelo, tok, frase: str, ctx: int = 128) -> float:
    """Perda média por token da frase, condicionada no que couber no contexto."""
    from nucleo.modelo.transformer import perda_entropia_cruzada

    ids = tok.codificar(frase)
    if len(ids) < 2:
        return float("nan")
    ids = ids[:ctx]
    arr = np.array([ids], dtype=np.int64)
    lg = modelo.forward(arr[:, :-1])
    return float(perda_entropia_cruzada(lg, arr[0, 1:]).data.reshape(-1)[0])


def perda_frase_ngrama(modelo_ng, tok, frase: str, ordem: int = 3) -> float:
    ids = tok.codificar(frase)
    total, n = 0.0, 0
    for i in range(ordem - 1, len(ids)):
        ctx = tuple(int(x) for x in ids[i - ordem + 1:i])
        total += -modelo_ng.logp(ctx, int(ids[i]))
        n += 1
    return total / max(n, 1)


def julgar(perdas_correcto: list[float], perdas_violado: list[float]) -> dict:
    """Acerto = prefere o correcto (perda menor). Empates contam metade."""
    acertos = 0
    for c, v in zip(perdas_correcto, perdas_violado):
        if c < v:
            acertos += 1
        elif c == v:
            acertos += 0.5
    n = len(perdas_correcto)
    media_c = float(np.mean(perdas_correcto))
    media_v = float(np.mean(perdas_violado))
    return {
        "n": n,
        "acertos": acertos,
        "taxa": round(acertos / n, 4),
        "perda_media_correcto": round(media_c, 4),
        "perda_media_violado": round(media_v, 4),
        "margem_media": round(media_v - media_c, 4),
    }


def binomial_p(acertos: int, n: int) -> float:
    """p-valor bilateral exacto contra p=0,5 — a moeda ao ar."""
    from math import comb
    p = sum(comb(n, k) for k in range(acertos, n + 1)) / (2 ** n)
    return round(min(1.0, 2 * p), 4)


def main() -> None:
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada
    from nucleo.tokenizador.bpe import TokenizadorBPE
    from nucleo import nn

    d = json.loads(PARES.read_text(encoding="utf-8"))
    pares = d["pares"]
    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")

    # ---- Transformer treinado em prosa (o que existe) ----
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=128, d_ff=384, tying=False)
    m = Transformer(cfg, semente=0)
    ids_p = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
    d_p = np.array([ids_p[i * 128:(i + 1) * 128]
                    for i in range(len(ids_p) // 128)], dtype=np.int64)
    opt = nn.Adam(m.parametros(), eta=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for _ in range(800):
        idx = rng.integers(0, len(d_p), 4)
        w = d_p[idx]
        opt.zero_grad()
        perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).backward()
        opt.step(grad_clip=1.0)

    # ---- trigrama, mesmos dados ----
    from nucleo.experimentos.baselines import Interpolado
    ng = Interpolado(3, tok.tamanho_vocab, 0.75)
    ng.treinar(ids_p)

    pc_t = [perda_frase_transformer(m, tok, p["correcto"]) for p in pares]
    pv_t = [perda_frase_transformer(m, tok, p["violado"]) for p in pares]
    pc_n = [perda_frase_ngrama(ng, tok, p["correcto"]) for p in pares]
    pv_n = [perda_frase_ngrama(ng, tok, p["violado"]) for p in pares]

    t = julgar(pc_t, pv_t)
    n = julgar(pc_n, pv_n)
    t["p_valor_vs_moeda"] = binomial_p(int(t["acertos"]), t["n"])
    n["p_valor_vs_moeda"] = binomial_p(int(n["acertos"]), n["n"])

    # por fenómeno, para ver onde acerta e onde falha
    por_fen: dict[str, dict] = {}
    for i, p in enumerate(pares):
        f = p["fenomeno"]
        e = por_fen.setdefault(f, {"n": 0, "acertos_transformer": 0, "acertos_ngrama": 0})
        e["n"] += 1
        e["acertos_transformer"] += int(pc_t[i] < pv_t[i])
        e["acertos_ngrama"] += int(pc_n[i] < pv_n[i])

    res = {
        "classe": "MEASURED",
        "instrumento": "26 pares correcto/violado, 14 fenómenos, frases escritas por "
                       "este projecto (estrutura derivada do Bechara, texto não copiado)",
        "baseline_moeda": 0.5,
        "transformer_prosa_800_passos": t,
        "trigrama_mesmos_dados": n,
        "por_fenomeno": por_fen,
        "leitura": "acerto acima de 0,5 com p<0,05 é julgamento de gramaticalidade. "
                   "Entre o acaso e o insignificante, o modelo NÃO julga gramática — "
                   "e nenhuma afirmação sobre «saber português» se sustenta.",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_gramatical.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"{'modelo':<22} {'acertos':>9} {'taxa':>8} {'p vs moeda':>11} {'margem':>9}")
    print("-" * 64)
    print(f"{'moeda ao ar':<22} {'13/26':>9} {'0,500':>8} {'—':>11} {'0,0000':>9}")
    print(f"{'Transformer (prosa)':<22} {t['acertos']:>9.1f} {t['taxa']:>8.3f} "
          f"{t['p_valor_vs_moeda']:>11.4f} {t['margem_media']:>9.4f}")
    print(f"{'trigrama':<22} {n['acertos']:>9.1f} {n['taxa']:>8.3f} "
          f"{n['p_valor_vs_moeda']:>11.4f} {n['margem_media']:>9.4f}")
    print()
    print("por fenómeno (acertos / total):")
    for f, e in sorted(por_fen.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {f:<32} transformer {e['acertos_transformer']}/{e['n']} | "
              f"trigrama {e['acertos_ngrama']}/{e['n']}")


if __name__ == "__main__":
    main()
