"""O modelo está a sobreajustar? — a pergunta que a curva de treino levanta.

O treino de 1500 passos terminou com perda_treino 4,09 e perda_val 5,90.
O resumo anterior registou só o «5,90 final». A curva mostra outra coisa:

    passo  550: val 5,729  (mínimo da região)
    passo 1500: val 5,896  (o valor registado, 0,17 nats PIOR)

O melhor ponto esteve a um terço do treino. Isto mede-se, não se afirma:
treina-se com 3 seeds, guarda-se a curva completa, e reporta-se o MÍNIMO
de validação com o passo em que ocorreu — não o valor final.

MÉTRICA DE SOBREAJUSTE
    folga = perda_val - perda_treino
Compara-se com o intervalo de ruído entre seeds. Uma única seed não
distingue sobreajuste de flutuação.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]


def treinar(ids: np.ndarray, vocab: int, passos: int, semente: int,
            ctx: int = 128, lote: int = 2, lr: float = 3e-3,
            tying: bool = True) -> dict:
    from nucleo import nn
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada

    total = len(ids) // (ctx + 1)
    dados = np.array([ids[i * (ctx + 1):(i + 1) * (ctx + 1)] for i in range(total)])
    corte = int(total * 0.9)
    treino, val = dados[:corte], dados[corte:]

    cfg = Config(vocab=vocab, d_model=128, n_camadas=4, n_cabecas=4,
                 ctx=ctx, d_ff=384, tying=tying)
    m = Transformer(cfg, semente=semente)
    opt = nn.Adam(m.parametros(), eta=lr, weight_decay=0.01)
    rng = np.random.default_rng(semente)

    hist = []
    for passo in range(1, passos + 1):
        bi = rng.integers(0, len(treino), lote)
        w = treino[bi]
        opt.zero_grad()
        lg = m.forward(w[:, :-1])
        alvo = w[:, 1:].reshape(-1)
        perda = perda_entropia_cruzada(lg, alvo)
        perda.backward()
        opt.step(grad_clip=1.0)
        if passo % 50 == 0:
            lv = np.mean([float(perda_entropia_cruzada(
                m.forward(v[None, :-1]), v[None, 1:].reshape(-1)).data.reshape(-1)[0])
                for v in val])
            hist.append({"passo": passo,
                         "perda_treino": round(float(perda.data.reshape(-1)[0]), 4),
                         "perda_val": round(float(lv), 4)})
    etiqueta = f"{'tying' if tying else 'semblying'}_seed{semente}"
    caminho_ckpt = m.guardar(RAIZ / "nucleo" / "experimentos" / "checkpoints" / etiqueta)
    melhor = min(hist, key=lambda h: h["perda_val"])
    return {"semente": semente, "tying": tying, "n_parametros": m.n_parametros(),
            "checkpoint": str(caminho_ckpt), "historico": hist,
            "melhor_val": melhor["perda_val"], "melhor_passo": melhor["passo"],
            "final_val": hist[-1]["perda_val"], "final_treino": hist[-1]["perda_treino"]}


def main() -> None:
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    ids = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)

    resultados = []
    for tying in (True, False):
        for semente in (0, 1, 2):
            t0 = time.time()
            r = treinar(ids, tok.tamanho_vocab, passos=1000, semente=semente, tying=tying)
            r["segundos"] = round(time.time() - t0, 1)
            print(f"tying={tying} seed {semente}: melhor_val {r['melhor_val']:.4f} @ "
                  f"{r['melhor_passo']}  |  final_val {r['final_val']:.4f}  "
                  f"| {r['n_parametros']:,} params  ({r['segundos']}s)")
            resultados.append(r)

    res = {"classe": "MEASURED", "passos": 1000, "seeds": 3, "ctx": 128,
           "lote": 2, "lr": 3e-3, "resultados": resultados}

    for tying in (True, False):
        sel = [r for r in resultados if r["tying"] == tying]
        fins = [r["final_val"] for r in sel]
        mel = [r["melhor_val"] for r in sel]
        gan = [r["final_val"] - r["melhor_val"] for r in sel]
        chave = "tying" if tying else "sem_tying"
        res[chave] = {
            "n_parametros": sel[0]["n_parametros"],
            "final_val_media": round(float(np.mean(fins)), 4),
            "final_val_desvio": round(float(np.std(fins, ddof=1)), 4),
            "final_val_por_seed": fins,
            "melhor_val_media": round(float(np.mean(mel)), 4),
            "perda_ao_parar_media": round(float(np.mean(gan)), 4),
            "seeds_que_pioram": sum(1 for g in gan if g > 0),
            "passo_do_minimo": [r["melhor_passo"] for r in sel],
        }

    t = res["tying"]; st = res["sem_tying"]
    dif = st["final_val_media"] - t["final_val_media"]
    res["diferenca_sem_tying_menos_tying"] = round(float(dif), 4)
    res["nota"] = ("A/B com os mesmos dados, ordem de lotes por seed, lr, lote e lr "
                   "constantes. A única variável é tying. Diferença positiva = "
                   "sem tying é MELHOR; negativa = tying é melhor.")
    # intervalo honesto: usa os desvios dos dois braços
    import math
    se = math.sqrt(t["final_val_desvio"]**2/3 + st["final_val_desvio"]**2/3)
    res["erro_padrao_da_diferenca"] = round(float(se), 4)
    res["diferenca_em_erros_padrao"] = round(float(dif / se), 2) if se > 0 else None
    (RAIZ / "nucleo" / "experimentos" / "resultado_sobreajuste.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print()
    print(f"{'':<12} {'params':>10} {'val final':>20} {'piora ao parar':>16}")
    print("-" * 64)
    for k, rot in (("tying", "tying"), ("sem_tying", "sem tying")):
        r = res[k]
        print(f"{rot:<12} {r['n_parametros']:>10,} "
              f"{r['final_val_media']:>13.4f} ±{r['final_val_desvio']:.4f} "
              f"{r['perda_ao_parar_media']:>+14.4f}")
    print()
    print(f"diferença (sem tying - tying): {res['diferenca_sem_tying_menos_tying']:+.4f} nats "
          f"= {res['diferenca_em_erros_padrao']:+.2f} erros-padrão")
    print(f"passos do mínimo: tying {res['tying']['passo_do_minimo']} | "
          f"sem tying {res['sem_tying']['passo_do_minimo']}")


if __name__ == "__main__":
    main()
