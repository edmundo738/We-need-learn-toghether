"""Ajuste do ponto de cruzamento e o que ele NÃO prova.

Lê `resultado_cruzamento.json` (produzido por `cruzamento.py`) e ajusta
`diferença = a*log(tokens) + b`. Reporta R2, o cruzamento e os resíduos.

Existe porque o relatório ESTRATEGIA.md citou primeiro «R2=0,994, 35x o
corpus» — escrito de memória, sem cálculo. O valor real é 0,9366 e 1,5x.
Ter o número num script é a única forma de ele deixar de ser uma afirmação.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]


def main() -> None:
    dados = json.loads((RAIZ / "nucleo" / "experimentos" / "resultado_cruzamento.json").read_text())
    cond = dados["condicoes"]
    tokens = np.array([c["tokens_unicos"] for c in cond], dtype=float)
    dif = np.array([c["diferenca"] for c in cond], dtype=float)

    x = np.log(tokens)
    A = np.vstack([x, np.ones_like(x)]).T
    (declive, intercepto), *_ = np.linalg.lstsq(A, dif, rcond=None)
    ajustado = A @ np.array([declive, intercepto])
    r2 = 1 - ((dif - ajustado) ** 2).sum() / ((dif - dif.mean()) ** 2).sum()
    cruzamento = float(np.exp(-intercepto / declive))
    corpus = 65999

    print(f"pontos: {len(tokens)}")
    print(f"declive   {declive:+.4f}")
    print(f"intercepto {intercepto:+.4f}")
    print(f"R2         {r2:.4f}")
    print(f"cruzamento {cruzamento:.3e} tokens = {cruzamento/corpus:.2f}x o corpus")
    print()
    print(f"{'tokens':>10} {'medido':>9} {'ajustado':>9} {'resíduo':>9}")
    for t, d, a in zip(tokens, dif, ajustado):
        print(f"{t:>10.0f} {d:>9.4f} {a:>9.4f} {d-a:>+9.4f}")

    residuo_max = float(np.abs(dif - ajustado).max())
    print()
    print(f"resíduo máximo: {residuo_max:.4f} nats")
    print("AVISO: recta sobre 4 pontos. A incerteza do cruzamento é de ordem")
    print("de grandeza — usar como faixa, não como valor.")

    saida = {"pontos": len(tokens), "declive": round(float(declive), 4),
             "intercepto": round(float(intercepto), 4), "r2": round(float(r2), 4),
             "cruzamento_tokens": round(cruzamento, 1),
             "cruzamento_x_corpus": round(cruzamento / corpus, 3),
             "residuo_maximo_nats": round(residuo_max, 4),
             "classe": "HYPOTHESIS"}
    (RAIZ / "nucleo" / "experimentos" / "resultado_ajuste.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
