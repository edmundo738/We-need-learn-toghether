"""Aplicar a medição de memorização a uma geração guardada.

Uso: python -m nucleo.experimentos.memorizacao_gpu [ficheiro_resultado]

Lê `resultado_corpus_real.json`, mede os n-gramas a 5/10/15/20 e a maior
cópia exacta, e grava `resultado_memorizacao.json`. Reproduz a tabela da
auditoria.
"""

from __future__ import annotations

import json
from pathlib import Path

from nucleo.avaliacao.memorizacao import (
    distribuicao,
    maior_copia_exata,
    sobreposicao_aproximada,
)
from nucleo.tokenizador.bpe import TokenizadorBPE

RAIZ = Path(__file__).resolve().parents[2]


def main() -> None:
    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    ids = json.loads((RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text())
    r = json.loads((RAIZ / "nucleo" / "experimentos" / "resultado_corpus_real.json").read_text())

    saida = {"amostras": [], "agregado": {}}
    totais: dict[int, list[int]] = {}
    for a in r["amostras"]:
        ger = tok.codificar(a["gerado"])
        linha = {"prompt": a["prompt"], "tokens_gerados": len(ger)}
        for res in distribuicao(ger, ids, ordens=(5, 10, 15, 20)):
            t = totais.setdefault(res.ordem, [0, 0])
            t[0] += res.ngramas_gerados
            t[1] += res.ngramas_no_corpus
        n, trecho = maior_copia_exata(ger, ids)
        linha["maior_copia_exata"] = n
        linha["trecho_copiado"] = tok.descodificar(trecho) if trecho else ""
        linha["sobreposicao_aproximada_10"] = sobreposicao_aproximada(ger, ids, 10, 2)
        saida["amostras"].append(linha)

    for o in sorted(totais):
        g, n = totais[o]
        saida["agregado"][f"ordem_{o}"] = {
            "gerados": g, "no_corpus": n,
            "fraccao": round(n / g, 4) if g else 0.0}

    destino = RAIZ / "nucleo" / "experimentos" / "resultado_memorizacao.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False))

    print(f"{'ordem':>6} {'gerados':>8} {'no corpus':>10} {'fracção':>9}")
    for o, (g, n) in sorted(totais.items()):
        print(f"{o:>6} {g:>8} {n:>10} {n/g*100 if g else 0:>8.2f}%")
    maior = max(a["maior_copia_exata"] for a in saida["amostras"])
    print(f"\nmaior cópia exacta: {maior} tokens")
    print(f"gravado em {destino}")


if __name__ == "__main__":
    main()
