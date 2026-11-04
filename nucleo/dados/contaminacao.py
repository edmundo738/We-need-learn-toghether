"""Verificação de contaminação entre hold-out e corpus de treino.

Porque isto existe
------------------
O relatório da Fase 2 afirmou que o texto de validação foi escrito de raiz e
nunca visto no treino. Isso é **plausível**, não é **provado** — e uma medição
de generalização sobre dados contaminados mede memorização disfarçada de
aprendizagem. É a diferença entre os dois tipos de erro que o projecto inteiro
tenta não cometer.

O que se mede
-------------
Sobreposição de **n-gramas de palavras** entre o hold-out e cada ficheiro do
corpus, para n = {5, 8, 13, 20}.

Porque estes n
--------------
- n=5: apanha frases-feitas curtas («a acção de uma nação»). Abaixo disto, o
  português tem tanta colisão natural que o número deixa de significar algo.
- n=8: apanha citações médias.
- n=13 e n=20: apanham parágrafos copiados — a forma grave.

Um n-grama que apareça **uma vez** em 3 MB não é contaminação, é coincidência
gramatical: «de que» e «não é» repetem-se naturalmente. O relatório distingue
esses casos em vez de somar tudo num número alarmante.

Uso:
    python -m nucleo.dados.contaminacao
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Normalização mínima: só o que não muda o conteúdo. Não se tiram acentos nem
# se passa a minúsculas no *texto*, mas a comparação de n-gramas é feita em
# minúsculas — «A Nação» e «a nação» no início de frase não são contaminação.
_PALAVRA = re.compile(r"[^\W\d_]+", re.UNICODE)


def palavras(texto: str) -> list[str]:
    return [p.lower() for p in _PALAVRA.findall(texto)]


def ngramas(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def analisar(texto_hold: str, corpus: dict[str, str],
             ns: tuple[int, ...] = (5, 8, 13, 20)) -> dict:
    h_tokens = palavras(texto_hold)
    resultado: dict = {
        "holdout_palavras": len(h_tokens),
        "ns": list(ns),
        "por_n": {},
        "ficheiros": {},
        "veredicto": "",
    }

    # Contagem por ficheiro, para localizar a fonte se houver contaminação.
    corpus_tokens = {nome: palavras(texto) for nome, texto in corpus.items()}
    total_palavras = sum(len(t) for t in corpus_tokens.values())
    resultado["corpus_palavras"] = total_palavras

    for n in ns:
        h_ng = ngramas(h_tokens, n)
        if not h_ng:
            continue
        encontrados: dict[str, list[str]] = {}
        for nome, tokens in corpus_tokens.items():
            c_ng = ngramas(tokens, n)
            inter = h_ng & c_ng
            if inter:
                encontrados[nome] = [" ".join(g) for g in sorted(inter)][:10]
        total = sum(len(v) for v in encontrados.values())
        resultado["por_n"][n] = {
            "ngramas_no_holdout": len(h_ng),
            "ngramas_encontrados": total,
            "fracao": round(total / len(h_ng), 4),
        }
        if encontrados:
            resultado["ficheiros"].setdefault(n, encontrados)

    # Veredicto. Limiares definidos ANTES de ver os números, e conservadores:
    # acima de 1% de sobreposição em n=13 há parágrafos copiados.
    frac13 = resultado["por_n"].get(13, {}).get("fracao", 0.0)
    frac20 = resultado["por_n"].get(20, {}).get("fracao", 0.0)
    frac5 = resultado["por_n"].get(5, {}).get("fracao", 0.0)

    if frac13 > 0.01 or frac20 > 0.01:
        resultado["veredicto"] = "CONTAMINADO — há parágrafos do hold-out no corpus"
    elif frac8 := resultado["por_n"].get(8, {}).get("fracao", 0.0) > 0.02:
        resultado["veredicto"] = "SUSPEITO — há n-gramas de 8 repetidos"
    elif frac5 > 0.10:
        resultado["veredicto"] = (
            "COLISÃO NATURAL — frases curtas comuns em português; "
            "não é contaminação, mas recomenda-se um hold-out maior"
        )
    else:
        resultado["veredicto"] = "LIMPO — sem sobreposição relevante"
    return resultado


def main() -> int:
    manifesto = json.loads((RAIZ / "dados" / "manifest.json").read_text())
    hold_out_path = RAIZ / "nucleo" / "dados" / "validacao_tokenizador.txt"
    bruto = hold_out_path.read_text(encoding="utf-8")
    hold = "\n".join(l for l in bruto.splitlines()
                     if l.strip() and not l.lstrip().startswith("#"))

    corpus = {}
    for entrada in manifesto["ficheiros"]:
        caminho = RAIZ / entrada["caminho"]
        if caminho.exists():
            corpus[entrada["caminho"].replace("dados/", "")] = \
                caminho.read_text(encoding="utf-8", errors="replace")

    r = analisar(hold, corpus)

    print("=" * 70)
    print("VERIFICAÇÃO DE CONTAMINAÇÃO — hold-out vs corpus de treino")
    print("=" * 70)
    print(f"hold-out : {r['holdout_palavras']:>8,} palavras")
    print(f"corpus   : {r['corpus_palavras']:>8,} palavras em {len(corpus)} ficheiros")
    print()
    print(f"{'n':>3} {'n-gramas':>10} {'encontrados':>12} {'fracção':>9}")
    print("-" * 38)
    for n in r["ns"]:
        d = r["por_n"].get(n, {})
        if not d:
            continue
        print(f"{n:>3} {d['ngramas_no_holdout']:>10,} "
              f"{d['ngramas_encontrados']:>12,} {100*d['fracao']:>8.2f}%")

    if r["ficheiros"]:
        print("\nOnde aparecem (primeiros 5 por n):")
        for n, achados in r["ficheiros"].items():
            print(f"  n={n}:")
            for nome, lista in achados.items():
                for g in lista[:5]:
                    print(f"    · {nome}: «{g}»")

    print("\n" + "=" * 70)
    print(f"VEREDICTO: {r['veredicto']}")
    print("=" * 70)

    saida = RAIZ / "nucleo" / "dados" / "resultado_contaminacao.json"
    saida.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\nguardado em {saida.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
