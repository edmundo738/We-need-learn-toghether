"""Atribuição de autoria por medição — nada aqui é inventado.

Método:
  1. Burrows' Delta sobre as 150 palavras mais frequentes (que em português são
     quase só palavras-funcionais), com z-scores por característica.
  2. Classificação leave-one-out: cada documento é atribuído ao vizinho mais
     próximo entre os restantes.
  3. Sobreposição real de n-gramas de caracteres (n=6, n=10) entre os textos
     questionados e cada obra — o teste que a resposta anterior fingiu ter feito.
  4. Métricas de superfície: comprimento de frase, densidade de pontuação,
     type-token ratio, hapax, densidade de 1.ª pessoa.

Uso:  python -m nucleo.experimentos.autoria_literaria
Saída: nucleo/experimentos/resultado_autoria.json
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
OBRAS = RAIZ / "dados" / "obras"
AUTOR = OBRAS / "autor"
PUBLICO = RAIZ / "dados" / "publico_limpo"

TOKEN = re.compile(r"[a-záàâãéêíóôõúüç]+")


def tokens(texto: str) -> list[str]:
    return TOKEN.findall(texto.lower())


def carregar_fontes() -> dict[str, str]:
    fontes: dict[str, str] = {}

    # --- obras do autor em teste -------------------------------------------
    ae = "".join(
        (OBRAS / f"apenas_existo_parte{i}.txt").read_text(encoding="utf-8")
        for i in (1, 2, 3)
    )
    fontes["AE_APENAS_EXISTO"] = ae

    h99 = "".join(
        (OBRAS / f"H99_Parte_{p}.txt").read_text(encoding="utf-8")
        for p in ("I", "II", "III", "IV")
    )
    fontes["H99"] = h99

    fontes["LUNA_excerto"] = (OBRAS / "depths_luna_EXCERTO.txt").read_text(encoding="utf-8")

    # --- textos em questão --------------------------------------------------
    if AUTOR.exists():
        fontes["Q_bencao_de_dor"] = (AUTOR / "uma_bencao_de_dor.txt").read_text(encoding="utf-8")
        fontes["Q_edmundo_sem_ia"] = (AUTOR / "edmundo_sem_ia.txt").read_text(encoding="utf-8")
        fontes["Q_atraves_de_alguem"] = (AUTOR / "atraves_de_alguem_que_erra.txt").read_text(
            encoding="utf-8"
        )
        fontes["Q_aventuras_zulu"] = (AUTOR / "aventuras_aleatorias_da_zulu.txt").read_text(
            encoding="utf-8"
        )

    # --- controlo negativo: outro autor, mesma língua ------------------------
    for nome in ("dom_casmurro", "bras_cubas", "cartomante", "alienista"):
        p = PUBLICO / f"{nome}.txt"
        if p.exists():
            fontes[f"CTRL_{nome}"] = p.read_text(encoding="utf-8")

    return fontes


def perfil_delta(fontes: dict[str, str], n_palavras: int = 150) -> dict:
    contagens = {nome: tokens(t) for nome, t in fontes.items()}

    total: dict[str, int] = {}
    for toks in contagens.values():
        for t in toks:
            total[t] = total.get(t, 0) + 1

    vocab = [p for p, _ in sorted(total.items(), key=lambda kv: -kv[1])[:n_palavras]]

    freq_rel: dict[str, list[float]] = {}
    for nome, toks in contagens.items():
        n = len(toks)
        c = {p: 0 for p in vocab}
        for t in toks:
            if t in c:
                c[t] += 1
        freq_rel[nome] = [(c[p] / n) * 1000.0 if n else 0.0 for p in vocab]

    nomes = list(freq_rel)
    medias, desvios = [], []
    for j in range(len(vocab)):
        col = [freq_rel[n][j] for n in nomes]
        medias.append(statistics.fmean(col))
        sd = statistics.pstdev(col)
        desvios.append(sd if sd > 0 else 1e-9)

    z = {n: [(freq_rel[n][j] - medias[j]) / desvios[j] for j in range(len(vocab))] for n in nomes}

    delta = {}
    for a in nomes:
        delta[a] = {}
        for b in nomes:
            if a == b:
                continue
            delta[a][b] = round(
                statistics.fmean(abs(za - zb) for za, zb in zip(z[a], z[b])), 4
            )

    return {"vocabulario": vocab, "delta": delta, "freq_rel": freq_rel}


def leave_one_out(delta: dict[str, dict[str, float]]) -> dict[str, str]:
    """Vizinho mais próximo, excluindo o próprio documento."""
    return {a: min(distancias, key=distancias.get) for a, distancias in delta.items()}


def ngramas(texto: str, n: int) -> set[str]:
    limpo = re.sub(r"\s+", " ", texto.lower())
    return {limpo[i : i + n] for i in range(len(limpo) - n + 1)}


def sobreposicao(q: str, ref: str, ns=(6, 10)) -> dict:
    saida = {}
    for n in ns:
        gq, gr = ngramas(q, n), ngramas(ref, n)
        inter = gq & gr
        saida[f"n{n}"] = {
            "comuns": len(inter),
            "da_query": round(100.0 * len(inter) / max(1, len(gq)), 2),
            "da_ref": round(100.0 * len(inter) / max(1, len(gr)), 4),
        }
    return saida


def superficie(texto: str) -> dict:
    toks = tokens(texto)
    n = len(toks)
    frases = [f.strip() for f in re.split(r"[.!?…]+", texto) if f.strip()]
    comprimentos = [len(tokens(f)) for f in frases]
    comprimentos = [c for c in comprimentos if c > 0]
    mil = n / 1000.0 if n else 1.0

    tipos = set(toks)
    hapax = sum(1 for t in tipos if toks.count(t) == 1) if n else 0
    primeiros = tokens(texto)[:1000]

    def por_mil(padrao: str) -> float:
        return round(len(re.findall(padrao, texto, flags=re.I)) / mil, 2)

    return {
        "palavras": n,
        "palavras_unicas": len(tipos),
        "ttr_1000": round(len(set(primeiros)) / max(1, len(primeiros)), 4),
        "hapax_pct": round(100.0 * hapax / max(1, len(tipos)), 2),
        "frase_media_palavras": round(statistics.fmean(comprimentos), 2) if comprimentos else 0,
        "frase_desvio": round(statistics.pstdev(comprimentos), 2) if len(comprimentos) > 1 else 0,
        "frases_40mais_pct": round(
            100.0 * sum(1 for c in comprimentos if c >= 40) / max(1, len(comprimentos)), 2
        ),
        "travessao_por_mil": por_mil("—"),
        "reticencias_por_mil": por_mil(r"\.\.\."),
        "exclamacao_por_mil": por_mil("!"),
        "interrogacao_por_mil": por_mil(r"\?"),
        "eu_por_mil": por_mil(r"\beu\b"),
        "kkk": por_mil("kkk"),
        "ingles_por_mil": por_mil(
            r"\b(the|you|your|and|is|are|of|this|that|with|what|for|not|but|it|to|in)\b"
        ),
    }


def main() -> None:
    fontes = carregar_fontes()
    if not fontes:
        raise SystemExit("sem fontes")

    perfil = perfil_delta(fontes)
    delta = perfil["delta"]
    vizinho = leave_one_out(delta)

    obras = [n for n in fontes if n.startswith(("AE_", "H99", "LUNA"))]
    controlos = [n for n in fontes if n.startswith("CTRL_")]
    querys = [n for n in fontes if n.startswith("Q_")]

    resumo_delta = {}
    for q in querys:
        resumo_delta[q] = {
            "obra_mais_proxima": min(obras, key=lambda o: delta[q][o]),
            "dist_a_cada_obra": {o: delta[q][o] for o in obras},
            "controlo_mais_proximo": min(controlos, key=lambda c: delta[q][c]) if controlos else None,
            "dist_min_controlo": min((delta[q][c] for c in controlos), default=None),
        }

    # distância entre as próprias obras e entre os controlos, como escala
    intra_obras = [
        delta[a][b] for i, a in enumerate(obras) for b in obras[i + 1 :]
    ]
    intra_ctrl = [
        delta[a][b] for i, a in enumerate(controlos) for b in controlos[i + 1 :]
    ]

    referencias = {n: fontes[n] for n in obras + controlos}
    sobreposicoes = {}
    for q in querys:
        sobreposicoes[q] = {
            ref: sobreposicao(fontes[q], referencias[ref]) for ref in referencias
        }

    saida = {
        "nota": (
            "Delta = média das diferenças absolutas de z-score sobre as 150 palavras "
            "mais frequentes. Quanto mais baixo, mais parecido."
        ),
        "escala": {
            "delta_medio_entre_obras_do_mesmo_autor": round(statistics.fmean(intra_obras), 4),
            "delta_medio_entre_obras_de_Machado": round(statistics.fmean(intra_ctrl), 4)
            if intra_ctrl
            else None,
            "delta_medio_obras_vs_controlo": round(
                statistics.fmean(delta[a][c] for a in obras for c in controlos), 4
            )
            if controlos
            else None,
        },
        "vizinho_mais_proximo": vizinho,
        "delta_completo": delta,
        "querys": resumo_delta,
        "sobreposicao_ngramas": sobreposicoes,
        "superficie": {n: superficie(t) for n, t in fontes.items()},
        "vocabulario_delta": perfil["vocabulario"][:40],
    }

    destino = Path(__file__).with_name("resultado_autoria.json")
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== ESCALA (Delta médio) ===")
    for k, v in saida["escala"].items():
        print(f"  {k:48s} {v}")
    print("\n=== DISTÂNCIA DE CADA TEXTO ÀS OBRAS ===")
    for q in querys:
        d = resumo_delta[q]
        print(f"\n  {q}  -> mais próximo: {d['obra_mais_proxima']}")
        for o in obras:
            print(f"      {o:18s} {d['dist_a_cada_obra'][o]}")
    print("\n=== VIZINHO MAIS PRÓXIMO (leave-one-out) ===")
    for n, v in vizinho.items():
        print(f"  {n:24s} -> {v}")
    print(f"\nJSON: {destino}")


if __name__ == "__main__":
    main()
