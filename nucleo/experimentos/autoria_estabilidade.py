"""Estabilidade e teste nulo para a atribuição por entropia cruzada.

Duas perguntas que um número sozinho não responde:

  1. ESTABILIDADE — se eu cortar o texto em questão ao meio, as duas metades
     apontam para o mesmo livro?  Uma atribuição que muda conforme o corte
     não é uma atribuição.

  2. NULO — qual é a margem que se obtém por acaso?  Gero blocos sintéticos
     misturando os dois livros em proporções aleatórias e vejo que margens
     produzem.  Se a margem observada cai dentro dessa distribuição, não há
     sinal.

Uso:  python -m nucleo.experimentos.autoria_estabilidade
Saída: nucleo/experimentos/resultado_autoria_estabilidade.json
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from nucleo.experimentos.autoria_entropia import (
    AUTOR,
    OBRAS,
    entropia,
    modelo_de,
    normalizar,
)

SEMENTE = 20261101


def margem(bpc: dict, a: str, b: str) -> float:
    """Segunda melhor menos a melhor (nas duas obras)."""
    ordenado = sorted((bpc[a], bpc[b]))
    return round(ordenado[1] - ordenado[0], 4)


def main() -> None:
    ae = "".join((OBRAS / f"apenas_existo_parte{i}.txt").read_text("utf-8") for i in (1, 2, 3))
    h99 = "".join((OBRAS / f"H99_Parte_{p}.txt").read_text("utf-8") for p in ("I", "II", "III", "IV"))

    todos = normalizar(ae) + normalizar(h99)
    vocab = len(set(todos))

    m_ae = modelo_de(ae, vocab)
    m_h99 = modelo_de(h99, vocab)

    def decide(texto: str) -> dict:
        b = {"AE": entropia(m_ae, texto), "H99": entropia(m_h99, texto)}
        vencedor = min(b, key=b.get)
        return {
            "bits": {k: round(v, 4) for k, v in b.items()},
            "vencedor": vencedor,
            "margem": margem(b, "AE", "H99"),
        }

    # ---- 1. estabilidade: metades, terços e quartos -------------------------
    estabilidade = {}
    for ficheiro in sorted(AUTOR.glob("*.txt")):
        texto = ficheiro.read_text("utf-8")
        cortes = {}
        for nome_corte, n_partes in (("metades", 2), ("terços", 3), ("quartos", 4)):
            tam = len(texto) // n_partes
            partes = [texto[i * tam : (i + 1) * tam] for i in range(n_partes)]
            decisoes = [decide(p) for p in partes if len(p) > 200]
            votos = [d["vencedor"] for d in decisoes]
            cortes[nome_corte] = {
                "votos": votos,
                "consistente": len(set(votos)) == 1,
                "margens": [d["margem"] for d in decisoes],
            }
        inteiro = decide(texto)
        estabilidade[ficheiro.stem] = {"inteiro": inteiro, "cortes": cortes}

    # ---- 2. referência: margens de blocos REAIS deixados de fora -----------
    # Um texto desconhecido sem sinal deve produzir margens parecidas com as de
    # um bloco genuíno.  Baralhar palavras seria um nulo errado: destrói a
    # sintaxe local e inflaciona artificialmente a margem.
    rng = random.Random(SEMENTE)
    tam = int(statistics.fmean([len(p.read_text("utf-8")) for p in AUTOR.glob("*.txt")]) or 4000)

    def margens_de_blocos(texto: str, etiqueta: str, n_blocos: int = 8) -> list[float]:
        saida = []
        restante = texto
        for _ in range(n_blocos):
            if len(restante) < tam * 2:
                break
            ini = rng.randrange(0, len(restante) - tam)
            bloco = restante[ini : ini + tam]
            restante = restante[:ini] + restante[ini + tam :]
            m_self = modelo_de(restante, vocab)
            b = {
                etiqueta: entropia(m_self, bloco),
                "outro": entropia(m_h99 if etiqueta == "AE" else m_ae, bloco),
            }
            saida.append(round(abs(b[etiqueta] - b["outro"]), 4))
        return sorted(saida)

    referencia = {
        "AE_blocos_reais": margens_de_blocos(ae, "AE"),
        "H99_blocos_reais": margens_de_blocos(h99, "H99"),
    }

    saida = {
        "estabilidade": estabilidade,
        "referencia_blocos_reais": {
            "tamanho_bloco": tam,
            **referencia,
            "nota": (
                "Margens que produz um bloco que SABEMOS ser do mesmo autor. "
                "Um texto em questão com margem dentro destes intervalos é "
                "indistinguível de material autêntico; muito abaixo, não tem sinal."
            ),
        },
    }

    destino = Path(__file__).with_name("resultado_autoria_estabilidade.json")
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== ESTABILIDADE (o texto aponta sempre para o mesmo livro?) ===")
    for nome, e in estabilidade.items():
        i = e["inteiro"]
        print(f"\n  {nome}")
        print(f"    inteiro:  {i['vencedor']:5s} margem {i['margem']:.4f}   {i['bits']}")
        for corte, c in e["cortes"].items():
            marca = "OK " if c["consistente"] else "MUDA"
            print(f"    {corte:10s} {marca} votos={c['votos']} margens={c['margens']}")

    print("\n=== REFERÊNCIA: margens de blocos REAIS (mesmo autor, sabido) ===")
    for k, v in saida["referencia_blocos_reais"].items():
        if k.startswith("tamanho") or k == "nota":
            continue
        print(f"  {k:18s} {v}")
    print(f"\nJSON: {destino}")


if __name__ == "__main__":
    main()
