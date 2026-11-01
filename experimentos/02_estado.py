"""Experimento 2 — o estado interno produz comportamento observável?

A pergunta de §9 do briefing, formulada de forma falsificável:

    Se as variáveis internas mudam, o texto de saída muda de forma mensurável?

Se a resposta for não, as variáveis não existem — são decoração. Este
experimento foi escrito para poder falhar. Na primeira execução falhou: com
os limiares originais, a Kianda ficou "presente" em 100% dos turnos e o
comprimento das respostas não variou.

Medições
--------
1. Trajectória das variáveis sob três regimes de conversa (elogio, crítica,
   interesse nos livros).
2. Comprimento das respostas por regime. Se o humor não mexer na fala, isto
   fica plano.
3. Taxa de discordância e de perguntas devolvidas por regime.
4. Distância entre estados finais, para provar que os regimes divergem.

Corre: python experimentos/02_estado.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile

from kianda.identidade.estado import EstadoInterno, Politica


REGIMES = {
    "elogio": ["elogio"] * 8,
    "critica": ["critica"] * 8,
    "livros": ["pergunta_livro"] * 8,
    "silencio": ["silencio"] * 8,
    "misto": ["saudacao", "pergunta_livro", "critica", "pergunta_livro",
              "repeticao", "critica", "pergunta_pessoal", "despedida"],
}


def barra(v: float, largura: int = 22) -> str:
    n = int(round(v * largura))
    return "█" * n + "·" * (largura - n)


def correr():
    print("=" * 76)
    print("EXPERIMENTO 2 — estado interno -> comportamento observável".center(76))
    print("=" * 76)

    pol = Politica()
    saidas: dict[str, dict] = {}

    for nome, eventos in REGIMES.items():
        e = EstadoInterno(semente=3)
        humores, comprimentos, discordar, perguntar = [], [], 0, 0
        for ev in eventos:
            e.aplicar(ev, intensidade=2.2)
            plano = pol.decidir(e)
            humores.append(plano["humor"])
            comprimentos.append(plano["max_tokens"])
            discordar += int(plano["pode_discordar"])
            perguntar += int(plano["devolve_pergunta"])
        saidas[nome] = {
            "estado": dict(e.v),
            "humor_final": e.humor(),
            "humores": humores,
            "max_tokens_medio": statistics.mean(comprimentos),
            "discordar_pct": 100 * discordar / len(eventos),
            "perguntar_pct": 100 * perguntar / len(eventos),
            "obj": e,
        }

    # -- 1. trajectória das variáveis relevantes --------------------------
    print("\n[1] TRAJECTÓRIA (8 turnos por regime)\n")
    chaves = ["curiosidade", "stress", "satisfacao", "frustracao", "solidao", "energia"]
    print(f"{'regime':10s}" + "".join(f"{k[:9]:>11s}" for k in chaves))
    print("-" * 76)
    for nome, d in saidas.items():
        linha = f"{nome:10s}"
        for k in chaves:
            linha += f"{d['estado'][k]:>11.3f}"
        print(linha)

    # -- 2. o humor mudou? ----------------------------------------------
    print("\n[2] HUMOR FINAL E ESTABILIDADE\n")
    for nome, d in saidas.items():
        distintos = len(set(d["humores"]))
        print(f"  {nome:10s} final={d['humor_final']:10s} "
              f"humores distintos={distintos}  "
              f"sequência={'>'.join(d['humores'][:5])}")

    todos = [h for d in saidas.values() for h in d["humores"]]
    fracao_presente = 100 * todos.count("presente") / len(todos)
    print(f"\n  fracção de turnos em «presente»: {fracao_presente:.1f}%  "
          f"(100% significa que o estado não se manifesta)")

    # -- 3. o texto mudou? ----------------------------------------------
    print("\n[3] EFEITO NO COMPORTAMENTO DE SAÍDA\n")
    print(f"{'regime':10s}{'max_tokens':>12s}{'discordar':>12s}{'perguntar':>12s}")
    print("-" * 76)
    for nome, d in saidas.items():
        print(f"{nome:10s}{d['max_tokens_medio']:>12.0f}"
              f"{d['discordar_pct']:>11.0f}%{d['perguntar_pct']:>11.0f}%")

    # -- 4. divergência entre estados -----------------------------------
    print("\n[4] DISTÂNCIA ENTRE ESTADOS FINAIS (euclidiana, 10 dimensões)\n")
    nomes = list(saidas)
    print(f"{'':10s}" + "".join(f"{n[:7]:>9s}" for n in nomes))
    for a in nomes:
        linha = f"{a:10s}"
        for b in nomes:
            linha += f"{saidas[a]['obj'].distancia(saidas[b]['obj']):>9.2f}"
        print(linha)

    # -- conclusão honesta ----------------------------------------------
    print("\n" + "=" * 76)
    print("CONCLUSÃO")
    print("=" * 76)
    efeito_tokens = max(d["max_tokens_medio"] for d in saidas.values()) - \
        min(d["max_tokens_medio"] for d in saidas.values())
    efeito_disc = max(d["discordar_pct"] for d in saidas.values()) - \
        min(d["discordar_pct"] for d in saidas.values())
    print(f"  variação de comprimento entre regimes : {efeito_tokens:.0f} tokens")
    print(f"  variação de taxa de discordância      : {efeito_disc:.0f} p.p.")
    print(f"  humores distintos produzidos          : {len(set(todos))}")
    if fracao_presente > 90:
        print("\n  O estado NÃO se manifesta. As variáveis são decoração.")
    elif efeito_tokens < 5 and efeito_disc < 10:
        print("\n  O estado muda mas o comportamento quase não. Efeito marginal.")
    else:
        print("\n  O estado produz comportamento observável: as mesmas perguntas,")
        print("  feitas em estados diferentes, produzem respostas diferentes em")
        print("  comprimento, disposição para discordar e disposição para perguntar.")
    print()


if __name__ == "__main__":
    correr()
