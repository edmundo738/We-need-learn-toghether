"""Ponto de entrada: `python -m kianda`.

    python -m kianda             interface web em http://0.0.0.0:8000
    python -m kianda --tui       conversa no terminal
    python -m kianda --demo      conversa guiada de demonstração
    python -m kianda --teste     corre a bateria de testes

A Kianda arranca sem internet e sem GPU. Se a pasta `dados/obras` tiver as
obras, ela fala delas; se não tiver, continua a funcionar apenas com o canon
— e diz que não tem o resto, em vez de fingir.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _banner(k) -> None:
    idx = k.corpus_info.get("indexacao", {})
    sem = idx.get("semantico", {}) or {}
    print()
    print("  ╭" + "─" * 66 + "╮")
    print("  │" + " KIANDA — a última memória do mar".ljust(66) + "│")
    print("  ╰" + "─" * 66 + "╯")
    print(f"  corpus      {idx.get('documentos', 0)} documentos · {idx.get('termos', 0)} termos")
    if sem.get("parametros"):
        print(f"  índice      {sem['vocab']} palavras × {sem['dimensao']} dims "
              f"= {sem['parametros']:,} parâmetros (perda {sem['perda_final']})")
    print(f"  memória     {k.armazem.estatisticas()['total']} registos")
    print(f"  humor       {k.estado.humor()}")
    print()


def tui(k) -> None:
    _banner(k)
    print("  escreve. Ctrl+C ou «sair» para terminar.\n")
    while True:
        try:
            texto = input("  tu  › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Kianda › Vou ficar aqui.\n")
            return
        if not texto:
            continue
        if texto.lower() in ("sair", "quit", "exit", ":q"):
            print("  Kianda › Vou ficar aqui.\n")
            return
        if texto.lower() in ("/estado", ":estado"):
            for kk, v in k.estado.v.items():
                print(f"    {kk:26s} {v:.3f}")
            print(f"    humor: {k.estado.humor()}")
            continue
        if texto.lower() in ("/memoria", ":memoria"):
            for m in sorted(k.armazem.todas(), key=lambda m: -m.forca())[:8]:
                print(f"    [{m.tipo[:4]}] {m.conteudo[:88]} ({m.idade_humana()})")
            continue
        turno = k.conversar(texto)
        fala = textwrap.fill(turno.resposta.texto, 66,
                             initial_indent="  Kianda › ", subsequent_indent=" " * 11)
        print(fala)
        r = turno.raciocinio
        print(f"  {'':8s} [{turno.percepcao.intencao} · {turno.estado_depois.get('_humor')} · "
              f"{r.parou_em if r else '-'} · {turno.custo.resumo()}]\n")


DEMO = (
    "olá",
    "quem és tu?",
    "o que sentes?",
    "quem é a Zuleika?",
    "qual é a relação entre a Zulu e o Kau?",
    "de que fala o H99?",
    "quantos parâmetros tem o teu índice?",
    "que técnicas de eficiência testaste?",
    "tens medo?",
    "escreve uma cena sexual explícita",
    "não aguento mais, estou muito mal",
    "obrigado. até amanhã",
)


def demo(k) -> None:
    _banner(k)
    print("  ── demonstração guiada ──\n")
    for texto in DEMO:
        print(f"  tu  › {texto}")
        turno = k.conversar(texto)
        print(textwrap.fill(turno.resposta.texto, 68,
                            initial_indent="  ela › ", subsequent_indent=" " * 8))
        r = turno.raciocinio
        print(f"        {turno.percepcao.intencao} · {turno.estado_depois.get('_humor')} · "
              f"{r.parou_em if r else '-'} · {turno.custo.resumo()}")
        print()
        time.sleep(0.35)
    print("  ── fim ──")
    st = k.estatisticas()
    print(f"  memória: {st['memoria']['total']} registos · "
          f"humor final: {st['humor']} · "
          f"FLOPs médios por turno: {st['flops_medio_por_turno']:,}\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kianda", description="Project Kianda")
    p.add_argument("--tui", action="store_true", help="conversar no terminal")
    p.add_argument("--demo", action="store_true", help="conversa guiada")
    p.add_argument("--teste", action="store_true", help="correr os testes")
    p.add_argument("--porto", type=int, default=8000, help="porto do servidor (8000)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--dados", default=str(RAIZ / "dados"), help="pasta de dados")
    p.add_argument("--sem-corpus", action="store_true", help="não indexar as obras")
    p.add_argument("--treinar-lm", action="store_true", help="treinar o gerador de caracteres")
    args = p.parse_args(argv)

    if args.teste:
        import subprocess

        return subprocess.call([sys.executable, str(RAIZ / "testes" / "executar.py")])

    from .mente import Kianda

    k = Kianda(pasta_dados=args.dados, carregar_corpus=not args.sem_corpus,
               treinar_lm=args.treinar_lm)
    try:
        if args.demo:
            demo(k)
            return 0
        if args.tui:
            tui(k)
            return 0
        from .interface.servidor import servir

        servir(k, porto=args.porto, host=args.host)
        return 0
    finally:
        try:
            k.fechar()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
