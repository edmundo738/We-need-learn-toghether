"""Experimento 1 — a hipótese central, medida.

    "Uma arquitectura modular, adaptativa, com memória persistente,
     computação condicional, modelos especializados e mecanismos de
     aprendizagem pode atingir determinadas capacidades cognitivas com
     menor custo computacional do que uma arquitectura que depende
     exclusivamente do aumento de escala de um único modelo."

Corre: python experimentos/01_benchmark.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kianda.benchmark.banco import Banco, CONJUNTO
from kianda.mente import Kianda


def main():
    print("=" * 74)
    print("EXPERIMENTO 1 — custo x capacidade".center(74))
    print("=" * 74)

    pasta = Path(tempfile.mkdtemp(prefix="kianda-bench-"))
    print(f"\n[1/5] a construir a Kianda em {pasta}")
    k = Kianda(pasta_dados=pasta)
    idx = k.corpus_info.get("indexacao", {})
    print(f"      corpus: {idx.get('documentos', 0)} documentos, "
          f"{idx.get('termos', 0)} termos")
    sem = idx.get("semantico", {})
    if sem and "parametros" in sem:
        print(f"      indice semantico: {sem['vocab']} palavras x {sem['dimensao']} dims "
              f"= {sem['parametros']:,} parametros, perda {sem['perda_inicial']} -> {sem['perda_final']}")

    banco = Banco(pasta / "bench")

    print(f"\n[2/5] a medir a Kianda em {len(CONJUNTO)} perguntas")
    mk = banco.medir_kianda(k)
    print(f"      exactidao {mk.exactidao:.1%}  .  {mk.flops_mflop:.3f} MFLOP  .  "
          f"{mk.tempo_ms/1000:.2f} s")

    print(f"\n[3/5] a treinar a linha de base monolitica do zero")
    mb = banco.medir_baseline(k)
    info = banco._info_baseline
    print(f"      {info['parametros']:,} parametros  .  perda final {info['perda_final']:.4f}")
    print(f"      exactidao {mb.exactidao:.1%}  .  {mb.flops_mflop:.3f} MFLOP")

    print(f"\n[4/5] abencoes: desligar uma peca de cada vez")
    abl = banco.ablacao(k)
    for nome, d in abl.items():
        print(f"      {nome:20s} exactidao {d['exactidao']:>6.1%}  "
              f"{d['flops_mflop']:>9.3f} MFLOP")

    print(f"\n[5/6] generalizacao: parafrases que a linha de base nunca viu")
    gen = banco.medir_generalizacao(k)
    print(f"      kianda       exactidao {gen['kianda']['exactidao']:>6.1%}  "
          f"{gen['kianda']['flops_mflop']:>9.3f} MFLOP")
    print(f"      linha base   exactidao {gen['baseline']['exactidao']:>6.1%}  "
          f"{gen['baseline']['flops_mflop']:>9.3f} MFLOP")
    print("      (a diferenca aqui e o efeito medido da ligacao de entidades)")

    print(f"\n[6/6] eficiencia do nucleo")
    ef = banco.medir_eficiencia_nucleo()
    for nome, d in ef.items():
        print(f"      {nome:16s} {json.dumps(d, ensure_ascii=False)}")

    print()
    print(banco.relatorio())

    print("\nDETALHE POR PERGUNTA (kianda)")
    print("-" * 74)
    for d in mk.detalhes:
        marca = "OK " if d["ok"] else "ERRO"
        print(f" {marca} {d['pergunta'][:44]:44s} {d['parou_em']:10s} {d['regime']:8s} "
              f"{d['humor']}")
        if not d["ok"]:
            print(f"      -> {d['resposta'][:110]}")

    alvo = banco.guardar(extra={"ablacoes": abl, "nucleo": ef,
                                "generalizacao": {kk: vv for kk, vv in gen.items() if kk != "_base"}})
    print(f"\nresultados guardados em {alvo}")
    k.fechar()


if __name__ == "__main__":
    main()
