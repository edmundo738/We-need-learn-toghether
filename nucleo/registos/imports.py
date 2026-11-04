"""Registo de imports — o estado que não pode ser perdido numa migração.

Porque este ficheiro existe
---------------------------
A condição do Edmundo para o Movimento 1: *«Antes de qualquer alteração,
registar todos os imports existentes.»*

Um registo que vive num relatório morre na sessão seguinte. Este vive num
artefacto versionado e é **gerado por um script**, não escrito à mão — por isso
pode ser reproduzido em qualquer commit e comparado.

O que regista
-------------
Para cada ficheiro `.py` do repositório, a lista exacta de módulos internos que
importa. Com isso consigo responder a três perguntas em qualquer momento:

1. Quem quebra se eu mover `X`? (importadores de `X`)
2. O núcleo novo é independente? (importadores de `kianda` a partir de `nucleo`)
3. A migração ficou completa? (comparar antes/depois)

O que NÃO faz
-------------
Não uso `ast` sobre o texto: imports condicionais dentro de funções escapavam a
uma análise de topo. Percorro o `ast` completo e guardo a linha de cada um, para
que o registo sirva para localizar o sítio e não só para contar.

Uso:
    python -m nucleo.registos.imports --guardar antes
    python -m nucleo.registos.imports --comparar antes
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
PASTA = Path(__file__).resolve().parent

# Directórios que não são código do projecto.
IGNORAR = {
    ".git", "__pycache__", ".pytest_cache", "kianda.egg-info",
    ".venv", "venv", "node_modules", "artefactos",
}


def _ficheiros() -> list[Path]:
    saida = []
    for p in RAIZ.rglob("*.py"):
        if any(parte in IGNORAR for parte in p.relative_to(RAIZ).parts):
            continue
        saida.append(p)
    return sorted(saida)


def _modulo_do_ficheiro(caminho: Path) -> str:
    """Caminho -> nome de módulo importável (`a/b/c.py` -> `a.b.c`)."""
    rel = caminho.relative_to(RAIZ).with_suffix("")
    partes = list(rel.parts)
    if partes and partes[-1] == "__init__":
        partes = partes[:-1]
    return ".".join(partes)


def _resolver(nome: str, nivel: int, origem: str) -> str:
    """Resolve um import (possivelmente relativo) para um módulo absoluto.

    `from ..nucleo.autograd import X` dentro de `kianda.mente` é relativo: o
    `nome` que o `ast` dá é «nucleo.autograd» e o `nivel` é 2. Sem resolver, o
    registo compararia maçãs com laranjas — e foi exactamente a confusão que
    tornou a colisão `kianda/nucleo` vs `nucleo` difícil de ver.
    """
    if nivel == 0:
        return nome
    partes = origem.split(".")
    # origem é um módulo; um pacote tem __init__. Simplificação aceite: subir
    # `nivel` níveis a partir do módulo que contém o ficheiro.
    base = partes[: max(len(partes) - nivel, 0)]
    if nome:
        base = base + nome.split(".")
    return ".".join(base)


def recolher() -> dict:
    ficheiros = _ficheiros()
    registo: dict[str, dict] = {}

    for caminho in ficheiros:
        rel = str(caminho.relative_to(RAIZ))
        origem = _modulo_do_ficheiro(caminho)
        try:
            arvore = ast.parse(caminho.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            registo[rel] = {"erro": f"SyntaxError: {e}"}
            continue

        imports = []
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                for a in no.names:
                    imports.append({"modulo": a.name, "nivel": 0,
                                    "linha": no.lineno, "tipo": "import"})
            elif isinstance(no, ast.ImportFrom):
                modulo = _resolver(no.module or "", no.level, origem)
                imports.append({
                    "modulo": modulo,
                    "nivel": no.level,
                    "linha": no.lineno,
                    "tipo": "from",
                    "nomes": [a.name for a in no.names],
                })

        registo[rel] = {
            "modulo": origem,
            "imports_internos": [i for i in imports if _e_interno(i["modulo"])],
            "imports_externos": sorted({
                i["modulo"].split(".")[0] for i in imports
                if not _e_interno(i["modulo"])
            }),
        }
    return registo


def _e_interno(modulo: str) -> bool:
    raiz = modulo.split(".")[0]
    return raiz in {"kianda", "nucleo", "dados", "testes", "experimentos"}


def _quem_importa(modulo_alvo: str, registo: dict) -> list[dict]:
    """Todos os sítios que importam `modulo_alvo` ou um sub-módulo dele."""
    saida = []
    for ficheiro, info in registo.items():
        for imp in info.get("imports_internos", []):
            m = imp["modulo"]
            if m == modulo_alvo or m.startswith(modulo_alvo + "."):
                saida.append({
                    "ficheiro": ficheiro,
                    "linha": imp["linha"],
                    "modulo": m,
                    "nomes": imp.get("nomes", []),
                })
    return saida


def resumo(registo: dict) -> dict:
    consumidores = {m: _quem_importa(m, registo)
                    for m in ("kianda.nucleo", "kianda", "nucleo")}
    # O núcleo novo viola a independência se importar kianda.
    violacoes = [
        {"ficheiro": f, "linha": i["linha"], "modulo": i["modulo"]}
        for f, info in registo.items()
        if f.startswith("nucleo/") or f.startswith("nucleo" + "\\")
        for i in info.get("imports_internos", [])
        if i["modulo"].split(".")[0] == "kianda"
    ]
    return {
        "total_ficheiros": len(registo),
        "importadores_de_kianda_nucleo": len(consumidores["kianda.nucleo"]),
        "consumidores_kianda_nucleo": consumidores["kianda.nucleo"],
        "violacoes_de_independencia": violacoes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guardar", metavar="NOME", help="grava o registo em registos/imports_<NOME>.json")
    ap.add_argument("--comparar", metavar="NOME", help="compara com um registo anterior")
    args = ap.parse_args(argv)

    registo = recolher()
    r = resumo(registo)
    documento = {"data": date.today().isoformat(), "resumo": r, "ficheiros": registo}

    print(f"ficheiros analisados                : {r['total_ficheiros']}")
    print(f"importadores de kianda.nucleo       : {r['importadores_de_kianda_nucleo']}")
    print(f"violações de independência do núcleo: {len(r['violacoes_de_independencia'])}")
    if r["violacoes_de_independencia"]:
        for v in r["violacoes_de_independencia"]:
            print(f"    · {v['ficheiro']}:{v['linha']} -> {v['modulo']}")

    if args.guardar:
        PASTA.mkdir(exist_ok=True)
        alvo = PASTA / f"imports_{args.guardar}.json"
        alvo.write_text(json.dumps(documento, indent=1, ensure_ascii=False))
        print(f"\nguardado: {alvo.relative_to(RAIZ)}")

    if args.comparar:
        anterior = json.loads((PASTA / f"imports_{args.comparar}.json").read_text())
        antes = {x["ficheiro"] for x in anterior["resumo"]["consumidores_kianda_nucleo"]}
        agora = {x["ficheiro"] for x in r["consumidores_kianda_nucleo"]}
        print(f"\ncomparação com «{args.comparar}»:")
        print(f"  importadores migrados : {len(antes - agora)}")
        print(f"  importadores restantes: {len(agora)}")
        if agora:
            for f in sorted(agora):
                print(f"    · {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
