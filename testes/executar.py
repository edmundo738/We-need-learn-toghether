"""Corre todos os testes sem pytest (só stdlib)."""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def carregar(caminho: Path):
    spec = importlib.util.spec_from_file_location(caminho.stem, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    total = falhas = 0
    for f in sorted(AQUI.glob("test_*.py")):
        print(f"\n{f.name}")
        try:
            mod = carregar(f)
        except Exception:
            print(f"  ERRO ao importar: {traceback.format_exc().splitlines()[-1]}")
            falhas += 1
            continue
        for nome in sorted(dir(mod)):
            if not nome.startswith("test_"):
                continue
            fn = getattr(mod, nome)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                print(f"  ok    {nome}")
            except AssertionError as e:
                falhas += 1
                print(f"  FALHA {nome}: {e}")
            except Exception as e:
                falhas += 1
                print(f"  ERRO  {nome}: {type(e).__name__}: {e}")
    print(f"\n{'='*60}\n{total - falhas}/{total} verdes"
          f"{'  — tudo em ordem' if not falhas else f'  — {falhas} falhas'}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
