"""Corre a suite sem pytest — mas SEM divergir da suite do pytest.

Porque este ficheiro foi reescrito
----------------------------------
Esta versão anterior era um mini-runner que importava cada `test_*.py` e
chamava todas as funções `test_*` sem argumentos. Funcionava enquanto os testes
não tinham fixtures nem parametrização. Quando passei a ter as duas coisas,
este runner ficou a reportar 11 falhas que **não existiam** — chamava funções
que precisavam de `servidor` ou `perfil` e recebia `TypeError`.

O perigo não era o erro: era ter duas suites. Um dia uma dizia verde e a outra
vermelho, e alguém escolhia a que gostava. A correcção é não ter duas.

Agora este script só delega no pytest. Se o pytest não estiver instalado, diz
exactamente o que fazer em vez de correr uma versão degradada.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def main() -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest não está instalado. A suite completa precisa dele:")
        print("    pip install pytest hypothesis")
        print()
        print("Não existe um modo degradado de propósito: uma segunda forma de")
        print("correr os testes acabaria a divergir desta, e foi o que aconteceu.")
        return 2

    return subprocess.call([sys.executable, "-m", "pytest", str(AQUI), "-q", "-p", "no:warnings"])


if __name__ == "__main__":
    raise SystemExit(main())
