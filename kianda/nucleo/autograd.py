"""Compatibilidade: `kianda.nucleo.autograd` -> `nucleo.autograd`.

Porque isto é um reexport e não uma cópia
-----------------------------------------
A implementação vive agora em `nucleo/autograd.py`, fora de `kianda/`, para que
o núcleo seja estruturalmente independente do sistema antigo. Mas 11 ficheiros
ainda importam por este caminho (registado em
`nucleo/registos/imports_antes.json`), e a regra do Movimento 1 é migrar **sem
quebrar a API experimental**.

Este módulo existe para que os dois caminhos devolvam **o mesmo objecto**:

    from kianda.nucleo.autograd import Tensor as A
    from nucleo.autograd        import Tensor as B
    assert A is B          # não é igual: é o mesmo

Um reexport com `from ... import *` cumpre a identidade; uma cópia não. A
diferença importa porque duas cópias da classe `Tensor` fazem `isinstance`
falhar de formas que se diagnosticam mal — o erro aparece longe da causa.

Quando remover
--------------
Quando `python -m nucleo.registos.imports --comparar antes` mostrar zero
importadores restantes. Enquanto houver um, este ficheiro fica.
"""

from nucleo.autograd import *          # noqa: F403
from nucleo.autograd import (          # nomes explícitos: ajudam o IDE e o leitor
    Custo,
    Tensor,
    entropia_cruzada,
    matmul,
    matmul_int8,
    quantizar_int8,
)
