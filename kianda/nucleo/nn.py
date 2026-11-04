"""Compatibilidade: `kianda.nucleo.nn` -> `nucleo.nn`.

Ver a nota em `kianda/nucleo/autograd.py` — a razão é a mesma, e a exigência de
identidade de objecto também: `kianda.nucleo.nn.Adam is nucleo.nn.Adam`.

Este caminho tem um consumidor que merece atenção: `benchmark/banco.py:522` usa
`__import__("kianda.nucleo.nn", fromlist=["LoRA"])` — um import por string, que
uma análise de `ast` sobre `import`/`from` **não** apanha. Foi encontrado por
`grep`, não pelo registo automático, e é a razão de o registo não substituir
uma busca textual.
"""

from nucleo.nn import *                # noqa: F403
from nucleo.nn import (                # noqa: F401
    Adam,
    Linear,
    LoRA,
    MLP,
    Modulo,
    calibrar_escala,
    destilar,
    inicializar_he,
    inicializar_xavier,
    podar_estruturado,
)
