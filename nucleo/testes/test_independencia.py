"""Testes de regressão da migração do núcleo — Movimento 1.

A condição do Edmundo:

    «Nenhuma funcionalidade será considerada concluída apenas porque o código
    foi escrito. Terá de existir um teste que demonstre que funciona.»

Este ficheiro existe para que a independência do núcleo seja uma **propriedade
verificada**, não uma intenção declarada. Sem ele, «o núcleo não depende do
sistema antigo» é uma frase que ninguém consegue contestar nem confirmar.

O que é testado
---------------
1. O núcleo importa sem que `kianda` exista no `sys.path`. Se um dia alguém
   acrescentar `from kianda...` a um módulo do núcleo, este teste falha — e falha
   antes de produzir um resultado que seria inútil como prova.
2. Os símbolos públicos de `autograd` e `nn` continuam nos mesmos nomes.
3. A API antiga (`kianda.nucleo.autograd`) continua a funcionar enquanto os
   consumidores não migram — a migração não pode quebrar quem já usava.
4. O cálculo continua correcto: gradiente de uma multiplicação conhecida.

O teste (1) é o importante. Os outros três protegem contra regressões enquanto
se move código.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))


# --------------------------------------------------------------------------
# 1. Independência estrutural — a propriedade central
# --------------------------------------------------------------------------


def test_nucleo_importa_sem_kianda_no_path() -> None:
    """O núcleo tem de funcionar num processo onde `kianda` é inalcançável.

    Executado num subprocesso com `sys.path` reduzido ao necessário e com uma
    armadilha: um meta-path hook que levanta erro se alguém tentar importar
    `kianda`. Sem a armadilha, bastaria `kianda` estar instalado no ambiente
    para o teste passar por acidente.
    """
    codigo = r"""
import sys, os

# Remover a RAIZ do repositório do sys.path, mas NÃO o site-packages do
# ambiente virtual — que também vive dentro da árvore quando o .venv está
# no projecto, e removê-lo apagava o numpy. Filtra-se a raiz exacta.
_RAIZ = os.path.abspath('.').replace(os.sep, '/')
_venv = os.path.abspath(sys.prefix).replace(os.sep, '/')
sys.path = [p for p in sys.path
            if not (p and os.path.abspath(p).replace(os.sep, '/').startswith(_RAIZ)
                    and not p.startswith(_venv))]

class Bloqueio:
    def find_module(self, nome, caminho=None):
        if nome == 'kianda' or nome.startswith('kianda.'):
            raise AssertionError(
                'VIOLACAO DE INDEPENDENCIA: o nucleo importou ' + nome)
        return None
    def find_spec(self, nome, caminho=None, alvo=None):
        return self.find_module(nome, caminho)

sys.meta_path.insert(0, Bloqueio())

import nucleo.tokenizador.bpe as bpe
from nucleo import autograd, nn

t = bpe.TokenizadorBPE()
t.treinar('abc abc abc def', vocab_alvo=300)
assert t.descodificar(t.codificar('abc def')) == 'abc def'

x = autograd.Tensor(__import__('numpy').array([[1.0, 2.0]], dtype='float32'), True)
w = autograd.Tensor(__import__('numpy').array([[3.0], [4.0]], dtype='float32'), True)
autograd.matmul(x, w).sum().backward()
assert w.grad is not None
print('OK')
"""
    r = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=str(RAIZ), capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, (
        f"o núcleo não é independente:\nstdout={r.stdout}\nstderr={r.stderr}"
    )
    assert "OK" in r.stdout


# Ficheiros de `nucleo/` que podem importar `kianda`, com a razão declarada.
#
# Isto não é uma excepção para o teste passar — é a fronteira, escrita. Uma lista
# de excepções explícita é auditável; uma regra vaga («as ferramentas podem») não.
#
# As duas entradas existem porque estas ferramentas **medem o sistema antigo**:
# `validacao_gerador.py` mede o gerador de `voz.py`, que ainda vive em `kianda/`.
# Quando o gerador migrar, as entradas saem. `test_excecoes_estao_correntes`
# falha se um ficheiro listado deixar de precisar da excepção.
EXCECOES = {
    "nucleo/experimentos/validacao_gerador.py":
        "mede o GeradorNeuronal de kianda/linguagem/voz.py, que ainda não migrou",
    "nucleo/testes/test_tokenizador.py":
        "o corpus de treino vem de kianda.linguagem.corpus até existir leitor próprio",
}


def test_nenhum_ficheiro_do_nucleo_importa_kianda() -> None:
    """Verificação estática, complementar ao teste dinâmico acima.

    O teste anterior prova que *funciona* sem `kianda`; este prova que *não
    tenta*, excepto onde está declarado que precisa. São propriedades
    diferentes: um import opcional dentro de um `try` passaria o primeiro e
    falharia o segundo.
    """
    import ast

    violacoes = []
    for caminho in sorted((RAIZ / "nucleo").rglob("*.py")):
        if "__pycache__" in caminho.parts:
            continue
        rel = str(caminho.relative_to(RAIZ))
        if rel in EXCECOES:
            continue
        arvore = ast.parse(caminho.read_text(encoding="utf-8", errors="replace"))
        for no in ast.walk(arvore):
            modulos = []
            if isinstance(no, ast.Import):
                modulos = [a.name for a in no.names]
            elif isinstance(no, ast.ImportFrom) and no.level == 0:
                modulos = [no.module or ""]
            for m in modulos:
                if m == "kianda" or m.startswith("kianda."):
                    violacoes.append(f"{rel}:{no.lineno} -> {m}")
    assert not violacoes, (
        "imports de kianda dentro do núcleo (não declarados em EXCECOES):\n  "
        + "\n  ".join(violacoes))


def test_excecoes_estao_correntes() -> None:
    """Uma excepção que já não é necessária é dívida disfarçada de política.

    Se um ficheiro listado em EXCECOES deixar de importar `kianda`, este teste
    falha e obriga a removê-lo da lista. Sem isto, a lista só cresce.
    """
    import ast

    for rel in EXCECOES:
        caminho = RAIZ / rel
        if not caminho.exists():
            pytest.fail(f"{rel} está em EXCECOES mas já não existe")
        arvore = ast.parse(caminho.read_text(encoding="utf-8", errors="replace"))
        usa = False
        for no in ast.walk(arvore):
            modulos = []
            if isinstance(no, ast.Import):
                modulos = [a.name for a in no.names]
            elif isinstance(no, ast.ImportFrom) and no.level == 0:
                modulos = [no.module or ""]
            if any(m == "kianda" or m.startswith("kianda.") for m in modulos):
                usa = True
                break
        assert usa, (
            f"{rel} está em EXCECOES mas já não importa kianda — remover da lista")


# --------------------------------------------------------------------------
# 2. Símbolos públicos preservados
# --------------------------------------------------------------------------

SIMBOLOS_AUTOGRAD = [
    "Custo", "Tensor", "matmul", "matmul_int8", "quantizar_int8",
    "entropia_cruzada",
]
SIMBOLOS_NN = [
    "Modulo", "Linear", "MLP", "Adam", "LoRA", "inicializar_he",
    "inicializar_xavier", "podar_estruturado", "destilar", "calibrar_escala",
]


def test_simbolos_de_autograd_presentes() -> None:
    from nucleo import autograd

    faltam = [s for s in SIMBOLOS_AUTOGRAD if not hasattr(autograd, s)]
    assert not faltam, f"símbolos desapareceram de autograd: {faltam}"


def test_simbolos_de_nn_presentes() -> None:
    from nucleo import nn

    faltam = [s for s in SIMBOLOS_NN if not hasattr(nn, s)]
    assert not faltam, f"símbolos desapareceram de nn: {faltam}"


# --------------------------------------------------------------------------
# 3. A API antiga continua a funcionar (compatibilidade durante a migração)
# --------------------------------------------------------------------------


def test_api_antiga_ainda_funciona() -> None:
    """`kianda.nucleo.autograd` não pode quebrar enquanto houver consumidores.

    Três ficheiros do núcleo importam por este caminho. Se o caminho antigo
    desaparecer sem que eles migrem, o repositório fica partido — e o registo de
    imports é que diz quando é seguro removê-lo.
    """
    codigo = r"""
import sys
sys.path.insert(0, '.')
from kianda.nucleo import autograd as antigo_auto
from kianda.nucleo import nn as antigo_nn
from nucleo import autograd as novo_auto
from nucleo import nn as novo_nn
assert antigo_auto.Tensor is novo_auto.Tensor, 'duas classes Tensor'
assert antigo_nn.Adam is novo_nn.Adam, 'duas classes Adam'
print('OK')
"""
    r = subprocess.run([sys.executable, "-c", codigo], cwd=str(RAIZ),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"a API antiga quebrou:\n{r.stderr}"
    assert "OK" in r.stdout


# --------------------------------------------------------------------------
# 4. O cálculo continua correcto
# --------------------------------------------------------------------------


def test_gradiente_numerico_conhecido() -> None:
    """Uma multiplicação cujo gradiente se calcula à mão.

    Sem isto, uma migração que partisse o autograd em silêncio passaria os
    testes de símbolos (os nomes existem) e de imports (não há violações).
    """
    from nucleo.autograd import Tensor, matmul

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), False)
    w = Tensor(np.array([[1.0], [10.0]], dtype=np.float32), True)
    y = matmul(x, w)          # [[1+20], [3+40]] = [[21], [43]]
    perda = y.sum()           # 64
    perda.backward()

    # d(perda)/d(w[i,j]) = soma de x[:, i] ; aqui só há j=0
    esperado = np.array([[4.0], [6.0]], dtype=np.float32)   # x[:,0].sum()=4, x[:,1].sum()=6
    assert np.allclose(w.grad, esperado), f"gradiente errado: {w.grad} != {esperado}"
    assert np.isclose(float(perda.data.sum()), 64.0)


def test_tokenizador_importa_e_usa_se() -> None:
    """O tokenizador é do núcleo e tem de funcionar sozinho."""
    from nucleo.tokenizador.bpe import TokenizadorBPE

    t = TokenizadorBPE()
    t.treinar("o mar o mar o mar a praia a praia", vocab_alvo=300)
    texto = "o mar e a praia"
    assert t.descodificar(t.codificar(texto)) == texto
