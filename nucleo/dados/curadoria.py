"""Filtros de curadoria determinísticos, no espírito do Dolma/Gopher.

PORQUE ISTO EXISTE
------------------
O Dolma (allenai) e o GigaVerbo (Tucano) não usam só «texto grande»: passam
o corpus por filtros. O Tucano treina um classificador BERT + XGBoost para
decidir o que é texto de qualidade. Eu não tenho dados rotulados nem GPU para
treinar esse classificador.

Mas metade dos filtros do Dolma **não precisam de modelo nenhum** — são
regras sobre o texto (`taggers/gopher.py`, lido em /tmp/estudo/dolma):
comprimento de palavra, razão símbolo/palavra, fracção de linhas com bullet,
fracção de caracteres em n-gramas repetidos, fracção de linhas duplicadas.

Esses implementam-se aqui, determinísticos e testáveis. É a parte que se pode
fazer com os recursos que existem, e a que o corpus actual nunca passou.

O QUE NÃO SE COPIA
------------------
Nenhum código do Dolma é copiado. O que se copia é **a ideia de filtrar**, e
os limiares são escolhidos e documentados, não herdados às cegas. Os limiares
do Gopher foram afinados para inglês da web; o nosso corpus é literatura em
português angolano, e um limiar de «fracção de palavras com carácter alfa»
que faz sentido na web remove prosa legítima. Por isso os limiares são
parâmetros, com valores por omissão justificados e medição do que removem.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

PALAVRA = re.compile(r"\S+")
SIMBOLOS = set("#…")
BULLETS = {"*", "-", "•"}


@dataclass
class Metricas:
    """O que se mediu num documento. Tudo em contagens, nada estimado."""

    caracteres: int = 0
    palavras: int = 0
    comprimento_mediano_palavra: float = 0.0
    razao_simbolo_palavra: float = 0.0
    fraccao_palavras_com_alfa: float = 0.0
    fraccao_linhas_bullet: float = 0.0
    fraccao_linhas_reticencias: float = 0.0
    fraccao_linhas_duplicadas: float = 0.0
    fracao_caracteres_em_ngrama_repetido: float = 0.0


def _mediana(vals: list[float]) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def medir(texto: str, n_ngrama: int = 10) -> Metricas:
    """Mede um documento. Sem julgar: só conta."""
    m = Metricas(caracteres=len(texto))
    palavras = PALAVRA.findall(texto)
    m.palavras = len(palavras)
    if not palavras:
        return m

    m.comprimento_mediano_palavra = _mediana([len(p) for p in palavras])
    m.razao_simbolo_palavra = sum(
        1 for p in palavras if any(s in p for s in SIMBOLOS)) / len(palavras)
    m.fraccao_palavras_com_alfa = sum(
        1 for p in palavras if any(c.isalpha() for c in p)) / len(palavras)

    linhas = texto.split("\n")
    nao_vazias = [l for l in linhas if l.strip()]
    if nao_vazias:
        m.fraccao_linhas_bullet = sum(
            1 for l in nao_vazias if l.strip()[:1] in BULLETS) / len(nao_vazias)
        m.fraccao_linhas_reticencias = sum(
            1 for l in nao_vazias if l.rstrip().endswith("…")) / len(nao_vazias)
        contagem = Counter(l.strip() for l in nao_vazias)
        dup = sum(c for c in contagem.values() if c > 1)
        m.fraccao_linhas_duplicadas = dup / len(nao_vazias)

    # Repetição de n-gramas de CARACTERES: assinatura de listas, menus, spam.
    #
    # Cuidado com a aritmética, que aqui deu 1,158 numa prosa legítima (mais
    # repetido do que texto — impossível). A causa: n-gramas de caracteres
    # SOBREPÕEM-SE, pelo que contar (c-1)*n por grama multiplica a mesma
    # posição várias vezes. A medida correcta é a fracção de POSIÇÕES cujo
    # n-grama já apareceu antes:
    if len(texto) > n_ngrama:
        gramas = Counter(texto[i:i + n_ngrama] for i in range(len(texto) - n_ngrama))
        posicoes_repetidas = sum(c - 1 for c in gramas.values() if c > 1)
        m.fracao_caracteres_em_ngrama_repetido = (
            posicoes_repetidas / (len(texto) - n_ngrama))
    return m


@dataclass
class Limiares:
    """Limiares de rejeição. Valores por omissão anotados com a razão."""

    min_palavras: int = 50          # menos que isto não ensina estrutura
    max_razao_simbolo_palavra: float = 0.10
    min_fraccao_palavras_com_alfa: float = 0.80
    max_fraccao_linhas_bullet: float = 0.90
    max_fraccao_linhas_reticencias: float = 0.30
    max_fracao_ngrama_repetido: float = 0.30


@dataclass
class Veredicto:
    aceite: bool
    motivos: list[str] = field(default_factory=list)
    metricas: Metricas | None = None


def avaliar_documento(texto: str, lim: Limiares | None = None) -> Veredicto:
    lim = lim or Limiares()
    m = medir(texto)
    motivos = []
    if m.palavras < lim.min_palavras:
        motivos.append(f"poucas_palavras:{m.palavras}")
    if m.razao_simbolo_palavra > lim.max_razao_simbolo_palavra:
        motivos.append(f"simbolos:{m.razao_simbolo_palavra:.3f}")
    if m.fraccao_palavras_com_alfa < lim.min_fraccao_palavras_com_alfa:
        motivos.append(f"sem_alfa:{m.fraccao_palavras_com_alfa:.3f}")
    if m.fraccao_linhas_bullet > lim.max_fraccao_linhas_bullet:
        motivos.append(f"bullets:{m.fraccao_linhas_bullet:.3f}")
    if m.fraccao_linhas_reticencias > lim.max_fraccao_linhas_reticencias:
        motivos.append(f"reticencias:{m.fraccao_linhas_reticencias:.3f}")
    if m.fracao_caracteres_em_ngrama_repetido > lim.max_fracao_ngrama_repetido:
        motivos.append(f"repeticao:{m.fracao_caracteres_em_ngrama_repetido:.3f}")
    return Veredicto(aceite=not motivos, motivos=motivos, metricas=m)


def deduplicar_linhas(texto: str, min_chars: int = 40) -> tuple[str, int]:
    """Remove linhas exactamente repetidas (>= min_chars).

    O Gopher do Dolma conta a fracção de linhas duplicadas como sinal de
    qualidade; aqui usamos a mesma medição para **remover** em vez de rejeitar
    o documento inteiro — uma obra pode ter uma frase repetida e continuar
    a ser boa, e foi exactamente essa repetição que contaminou a validação
    (7 de 511 janelas, ver AUDITORIA-FASE2.md).
    """
    vistas: set[str] = set()
    saida = []
    removidas = 0
    for linha in texto.split("\n"):
        chave = linha.strip()
        if len(chave) >= min_chars:
            if chave in vistas:
                removidas += 1
                continue
            vistas.add(chave)
        saida.append(linha)
    return "\n".join(saida), removidas
