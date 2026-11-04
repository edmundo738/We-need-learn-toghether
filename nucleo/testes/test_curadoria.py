"""Testes dos filtros de curadoria.

Cada teste corresponde a um defeito real ou a uma armadilha medida.
"""

from __future__ import annotations

import pytest

from nucleo.dados.curadoria import (
    Limiares,
    avaliar_documento,
    deduplicar_linhas,
    medir,
)


def test_repeticao_nao_pode_passar_de_um():
    """A fracção de posições repetidas está em [0,1] por construção.

    A primeira versão multiplicava (c-1) por n_ngrama sem contar a
    sobreposição dos n-gramas de caracteres e devolvia 1,158 numa prosa
    legítima — mais caracteres repetidos do que texto. Impossível, e o
    filtro rejeitava as cinco obras do corpus.
    """
    texto = open("dados/obras/apenas_existo_parte1.txt", encoding="utf-8").read()
    m = medir(texto)
    assert 0.0 <= m.fracao_caracteres_em_ngrama_repetido <= 1.0


def test_prosa_real_e_aceite():
    import glob
    for f in sorted(glob.glob("dados/obras/*.txt")):
        texto = open(f, encoding="utf-8").read()
        v = avaliar_documento(texto)
        assert v.aceite, f"{f} rejeitada por {v.motivos}"


def test_spam_repetitivo_e_rejeitado():
    spam = "Item " * 500 + "\n" + "Menu " * 500
    v = avaliar_documento(spam)
    assert not v.aceite
    assert any(m.startswith("repeticao") for m in v.motivos)


def test_texto_curto_e_rejeitado():
    v = avaliar_documento("Duas palavras.")
    assert not v.aceite
    assert any(m.startswith("poucas_palavras") for m in v.motivos)


def test_texto_sem_letras_e_rejeitado():
    v = avaliar_documento("1234 5678 9012 " * 50)
    assert not v.aceite
    assert any(m.startswith("sem_alfa") for m in v.motivos)


def test_lista_de_bullets_e_rejeitada():
    v = avaliar_documento("\n".join(["- item de uma lista"] * 200))
    assert not v.aceite


def test_metricas_sao_contagens_nao_estimativas():
    """medir() não julga: conta. Um texto de 10 palavras tem 10 palavras."""
    texto = "um dois três quatro cinco seis sete oito nove dez"
    m = medir(texto)
    assert m.palavras == 10
    assert m.caracteres == len(texto)
    assert m.fraccao_palavras_com_alfa == 1.0


def test_deduplicar_remove_linhas_repetidas():
    linha = "Esta é uma linha suficientemente longa para ser deduplicada."
    texto = f"{linha}\noutra linha\n{linha}\nmais uma\n{linha}"
    saida, removidas = deduplicar_linhas(texto)
    assert removidas == 2
    assert saida.count(linha) == 1


def test_deduplicar_preserva_linhas_curtas_repetidas():
    """Linhas curtas repetidas podem ser diálogo legítimo («— Sim.»)."""
    texto = "— Sim.\n— Não.\n— Sim.\n— Não."
    saida, removidas = deduplicar_linhas(texto)
    assert removidas == 0
    assert saida == texto


def test_deduplicar_corrige_a_contaminacao_medida():
    """A frase que contaminou treino/validação tem de ser removida.

    Do relatório de auditoria: 7 janelas de validação continham esta frase
    porque aparece duas vezes na obra. Depois de deduplicar, uma só.
    """
    frase = "Sou Zuleika Mendes. Vinte e seis anos. Saí do Rangel naquela manhã."
    texto = f"{frase}\nTexto no meio da obra.\n{frase}"
    saida, removidas = deduplicar_linhas(texto)
    assert removidas == 1
    assert saida.count(frase) == 1


def test_limiares_sao_configuraveis():
    """O mesmo texto passa com um limiar e falha com outro mais exigente.

    Nota: "palavra "*60 NÃO serve — é repetição 0,983 e o filtro rejeita-o
    com razão. Usa-se prosa variada.
    """
    # Prosa real de uma obra, truncada a duas extensões diferentes.
    texto = open("dados/obras/apenas_existo_parte1.txt", encoding="utf-8").read()
    assert avaliar_documento(texto[:20000]).aceite
    assert not avaliar_documento(
        texto[:20000], Limiares(min_palavras=100000)).aceite
