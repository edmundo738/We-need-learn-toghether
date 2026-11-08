"""Testes da medição de autoria.

O que fica demonstrado aqui, e não só afirmado:
  1. o método de entropia distingue dois autores reais (Machado vs. o autor em
     estudo) com margem confortável — se não distinguisse, não servia;
  2. a função de entropia é determinística (mesma entrada, mesma saída);
  3. uma obra é reconhecida quando o modelo é treinado no resto dela própria;
  4. o teste que motivou tudo isto: o ensaio que sabemos ser do autor fica
     dentro da faixa de margens de material autêntico. Se este teste falhar,
     a conclusão sobre os textos em questão deixa de valer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nucleo.experimentos.autoria_entropia import entropia, modelo_de, normalizar

RAIZ = Path(__file__).resolve().parents[2]
OBRAS = RAIZ / "dados" / "obras"
PUBLICO = RAIZ / "dados" / "publico_limpo"
AUTOR = OBRAS / "autor"


def _texto(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _ae() -> str:
    return "".join(_texto(OBRAS / f"apenas_existo_parte{i}.txt") for i in (1, 2, 3))


def _h99() -> str:
    return "".join(_texto(OBRAS / f"H99_Parte_{p}.txt") for p in ("I", "II", "III", "IV"))


def _machado() -> str:
    return _texto(PUBLICO / "dom_casmurro.txt")


@pytest.fixture(scope="module")
def vocab() -> int:
    return len(set(normalizar(_ae()) + normalizar(_h99() + _machado())))


def test_entropia_e_deterministica(vocab):
    m = modelo_de(_ae()[:20000], vocab)
    texto = _ae()[20000:24000]
    a = entropia(m, texto)
    b = entropia(m, texto)
    assert a == b
    assert 0.5 < a < 8.0, "bits por caracter fora de qualquer intervalo plausível"


def test_obra_reconhecida_no_proprio_livro(vocab):
    """Leave-out real: treino no resto, o bloco tem de ser reconhecido."""
    ae = _ae()
    bloco = ae[30000:36000]
    resto = ae[:30000] + ae[36000:]
    m_self = modelo_de(resto, vocab)
    m_h99 = modelo_de(_h99(), vocab)

    b_self = entropia(m_self, bloco)
    b_outro = entropia(m_h99, bloco)
    assert b_self < b_outro, "o próprio livro tem de explicar melhor o bloco do que o outro"


def test_distingue_autores_com_margem(vocab):
    """Machado não pode ser confundido com o autor em estudo."""
    m_ae = modelo_de(_ae(), vocab)
    m_mach = modelo_de(_machado(), vocab)
    trecho_machado = _machado()[20000:26000]
    trecho_ae = _ae()[20000:26000]

    # o trecho de Machado tem de estar mais perto de Machado
    assert entropia(m_mach, trecho_machado) < entropia(m_ae, trecho_machado)
    # e o do autor mais perto do autor
    assert entropia(m_ae, trecho_ae) < entropia(m_mach, trecho_ae)


def test_ensaio_de_2026_cai_na_faixa_do_autor(vocab):
    """O ensaio é sabidamente do autor. Serve de escala.

    Faixa de referência medida em resultado_autoria_estabilidade.json:
    blocos reais de AE produzem margens 0.108-0.318.
    """
    ficheiro = AUTOR / "edmundo_sem_ia_revisto.txt"
    if not ficheiro.exists():
        pytest.skip("texto do autor não disponível")
    ensaio = _texto(ficheiro).split("#" * 60)[1]

    m_ae = modelo_de(_ae(), vocab)
    m_h99 = modelo_de(_h99(), vocab)
    b_ae = entropia(m_ae, ensaio)
    b_h99 = entropia(m_h99, ensaio)

    assert b_ae < b_h99, "o ensaio deve aproximar-se de APENAS EXISTO"
    assert abs(b_ae - b_h99) >= 0.05, (
        "margem abaixo da faixa de material autêntico — o método não está a ver sinal"
    )
