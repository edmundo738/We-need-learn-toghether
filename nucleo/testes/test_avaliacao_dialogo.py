"""A avaliação de diálogo esteve errada três vezes. Este ficheiro guarda o quê/porquê.

Um erro de avaliação não é um erro de modelo: é um erro de INSTRUMENTO, e é
pior, porque produz números que parecem resultados. O historial completo está
no docstring de `nucleo/experimentos/avaliacao_limpa.py`. Aqui ficam os factos
medidos e as asserções que impedem o regresso dos métodos errados.

MÉTODO CORRECTO (medido, `resultado_avaliacao_limpa.json`)
    janelas DENTRO de cada conversa, sem enchimento, sem atravessar fronteiras,
    cauda incompleta descartada e reportada.

    condição   treino    validação   folga     vs acaso 9,011
    v1         0,0301    9,1121      9,0820    acima
    v2         0,0177    8,4727      8,4550    abaixo
    ambas      0,1026    8,2711      8,1685    abaixo

CONCLUSÃO QUE OS NÚMEROS SUSTENTAM
    Treinar em 80 conversas melhora a validação de 9,11 para 8,27 — 0,74 nats
    abaixo do acaso, num vocabulário de 8 192. É melhor que nada, e é muito
    pouco: em conversas novas o modelo acerta ligeiramente mais que o uniforme.
    A perda de TREINO é 0,10. A folga continua a ser 8,17 nats: **decora.**

    Duplicar os dados (50 -> 80 conversas) reduziu a folga de 9,08 para 8,17 =
    0,90x. Não resolve. O regime é de decoração, não de aprendizagem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def limpa() -> dict:
    f = RAIZ / "nucleo" / "experimentos" / "resultado_avaliacao_limpa.json"
    if not f.exists():
        pytest.skip("correr nucleo.experimentos.avaliacao_limpa")
    return json.loads(f.read_text())


def _cond(d: dict, nome: str) -> dict:
    return next(r for r in d["condicoes"] if r["nome"] == nome)


def test_metodo_nao_usa_enchimento(limpa: dict) -> None:
    """O método 2 (padding mascarado na perda) estava errado: contamina a atenção."""
    assert "sem enchimento" in limpa["metodo"]
    assert "intra-conversa" in limpa["metodo"]
    # o descarte é reportado, não silencioso
    for c in limpa["condicoes"]:
        assert "tokens_descartados_treino" in c
        assert "tokens_descartados_val" in c


def test_todas_as_condicoes_estao_abaixo_do_acaso_ou_reportadas(limpa: dict) -> None:
    """v1 fica ACIMA do acaso. Isso tem de ser dito, não arredondado para «melhorou»."""
    acaso = limpa["acaso_log_vocab"]
    v1 = _cond(limpa, "v1")
    ambs = _cond(limpa, "ambas")
    assert v1["val_acima_do_acaso"] is True, "v1 deixou de estar acima do acaso"
    assert v1["perda_val"] > acaso
    assert ambs["val_acima_do_acaso"] is False, "ambas passou a estar acima do acaso"
    assert ambs["perda_val"] < acaso


def test_mais_dados_melhoram_mas_nao_resolvem(limpa: dict) -> None:
    """A distinção essencial: houve melhoria real E ela é insuficiente."""
    v1 = _cond(limpa, "v1")
    ambs = _cond(limpa, "ambas")
    # melhoria real
    assert ambs["perda_val"] < v1["perda_val"], "mais dados deixaram de ajudar"
    # mas a folga não colapsou
    razao = limpa["razao_folga_ambas_sobre_v1"]
    assert razao > 0.85, (
        f"a folga caiu a {razao:.2f}x: se cair abaixo de 0,85 a conclusão "
        "«duplicar não resolve» deixa de valer e o teste deve ser reescrito")
    assert _cond(limpa, "ambas")["folga"] > 7.0


def test_a_perda_de_treino_continua_a_provar_decoracao(limpa: dict) -> None:
    for c in limpa["condicoes"]:
        assert c["perda_treino"] < 0.5, (
            f"{c['nome']}: perda de treino {c['perda_treino']} — se subiu acima "
            "de 0,5 o modelo deixou de decorar e isto é outra experiência")


def test_melhoria_em_nats_e_pequena(limpa: dict) -> None:
    """Quantificar a melhoria em vez de a deixar como «desceu»."""
    acaso = limpa["acaso_log_vocab"]
    ambs = _cond(limpa, "ambas")
    ganho = acaso - ambs["perda_val"]
    assert 0 < ganho < 1.5, (
        f"ganho de {ganho:.2f} nats sobre o acaso: se passar de 1,5 nats o "
        "modelo começou a generalizar e a conclusão muda")


def test_nao_ha_janelas_que_atravessam_conversas(limpa: dict) -> None:
    """Sanidade aritmética: as janelas usadas têm de caber no texto das conversas."""
    for c in limpa["condicoes"]:
        assert c["janelas_treino"] == c["tokens_usados_treino"] // 128
        assert c["janelas_val"] == c["tokens_usados_val"] // 128
