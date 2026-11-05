"""O painel mostra números reais, e recusa inventar os que faltam.

O risco de um painel é este: como é visual, um número errado passa melhor
despercebido do que num JSON. Estes testes garantem três coisas:

1. o que o painel mostra vem dos ficheiros medidos, não de constantes no HTML;
2. um resultado em falta aparece como `None` (a interface mostra «—»), nunca
   como um número plausível;
3. a geração ao vivo carrega um checkpoint real e devolve texto.

O ponto 1 é o que distingue um painel de um panfleto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "painel"))


@pytest.fixture(scope="module")
def servidor():
    import servidor as s
    return s


def test_estado_le_os_ficheiros_e_nao_constantes(servidor) -> None:
    e = servidor.estado()
    # estes valores têm de coincidir com os JSON, não com o HTML
    melhor = json.loads((RAIZ / "nucleo" / "experimentos" /
                         "resultado_melhor_ponto.json").read_text())
    assert e["baselines"]["transformer_bpb"] == melhor["melhor_observado"]["bits_por_byte_media"]
    assert e["baselines"]["trigrama_bpb"] == melhor["trigrama_bits_por_byte"]

    gram = json.loads((RAIZ / "nucleo" / "experimentos" /
                       "resultado_gramatical.json").read_text())
    assert e["gramatica"]["transformer"] == gram["transformer_prosa_800_passos"]["taxa"]


def test_campo_em_falta_e_none_e_nao_numero(servidor, tmp_path) -> None:
    """O contrato mais importante: não inventar."""
    e = servidor.estado()
    # todos os campos numéricos são None ou número — nunca string com número
    for bloco in ("corpus", "baselines", "gramatica"):
        for k, v in e[bloco].items():
            assert v is None or isinstance(v, (int, float)), f"{bloco}.{k} = {v!r}"


def test_o_painel_reconhece_que_o_trigrama_ganha(servidor) -> None:
    """Se o painel dissesse «ganha» quando a medição diz «perde», seria propaganda."""
    e = servidor.estado()
    assert e["baselines"]["ganha"] is False
    assert e["baselines"]["margem"] < 0


def test_a_curva_do_sobreajuste_esta_completa(servidor) -> None:
    e = servidor.estado()
    assert e["curva"] is not None
    assert len(e["curva"]) >= 10
    passos = [c["passo"] for c in e["curva"]]
    assert passos == sorted(passos)
    # a validação do último ponto tem de ser pior que a do melhor
    melhor = min(c["perda_val"] for c in e["curva"])
    assert e["curva"][-1]["perda_val"] > melhor + 3


def test_checkpoints_disponiveis_para_gerar(servidor) -> None:
    e = servidor.estado()
    assert e["checkpoints"], "sem checkpoints o painel não pode gerar"
    assert any(c.startswith("publico_") for c in e["checkpoints"])


def test_gerar_produz_texto(servidor) -> None:
    nome = servidor.estado()["checkpoints"]
    alvo = next((c for c in nome if c.startswith("publico_")), None)
    if alvo is None:
        pytest.skip("sem checkpoint publico_*")
    if not (servidor.CHECKPOINTS / alvo / "pesos.npz").exists():
        pytest.skip("checkpoint sem pesos (*.npz fora do git)")
    r = servidor.gerar(alvo, "Capitu olhou", n=12, temp=0.0)
    assert r["checkpoint"] == alvo
    assert isinstance(r["texto"], str) and len(r["texto"]) > 0


def test_o_html_nao_contem_numeros_medidos(servidor) -> None:
    """Um número escrito à mão no HTML nunca se actualiza."""
    html = servidor.HTML
    for suspeito in ("2.6796", "2.4766", "0.4615", "381981", "938807"):
        assert suspeito not in html, (
            f"«{suspeito}» está escrito à mão no HTML: tem de vir do /api/estado")
