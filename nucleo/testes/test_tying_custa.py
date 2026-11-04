"""O weight tying NÃO melhora a perda neste modelo. Este teste fixa isso.

A sessão anterior registou «tying melhora o condicionamento» (verdade sobre
o gradiente) e daí passou, sem medir, a tratar o tying como ganho. A medição
A/B com 3 seeds e tudo o resto constante diz o contrário:

    tying     1 722 624 params  val 6,0911 ± 0,0339
    sem tying 2 771 200 params  val 5,7524 ± 0,0383
    diferença -0,3387 nats = -11,47 erros-padrão

As três seeds de tying ficam ACIMA das três de sem-tying: não há sobreposição.
`resultado_sobreajuste.json` guarda a medição; este teste verifica que os
números continuam a dizer o que o comentário diz.

O teste não re-treina (custaria 14 minutos). Verifica o ficheiro medido e,
separadamente, que a afirmação estrutural do tying continua verdadeira.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
MEDICAO = RAIZ / "nucleo" / "experimentos" / "resultado_sobreajuste.json"


@pytest.fixture(scope="module")
def medicao() -> dict:
    if not MEDICAO.exists():
        pytest.skip("resultado_sobreajuste.json ausente: correr nucleo.experimentos.sobreajuste")
    return json.loads(MEDICAO.read_text())


def test_medicao_tem_os_dois_bracos_com_3_seeds(medicao: dict) -> None:
    com = [r for r in medicao["resultados"] if r["tying"]]
    sem = [r for r in medicao["resultados"] if not r["tying"]]
    assert len(com) == 3 and len(sem) == 3
    assert {r["semente"] for r in com} == {r["semente"] for r in sem}, \
        "as seeds têm de ser as mesmas nos dois braços"


def test_tying_custa_perda_e_nao_e_ruido(medicao: dict) -> None:
    dif = medicao["diferenca_sem_tying_menos_tying"]
    assert dif < 0, "se o sinal mudou, o comentário deste teste está errado"
    assert abs(medicao["diferenca_em_erros_padrao"]) > 3, \
        "uma diferença abaixo de 3 erros-padrão não sustenta a conclusão"


def test_as_seeds_de_tying_ficam_todas_acima(medicao: dict) -> None:
    """O teste mais duro: sem sobreposição entre os dois braços."""
    com = sorted(r["final_val"] for r in medicao["resultados"] if r["tying"])
    sem = sorted(r["final_val"] for r in medicao["resultados"] if not r["tying"])
    assert min(com) > max(sem), (
        f"houve sobreposição: pior sem-tying {max(sem):.4f} vs melhor com-tying "
        f"{min(com):.4f} — nesse caso a conclusão enfraquece"
    )


def test_tying_poupa_milhao_e_meio_de_parametros(medicao: dict) -> None:
    com = medicao["tying"]["n_parametros"]
    sem = medicao["sem_tying"]["n_parametros"]
    assert sem - com == 8192 * 128, "a diferença tem de ser vocab*d_model"


def test_tying_ainda_esta_a_descer_quando_sem_tying_ja_parou(medicao: dict) -> None:
    """Sem tying o mínimo chega antes do fim; com tying, ao fim da corrida.

    É a leitura de capacidade: com mais parâmetros, o modelo esgota os dados
    mais cedo. Se isto deixar de ser verdade, a interpretação muda.
    """
    passos_com = medicao["tying"]["passo_do_minimo"]
    passos_sem = medicao["sem_tying"]["passo_do_minimo"]
    assert all(p == 1000 for p in passos_com), f"tying parou de descer: {passos_com}"
    assert any(p < 1000 for p in passos_sem), f"sem tying não saturou: {passos_sem}"


def test_tying_continua_a_ser_estruturalmente_correto() -> None:
    """A conclusão é sobre PERDA, não sobre o tying estar mal implementado."""
    from nucleo.modelo.transformer import Config, Transformer

    cfg = Config(vocab=64, d_model=16, n_camadas=2, n_cabecas=2, ctx=16,
                 d_ff=32, tying=True)
    m = Transformer(cfg, semente=0)
    assert np.allclose(m.cabeca.W.data, m.emb.peso.data.T), "a cabeça tem de espelhar o embedding"

    # gradiente flui pelos dois lados (é a razão pela qual se usa tying)
    import json as _json
    import tempfile
    ids = np.arange(16).reshape(1, 16) % 64
    lg = m.forward(ids[:, :-1])
    from nucleo.modelo.transformer import perda_entropia_cruzada
    perda = perda_entropia_cruzada(lg, ids[0, 1:])
    perda.backward()
    g = m.emb.peso.grad
    assert g is not None and float(np.abs(g).max()) > 0, "embedding sem gradiente com tying"
