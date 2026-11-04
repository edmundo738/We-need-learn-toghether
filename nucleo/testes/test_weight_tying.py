"""Weight tying: a cabeça de saída partilha a matriz do embedding.

É a correcção que a comparação com nanoGPT/LitGPT identificou e que
**reduz** parâmetros. Aqui testa-se que reduz os certos, que o gradiente
acumula dos dois lados, e que o optimizador não vê o mesmo tensor duas vezes.
"""

from __future__ import annotations

import numpy as np

from nucleo import nn
from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada


def _cfg(tying: bool):
    return Config(vocab=64, d_model=32, n_camadas=2, n_cabecas=2, ctx=12,
                  d_ff=64, tying=tying)


def test_tying_poupa_exactamente_vocab_vezes_d_model():
    m0 = Transformer(_cfg(False), semente=0)
    m1 = Transformer(_cfg(True), semente=0)
    poupados = m0.n_parametros() - m1.n_parametros()
    assert poupados == 64 * 32, f"poupou {poupados}, esperava {64*32}"


def test_tying_e_por_omissao():
    """As referências (nanoGPT, LitGPT moderno) usam tying; o padrão segue-as."""
    assert Config().tying is True


def test_parametros_nao_tem_tensores_duplicados():
    """Se a cabeça fosse também listada, o Adam aplicaria o update duas vezes.

    Este é o defeito silencioso que o tying pode introduzir: o modelo treina,
    a loss desce, e o passo efectivo é o dobro do configurado.
    """
    m = Transformer(_cfg(True), semente=0)
    ps = m.parametros()
    assert len({id(t) for t in ps.values()}) == len(ps)
    assert not any(t is m.cabeca.W for t in ps.values())


def test_cabeca_e_transposta_do_embedding():
    """A transposta é materializada a cada forward (ligação viva).

    O `W` da cabeça começa a zeros e é sincronizado no primeiro forward —
    de propósito: guardar uma cópia no construtor foi exactamente o bug que
    fazia o modelo carregado devolver outra saída.
    """
    m = Transformer(_cfg(True), semente=0)
    m.forward(np.array([[1, 2, 3]]))
    np.testing.assert_array_equal(m.cabeca.W.data, m.emb.peso.data.T)


def test_transposta_acompanha_o_dono_depois_do_forward():
    """Alterar emb.peso e voltar a fazer forward tem de mudar a cabeça."""
    m = Transformer(_cfg(True), semente=0)
    m.forward(np.array([[1, 2, 3]]))
    m.emb.peso.data[4, 2] += 0.5
    m.forward(np.array([[1, 2, 3]]))
    assert abs(float(m.cabeca.W.data[2, 4]) - float(m.emb.peso.data[4, 2])) < 1e-7


def test_gradiente_acumula_dos_dois_lados():
    """Perturbar um elemento de emb.peso tem de afectar (a) o embedding da
    entrada e (b) a distribuição de saída. Se só um contasse, o tying seria
    decorativo."""
    c = _cfg(True)
    m = Transformer(c, semente=0)
    ids = np.array([[1, 2, 3, 4]])
    alvos = np.array([2, 3, 4, 5])

    def perda():
        return float(perda_entropia_cruzada(m.forward(ids), alvos).data.reshape(-1)[0])

    for t in m.parametros().values():
        t.zero_grad()
    perda_entropia_cruzada(m.forward(ids), alvos).backward()
    grad = m.emb.peso.grad.copy()

    # o token 5 aparece como ALVO (só a cabeça o usa) e o 1 como ENTRADA
    # (só o embedding o usa). Ambos têm de ter gradiente não nulo.
    assert np.abs(grad[1]).sum() > 0, "entrada não contribui"
    assert np.abs(grad[5]).sum() > 0, "cabeça não contribui"


def test_gradiente_numerico_com_tying():
    """Diferenças finitas sobre emb.peso com a cabeça partilhada."""
    c = _cfg(True)
    m = Transformer(c, semente=3)
    ids = np.array([[1, 2, 3]])
    alvos = np.array([2, 3, 4])

    def perda():
        return float(perda_entropia_cruzada(m.forward(ids), alvos).data.reshape(-1)[0])

    for t in m.parametros().values():
        t.zero_grad()
    perda_entropia_cruzada(m.forward(ids), alvos).backward()

    base = m.emb.peso.data.copy()
    for idx in [(1, 0), (4, 5), (7, 3)]:
        ana = float(m.emb.peso.grad[idx])
        eps = 1e-3
        m.emb.peso.data[idx] = base[idx] + eps
        c1 = perda()
        m.emb.peso.data[idx] = base[idx] - eps
        c2 = perda()
        m.emb.peso.data[idx] = base[idx]
        num = (c1 - c2) / (2 * eps)
        assert abs(num - ana) / max(abs(num), abs(ana), 1e-12) < 0.05, (
            f"{idx}: num={num} ana={ana}")


def test_sem_tying_continua_possivel():
    """Desligar tem de continuar a funcionar — serve para comparar."""
    m = Transformer(_cfg(False), semente=0)
    assert "cabeca.W" in m.parametros()
    assert len({id(t) for t in m.parametros().values()}) == len(m.parametros())


def test_linear_rejeita_peso_com_forma_errada():
    rng = np.random.default_rng(0)
    from nucleo.autograd import Tensor
    errado = Tensor(np.zeros((5, 3), dtype=np.float32), True)
    try:
        nn.Linear(3, 4, rng, peso=errado)
    except ValueError:
        return
    raise AssertionError("devia ter rejeitado a forma errada")
