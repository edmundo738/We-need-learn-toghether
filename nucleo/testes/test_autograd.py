

# ==========================================================================
# Tensor.fatiar — acrescentado na auditoria da Fase 2
# ==========================================================================

def test_fatiar_valores_correctos():
    """fatiar(2,5) devolve exactamente as linhas 2,3,4."""
    import numpy as np
    from nucleo.autograd import Tensor
    x = Tensor(np.arange(12, dtype=np.float32).reshape(6, 2), True)
    f = x.fatiar(2, 5)
    assert f.data.shape == (3, 2)
    np.testing.assert_array_equal(f.data, x.data[2:5])


def test_fatiar_sem_fim_vai_ate_ao_final():
    import numpy as np
    from nucleo.autograd import Tensor
    x = Tensor(np.arange(8, dtype=np.float32).reshape(4, 2), True)
    np.testing.assert_array_equal(x.fatiar(1).data, x.data[1:])


def test_fatiar_gradiente_so_na_fatia():
    """O gradiente de x é zero fora da fatia. É a razão de existir da operação:
    sem isto não se consegue calcular a perda só numa parte do lote."""
    import numpy as np
    from nucleo.autograd import Tensor
    x = Tensor(np.ones((6, 3), dtype=np.float32), True)
    x.fatiar(2, 5).sum().backward()
    esperado = np.zeros((6, 3), dtype=np.float32)
    esperado[2:5] = 1.0
    np.testing.assert_allclose(x.grad, esperado, atol=1e-6)


def test_fatiar_gradiente_numerico():
    """Compara o gradiente de fatiar com diferenças finitas centrais."""
    import numpy as np
    from nucleo.autograd import Tensor
    rng = np.random.default_rng(11)
    base = rng.standard_normal((5, 3)).astype(np.float32)

    def perda(d):
        return float((Tensor(d, False).fatiar(1, 4) ** 2).sum().data.reshape(-1)[0])

    x = Tensor(base.copy(), True)
    (x.fatiar(1, 4) ** 2).sum().backward()

    eps = 1e-4
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            d = base.copy()
            d[i, j] += eps
            c1 = perda(d)
            d[i, j] -= 2 * eps
            c2 = perda(d)
            num = (c1 - c2) / (2 * eps)
            assert abs(num - x.grad[i, j]) < 1e-2, (i, j, num, x.grad[i, j])


def test_fatiar_fatia_vazia_nao_rebenta():
    import numpy as np
    from nucleo.autograd import Tensor
    x = Tensor(np.ones((4, 2), dtype=np.float32), True)
    f = x.fatiar(2, 2)
    assert f.data.shape == (0, 2)
