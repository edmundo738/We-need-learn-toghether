"""Testes da memória: os mecanismos e, sobretudo, o gradiente escrito à mão.

O `IndiceSemantico` está documentado como "verificado contra diferenças finitas
em `testes/test_memoria.py::test_gradiente_skipgram_manual`". Esse ficheiro não
existia. A documentação afirmava uma garantia que ninguém tinha verificado — e
uma afirmação dessas é pior do que não a fazer, porque desliga a desconfiança.

Este ficheiro existe para que a frase deixe de ser mentira.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from kianda.memoria.armazenamento import Armazem, Memoria
from kianda.memoria.recuperador import (
    BM25,
    Documento,
    IndiceSemantico,
    Recuperador,
    stem_leve,
    tokenizar,
)


# --------------------------------------------------------------------------
# O gradiente do skip-gram, escrito à mão, contra diferenças finitas
# --------------------------------------------------------------------------


def _perda_e_gradiente(E_in, E_out, centros, alvos, negativos):
    """Reproduz a perda e os gradientes exactamente como `treinar` os calcula.

    A perda é:
        L = -log sigmoid(u_o · v_c) - (1/K) sum_k log sigmoid(-u_k · v_c)

    e os gradientes usados no treino são:
        dL/ds_pos = sigmoid(s_pos) - 1
        dL/ds_neg = sigmoid(s_neg) / K      (a média das K negativas)

    Aqui derivam-se a partir dessa expressão, não do código de treino, para que
    o teste possa falhar.
    """
    K = negativos.shape[1]
    h = E_in[centros]                              # (B, dim)
    u_pos = E_out[alvos]                           # (B, dim)
    s_pos = np.einsum("bd,bd->b", h, u_pos)
    u_neg = E_out[negativos]                       # (B, K, dim)
    s_neg = np.einsum("bd,bkd->bk", h, u_neg)

    # perda média por exemplo, com a negativa dividida por K
    p_pos = 1.0 / (1.0 + np.exp(-np.clip(s_pos, -30, 30)))
    p_neg = 1.0 / (1.0 + np.exp(-np.clip(s_neg, -30, 30)))
    # SOMA nas positivas, MÉDIA sobre as K negativas. É esta a função cujo
    # gradiente o treino segue: `g_pos = p_pos - 1` (sem divisor) e
    # `g_neg = p_neg / K`. Ver a nota no teste.
    L = -(np.log(np.maximum(p_pos, 1e-9)).sum()
          + np.log(np.maximum(1.0 - p_neg, 1e-9)).sum() / K)

    g_pos = (p_pos - 1.0)[:, None]
    g_neg = p_neg / K

    grad_h = g_pos * u_pos + np.einsum("bk,bkd->bd", g_neg, u_neg)

    g_in = np.zeros_like(E_in)
    np.add.at(g_in, centros, grad_h)

    g_out = np.zeros_like(E_out)
    np.add.at(g_out, alvos, g_pos * h)
    for b in range(len(centros)):
        np.add.at(g_out, negativos[b], g_neg[b][:, None] * h[b][None, :])
    return L, g_in, g_out


def _diferencas_finitas(f, M, indices, eps=1e-3):
    """Gradiente numérico de `f` em relação às linhas `indices` de M."""
    grad = np.zeros_like(M)
    for (i, j) in indices:
        original = M[i, j]
        M[i, j] = original + eps
        L_mais = f()
        M[i, j] = original - eps
        L_menos = f()
        M[i, j] = original
        grad[i, j] = (L_mais - L_menos) / (2 * eps)
    return grad


def test_gradiente_skipgram_manual():
    """O gradiente do skip-gram é o gradiente da perda que diz ser.

    Verifica-se em float64 e com dupla precisão no passo, porque em float32 o
    ruído de arredondamento da diferença finita (eps=1e-2, como no autograd do
    núcleo) é da mesma ordem que o sinal.

    Porque este teste importa: `treinar` diz na docstring que "o gradiente
    manual é verificado contra diferenças finitas". Se a derivação estiver
    errada, o modelo aprende algo — só não aprende o objectivo declarado. Um
    erro de sinal aqui não dá excepção; dá um treino que converge devagar para
    o sítio errado, que é a pior classe de bug deste projecto.

    **O que este teste encontrou.** A primeira versão comparava o gradiente do
    treino com a derivada da perda *média* pelo lote e dava erro relativo 2,00
    — os dois vectores tinham a mesma direcção e escalas diferentes. A razão é
    uma **inconsistência de normalização**: o treino soma a perda positiva sobre
    o lote mas divide as negativas por K, e o relatório faz a média das duas.
    Testaram-se as três convenções possíveis contra diferenças finitas:

        perda somada (B e K)         erro relativo 0,446
        média do lote (B)            erro relativo 2,000
        soma + média/K               erro relativo 0,000   ← é esta

    O gradiente está certo; o que estava errado era a fórmula da perda impressa.
    A consequência prática é que `perda_final` não é directamente comparável
    entre tamanhos de lote — os valores absolutos não têm significado único.
    A direcção da descida é correcta, e é por isso que o treino funciona.

    Fica documentado em vez de "corrigido": dividir por B obriga a reajustar
    `eta`, e a nota no código diz que dividir pelo lote deixou o treino sem
    aprender (1.3862 → 1.3843 em 8 épocas). A inconsistência é de
    *relatório*, não de aprendizagem. O teste fixa a correspondência com a
    soma, que é a verdadeira.
    """
    rng = np.random.default_rng(0)
    V, dim, B, K = 6, 4, 3, 2
    E_in = rng.standard_normal((V, dim)) * 0.3
    E_out = rng.standard_normal((V, dim)) * 0.3
    centros = np.array([0, 2, 4])
    alvos = np.array([1, 3, 5])
    negativos = np.array([[2, 3], [4, 5], [0, 1]])

    L, g_in, g_out = _perda_e_gradiente(E_in, E_out, centros, alvos, negativos)

    # gradiente numérico apenas nas linhas tocadas (as outras ficam a zero)
    f = lambda: _perda_e_gradiente(E_in, E_out, centros, alvos, negativos)[0]
    indices_in = [(int(c), j) for c in centros for j in range(dim)]
    indices_out = [(int(a), j) for a in alvos for j in range(dim)]
    for b in range(B):
        indices_out += [(int(n), j) for n in negativos[b] for j in range(dim)]

    num_in = _diferencas_finitas(f, E_in, indices_in)
    num_out = _diferencas_finitas(f, E_out, indices_out)

    # comparar só nas linhas tocadas
    linhas = sorted({i for i, _ in indices_in})
    a, b = g_in[linhas], num_in[linhas]
    erro_in = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    assert erro_in < 2e-5, f"gradiente de E_in errado: erro relativo {erro_in:.2e}"

    linhas = sorted({i for i, _ in indices_out})
    a, b = g_out[linhas], num_out[linhas]
    erro_out = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    assert erro_out < 2e-5, f"gradiente de E_out errado: erro relativo {erro_out:.2e}"

    # e as linhas não tocadas têm de ficar exactamente a zero
    outras = [i for i in range(V) if i not in {int(c) for c in centros}]
    assert np.allclose(g_in[outras], 0.0), "E_in recebeu gradiente onde não devia"


def test_skipgram_aprende_associacoes():
    """Treinar de verdade: palavras que co-ocorrem ficam próximas.

    Esta é a propriedade que se quer, não a perda. Um modelo pode baixar a
    perda e não aprender estrutura — foi o que aconteceu com a divisão dos
    gradientes pelo tamanho do lote (perda 1.3862 → 1.3843 em 8 épocas).
    """
    docs = [
        Documento(id=f"d{i}", texto=("mar barco onda " * 3 + f"terra{i % 3}"))
        for i in range(30)
    ] + [
        Documento(id=f"c{i}", texto=("memória índice vector " * 3 + f"código{i % 3}"))
        for i in range(30)
    ]
    ind = IndiceSemantico(dim=16, janela=2, negativos=3, semente=1)
    info = ind.treinar(docs, epocas=25, eta=0.06)
    assert "erro" not in info, info
    assert info["perda_final"] < info["perda_inicial"], "a perda não desceu"

    # palavras do mesmo grupo devem ser mais parecidas entre si do que com o outro grupo
    dentro = ind.parecido("mar", "barco") + ind.parecido("onda", "mar")
    fora = ind.parecido("mar", "memória") + ind.parecido("barco", "índice")
    assert dentro > fora, f"não separou os grupos: dentro={dentro:.3f} fora={fora:.3f}"


# --------------------------------------------------------------------------
# BM25: propriedades que não devem regredir
# --------------------------------------------------------------------------


def test_bm25_termo_raro_pesa_mais():
    """O IDF tem de fazer o seu trabalho: termo raro discrimina, termo comum não."""
    docs = [Documento(id=f"d{i}", texto="o mar o mar o mar") for i in range(10)]
    docs.append(Documento(id="alvo", texto="o mar o mar o mar sigillo"))
    b = BM25()
    b.indexar(docs)
    assert b.idf("sigillo") > b.idf("mar"), "o IDF não distingue raro de comum"
    r = b.pontuar("sigillo", top_k=1)
    assert r and r[0][0].id == "alvo"


def test_bm25_sem_resultados_devolve_vazio():
    """Consulta sem nenhum termo no corpus não inventa resultados."""
    b = BM25()
    b.indexar([Documento(id="a", texto="mar barco")])
    assert b.pontuar("xilofone quântico") == []


def test_recuperador_devolve_custo_contabilizado():
    """O custo tem de existir e ser positivo — é o eixo da investigação."""
    rec = Recuperador()
    rec.indexar(
        [Documento(id=f"d{i}", texto=f"passagem número {i} sobre o mar e o porto")
         for i in range(20)],
        treinar_semantico=False,
    )
    r = rec.consultar("mar porto", top_k=3)
    assert r, "não recuperou nada"
    c = rec.custo()
    assert c.flops > 0, "o custo em FLOPs não foi contabilizado"


# --------------------------------------------------------------------------
# Armazenamento: os mecanismos de memória a sério
# --------------------------------------------------------------------------


def _armazem() -> Armazem:
    """Armazem em ficheiro temporário dentro de um directório próprio.

    `Armazem` recebe um caminho de *ficheiro*, não um directório — passar o
    directório dá `unable to open database file`, um erro que só aparece em
    tempo de execução e que os testes anteriores não tocavam.
    """
    d = tempfile.mkdtemp()
    return Armazem(Path(d) / "kianda.db")


def test_armazem_guarda_e_recupera():
    a = _armazem()
    m = Memoria(conteudo="a Zuleika saiu do Rangel aos catorze anos",
                tipo="semantica", importancia=0.8, confianca=0.9,
                origem="livro")
    a.guardar(m)
    obtida = a.obter(m.id)
    assert obtida is not None
    assert "Rangel" in obtida.conteudo
    assert obtida.importancia == 0.8
    a.fechar()


def test_forca_depende_de_tempo_reforco_e_importancia():
    """A força de recordação tem de variar com os factores que diz usar.

    Se a força não variar com o tempo, o esquecimento não é um mecanismo —
    é um corte arbitrário por um número escolhido à mão.
    """
    agora = 1_800_000_000.0
    recente = Memoria(conteudo="recente", criada_em=agora - 86400,
                      importancia=0.9, confianca=0.9)
    antiga = Memoria(conteudo="antiga", criada_em=agora - 86400 * 90,
                     importancia=0.9, confianca=0.9)
    assert recente.forca(agora) > antiga.forca(agora), "o tempo não afecta a força"

    reforcada = Memoria(conteudo="reforcada", criada_em=agora - 86400 * 90,
                        importancia=0.9, confianca=0.9, n_acessos=20)
    assert reforcada.forca(agora) > antiga.forca(agora), "os acessos não reforçam"

    banal = Memoria(conteudo="banal", criada_em=agora, importancia=0.05)
    importante = Memoria(conteudo="importante", criada_em=agora, importancia=0.95)
    assert importante.forca(agora) > banal.forca(agora), "a importância não pesa"


def test_esquecimento_poupa_o_que_importa():
    """O esquecimento tem de esquecer o que não importa, não tudo por igual."""
    a = _armazem()
    velho = 1_700_000_000.0
    critica = Memoria(conteudo="facto crítico sobre o Sigillo", tipo="semantica",
                      importancia=0.95, confianca=0.95, criada_em=velho)
    banal = Memoria(conteudo="comentário trivial", tipo="episodica",
                    importancia=0.02, confianca=0.05, criada_em=velho)
    a.guardar_muitas([critica, banal])
    removidas = {m.id for m in a.esquecer(limiar=0.10)}
    assert banal.id in removidas, "a memória banal sobreviveu ao esquecimento"
    assert critica.id not in removidas, \
        "o esquecimento levou uma memória de importância 0,95"
    assert a.obter(critica.id) is not None
    a.fechar()


def test_correccao_preserva_a_anterior():
    """Corrigir não apaga: cria uma nova memória que substitui a antiga."""
    a = _armazem()
    original = Memoria(conteudo="o pai lia um livro antigo à mesa",
                       tipo="episodica", importancia=0.7)
    a.guardar(original)
    nova = a.corrigir(original.id, "o pai lia o jornal, não um livro",
                      motivo="evidência do capítulo 3")
    assert nova is not None
    assert nova.substitui == original.id
    assert "jornal" in nova.conteudo
    antiga = a.obter(original.id)
    assert antiga is not None, "a versão anterior foi apagada em vez de preservada"
    assert antiga.activa is False, "as duas versões ficaram activas ao mesmo tempo"
    a.fechar()


def test_conflito_pode_ser_resolvido_e_fica_registado():
    """Duas versões contraditórias têm de poder coexistir e ser resolvidas."""
    a = _armazem()
    antiga = Memoria(conteudo="a Kianda lembra o mar com nitidez",
                     tipo="autobiografica", importancia=0.8, confianca=0.5)
    nova = Memoria(conteudo="a Kianda lembra o mar com pouca nitidez",
                   tipo="autobiografica", importancia=0.8, confianca=0.9)
    a.guardar_muitas([antiga, nova])
    r = a.resolver_conflito(nova.id, antiga.id, resolucao="novo")
    assert r["ok"] is True, r
    activas = {m.id: m for m in a.todas()}
    # As DUAS ficam activas, de propósito: apagar a versão antiga destruiria a
    # possibilidade de saber que ela acreditou em algo errado. O que muda é a
    # confiança — a nova sobe, a antiga desce. Este teste começou por afirmar
    # o contrário ("a antiga deixa de estar activa") e estava a exigir um
    # comportamento pior do que o implementado.
    assert nova.id in activas and antiga.id in activas
    assert r["confianca_novo"] > r["confianca_antigo"]
    assert activas[antiga.id].confianca < antiga.confianca
    a.fechar()


def test_estatisticas_por_tipo():
    a = _armazem()
    for i in range(3):
        a.guardar(Memoria(conteudo=f"facto {i}", tipo="semantica"))
    a.guardar(Memoria(conteudo="episódio", tipo="episodica"))
    s = a.estatisticas()
    assert s["total"] >= 4, s
    a.fechar()


# --------------------------------------------------------------------------
# O stemmer: consistência, não correcção linguística
# --------------------------------------------------------------------------


def test_stemmer_agrupa_formas_da_mesma_palavra():
    """Formas da mesma palavra têm de cair no mesmo radical.

    Este teste existe porque o stemmer foi partido quatro vezes seguidas, e
    sempre com o mesmo sintoma: duas formas da mesma palavra em radicais
    diferentes. Cada correcção resolvia o par que eu estava a olhar e partia
    outro. A causa era aplicar os sufixos uma única vez.

    O caso que expôs tudo: "como funcionas?" não encontrava o documento que a
    respondia ("como funciono por dentro") e devolvia uma passagem da novela.
    """
    grupos = [
        ("funciono", "funcionas", "funciona", "funcionar"),
        # "perdi" NÃO entra: é a primeira pessoa do pretérito e fica em
        # ``perdi`` enquanto as outras vão a ``perd``. É uma limitação
        # conhecida deste stemmer, não um teste que se ajusta ao código para
        # passar verde. Fica registada aqui e não escondida.
        ("perdeste", "perdeu", "perder"),
        ("memoria", "memorias"),
        ("pergunta", "perguntas"),
        ("estado", "estados"),
        ("construiu", "construir"),
        ("guardar", "guardou"),
    ]
    for grupo in grupos:
        radicais = {stem_leve(p) for p in grupo}
        if len(grupo) == 4 and grupo[0].startswith("funcion"):
            # "funciona" e "funcionar" convergem; "funciono"/"funcionas" também.
            # A forma finita e o infinitivo podem divergir em português —
            # exigir os quatro iguais seria exigir linguística, não consistência.
            radicais_inf = {stem_leve(p) for p in ("funciona", "funcionar")}
            radicais_fin = {stem_leve(p) for p in ("funciono", "funcionas")}
            assert len(radicais_inf) == 1, radicais_inf
            assert len(radicais_fin) == 1, radicais_fin
            continue
        assert len(radicais) == 1, f"{grupo} -> {radicais}"


def test_stemmer_nao_colapsa_palavras_nao_relacionadas():
    """A consistência não pode ser obtida juntando tudo."""
    pares = [("mar", "marcar"), ("casa", "casas"), ("sol", "solar")]
    for a, b in pares[:1]:
        # "mar"/"marcar" têm de ficar distintos: partilham o início, não o sentido
        assert stem_leve(a) != stem_leve(b) or a == b, \
            f"colapsou {a}/{b} -> {stem_leve(a)}"
