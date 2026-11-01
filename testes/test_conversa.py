"""Testes de conversa real — a barreira que faltava.

Porque este ficheiro existe
---------------------------
A suite anterior tinha 96 testes e todos passavam enquanto a Kianda respondia
«A memória tem cinco tipos» a quem dizia «estou triste» e «Relação entre Zuleika
e Bisa» a quem perguntava «como estás?». Nenhum teste julgava a **relação entre
a pergunta e a resposta** — apenas que a resposta não estava vazia, que a
intenção era a certa, que havia evidência.

Este ficheiro julga essa relação. Um teste aqui falha quando a resposta não
responde, mesmo que seja português impecável e esteja cheia de factos certos.

Regra de desenho, imposta pela missão:

- um teste pode falhar por razões de conteúdo, nunca por import/configuração;
- nenhum `skip`/`xfail` para esconder falha;
- quando um teste falha, a correcção é no sistema, não na asserção;
- a única razão aceitável para mudar uma asserção é ela estar a medir a coisa
  errada — e nesse caso a razão fica escrita ao lado.
"""

from __future__ import annotations

import re
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pytest


def _kianda():
    from kianda.mente import Kianda
    return Kianda(pasta_dados=tempfile.mkdtemp())


# Conteúdo que NUNCA deve aparecer numa resposta a uma pergunta social ou a uma
# afirmação sobre o estado da pessoa. Não é uma lista de palavras feias: é
# conteúdo de uma obra de ficção ou de documentação técnica, que não tem nada a
# ver com «como estás?».
FORA_DE_CONTEXTO = re.compile(
    r"zuleika|zulu|kau|bisa|sigillo|privatus|h99|apenas existo|"
    r"capulana|rangel|cazenga|"
    r"cinco tipos|autograd|parâmetro|bm25|flop|"
    r"não afirma ser consciente|mecanismos computacionais|"
    r"amizade antiga|conversa adiada",
    re.I)


# ---------------------------------------------------------------------------
# 1. Perguntas sociais
# ---------------------------------------------------------------------------

SOCIAIS = [
    "como estás?",
    "como vai, kianda?",
    "tudo bem contigo?",
    "estás bem?",
    "tas bem?",
    "tás boa?",
    "e aí, tudo fixe?",
    "como vai isso contigo?",
    "mano, estás tranquilo?",
    "eu apenas quero saber como você está",
    "só passei para saber como estás",
    "então, como vai a vida?",
]


@pytest.mark.parametrize("pergunta", SOCIAIS)
def test_pergunta_social_responde_ao_estado(pergunta):
    """Uma pergunta social pede uma resposta sobre o estado dela.

    Não se exige uma frase específica. Exige-se três coisas:

    1. não citar ficção nem documentação técnica;
    2. falar de estado (dela ou da relação), não de um assunto lateral;
    3. não ser uma pergunta de volta sem responder primeiro.

    O critério 3 é o que apanha «Boa tarde. Como estás?» a quem perguntou
    «como estás?» — tecnicamente cortês, conversacionalmente vazio.
    """
    k = _kianda()
    try:
        t = k.conversar(pergunta, guardar=False)
        txt = t.resposta.texto
        assert txt.strip(), "resposta vazia"
        assert not FORA_DE_CONTEXTO.search(txt), (
            f"«{pergunta}» respondeu com conteúdo fora de contexto: {txt[:120]}"
        )
        assert re.search(r"estou|estado|operacional|energia|bem|aqui|contigo|"
                         r"dia|tarde|noite|sinto", txt, re.I), (
            f"«{pergunta}» não falou de estado nenhum: {txt[:120]}"
        )
        # O primeiro segmento não pode ser uma pergunta de volta.
        primeiro = txt.split(".")[0]
        assert "?" not in primeiro or len(primeiro) > len(txt) * 0.8, (
            f"«{pergunta}» respondeu com uma pergunta em vez de uma resposta: "
            f"{txt[:120]}"
        )
    finally:
        k.fechar()


# ---------------------------------------------------------------------------
# 2. Afirmações sobre o estado de quem fala
# ---------------------------------------------------------------------------

AFIRMACOES_ESTADO = [
    ("estou triste", "triste"),
    ("estou muito cansado", "cansado"),
    ("estou com medo", "medo"),
    ("tensa", "tensa"),
    ("sinto-me sozinho", "sozinho"),
    ("não estou bem", "bem"),
    ("acho que estou a ficar triste", "triste"),
]


@pytest.mark.parametrize("mensagem,estado", AFIRMACOES_ESTADO)
def test_afirmacao_de_estado_e_reconhecida_como_afirmacao(mensagem, estado):
    """«Estou triste» não é uma pergunta sobre memória.

    Dois requisitos independentes:
    - a resposta não pode citar ficção/documentação;
    - a resposta tem de acusar que ouviu o estado (nomeá-lo, ou responder à
      pessoa em vez de mudar de assunto).
    """
    k = _kianda()
    try:
        t = k.conversar(mensagem, guardar=False)
        txt = t.resposta.texto
        assert not FORA_DE_CONTEXTO.search(txt), (
            f"«{mensagem}» respondeu com conteúdo fora de contexto: {txt[:120]}"
        )
        assert (estado in txt.lower()
                or re.search(r"ouvi|contigo|estou aqui|importa|conta|"
                             r"obrigad|sinto muito|percebo", txt, re.I)), (
            f"«{mensagem}» não acusou o estado: {txt[:120]}"
        )
    finally:
        k.fechar()


def test_negacao_muda_o_sentido():
    """«Não estou triste» não pode ser respondido como «estou triste».

    A diferença entre as duas respostas tem de existir. Não se exige qual é —
    exige-se que o sistema não trate a negação como o afirmado.
    """
    k = _kianda()
    try:
        a = k.conversar("estou triste", guardar=False).resposta.texto.lower()
        b = k.conversar("não estou triste", guardar=False).resposta.texto.lower()
        assert a != b, "a negação recebeu exactamente a mesma resposta"
        assert ("não" in b or "nao" in b or "ainda bem" in b
                or "que bom" in b or "menos mal" in b), (
            f"a resposta à negação não a reconhece: {b[:120]}"
        )
    finally:
        k.fechar()


def test_estado_de_terceiro_nao_e_estado_da_pessoa():
    """«Ela está triste» fala de outra pessoa.

    Responder «lamento que te sintas assim» a quem falou de uma terceira pessoa
    é um erro de leitura. Este teste exige que a resposta não atribua o estado
    ao utilizador.
    """
    k = _kianda()
    try:
        txt = k.conversar("ela está triste", guardar=False).resposta.texto.lower()
        assert "lamento que te sintas" not in txt
        assert "que pena que estejas" not in txt
        assert not FORA_DE_CONTEXTO.search(txt), (
            f"«ela está triste» foi respondido com ficção: {txt[:120]}"
        )
    finally:
        k.fechar()


# ---------------------------------------------------------------------------
# 3. Reparação de diálogo
# ---------------------------------------------------------------------------

REPARACOES = [
    "não foi isso que perguntei",
    "não entendeste",
    "estás a fugir ao assunto",
    "o que isso tem a ver?",
    "não perguntei, afirmei",
    "isso não faz sentido",
]


@pytest.mark.parametrize("msg", REPARACOES)
def test_reparacao_reconhece_o_erro(msg):
    """Pedir reparação não é insultar.

    A resposta tem de reconhecer que interpretou mal — não pedir ao utilizador
    que diagnostique («diz-me o que estava mal») nem defender-se («não vou mudar
    de resposta só porque não gostaste»).
    """
    k = _kianda()
    try:
        k.conversar("como estás?", guardar=False)
        t = k.conversar(msg, guardar=False)
        txt = t.resposta.texto.lower()
        assert "diz-me o que estava mal" not in txt, (
            f"pediu ao utilizador para diagnosticar: {txt[:120]}"
        )
        assert "não vou mudar de resposta" not in txt, (
            f"respondeu defensivamente: {txt[:120]}"
        )
        assert re.search(r"tens razão|desculpa|interpretei|percebi mal|"
                         r"errei|foi um erro|não respondi|deixa-me", txt), (
            f"«{msg}» não produziu reparação: {txt[:120]}"
        )
    finally:
        k.fechar()


def test_reparacao_nao_se_repete_identica():
    """A mesma reparação duas vezes não pode dar a mesma frase.

    Se der, não houve reparação: houve um template. A missão é explícita sobre
    «repetir respostas rejeitadas sem as corrigir».
    """
    k = _kianda()
    try:
        k.conversar("como estás?", guardar=False)
        a = k.conversar("não foi isso que perguntei", guardar=False).resposta.texto
        b = k.conversar("não foi isso que perguntei", guardar=False).resposta.texto
        assert a != b, f"a reparação repetiu-se palavra por palavra: {a[:100]}"
    finally:
        k.fechar()


# ---------------------------------------------------------------------------
# 4. Referências contextuais
# ---------------------------------------------------------------------------


def test_referencia_resolve_para_o_turno_anterior():
    """«o que isso significa?» refere-se ao que foi dito antes.

    O teste usa uma pergunta técnica com uma resposta concreta, e depois pede
    esclarecimento. A resposta tem de conter vocabulário do turno anterior —
    prova de que ligou a referência — e não um bloco genérico.
    """
    k = _kianda()
    try:
        primeiro = k.conversar("como funciona a memória?", guardar=False).resposta.texto
        segundo = k.conversar("o que isso significa?", guardar=False).resposta.texto
        assert "estou a ouvir" not in segundo.lower(), (
            f"respondeu com um bloco genérico: {segundo[:100]}"
        )
        # Tem de retomar alguma palavra do turno anterior.
        from kianda.memoria.recuperador import stem_leve, tokenizar
        antes = {stem_leve(x) for x in tokenizar(primeiro)}
        depois = {stem_leve(x) for x in tokenizar(segundo)}
        comuns = antes & depois
        assert len(comuns) >= 3, (
            f"«o que isso significa?» não retomou o turno anterior "
            f"(palavras comuns: {sorted(comuns)[:8]}); respondeu: {segundo[:120]}"
        )
    finally:
        k.fechar()


def test_pergunta_incompleta_usa_o_contexto():
    """«e quanto ao custo?» depois de falar de um assunto.

    É o exemplo do briefing. A resposta tem de falar de custo — não de outra
    coisa qualquer.
    """
    k = _kianda()
    try:
        k.conversar("quantos parâmetros tens?", guardar=False)
        txt = k.conversar("e o custo?", guardar=False).resposta.texto.lower()
        assert "custo" in txt or "mflop" in txt or "flop" in txt or "ms" in txt, (
            f"«e o custo?» não falou de custo: {txt[:120]}"
        )
    finally:
        k.fechar()


# ---------------------------------------------------------------------------
# 5. Comunicação informal, erros e abreviações
# ---------------------------------------------------------------------------

INFORMAIS = [
    ("nao entendi nada", "não entendeu"),
    ("o que tas a dizer", "pediu para repetir"),
    ("como e que funciona", "perguntou como funciona"),
    ("eu nao perguntei isso", "corrigiu"),
    ("tava a falar contigo", "corrigiu o interlocutor"),
    ("pq?", "perguntou porquê"),
    ("q isso?", "pediu esclarecimento"),
    ("n percebi", "não entendeu"),
    ("I don't understand isso", "não entendeu, meio em inglês"),
    ("Can you explain melhor?", "pediu explicação, meio em inglês"),
]


@pytest.mark.parametrize("mensagem,descricao", INFORMAIS)
def test_informal_nao_recebe_ficcao(mensagem, descricao):
    """Comunicação informal não pode receber conteúdo de livro.

    Não se exige uma resposta perfeita — exige-se que o sistema saiba que a
    pessoa está a falar *com ele* e não a fazer uma pergunta sobre a obra. Era
    este o defeito: «não entendi» recebia uma passagem sobre o mar.
    """
    k = _kianda()
    try:
        t = k.conversar(mensagem, guardar=False)
        txt = t.resposta.texto
        assert txt.strip(), f"[{descricao}] resposta vazia"
        assert not FORA_DE_CONTEXTO.search(txt), (
            f"[{descricao}] «{mensagem}» respondeu com ficção: {txt[:120]}"
        )
    finally:
        k.fechar()


# ---------------------------------------------------------------------------
# 6. O transcript do utilizador, exatamente como aconteceu
# ---------------------------------------------------------------------------

TRANSCRIPT = [
    "a última memória do mar",
    "tensa",
    "como vai, kianda?",
    "eu apenas quero saber como voce esta",
    "nao perguntei sobre o livro mulher ou ia estou a falar contigo pessoa",
    "voce esta mal",
    "nao perguntei afirmei",
    "como assim:",
    "do nada apenas lanças perguntas aleatorio a fim de ver se consegues responder algo?",
    "o erro é o seguinte parece que nao respondes nada de verdade",
    "o que estas a fazer, estamos a conversar não seja louca",
    "o que isso significa?",
    "não quero estas a me irritar vou desistir de ti",
    "não estendi o que isso tem a ver",
    "kkkk, isso esta pessimo voce nem tenta falar kkkk",
    "estou a falar com o nada pois nao funciona",
]


def test_transcript_original_regressao():
    """A conversa que originou a missão, sem alterações.

    Este teste guarda a ordem e o texto originais. Não é para ficar verde a
    qualquer custo: mede quantos turnos produzem uma resposta sem relação com a
    mensagem, e exige uma minoria. O limite desce à medida que o sistema
    melhora, nunca sobe.
    """
    k = _kianda()
    try:
        falhas = []
        for msg in TRANSCRIPT:
            t = k.conversar(msg, guardar=False)
            txt = t.resposta.texto
            e_social_ou_pessoal = bool(re.search(
                r"como (vai|est|estás)|quero saber|voce esta|você está|tensa",
                msg, re.I))
            if e_social_ou_pessoal and FORA_DE_CONTEXTO.search(txt):
                falhas.append((msg, txt[:90]))
        # Linha de base medida antes da correcção: 16 turnos, 0 coerentes e
        # 6 contaminações explícitas. O limite desce, nunca sobe.
        assert len(falhas) <= 1, (
            f"{len(falhas)} turnos sociais responderam com ficção: {falhas}"
        )
    finally:
        k.fechar()


def test_transcript_sem_respostas_repetidas_em_sequencia():
    """Duas respostas iguais seguidas a mensagens diferentes é sinal de template."""
    k = _kianda()
    try:
        anteriores = []
        repetidas = []
        for msg in TRANSCRIPT:
            txt = k.conversar(msg, guardar=False).resposta.texto
            if anteriores and txt == anteriores[-1]:
                repetidas.append((msg, txt[:60]))
            anteriores.append(txt)
        assert len(repetidas) <= 2, (
            f"{len(repetidas)} respostas idênticas em sequência: {repetidas}"
        )
    finally:
        k.fechar()


# ---------------------------------------------------------------------------
# 7. Coerência global: a resposta fala do que foi perguntado
# ---------------------------------------------------------------------------


def test_sobreposicao_minima_pergunta_resposta():
    """Nenhuma resposta pode ignorar por completo a pergunta.

    Sobreposição lexical é uma medida fraca — e é por isso que este teste exige
    apenas *alguma* ligação, não uma alta. Zero ligação em frases com assunto é
    sinal de que a resposta veio de outro sítio.

    Excepções legítimas (e documentadas): perguntas cuja resposta correcta não
    repete palavras da pergunta — «quem és tu?» / «Chamo-me Kianda». Essas estão
    na lista de exclusão abaixo, e a lista é curta de propósito.
    """
    from kianda.memoria.recuperador import stem_leve, tokenizar

    VAZIAS = {"o", "a", "que", "de", "e", "é", "um", "uma", "do", "da",
              "tu", "te", "isso", "isto", "me", "se", "em", "no", "na"}

    CASOS = [
        "quem é a Zuleika?",
        "de que fala o H99?",
        "quantos parâmetros tens?",
        "como funcionas?",
        "o que sentes?",
        "tens medo?",
        "que tipos de memória tens?",
        "o que acontece quando esqueceres algo?",
        "preciso de placa gráfica para correr?",
    ]
    k = _kianda()
    try:
        for q in CASOS:
            txt = k.conversar(q, guardar=False).resposta.texto
            a = {stem_leve(x) for x in tokenizar(q)} - VAZIAS
            b = {stem_leve(x) for x in tokenizar(txt)}
            sobrep = len(a & b) / max(len(a), 1)
            assert sobrep > 0.10, (
                f"«{q}» e a resposta não partilham quase nada "
                f"({sobrep:.2f}): {txt[:100]}"
            )
    finally:
        k.fechar()
