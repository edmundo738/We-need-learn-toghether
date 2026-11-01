"""Representação do turno: o que foi dito, por quem, e sobre o quê.

Porque este módulo existe
-------------------------
O sistema anterior reduzia cada mensagem a **uma etiqueta** (`intencao`) e
escolhia um template. Isso é suficiente para um formulário e insuficiente para
uma conversa, por três razões medidas:

1. **Não há sujeito.** «Estou triste», «Estás triste?», «Ela está triste» e
   «Não estou triste» recebiam a mesma etiqueta (`pergunta_pessoal`) e a mesma
   resposta. A informação de quem sente o quê estava no texto e era deitada
   fora na primeira linha do pipeline.

2. **Não há acto.** «o que sentes?» é uma pergunta; «estou triste» é uma
   afirmação. O classificador sabe distingui-las por acaso (uma cai em
   `pergunta_pessoal`, a outra também) e a composição não usa a distinção.

3. **Não há referências.** «o que isso significa?» não tem assunto próprio: o
   assunto está no turno anterior. Sem resolução, a única saída possível é um
   template genérico («Estou a ouvir. Continua.»).

Este módulo não classifica intenções — isso continua a ser o Naive Bayes. Ele
extrai a **estrutura** que a classificação não consegue representar: acto,
sujeito, tópico, referências e estado comunicado. A classificação decide *que
tipo de resposta*; a representação decide *do que é que a resposta fala*.

Regra de desenho: nada aqui inventa significado que não esteja no texto ou no
turno anterior. Quando não há evidência, o campo fica vazio — e a composição é
obrigada a lidar com o vazio em vez de o preencher com ficção.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..memoria.recuperador import normalizar, stem_leve, tokenizar

# ---------------------------------------------------------------------------
# Vocabulário fechado. Cada lista é uma decisão, e está escrita para poder ser
# contestada: se um teste falhar, sabe-se exactamente qual destas linhas mudar.
# ---------------------------------------------------------------------------

# Palavras que não trazem assunto. Usadas para decidir se um turno tem conteúdo
# próprio ou se vive do anterior.
_VAZIAS = frozenset("""
a o e é de do da dos das um uma uns umas que se em no na nos nas por para com
sem ao aos à às meu minha teu tua seu sua meu me te lhe nos vos
mais menos muito pouco ja já ainda tambem também so só bem mal
aqui ali lá ai aí isto isso aquilo este esta esse essa aquele aquela
coisa coisas negocio negócio cena tipo assim entao então ok okay
""".split())

# Marcadores de pergunta. Não é uma lista de intenções: é gramática.
_INTERROGATIVOS = frozenset("""
quem que qual quais quanto quanta quantos quantas onde quando como porque
porquê para_quê
""".split())
_INICIO_INTERROGATIVO = re.compile(
    r"^\s*(quem|qual|quais|quanto|quanta|quantos|quantas|onde|quando|como|"
    r"porque|porquê|o que|que|será|sera|podes|podia|consegues|sabes|diz|"
    r"explica|fala)\b", re.I)

# Sujeito gramatical. A ordem importa: verifica-se 1.ª pessoa antes de 2.ª
# porque «eu não estou» contém ambos os pronomes e o sujeito é «eu».
_P1 = re.compile(r"\b(eu|me|mim|comigo|estou|sou|tenho|sinto|ando|estava|era|"
                 r"fiquei|estou-me|sinto-me)\b", re.I)
_P2 = re.compile(r"\b(tu|te|ti|contigo|estás|estas|tas|tás|és|tens|sentes|"
                 r"andas|estavas|eras|ficaste|você|voce|vc|senhor|senhora|"
                 r"você|vocês|vos)\b|\b(como voce esta|como você está|"
                 r"tudo fixe|tudo bom|tudo bem)\b", re.I)
_P3 = re.compile(r"\b(ele|ela|eles|elas|dele|dela|deles|delas)\b", re.I)
_KIANDA = re.compile(r"\b(kianda|tu|você|voce|vc)\b", re.I)

# Negação — muda o significado de tudo o que vem a seguir.
_NEGACAO = re.compile(r"\b(não|nao|nunca|nem|jamais|n|nt|nada)\b", re.I)

# Manifestações de estado em 1.ª pessoa. É uma lista de *formas de dizer*, não
# de emoções: o objectivo é detectar que a pessoa está a falar de si, não
# diagnosticar o que ela sente. O valor emocional fica com quem fala.
# O adjectivo pode vir depois de um adverbio («muito cansado») ou de uma
# locucao («a ficar triste»). A primeira versao exigia que viesse logo a seguir
# ao auxiliar e por isso «estou muito cansado» nao era detectado.
_ESTADO_1P = re.compile(
    r"\b(?:estou|sinto|ando|fiquei|sinto-me|tô|tou)\s+"
    r"(?:muito |pouco |meio |bastante |tão |tao |super |mesmo )*"
    r"(?:a ficar |a sentir |a tornar-me )?"
    r"(bem|mal|triste|cansad[oa]|feliz|ansios[oa]|sozinh[oa]|perdid[oa]|"
    r"nervos[oa]|fart[oa]|exaust[oa]|preocupad[oa]|assustad[oa]|tens[oa]|"
    r"irritad[oa]|confus[oa]|melhor|pior|otim[oa]|optim[oa]|"
    r"desanimad[oa]|excitad[oa]|curios[oa])", re.I)
# Formas nominais: «tensa» sozinho, «cansado», «com medo».
_ESTADO_NOMINAL = re.compile(
    r"^\s*(triste|cansad[oa]|feliz|ansios[oa]|sozinh[oa]|perdid[oa]|nervos[oa]|"
    r"fart[oa]|exaust[oa]|preocupad[oa]|assustad[oa]|tenso|tensa|irritad[oa]|"
    r"confus[oa]|desanimad[oa]|excitad[oa]|bem|mal|otim[oa]|optim[oa]|"
    r"com (medo|sono|raiva|pressa|fome))\s*[.!?]*\s*$", re.I)

# Reparação de diálogo: o utilizador está a dizer que a resposta não serviu.
_REPARACAO = re.compile(
    r"(n[ãa]o (foi )?isso que (eu )?perguntei"
    r"|n[ãa]o (foi )?isso"
    r"|n[ãa]o perguntei"
    r"|n[ãa]o (me )?entendeste"
    r"|n[ãa]o (me )?entendeu"
    r"|n[ãa]o percebeste"
    r"|n[ãa]o percebi"
    r"|n[ãa]o entendi"
    r"|est[áa]s a fugir"
    r"|fugiste ao assunto"
    r"|o que (isso|isto) tem a ver"
    r"|isso n[ãa]o faz sentido"
    r"|n[ãa]o foi isso que pedi"
    r"|respondeste ao lado"
    r"|nada a ver com)", re.I)

# Pedido de esclarecimento: não é reparação, é «explica-me isso».
_ESCLARECIMENTO = re.compile(
    r"^\s*(o que (isso|isto|aquilo) (significa|quer dizer|é)"
    r"|o que queres dizer"
    r"|o que quer dizer"
    r"|como assim"
    r"|explica(me)?( isso| isto| melhor)?\b"
    r"|n[ãa]o (entendi|percebi|estou a entender)"
    r"|podes explicar"
    r"|mais devagar"
    r"|e\?)", re.I)

# Pedido de repetição: não é pedido de esclarecimento (explicar) nem pergunta
# nova. A pessoa diz que não recebeu a mensagem, ou que quer ouvi-la outra vez.
# Sem isto, «o que tas a dizer» caía na recuperação e respondia com um documento.
_PEDIDO_REPETICAO = re.compile(
    r"(o que (tas|est[áa]s|é que est[áa]s) a dizer"
    r"|o que (disseste|dizes|acabaste de dizer)"
    r"|repete|repetir|outra vez|n[ãa]o ouvi|n[ãa]o li"
    r"|como (disseste|dizes)|diz outra vez|manda outra vez"
    r"|what did you say|say again|repeat)", re.I)

# Referências anafóricas que precisam do turno anterior para terem sentido.
_REFERENCIA = re.compile(r"\b(isso|isto|aquilo|o mesmo|a mesma|disso|disto|"
                         r"nisso|nisto|tal|aquela coisa|essa coisa)\b", re.I)
# Referências a pessoas por pronome.
_REF_PRONOMINAL = re.compile(r"\b(ele|ela|eles|elas|o mesmo|a mesma)\b", re.I)

# Brincadeira / ironia: sinal de que o sentido literal não é o sentido.
_BRINCADEIRA = re.compile(r"(\bkkk+\b|\bha(ha)+\b|\brs+\b|😂|🤣|😅|\blol\b|"
                          r"\bbrincadeira\b|\ba brincar\b|\bzoeira\b|\bna boa\b)", re.I)

# Pedidos de ajuda para formular a ideia (secção 6G do briefing).
# Perguntas sobre o estado dela. Partilhada com a composição (`_e_pergunta_de_estado`
# em mente.py) — a lista fica aqui para não haver duas verdades.
_PERGUNTA_ESTADO = re.compile(
    r"(como (est[áa]s?|estas|vai|vais|vai isso|vai a vida|é que estás|e que estas)"
    r"|tudo (bem|fixe|bom)|tas bem|tás bem|tas boa|tás boa|est[áa]s bem|estas bem"
    r"|tranquil[oa]|s[óo] passei para saber|quero saber como|"
    r"como est[áa] contigo|como vai contigo)", re.I)

_FORMULAR = re.compile(r"(n[ãa]o sei (como )?(dizer|explicar|formular)"
                       r"|n[ãa]o consigo (explicar|dizer)"
                       r"|como (é que )?(se )?diz"
                       r"|n[ãa]o me sai)", re.I)


@dataclass
class Turno:
    """Representação estruturada de uma mensagem do utilizador.

    Os campos são independentes de propósito: uma frase pode ser ao mesmo tempo
    uma afirmação, uma manifestação de estado e uma continuação de um assunto.
    Forçar tudo a uma categoria foi o erro da versão anterior.
    """

    texto: str
    normalizado: str

    # -- acto comunicativo (não exclusivo) --------------------------------
    e_pergunta: bool = False
    e_afirmacao: bool = False
    e_correcao: bool = False
    e_pedido_esclarecimento: bool = False
    e_pedido_repeticao: bool = False
    e_pedido: bool = False
    e_brincadeira: bool = False

    # -- quem -------------------------------------------------------------
    sujeito: str = "desconhecido"   # utilizador | kianda | terceiro | desconhecido
    negado: bool = False
    hipotese: bool = False           # «acho que», «talvez», «se calhar»

    # -- estado comunicado -------------------------------------------------
    estado: str = ""                 # a forma exacta que a pessoa usou, ou ""
    estado_polaridade: int = 1       # +1 / -1 (negado)
    recusa_falar: bool = False       # «não quero falar disso»

    # -- assunto e referências ---------------------------------------------
    topicos: list[str] = field(default_factory=list)
    tem_conteudo_proprio: bool = True
    referencia_pendente: bool = False
    referencia_resolvida: str = ""
    referencia_origem: str = ""      # de que turno veio

    # -- para a composição -------------------------------------------------
    precisa_contexto: bool = False
    ambiguidade: bool = False

    def resumo(self) -> str:
        mapa = {
            "pergunta": self.e_pergunta,
            "afirmação": self.e_afirmacao,
            "correcção": self.e_correcao,
            "esclarecimento": self.e_pedido_esclarecimento,
            "pedido": self.e_pedido,
            "brincadeira": self.e_brincadeira,
        }
        actos = [n for n, v in mapa.items() if v]
        return (f"{self.sujeito}"
                f"{' (negado)' if self.negado else ''}"
                f" · {'/'.join(actos) or 'sem acto'}"
                f" · estado={self.estado or '-'}"
                f" · ref={self.referencia_resolvida or '-'}")


# Adjectivos de estado. Usados para (a) detectar «tensa» sozinho como
# afirmacao e (b) decidir se um topico e assunto ou estado. Ficam numa so lista
# porque a distincao entre elas nao e linguistica: e operacional.
PALAVRAS_ESTADO = frozenset("""
triste cansado cansada feliz ansioso ansiosa sozinho sozinha perdido perdida
nervoso nervosa farto farta exausto exausta preocupado preocupada assustado
assustada tenso tensa irritado irritada confuso confusa desanimado desanimada
excitado excitada bem mal otimo optimo otima cansaco raiva medo sono pressa
fome solidao saudade
""".split())


def _topicos(texto: str) -> list[str]:
    """Palavras com conteúdo, por ordem de aparição.

    Não é extração de tópico a sério: é o que sobra depois de tirar as palavras
    vazias. Serve para decidir *se* um turno tem assunto próprio e para dar à
    resolução de referências algo concreto a que se agarrar.
    """
    vistos, saida = set(), []
    for t in tokenizar(texto):
        r = stem_leve(t)
        if len(t) < 3 or t in _VAZIAS or r in _VAZIAS:
            continue
        if r in vistos:
            continue
        vistos.add(r)
        saida.append(t)
    return saida


def analisar(texto: str, historico: list[str] | None = None) -> Turno:
    """Constrói a representação de um turno.

    `historico` são as mensagens anteriores do utilizador, da mais antiga para a
    mais recente. É usado apenas para resolver referências — nunca para
    adivinhar intenção.
    """
    historico = historico or []
    cru = (texto or "").strip()
    norm = normalizar(cru)
    t = Turno(texto=cru, normalizado=norm)
    if not norm:
        return t

    topicos = _topicos(cru)
    t.topicos = topicos

    # -- actos ------------------------------------------------------------
    # O interrogativo tem de estar no INICIO ou haver ponto de interrogacao.
    # A versao anterior procurava interrogativos em qualquer posicao e por isso
    # «acho que estou a ficar triste» era pergunta (por causa do «que»).
    interrogativo = ("?" in cru or bool(_INICIO_INTERROGATIVO.match(cru)))
    t.e_pergunta = interrogativo
    t.e_correcao = bool(_REPARACAO.search(cru))
    t.e_pedido_esclarecimento = bool(_ESCLARECIMENTO.search(cru))
    t.e_pedido_repeticao = bool(_PEDIDO_REPETICAO.search(cru))
    t.e_pedido = bool(re.match(r"^\s*(diz|explica|fala|mostra|conta|ajuda|podes|"
                               r"podias|quero que)\b", cru, re.I))
    t.e_brincadeira = bool(_BRINCADEIRA.search(cru))

    # -- sujeito ----------------------------------------------------------
    # A ordem é a decisão: 1.ª pessoa antes de 2.ª porque «eu não estou bem»
    # contém «tu» em nenhum caso mas «estou» sempre; e «estás» contém «tas».
    # Hipotese e recusa primeiro: «acho que estou a ficar triste» tem «acho que»,
    # que contem «que» e fazia o turno passar por pergunta. Avaliar o sujeito
    # antes disto deixava o erro passar para a composicao.
    _e_hipotese = bool(re.search(r"\b(acho que|talvez|se calhar|pode ser|"
                                 r"será que|se eu|imagina)", cru, re.I))
    _e_recusa = bool(re.search(r"n[ãa]o quero falar|deixa|esquece|"
                               r"n[ãa]o fales disso", cru, re.I))
    if _e_hipotese:
        interrogativo = False

    if _P1.search(cru):
        t.sujeito = "utilizador"
    elif _P2.search(cru) or _KIANDA.search(cru):
        t.sujeito = "kianda"
    elif _P3.search(cru):
        t.sujeito = "terceiro"

    # Uma pergunta em 2.ª pessoa dirigida a ela não é uma afirmação.
    if t.e_pergunta and t.sujeito == "kianda":
        t.e_afirmacao = False
    elif t.sujeito == "utilizador" and not t.e_pergunta:
        t.e_afirmacao = True
    elif not t.e_pergunta and not t.e_correcao and not t.e_pedido_esclarecimento:
        t.e_afirmacao = bool(topicos) or bool(
            re.search(r"\b(é|e|está|esta|foi|tem|há|ha)\b", cru))

    # Uma pergunta sobre o estado dela é sobre ela, mesmo escrita na 1.ª pessoa
    # («eu só queria saber como estás»). O verbo está na 2.ª pessoa; o «eu»
    # pertence à moldura («eu quero saber»), não ao assunto. Sem esta regra, a
    # frase caía no ramo de afirmação do utilizador e ia buscar documentação.
    if _PERGUNTA_ESTADO.search(cru):
        t.sujeito = "kianda"
        t.e_pergunta = True

    # -- negação e hipótese ------------------------------------------------
    # «Nao estou triste» e uma negacao. Mas «nao quero falar disso» nao nega o
    # estado — recusa falar dele. Sao coisas diferentes e a versao anterior
    # tratava as duas como negacao.
    t.negado = bool(_NEGACAO.search(cru)) and not t.e_pergunta and not _e_recusa
    t.hipotese = _e_hipotese
    t.recusa_falar = _e_recusa

    # -- estado comunicado pela pessoa -------------------------------------
    _COM = re.search(r"\bcom\s+(medo|sono|raiva|pressa|fome|saudade)\b", cru, re.I)
    m = _ESTADO_1P.search(cru)
    if m:
        # Guardar o adjectivo, nao o auxiliar: «estou a ficar triste» -> «triste».
        # A versao anterior guardava «estou a ficar», que nao e um estado e nao
        # serve para responder a nada.
        # Procurar a primeira forma que seja realmente um estado (ou uma
        # locucao «com X»). Não basta a ultima palavra: em «estou muito
        # cansado hoje» era «hoje».
        t.estado = m.group(1).strip().lower()
    else:
        m = _ESTADO_NOMINAL.match(cru)
        if m and not t.e_pergunta:
            t.estado = m.group(1).strip()
    # Estado atribuído a outra pessoa: «ela está triste», «ele anda mal».
    if not t.estado and t.sujeito == "terceiro" and not t.e_pergunta:
        m3 = re.search(r"\b(est[áa]|anda|parece|ficou)\s+"
                       r"(?:muito |meio |bastante |tão |tao )?"
                       r"([a-záàâãéêíóôõúç]+)", cru, re.I)
        if m3 and stem_leve(m3.group(2)) in {stem_leve(x) for x in PALAVRAS_ESTADO}:
            t.estado = m3.group(2).lower()

    if not t.estado and _COM and not t.e_pergunta:
        t.estado = _COM.group(1).lower()
    # Um adjectivo de estado sozinho («tensa», «cansada») é a pessoa a falar de
    # si sem verbo. Sem sujeito explícito, o sujeito é quem escreveu — não é
    # adivinhação, é a única leitura possível de uma mensagem de uma palavra.
    if t.estado and t.sujeito == "desconhecido":
        t.sujeito = "utilizador"

    if t.estado and t.negado:
        t.estado_polaridade = -1
    # «estou triste, mas não quero falar disso» — a recusa prevalece
    if t.estado and t.recusa_falar:
        t.recusa_falar = True

    # -- conteúdo próprio e referências ------------------------------------
    # Um turno tem conteúdo próprio se tiver palavras com conteúdo para além
    # dos pronomes. «o que isso significa» tem só `significa`.
    uteis = [x for x in topicos if stem_leve(x) not in
             {"isso", "isto", "aquilo", "signific", "quer", "dizer", "explic"}]
    t.tem_conteudo_proprio = len(uteis) >= 2

    if _REFERENCIA.search(cru) and not t.tem_conteudo_proprio:
        t.referencia_pendente = True
        t.precisa_contexto = True
        # Resolver contra o que foi dito, por esta ordem: (1) a última fala da
        # Kianda, porque «o que isso significa?» aponta para o que ela acabou de
        # dizer; (2) a última mensagem do utilizador, se a dela não tiver
        # assunto.
        #
        # A ordem estava invertida e o custo era visível: «estou com medo» →
        # «o que isso significa?» resolvia o «isso» para «estou medo» — as
        # palavras do próprio utilizador — e a explicação falava do que ele
        # tinha dito, não do que ela tinha dito. Uma pergunta sobre a resposta
        # tem de se resolver contra a resposta.
        # Primeiro a voz da Kianda (marcada), depois a do utilizador.
        _da_kianda = [h for h in historico if h.startswith("kianda: ")]
        _do_user = [h for h in historico if not h.startswith("kianda: ")]
        for grupo in (_da_kianda[-2:], _do_user):
            for anterior in reversed(grupo):
                top = _topicos(anterior.replace("kianda: ", "", 1))
                if top:
                    t.referencia_resolvida = " ".join(top[:6])
                    t.referencia_origem = anterior
                    break
            if t.referencia_resolvida:
                break

    # -- ambiguidade --------------------------------------------------------
    # Só há ambiguidade quando o turno não tem conteúdo próprio E não há
    # histórico que o resolva. Nesse caso pedir esclarecimento é o correcto.
    if (not t.tem_conteudo_proprio and not t.referencia_resolvida
            and not t.estado and not t.e_correcao and not t.e_brincadeira):
        t.ambiguidade = True

    return t
