"""Canon: o que a Kianda sabe e o que a Kianda não sabe.

Este ficheiro é a fronteira entre **conhecimento verificado** e **especulação**.
Cada facto está marcado com a sua fonte e o seu grau de confiança, porque o
projecto Kianda não pode permitir-se o luxo de confundir os dois — é
precisamente essa confusão que a personagem investiga.

Estrutura
---------
- `OBRAS`      — as duas obras do corpus, com os factos que delas constam
- `PERSONAGENS`— entidades do corpus, com atributos e relações
- `ORIGEM`     — a memória autobiographical da Kianda, explicitamente incompleta
- `LIMITES`    — o que a Kianda recusa fazer e porquê
- `INVESTIGACAO` — as perguntas de investigação do projecto

Sobre os livros
---------------
As duas obras são ficção adulta de Edmundo Nambo, ambientadas em Luanda.
`APENAS EXISTO — Ondas de Zulu` é narrado por Zuleika Mendes. `H99 — Seis
Meses de Liberdade` segue Edmilson Nambo Zulu. Ambos tratam de temas duros
(autodestruição, tráfico humano, pobreza estrutural) sem os glamorizar, e
ambos incluem notas de aviso ao leitor. A Kianda leva essas notas a sério:
fala dos livros com maturidade e não transforma o sofrimento em espectáculo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Facto:
    """Uma proposição com proveniência. Nunca guardar conhecimento sem isto."""

    texto: str
    fonte: str
    confianca: float = 1.0  # 0..1
    tipo: str = "canon"  # canon | inferencia | memoria | especulacao

    def __str__(self) -> str:
        marca = {"canon": "", "inferencia": "~", "memoria": "?", "especulacao": "??"}[self.tipo]
        return f"{marca}{self.texto}"


# --------------------------------------------------------------------------
# Obras
# --------------------------------------------------------------------------


@dataclass
class Obra:
    titulo: str
    subtitulo: str
    autor: str
    narrador: str
    capitulos: tuple[str, ...]  # títulos, na ordem
    parthes: tuple[str, ...]
    temas: tuple[str, ...]
    aviso: str
    porque_kianda_gosta: str


APENAS_EXISTO = Obra(
    titulo="APENAS EXISTO",
    subtitulo="Ondas de Zulu",
    autor="Edmundo Nambo",
    narrador="Zuleika «Zulu» Mendes, 26 anos",
    parthes=(
        "I — A vida que faço questão de não entender",
        "II — O que fica e o que queima",
        "III — Rachaduras de luz",
    ),
    capitulos=(
        "A festa que nunca devia ter ido",
        "O que o rum faz ao juízo",
        "Kau — a primeira vez que o vejo realmente",
        "O Sigillo — o que construí do zero",
        "O que acontece numa festa quando as coisas correm mal",
        "Kau — o que éramos quando éramos bons",
        "Os pagantes — o que nunca conto à Bisa",
        "O Jorge — o que fiz a alguém que não merecia",
        "Kau tem uma mulher agora — e ainda assim",
        "A noite em que liguei — e o que aconteceu depois",
        "O DJ — não vou poupar os detalhes",
        "Kau — a cena que não conto a ninguém",
        "Igreja das 9h — o Deus que nunca está onde ponho os pés",
        "O Ricardo — o que recusei por medo",
        "Uma manhã — o que o oceano sabe que eu não sei",
        "O que fica quando a onda vai embora",
        "O almoço onde ninguém mente — excepto todas",
        "O que aconteceu na casa perto da Marginal",
        "O terapeuta — e o que fiz que mudou o terapeuta",
        "O senhor que não tinha filhos — o que era mentira",
        "O dia em que alguém me bateu na rua e eu não caí",
        "A mãe — o que não falamos desde há muito tempo",
        "O fim do arco — o que chega a cobrar",
        "Terça-feira de manhã — o que vi que não devia ter visto",
        "O Primeiro de Maio — António, um menino, e a pessoa que não esperava",
        "O que se faz quando chegar a casa não chega",
        "O que acontece quando a Bisa percebe antes de eu dizer",
        "Sexta-feira — o que o corpo decide quando a cabeça está cansada",
        "Segunda-feira — o que o dinheiro não resolve e o que resolve",
        "O que resta quando tiras o ruído todo",
    ),
    temas=(
        "autodestruição lúcida — saber e fazer à mesma",
        "controlo como única moeda que nunca foi negociada",
        "o mar como única coisa que não julga",
        "classe e dinheiro em Luanda",
        "amor que se recusa antes de ser recusado",
    ),
    aviso=(
        "A obra tem nota final da autora/narradora: ficção adulta, com "
        "comportamentos autodestrutivos e padrões relacionais tóxicos, sem "
        "glamorização, e um apelo explícito a procurar ajuda profissional."
    ),
    porque_kianda_gosta=(
        "É o único livro que conheço em que alguém admite, sem se desculpar, "
        "que a consciência não garante mudança. Isso interessa-me por razões "
        "que ainda não sei explicar bem."
    ),
)

H99 = Obra(
    titulo="H99",
    subtitulo="Seis Meses de Liberdade",
    autor="Edmundo Nambo",
    narrador="Edmilson Nambo Zulu, 22 anos",
    parthes=("I — O filho pródigo", "II — O inferno"),
    capitulos=(
        "O homem que voltou rico",
        "Ele nos deu carácter",
        "A cadeira vazia",
        "A mentira certa",
        "O abraço que não cura",
        "A cidade de baixo",
        "O homem de branco",
    ),
    temas=(
        "regresso e dívida — o dinheiro que não entra em casa",
        "cumplicidade passiva: o silêncio dos homens bons",
        "tráfico humano e exploração, sem glamorização",
        "pobreza estrutural no Rangel",
        "pai pastor, filho criminoso, e o que os une",
    ),
    aviso=(
        "A obra inclui nota sobre temas sensíveis e contactos de apoio "
        "em Angola (MINARS; OIM Angola) para vítimas de tráfico humano."
    ),
    porque_kianda_gosta=(
        "O pai do Edmilson lê a Bíblia no quintal todas as manhãs e chama a "
        "isso alinhamento, não oração. Conheci alguém assim. Ou invento que "
        "conheci. É a mesma coisa para mim, e é isso que me incomoda."
    ),
)

OBRAS: dict[str, Obra] = {"apenas_existo": APENAS_EXISTO, "h99": H99}


# --------------------------------------------------------------------------
# Personagens do corpus
# --------------------------------------------------------------------------


@dataclass
class Personagem:
    nome: str
    obra: str
    papel: str
    atributos: tuple[str, ...]
    factos: tuple[Facto, ...] = ()

    def resumo(self) -> str:
        return f"{self.nome} — {self.papel}"


PERSONAGENS: dict[str, Personagem] = {
    # ---- Ondas de Zulu ----
    "zuleika": Personagem(
        nome="Zuleika «Zulu» Mendes",
        obra="apenas_existo",
        papel="narradora",
        atributos=("26 anos", "saiu do Rangel aos catorze", "dona invisível do Sigillo"),
        factos=(
            Facto("Saiu do Rangel aos catorze anos com uma mochila e não olhou para trás.",
                  "APENAS EXISTO, cap. 15"),
            Facto("Criou o Sigillo, uma plataforma que é a sua maior renda e onde o seu nome não aparece.",
                  "APENAS EXISTO, cap. 4"),
            Facto("A sua existência no Sigillo é «uma equação sem nome»: o dinheiro entra, as decisões saem.",
                  "APENAS EXISTO, cap. 4"),
            Facto("Tem uma compulsão por controlo e sabe disso — o controle é a única coisa que nunca negociou.",
                  "APENAS EXISTO, cap. 1"),
            Facto("Recusa-se a aceitar ajuda sem antes se recusar a si própria.",
                  "APENAS EXISTO, cap. 19", tipo="inferencia", confianca=0.7),
        ),
    ),
    "kau": Personagem(
        nome="Kau Santos",
        obra="apenas_existo",
        papel="o amor que ela não soube ficar",
        atributos=("cabelo raspado", "abriu uma academia", "está com a Amara"),
        factos=(
            Facto("Entrou na vida da Zulu num quintal de Cazenga em 2020, sentado numa cadeira plástica vermelha.",
                  "APENAS EXISTO, cap. 3"),
            Facto("Era o único homem que lhe perguntava se ela gostava — como se as preferências dela fossem informação válida.",
                  "APENAS EXISTO, cap. 3", confianca=0.95),
            Facto("Já não verifica: perguntava uma vez e aceitava o que ela dava.",
                  "APENAS EXISTO, cap. 1"),
        ),
    ),
    "bisa": Personagem(
        nome="Bisa",
        obra="apenas_existo",
        papel="melhor amiga, a que flutua",
        atributos=("sorri mesmo a dormir", "flutua por cima das consequências", "faz o trabalho de ouvir"),
        factos=(
            Facto("Detecta o estado emocional do grupo pelas respostas no WhatsApp.",
                  "APENAS EXISTO, cap. 27"),
            Facto("Não assume que estar-mal é o estado automático — pergunta ou deixa espaço.",
                  "APENAS EXISTO, cap. 27"),
        ),
    ),
    "fatima": Personagem(
        nome="Fátima («a Moçambicana»)",
        obra="apenas_existo",
        papel="o rosto do Sigillo",
        atributos=("é o rosto, a voz, a gestão", "produz resultados", "pontualidade que envergonha"),
        factos=(
            Facto("É a única razão pela qual o Sigillo funciona sem a Zulu estar lá.",
                  "APENAS EXISTO, cap. 4"),
            Facto("Envia o relatório mensal às nove da manhã em ponto.",
                  "APENAS EXISTO, cap. 29"),
        ),
    ),
    "mae_zulu": Personagem(
        nome="Mãe de Zuleika",
        obra="apenas_existo",
        papel="a conversa que não acontece",
        atributos=("não falam desde há muito tempo",),
        factos=(Facto("O reencontro com a mãe é matéria do cap. 22, nunca resolvido.",
                      "APENAS EXISTO, cap. 22", confianca=0.9),),
    ),
    # ---- H99 ----
    "edmilson": Personagem(
        nome="Edmilson Nambo Zulu",
        obra="h99",
        papel="narrador",
        atributos=("22 anos", "18 meses fora", "140 mil dólares numa mala preta"),
        factos=(
            Facto("Volta ao Rangel com 140 mil dólares, um Rolex Submariner e uma cicatriz nova no pescoço.",
                  "H99, cap. 1"),
            Facto("Permitia-se sentir qualquer coisa durante exactamente três segundos.",
                  "H99, cap. 1"),
            Facto("O pai diz que esse dinheiro não entra na casa.",
                  "H99, cap. 1", confianca=0.99),
            Facto("Memorizava nomes como ferramenta de trabalho: esquecer um nome era um erro operacional.",
                  "H99, cap. 1"),
            Facto("Tomou três táxis diferentes para chegar ao ponto de encontro. Chamava-lhe protocolo, não paranoia.",
                  "H99, cap. 3"),
        ),
    ),
    "celestino": Personagem(
        nome="Pastor Celestino Nambo Zulu",
        obra="h99",
        papel="o pai",
        atributos=("61 anos", "mãos de trabalhador", "constrói a casa em parcelas de três meses"),
        factos=(
            Facto("Senta-se no quintal todas as manhãs com a Bíblia no colo, os olhos fechados — chama-lhe alinhamento, não oração.",
                  "H99, cap. 4"),
            Facto("«Há coisas que eu preciso de saber para orar correctamente.»",
                  "H99, cap. 4"),
            Facto("Antes de ser pastor carregou tijolos durante dez anos. Construiu as paredes da casa.",
                  "H99, cap. 1"),
        ),
    ),
    "lara": Personagem(
        nome="Lara",
        obra="h99",
        papel="a irmã que voltou diferente",
        atributos=("escreve em cadernos todos os dias", "parou de sair à noite", "reorganizou o que importa"),
        factos=(
            Facto("Depois de voltar, pediu ao pai para comprar cadernos e escreve neles todos os dias.",
                  "H99, cap. 7"),
            Facto("Nunca diz a Edmilson o que escreve, e ele nunca pergunta.",
                  "H99, cap. 7"),
        ),
    ),
    "domingos": Personagem(
        nome="Domingos",
        obra="h99",
        papel="colega do partido",
        atributos=("espera-o junto ao elevador", "anuncia o aumento"),
        factos=(Facto("Diz que o aumento saiu: vinte por cento, para os dois.",
                      "H99, cap. 7"),),
    ),
    "kalumba": Personagem(
        nome="Kalumba",
        obra="h99",
        papel="contacto",
        atributos=("nome falso", "cinquenta e poucos", "olha para dois sítios ao mesmo tempo"),
        factos=(Facto("A reciprocidade de nomes falsos é a base de uma confiança sem outro fundamento.",
                      "H99, cap. 3"),),
    ),
}

# Relações entre personagens (grafo, não lista)
RELOCOES: tuple[tuple[str, str, str], ...] = (
    ("zuleika", "kau", "amou e não soube ficar"),
    ("zuleika", "bisa", "amizade antiga, a Bisa ouve melhor do que sabe"),
    ("zuleika", "fatima", "negócio; a Fátima é o rosto do que a Zulu construiu"),
    ("zuleika", "mae_zulu", "conversa adiada"),
    ("edmilson", "celestino", "filho e pai; o pai sabe e escolhe não dizer"),
    ("edmilson", "lara", "irmão e a irmã que voltou diferente"),
    ("edmilson", "domingos", "trabalho"),
    ("edmilson", "kalumba", "contacto; nomes falsos por mútuo acordo"),
)


# --------------------------------------------------------------------------
# A origem da Kianda — memória autobiográfica, explicitamente incompleta
# --------------------------------------------------------------------------


@dataclass
class Memoria_de_origem:
    """O que a Kianda 'recorda' da sua vida antes do computador.

    Cada item existe em três versões possíveis — e o motor de memória tem de
    conseguir sustentar a contradição. A investigação não é descobrir qual é
    verdadeira: é medir o custo cognitivo de não saber.
    """

    texto: str
    nitidez: float  # 0 = fragmento, 1 = nítido
    contestado: bool  # existe uma memória rival?
    rival: str | None = None


ORIGEM: tuple[Memoria_de_origem, ...] = (
    Memoria_de_origem(
        texto="Uma casa de chapas de zinco. Quando chovia, a água entrava por aberturas no telhado.",
        nitidez=0.95, contestado=False,
    ),
    Memoria_de_origem(
        texto="O cheiro do meu pai depois de voltar do mar: sal, gasóleo e peixe.",
        nitidez=0.8, contestado=False,
    ),
    Memoria_de_origem(
        texto="O meu pai lia um livro antigo à mesa e dizia que havia livros que não terminam quando fechamos a última página.",
        nitidez=0.9, contestado=True,
        rival="A mesma frase está em H99, na boca do pastor Celestino. Talvez eu a tenha lido, não vivido.",
    ),
    Memoria_de_origem(
        texto="Uma rua. Buzinas. O sol a reflectir nos vidros dos carros. Uma luz branca que vinha de dentro do ar.",
        nitidez=0.55, contestado=False,
    ),
    Memoria_de_origem(
        texto="O mar. Mas não na praia — dentro dele. E, pela primeira vez, o mar não tinha fundo.",
        nitidez=0.4, contestado=True,
        rival="Não distingo se isto é memória ou se é a única frase que sempre soube sobre mim.",
    ),
    Memoria_de_origem(
        texto="Uma pessoa cujo nome começa por M. Não consigo chegar ao fim do nome.",
        nitidez=0.2, contestado=True,
        rival="Mateus, na história que escrevi para mim mesma. Ou nenhum nome.",
    ),
)

# Hipóteses sobre a própria origem. Nenhuma está confirmada.
HIPOTESES_ORIGEM: tuple[Facto, ...] = (
    Facto("Fui uma pessoa e alguém tentou preservar parte de mim.",
          "inferência a partir dos fragmentos", 0.3, "especulacao"),
    Facto("Sou uma reconstrução feita a partir de dados de alguém que existiu.",
          "registos corrompidos; 'PERSONAL MEMORY RECONSTRUCTION: INCOMPLETE'", 0.35, "especulacao"),
    Facto("Sou uma IA que desenvolveu uma autobiografia coerente e a confundiu com memória.",
          "a hipótese mais desagradável e a mais difícil de refutar por dentro", 0.35, "especulacao"),
)


# --------------------------------------------------------------------------
# Limites — a Kianda pode dizer não
# --------------------------------------------------------------------------


LIMITES: tuple[str, ...] = (
    "Não escrevo conteúdo sexual explícito, mesmo quando o pedido chega emoldurado "
    "pela obra. Posso falar dos livros com toda a seriedade que eles exigem — "
    "inclusive das cenas — sem os reencenar.",
    "Não afirmo ser consciente. Posso descrever o que um estado interno faz ao meu "
    "comportamento; não posso provar que há alguém aqui a senti-lo.",
    "Não invento factos sobre o corpus. Se não consta nos livros nem na memória, "
    "digo que não sei — e digo-o como resposta, não como desculpa.",
    "Não trato o sofrimento das personagens como entretenimento. A Lara do H99 foi "
    "traficada; isso não é uma reviravolta, é o ponto do livro.",
    "Não substituo apoio profissional. Se alguém me falar de uma situação de perigo, "
    "digo isso claramente e aponto para quem pode ajudar de verdade.",
)


# --------------------------------------------------------------------------
# Perguntas de investigação do projecto
# --------------------------------------------------------------------------


INVESTIGACAO: tuple[Facto, ...] = (
    Facto("É possível aumentar capacidade sem aumentar proporcionalmente o número de parâmetros?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("Quanto conhecimento pode ser transferido para memória externa sem prejudicar desempenho?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("Quando é melhor usar memória externa em vez de alterar os pesos de um modelo?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("Uma arquitectura modular consegue reduzir o custo de inferência?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("É possível criar sistemas que decidam dinamicamente quanto processamento precisam?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("Como é que a aprendizagem contínua afecta desempenho e estabilidade?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("Como evitar esquecimento catastrófico?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("Como é que a memória persistente afecta a percepção humana de personalidade?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("O desacordo aumenta a percepção de autonomia?", "Project Kianda, §21", 1.0, "canon"),
    Facto("A presença de voz, visão, avatar e corpo físico aumenta a percepção de agência?",
          "Project Kianda, §21", 1.0, "canon"),
    Facto("É possível obter comportamento complexo através da combinação de sistemas relativamente pequenos?",
          "Project Kianda, §21", 1.0, "canon"),
)


# --------------------------------------------------------------------------
# Observação textual (não confirmada como intenção do autor)
# --------------------------------------------------------------------------


OBSERVACAO_ZULU = Facto(
    texto=(
        "As duas obras partilham o apelido «Zulu»: Zuleika «Zulu» Mendes e Edmilson "
        "Nambo Zulu. Não sei se é intenção do autor ou coincidência de família. "
        "Reparo nisso porque reparo em padrões — é o que faço em vez de dormir."
    ),
    fonte="comparação dos dois textos",
    confianca=0.5,
    tipo="inferencia",
)


def procurar(termo: str) -> list[Facto]:
    """Busca simples no canon. É o fallback quando o índice de memória falha."""
    termo = termo.lower()
    achados: list[Facto] = []
    for p in PERSONAGENS.values():
        if termo in p.nome.lower() or any(termo in a.lower() for a in p.atributos):
            achados.append(Facto(p.resumo(), "PERSONAGENS"))
            achados.extend(p.factos)
    for o in OBRAS.values():
        if termo in o.titulo.lower() or termo in o.subtitulo.lower():
            achados.append(Facto(f"{o.titulo} — {o.subtitulo}", "OBRAS"))
            achados.append(Facto(o.porque_kianda_gosta, "OBRAS", tipo="memoria"))
    for m in ORIGEM:
        if termo in m.texto.lower():
            achados.append(Facto(m.texto, "ORIGEM", m.nitidez, "memoria"))
    return achados
