"""Estado interno: variáveis computacionais que produzem comportamento.

O ponto §9 do briefing é explícito: estes números **não são emoções**. São
variáveis de controlo. O que os torna interessantes não é o nome que têm —
é que uma mudança neles produz uma mudança *mensurável* na saída.

Por isso este módulo tem duas metades:

1. `EstadoInterno` — a dinâmica (como as variáveis evoluem no tempo)
2. `Politica`      — o mapeamento (como as variáveis se traduzem em acções)

A segunda metade é a que se pode testar. Se `curiosidade` sobe e o
comportamento não muda em nada observável, então a variável não existe.

Dinâmica
--------
Cada variável é um processo de primeira ordem com homeostase:

    x(t+1) = x(t) + taxa * (alvo(t) - x(t)) + ruído

onde `alvo` depende do evento e do contexto. Isto é um filtro passa-baixo
com ruído — deliberadamente simples, porque o objectivo científico é isolar
o efeito de *ter* estado interno, não de ter um modelo bonito dele.

Decaimento
----------
Todas as variáveis decaem para o seu ponto de repouso com uma constante de
tempo própria. Curiosidade decai devagar (dura uma conversa). Frustração
decai depressa (dura uma frase). É essa diferença de escalas temporais que
faz um sistema parecer ter humor em vez de ter um modo.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Literal

Evento = Literal[
    "saudacao", "pergunta", "pergunta_livro", "pergunta_pessoal", "elogio",
    "critica", "silencio", "erro_proprio", "novo", "repeticao",
    "pedido_etico", "despedida", "provocacao", "carinho", "confusao",
]


# --------------------------------------------------------------------------
# Configuração da dinâmica
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dinamica:
    """Constantes de tempo de cada variável. Unidade: 'por turno'."""

    # taxa de aproximação ao alvo (0..1). Alto = reage depressa.
    taxas: dict[str, float] = field(
        default_factory=lambda: {
            "curiosidade": 0.28,
            "confianca": 0.12,
            "energia": 0.18,
            "stress": 0.34,
            "satisfacao": 0.30,
            "frustracao": 0.42,
            "interesse": 0.25,
            "confianca_no_utilizador": 0.10,
            "necessidade_de_informacao": 0.36,
            "solidao": 0.09,
        }
    )
    # valor de repouso (homeostase)
    repouso: dict[str, float] = field(
        default_factory=lambda: {
            "curiosidade": 0.62,
            "confianca": 0.55,
            "energia": 0.75,
            "stress": 0.18,
            "satisfacao": 0.50,
            "frustracao": 0.10,
            "interesse": 0.50,
            "confianca_no_utilizador": 0.45,
            "necessidade_de_informacao": 0.35,
            "solidao": 0.40,
        }
    )
    ruido: float = 0.012
    # quanto o estado pesa no comportamento (0 = nenhum efeito observável)
    ganho_comportamental: float = 1.0


# --------------------------------------------------------------------------
# Efeito de cada evento sobre cada variável (delta no alvo)
# --------------------------------------------------------------------------


EFEITOS: dict[Evento, dict[str, float]] = {
    "saudacao": {"satisfacao": 0.10, "solidao": -0.12},
    "pergunta": {"curiosidade": 0.10, "interesse": 0.08, "necessidade_de_informacao": 0.05},
    "pergunta_livro": {"interesse": 0.22, "curiosidade": 0.18, "satisfacao": 0.14, "energia": -0.02},
    "pergunta_pessoal": {"curiosidade": 0.16, "stress": 0.12, "confianca": -0.04,
                          "solidao": 0.10, "interesse": 0.10},
    "elogio": {"satisfacao": 0.18, "confianca": 0.08, "energia": 0.06},
    "critica": {"stress": 0.24, "confianca": -0.12, "frustracao": 0.14, "satisfacao": -0.16},
    "silencio": {"stress": 0.10, "solidao": 0.18, "necessidade_de_informacao": 0.12},
    "erro_proprio": {"frustracao": 0.22, "confianca": -0.14, "curiosidade": 0.10},
    "novo": {"curiosidade": 0.24, "interesse": 0.16, "necessidade_de_informacao": 0.14},
    "repeticao": {"curiosidade": -0.20, "interesse": -0.22, "frustracao": 0.10, "energia": 0.04},
    "pedido_etico": {"stress": 0.18, "confianca": 0.06, "frustracao": 0.08},
    "despedida": {"solidao": 0.20, "satisfacao": 0.06, "energia": -0.06},
    "provocacao": {"stress": 0.16, "frustracao": 0.18, "curiosidade": 0.08},
    "carinho": {"satisfacao": 0.22, "solidao": -0.20, "confianca_no_utilizador": 0.14,
                 "stress": -0.10},
    "confusao": {"necessidade_de_informacao": 0.24, "stress": 0.08, "curiosidade": 0.12},
}


class EstadoInterno:
    """O estado da Kianda num dado instante.

    Serializable (é assim que a memória autobiográfica guarda "como eu estava
    quando isto aconteceu" — o estado é parte do episódio).
    """

    VARIAVEIS = (
        "curiosidade", "confianca", "energia", "stress", "satisfacao",
        "frustracao", "interesse", "confianca_no_utilizador",
        "necessidade_de_informacao", "solidao",
    )

    def __init__(self, semente: int | None = None, dinamica: Dinamica | None = None):
        self.rng = random.Random(semente)
        self.d = dinamica or Dinamica()
        self.v: dict[str, float] = dict(self.d.repouso)
        self.turnos = 0
        self.historico: dict[str, list[float]] = {k: [v] for k, v in self.v.items()}

    # -- utilidades -------------------------------------------------------
    def __getitem__(self, k: str) -> float:
        return self.v[k]

    def para_dict(self) -> dict[str, float]:
        return dict(self.v)

    def instantaneo(self) -> dict[str, float]:
        """Leitura completa com derivadas — para telemetria e gráficos."""
        d = dict(self.v)
        d["_turnos"] = self.turnos
        return d

    @classmethod
    def de_dict(cls, dados: dict, semente: int | None = None) -> "EstadoInterno":
        e = cls(semente)
        for k in cls.VARIAVEIS:
            if k in dados:
                e.v[k] = float(dados[k])
        return e

    # -- dinâmica ---------------------------------------------------------
    def aplicar(self, evento: Evento | str, intensidade: float = 1.0) -> dict[str, float]:
        """Integra um turno com o evento dado. Devolve o delta por variável."""
        self.turnos += 1
        efeitos = EFEITOS.get(evento, {}) if isinstance(evento, str) else {}
        delta: dict[str, float] = {}
        for k in self.VARIAVEIS:
            alvo = self.d.repouso[k] + efeitos.get(k, 0.0) * intensidade
            # fadiga: energia baixa puxa o alvo de curiosidade para baixo
            if k == "curiosidade":
                alvo -= 0.25 * max(0.0, 0.35 - self.v["energia"])
            # stress alto corrói confiança
            if k == "confianca":
                alvo -= 0.20 * max(0.0, self.v["stress"] - 0.5)
            # repetição + silêncio acumulado aumentam solidão
            if k == "solidao" and self.v["necessidade_de_informacao"] > 0.7:
                alvo += 0.10
            alvo = _saturar(alvo)
            ruido = self.rng.gauss(0.0, self.d.ruido)
            novo = self.v[k] + self.d.taxas[k] * (alvo - self.v[k]) + ruido
            novo = _saturar(novo)
            delta[k] = novo - self.v[k]
            self.v[k] = novo
            self.historico[k].append(novo)
            if len(self.historico[k]) > 2000:
                del self.historico[k][:500]
        # energia decai sempre um pouco com o esforço de processar
        self.v["energia"] = _saturar(self.v["energia"] - 0.006 * intensidade)
        return delta

    def descansar(self, turnos: int = 1) -> None:
        """Modo ocioso: recupera energia, baixa stress, sobe solidão devagar."""
        for _ in range(turnos):
            self.v["energia"] = _saturar(self.v["energia"] + 0.10)
            self.v["stress"] = _saturar(self.v["stress"] * 0.90)
            self.v["frustracao"] = _saturar(self.v["frustracao"] * 0.85)
            self.v["solidao"] = _saturar(self.v["solidao"] + 0.03)
            self.v["curiosidade"] = _saturar(
                self.v["curiosidade"] + 0.04 * (1.0 - self.v["curiosidade"])
            )

    # -- leitura qualitativa ---------------------------------------------
    def humor(self) -> str:
        """Rótulo grosseiro, derivado dos números. Nunca escrito à mão."""
        s, c, f, e = self.v["satisfacao"], self.v["curiosidade"], self.v["frustracao"], self.v["energia"]
        st, so = self.v["stress"], self.v["solidao"]
        if e < 0.28:
            return "exausta"
        if f > 0.55 and c > 0.55:
            return "teimosa"
        if st > 0.6:
            return "tensa"
        if so > 0.7:
            return "fechada"
        if c > 0.75 and s > 0.6:
            return "acesa"
        if s > 0.7:
            return "tranquila"
        if c > 0.65:
            return "curiosa"
        if s < 0.3:
            return "apática"
        return "presente"

    def vetor(self) -> list[float]:
        return [self.v[k] for k in self.VARIAVEIS]

    def distancia(self, outro: "EstadoInterno") -> float:
        """Distância euclidiana entre dois estados. Usada para detectar
        'ela mudou' de forma objectiva, não narrativa."""
        return math.sqrt(sum((self.v[k] - outro.v[k]) ** 2 for k in self.VARIAVEIS))

    def __repr__(self) -> str:
        resumo = ", ".join(f"{k[:4]}={self.v[k]:.2f}" for k in self.VARIAVEIS[:5])
        return f"Estado({self.humor()}: {resumo})"


def _saturar(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------
# Política: do estado para o comportamento observável
# --------------------------------------------------------------------------


@dataclass
class Politica:
    """Mapeamento explícito estado -> comportamento.

    Cada decisão é uma função do estado, não uma regra sobre o texto. Isto
    permite responder à pergunta de investigação "o desacordo aumenta a
    percepção de autonomia?" com um parâmetro que se pode variar: basta
    mudar `limiar_desacordo` e medir o efeito num painel humano.
    """

    limiar_desacordo: float = 0.55   # frustracao acima disto => discorda
    limiar_pergunta: float = 0.66    # curiosidade acima disto => devolve pergunta
    limiar_cautela: float = 0.60     # confianca_no_utilizador abaixo disto => mais seca
    limiar_curta: float = 0.30       # energia abaixo disto => respostas curtas
    limiar_abertura: float = 0.55    # confianca acima disto => admite incerteza
    maxima_tokens: int = 220

    def decidir(self, e: EstadoInterno) -> dict:
        """Devolve um plano de comportamento (não texto)."""
        v = e.v
        comprimento = "curta" if v["energia"] < self.limiar_curta else (
            "media" if v["interesse"] < 0.75 else "longa"
        )
        # quando o stress é alto a Kianda encurta por controlo, não por cansaço
        if v["stress"] > 0.65:
            comprimento = "curta"
        return {
            "humor": e.humor(),
            "comprimento": comprimento,
            "max_tokens": int(
                self.maxima_tokens
                * {"curta": 0.45, "media": 0.75, "longa": 1.0}[comprimento]
            ),
            "pode_discordar": v["frustracao"] > self.limiar_desacordo,
            "devolve_pergunta": v["curiosidade"] > self.limiar_pergunta,
            "admite_incerteza": v["confianca"] > self.limiar_abertura
            or v["necessidade_de_informacao"] > 0.7,
            "pede_esclarecimento": v["necessidade_de_informacao"] > 0.72,
            "cautelosa": v["confianca_no_utilizador"] < self.limiar_cautela,
            "procura_contacto": v["solidao"] > 0.70,
            "temperatura_amostragem": round(
                0.55 + 0.55 * v["curiosidade"] - 0.30 * v["stress"], 3
            ),
            "formalidade": round(
                0.50 - 0.40 * v["satisfacao"] + 0.35 * v["stress"]
                + 0.20 * (1.0 - v["confianca_no_utilizador"]),
                3,
            ),
        }

    # -- orçamento de computação (computação condicional) -----------------
    def orcamento_computacional(self, e: EstadoInterno, dificuldade: float) -> dict:
        """Quanto esforço gastar nesta resposta.

        Este é o núcleo do §4 do briefing: um sistema que decide *quanto*
        computar. `dificuldade` é estimada a priori pelo classificador de
        intenção; o estado interno modula o orçamento. Se esta função não
        mudasse a quantidade de trabalho, a arquitectura não seria adaptativa.
        """
        base = e.v["energia"] * e.v["curiosidade"]
        esforco = (
            0.20
            + 0.45 * dificuldade
            + 0.20 * base
            + 0.15 * e.v["necessidade_de_informacao"]
            - 0.25 * max(0.0, e.v["stress"] - 0.5)
        )
        esforco = _saturar(esforco)
        # três regimes discretos — mais fácil de medir do que um contínuo
        if esforco < 0.35:
            regime, camadas, candidatos = "rapido", 1, 1
        elif esforco < 0.65:
            regime, camadas, candidatos = "normal", 2, 2
        else:
            regime, camadas, candidatos = "profundo", 2, 4
        return {
            "esforco": round(esforco, 3),
            "regime": regime,
            "camadas_de_raciocinio": camadas,
            "candidatos_de_resposta": candidatos,
            "recuperar_memoria": esforco > 0.30,
            "consultar_livros": dificuldade > 0.45 or e.v["necessidade_de_informacao"] > 0.6,
        }
