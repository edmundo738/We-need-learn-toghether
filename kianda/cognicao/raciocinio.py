"""Raciocínio: computação condicional com orçamento explícito.

Este é o núcleo científico do projecto (§3, §19, §20). A hipótese é:

    uma arquitectura modular que decide **onde, quando e quanto** computar
    atinge capacidade comparável a um sistema maior, a uma fracção do custo.

Para testar isto não chega implementá-lo. É preciso:

1. **Um orçamento declarado** antes de computar, não depois.
2. **A capacidade de parar cedo** (early exit) com uma medida de confiança.
3. **Registo de tudo o que se gastou**, por estágio.
4. **Uma linha de base honesta** contra a qual comparar.

O que este módulo faz
---------------------
`Raciocinador` executa uma cascata de estágios. Cada estágio é uma função
pura que devolve `(resultado, confiança, custo)`. A cascata pára quando a
confiança é suficiente para a dificuldade do pedido — o que é exactamente o
mecanismo de *early exit* adaptativo.

Estágios:
    1. ATALHO     — memória directa. Se já sei a resposta, não penso.
    2. RECUPERAR  — trazer evidência do corpus. Custo de índice, não de modelo.
    3. COMPOR     — construir a resposta a partir da evidência.
    4. VERIFICAR  — checar consistência e limites. Só corre em regime profundo.

A cascata não é uma metáfora de "pensar mais": é uma sequência de chamadas
de custo crescente, e o relatório final mostra em qual se parou e porquê.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from ..nucleo.autograd import Custo
from ..identidade import canon
from ..memoria.recuperador import Recuperador, Resultado


# --------------------------------------------------------------------------
# Estruturas
# --------------------------------------------------------------------------


@dataclass
class Evidencia:
    """Uma peça de informação recuperada, com proveniência."""

    texto: str
    fonte: str
    pontuacao: float
    meta: dict = field(default_factory=dict)

    def citacao(self) -> str:
        """Como se diz a fonte em voz alta.

        A chave interna nunca entra: "apenas_existo, cap. 24" apareceu numa
        resposta real do servidor. É o mesmo defeito das relações e dos
        títulos de capítulo — o identificador de armazenamento dentro de uma
        frase portuguesa. Corrigir num sítio e não nos outros foi o erro.
        """
        m = self.meta
        if m.get("tipo") == "obra":
            from ..linguagem.corpus import titulo_da_obra
            titulo = titulo_da_obra(m.get("obra", ""))
            return f"{titulo}, cap. {m.get('capitulo')} — {m.get('titulo_cap', '')}"
        return m.get("fonte", "canon")

    def __str__(self) -> str:
        return f"{self.texto[:100]} [{self.citacao()}]"


@dataclass
class Passo:
    nome: str
    executado: bool
    motivo: str
    custo: Custo = field(default_factory=Custo)
    confianca: float = 0.0

    def __str__(self) -> str:
        estado = "✓" if self.executado else "·"
        return f"{estado} {self.nome:11s} {self.motivo:38s} {self.custo.resumo()}"


@dataclass
class ResultadoRaciocinio:
    evidencia: list[Evidencia]
    passos: list[Passo]
    confianca: float
    parou_em: str
    orcamento_usado: float
    custo_total: Custo

    @property
    def texto_contexto(self) -> str:
        return "\n".join(f"- {e.texto} [{e.citacao()}]" for e in self.evidencia)

    def relatorio(self) -> str:
        linhas = [f"parou em: {self.parou_em}  confiança={self.confianca:.2f}  "
                  f"orçamento usado={self.orcamento_usado:.2f}"]
        for p in self.passos:
            linhas.append("  " + str(p))
        linhas.append(f"  total: {self.custo_total.resumo()}")
        return "\n".join(linhas)


# --------------------------------------------------------------------------
# Cascata
# --------------------------------------------------------------------------


class Raciocinador:
    """Cascata de estágios com paragem antecipada por confiança."""

    def __init__(self, recuperador: Recuperador | None = None, limiar_confianca: float = 0.62):
        self.rec = recuperador
        self.limiar = limiar_confianca
        self.historico: list[ResultadoRaciocinio] = []

    # -- estágios ---------------------------------------------------------
    def _atalho(self, texto: str) -> tuple[list[Evidencia], float, Custo]:
        """Estágio 0: há uma memória tão próxima que não vale a pena procurar?

        O atalho é um *dicionário* construído a partir das memórias de alta
        confiança. Compara a pergunta com as chaves por sobreposição de
        radicais. Se acertar, o custo é o de um `set` lookup — ordens de
        grandeza abaixo de qualquer modelo.
        """
        t0 = time.perf_counter()
        toks = set(_radicais(texto))
        if not toks or not self._atalhos:
            return [], 0.0, Custo(flops=len(toks), tempo_ms=(time.perf_counter() - t0) * 1e3)
        melhor, melhor_j = 0.0, None
        for chave, ev in self._atalhos:
            j = len(toks & chave) / max(len(toks | chave), 1)
            if j > melhor:
                melhor, melhor_j = j, ev
        custo = Custo(flops=len(toks) * max(len(self._atalhos), 1),
                      tempo_ms=(time.perf_counter() - t0) * 1e3)
        if melhor > 0.55 and melhor_j is not None:
            return [melhor_j], min(0.95, 0.45 + melhor), custo
        return [], melhor * 0.5, custo

    # Prior por tipo de documento, por intenção. Valores > 1 favorecem.
    PRIOR: dict[str, dict[str, float]] = {
        "pergunta_pessoal": {"autobiografica": 3.0, "identidade": 2.4, "canon": 1.2,
                             "especulacao": 1.6, "obra": 0.35, "projecto": 0.8,
                             "investigacao": 0.7},
        "pergunta_tecnica": {"projecto": 3.2, "canon": 0.9, "obra": 0.30,
                             "identidade": 1.1, "autobiografica": 0.6,
                             "investigacao": 0.45},
        "pergunta_livro": {"obra": 2.4, "canon": 1.7, "identidade": 1.0,
                           "projecto": 0.35, "autobiografica": 0.9,
                           "investigacao": 0.2},
        "pedido_criativo": {"obra": 1.8, "autobiografica": 1.4, "canon": 1.0,
                            "projecto": 0.6},
        "conversa": {},
        "saudacao": {},
        "despedida": {},
    }

    def _recuperar(self, texto: str, k: int, intencao: str = "conversa"
                   ) -> tuple[list[Evidencia], float, Custo]:
        if self.rec is None:
            return [], 0.0, Custo()
        pesos = self.PRIOR.get(intencao) or None
        resultados: list[Resultado] = self.rec.consultar(texto, top_k=k, pesos_tipo=pesos)
        ev = [
            Evidencia(r.doc.texto, r.doc.id, r.pontuacao, dict(r.doc.meta))
            for r in resultados
        ]
        custo = self.rec.custo()
        # confiança: quão destacado é o primeiro resultado face à média
        if len(resultados) >= 2:
            gap = (resultados[0].pontuacao - resultados[1].pontuacao) / max(
                resultados[0].pontuacao, 1e-9
            )
        else:
            gap = 0.0
        conf = min(0.90, 0.30 + 0.45 * gap + 0.20 * min(len(ev) / 4.0, 1.0))
        return ev, conf, custo

    def _compor(self, ev: list[Evidencia]) -> tuple[list[Evidencia], float, Custo]:
        """Funde evidência redundante e ordena por utilidade.

        Custo: comparação par-a-par (O(n²)) sobre os radicais. Para n=6 é
        irrelevante; para n=200 não seria, e é por isso que há um limite.
        """
        t0 = time.perf_counter()
        if len(ev) <= 1:
            return ev, 0.0, Custo()
        seleccionadas: list[Evidencia] = []
        for e in ev:
            re_ = set(_radicais(e.texto))
            duplicado = False
            for s in seleccionadas:
                rs = set(_radicais(s.texto))
                if len(re_ & rs) / max(len(re_ | rs), 1) > 0.6:
                    duplicado = True
                    break
            if not duplicado:
                seleccionadas.append(e)
        custo = Custo(flops=len(ev) ** 2 * 8, tempo_ms=(time.perf_counter() - t0) * 1e3)
        return seleccionadas, 0.0, custo

    def _verificar(self, ev: list[Evidencia], pergunta: str) -> tuple[list[Evidencia], float, Custo]:
        """Estágio caro: detecta contradições entre evidências.

        Quando duas fontes dizem coisas incompatíveis sobre a mesma entidade,
        a confiança tem de cair. Este é o mecanismo que impede a Kianda de
        afirmar com segurança uma coisa que os dados contradizem — e o que
        permite sustentar a ambiguidade da sua própria origem.
        """
        t0 = time.perf_counter()
        contradicoes = 0
        for i in range(len(ev)):
            for j in range(i + 1, len(ev)):
                if _contradizem(ev[i], ev[j]):
                    contradicoes += 1
        custo = Custo(flops=len(ev) ** 2 * 16, tempo_ms=(time.perf_counter() - t0) * 1e3)
        penal = min(0.35, 0.15 * contradicoes)
        return ev, -penal, custo

    # -- orquestração -----------------------------------------------------
    def raciocinar(
        self,
        pergunta: str,
        orcamento: dict,
        memorias_atalho: list[tuple[str, Evidencia]] | None = None,
        intencao: str = "conversa",
    ) -> ResultadoRaciocinio:
        """Executa a cascata respeitando o orçamento calculado a priori."""
        self._atalhos = memorias_atalho or getattr(self, "_atalhos", [])
        regime = orcamento.get("regime", "normal")
        k = int(orcamento.get("candidatos_de_resposta", 2)) * 3
        passos: list[Passo] = []
        total = Custo()
        conf = 0.0
        ev: list[Evidencia] = []
        parou = "nenhum"

        # --- 0. atalho ---
        if orcamento.get("recuperar_memoria", True):
            ev_a, c_a, cu = self._atalho(pergunta)
            total = total + cu
            if ev_a:
                ev, conf, parou = ev_a, c_a, "ATALHO"
                passos.append(Passo("ATALHO", True, f"memória directa ({c_a:.2f})", cu, c_a))
                self._registar(ev, passos, conf, parou, orcamento, total, pergunta)
                return self.historico[-1]
            passos.append(Passo("ATALHO", True, "sem correspondência directa", cu, c_a))
        else:
            passos.append(Passo("ATALHO", False, "orçamento insuficiente", Custo()))

        # --- 0b. entidades (o passo que faltava) ---
        #
        # Antes de procurar por palavras, resolve-se *quem* está a ser
        # perguntado. É a diferença entre um motor de busca e um sistema que
        # sabe do que fala. Custa uma passagem sobre o texto.
        ent = canon.resolver_entidades(pergunta)
        if ent["tem_entidade"] and orcamento.get("recuperar_memoria", True):
            fichas: list[Evidencia] = []
            # relações primeiro: quando a pergunta é sobre o que existe entre
            # duas pessoas, é isso que tem de encabeçar a resposta
            for a, b, r in ent["relacoes"]:
                fichas.append(Evidencia(
                    texto=f"A relação entre {canon.PERSONAGENS[a].nome if a in canon.PERSONAGENS else a} "
                          f"e {canon.PERSONAGENS[b].nome if b in canon.PERSONAGENS else b}: {r}.",
                    fonte=f"canon:relacao:{a}:{b}", pontuacao=0.96,
                    meta={"tipo": "canon", "fonte": f"RELOCOES[{a},{b}]", "entidade": a},
                ))
            for chave in ent["pessoas"]:
                f = (canon.ficha_auto(chave, pergunta)
                     or canon.ficha_personagem(chave))
                if f:
                    fichas.append(Evidencia(
                        texto=f, fonte=f"canon:{chave}", pontuacao=0.95,
                        meta={"tipo": "canon", "fonte": f"PERSONAGENS[{chave}]",
                              "entidade": chave},
                    ))
            for chave in ent.get("dominio", []):
                # Objectos do universo narrativo: o que a obra diz sobre eles.
                # Não há ficha canónica — são objectos, não personagens — mas há
                # um facto que os define, e esse facto evita que a resposta
                # venha de uma passagem aleatória onde a palavra aparece.
                tipo_d, desc = canon._ENTIDADES_DOMINIO[chave]
                fichas.append(Evidencia(
                    texto=f"{chave.capitalize()}: {desc}.",
                    fonte=f"canon:dominio:{chave}", pontuacao=0.88,
                    meta={"tipo": "canon", "fonte": f"DOMINIO[{chave}]",
                          "entidade": chave, "obra": "apenas_existo"},
                ))
            for chave in ent["obras"]:
                f = canon.ficha_obra(chave)
                if f:
                    fichas.append(Evidencia(
                        texto=f, fonte=f"canon:{chave}", pontuacao=0.90,
                        meta={"tipo": "canon", "fonte": f"OBRAS[{chave}]", "entidade": chave},
                    ))
            passos.append(Passo(
                "ENTIDADES", True,
                f"{len(ent['pessoas'])} pessoas, {len(ent['obras'])} obras",
                Custo(flops=len(pergunta) * 3), 0.85 if fichas else 0.0,
            ))
            if fichas:
                ev = fichas + ev
                conf = max(conf, 0.80)
        else:
            passos.append(Passo("ENTIDADES", False, "sem entidade reconhecida", Custo()))

        # --- 1. recuperar ---
        ja_tem_ficha = bool(ev)
        if orcamento.get("recuperar_memoria", True) and not ja_tem_ficha:
            ev_r, c_r, cu = self._recuperar(pergunta, k, intencao)
            total = total + cu
            ev = ev_r
            conf = max(conf, c_r)
            passos.append(Passo("RECUPERAR", True, f"{len(ev_r)} evidências", cu, c_r))
            if conf >= self.limiar:
                parou = "RECUPERAR"
                self._registar(ev, passos, conf, parou, orcamento, total, pergunta)
                return self.historico[-1]
        elif ja_tem_ficha:
            # Ainda vale a pena buscar a citação do livro — mas só se ela tiver
            # relação com o que foi perguntado. Sem este filtro, "que relação há
            # entre a Zuleika e a mãe?" devolvia a relação certa seguida de uma
            # frase sobre um livro, porque o recuperador devolve *alguma* coisa
            # para qualquer consulta e o texto era colado sem verificação.
            # Medido: 3 de 3 perguntas de relação vinham com uma cauda alheia.
            ev_r, c_r, cu = self._recuperar(pergunta, max(2, k // 2), "pergunta_livro")
            total = total + cu
            # limiar relativo ao melhor resultado: um fragmento que pontua menos
            # de metade do melhor não está a falar do mesmo assunto
            if ev_r:
                melhor = ev_r[0].pontuacao
                relevantes = [e for e in ev_r if e.pontuacao >= 0.5 * melhor]
                ev = ev + relevantes[:2]
            passos.append(Passo(
                "RECUPERAR", True,
                f"{min(2, len(relevantes)) if ev_r else 0} citação(ões) do corpus",
                cu, c_r,
            ))
        else:
            passos.append(Passo("RECUPERAR", False, "paragem antecipada", Custo()))

        # --- 2. compor ---
        if regime != "rapido" and conf < self.limiar:
            ev_c, _, cu = self._compor(ev)
            total = total + cu
            ganho = 1.0 if len(ev_c) < len(ev) else 0.0
            ev = ev_c
            conf = min(0.92, conf + 0.04 * ganho)
            passos.append(Passo("COMPOR", True, f"{len(ev)} após fusão", cu, conf))
        else:
            passos.append(Passo(
                "COMPOR", False,
                "regime rápido" if regime == "rapido" else "confiança suficiente",
                Custo(),
            ))

        # --- 3. verificar (só em regime profundo) ---
        if regime == "profundo":
            ev_v, ajuste, cu = self._verificar(ev, pergunta)
            total = total + cu
            conf = max(0.0, min(0.98, conf + ajuste))
            passos.append(Passo(
                "VERIFICAR", True,
                f"ajuste {ajuste:+.2f}" if ajuste else "sem contradições", cu, conf,
            ))
            parou = "VERIFICAR"
        else:
            passos.append(Passo("VERIFICAR", False, "orçamento não o justifica", Custo()))
            parou = parou if parou != "nenhum" else "COMPOR"

        self._registar(ev, passos, conf, parou, orcamento, total, pergunta)
        return self.historico[-1]

    def _registar(self, ev, passos, conf, parou, orcamento, total, pergunta) -> None:
        self.historico.append(ResultadoRaciocinio(
            evidencia=ev,
            passos=passos,
            confianca=round(conf, 3),
            parou_em=parou,
            orcamento_usado=orcamento.get("esforco", 0.0),
            custo_total=total,
        ))
        self._ultima_pergunta = pergunta

    # -- estatísticas -----------------------------------------------------
    def estatisticas(self) -> dict:
        if not self.historico:
            return {"n": 0}
        n = len(self.historico)
        por_paragem: dict[str, int] = {}
        for r in self.historico:
            por_paragem[r.parou_em] = por_paragem.get(r.parou_em, 0) + 1
        return {
            "n": n,
            "confianca_media": round(sum(r.confianca for r in self.historico) / n, 3),
            "flops_medio": round(sum(r.custo_total.flops for r in self.historico) / n),
            "por_paragem": por_paragem,
            "poupanca_estimada_pct": round(
                100 * por_paragem.get("ATALHO", 0) / n, 1
            ),
        }


def _radicais(texto: str) -> list[str]:
    from ..memoria.recuperador import stem_leve, tokenizar

    return [stem_leve(t) for t in tokenizar(texto)]


_NEGACOES = ("não", "nunca", "jamais", "sem", "nem", "tampouco")


def _contradizem(a: Evidencia, b: Evidencia) -> bool:
    """Heurística declarada: mesma entidade + polaridade oposta.

    Não é detecção de contradição semântica. É um sinal barato que apanha o
    caso mais comum no corpus: a mesma frase afirmada e negada, ou dois
    registos com confiança declarada muito diferente.
    """
    ma, mb = a.meta, b.meta
    if ma.get("entidade") and ma.get("entidade") == mb.get("entidade"):
        if ma.get("polaridade") and mb.get("polaridade"):
            return ma["polaridade"] != mb["polaridade"]
    ra, rb = set(_radicais(a.texto)), set(_radicais(b.texto))
    sobreposicao = len(ra & rb) / max(len(ra | rb), 1)
    if sobreposicao > 0.5:
        neg_a = any(n in a.texto.lower().split() for n in _NEGACOES)
        neg_b = any(n in b.texto.lower().split() for n in _NEGACOES)
        if neg_a != neg_b:
            return True
    return False
