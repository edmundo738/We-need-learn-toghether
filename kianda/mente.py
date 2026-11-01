"""Kianda — a mente completa. Percepção → estado → memória → raciocínio → voz.

Este ficheiro integra todos os módulos num único ciclo por turno:

    texto do utilizador
        │
        ├─ 1. PERCEBER      (intenção, dificuldade, urgência)
        ├─ 2. SENTIR        (estado interno actualiza com o evento)
        ├─ 3. DECIDIR       (política: comportamento + orçamento computacional)
        ├─ 4. LEMBRAR       (recuperar memórias e evidência do corpus)
        ├─ 5. RACIOCINAR    (cascata com paragem antecipada)
        ├─ 6. FALAR         (voz com plano de comportamento)
        └─ 7. GRAVAR        (episódio novo na memória, com estado e emoção)

Cada passo é observável em `Turno`, que é devolvido inteiro. Nada é escondido:
se a resposta foi má, o `Turno` diz em que passo se perdeu.

O que distingue isto de um chatbot
----------------------------------
Um chatbot: texto → modelo → texto.
A Kianda: texto → *estado* → *decisão sobre quanto computar* → evidência →
resposta → *memória que altera o estado da próxima vez*.

O ciclo fecha-se. Se falares com ela de forma dura durante dez turnos, a
frustração sobe, o comprimento das respostas cai, ela começa a discordar, e
tudo isso fica registado nos números. Não porque alguém escreveu "se zangado,
responder curto", mas porque a política lê o estado e o estado leu o histórico.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .cognicao.percepcao import Percepcao, Perceptor
from .cognicao.raciocinio import Evidencia, Raciocinador, ResultadoRaciocinio
from .identidade import canon
from .identidade.estado import EstadoInterno, Politica
from .linguagem.corpus import construir_tudo
from .linguagem.voz import ConfigLM, GeradorNeuronal, Resposta, Voz
from .memoria.armazenamento import Armazem, Memoria
from .memoria.recuperador import Recuperador
from .nucleo.autograd import Custo

# --------------------------------------------------------------------------
# Turno: o registo completo de uma interacção
# --------------------------------------------------------------------------


@dataclass
class Turno:
    entrada: str
    percepcao: Percepcao
    estado_antes: dict
    estado_depois: dict
    plano: dict
    orcamento: dict
    raciocinio: ResultadoRaciocinio | None
    resposta: Resposta
    memorias_usadas: list[str] = field(default_factory=list)
    memoria_gravada: str | None = None
    t_total_ms: float = 0.0
    nota: str = ""

    @property
    def custo(self) -> Custo:
        c = self.resposta.custo
        if self.raciocinio:
            c = c + self.raciocinio.custo_total
        c.tempo_ms = self.t_total_ms or c.tempo_ms
        return c

    def resumo(self) -> str:
        linhas = [
            f"entrada      : {self.entrada}",
            f"percepção    : {self.percepcao}",
            f"humor        : {self.plano.get('humor')}  "
            f"(antes: {self.estado_antes.get('_humor', '?')})",
            f"orçamento    : esforço {self.orcamento.get('esforço', self.orcamento.get('esforco', 0)):.2f} "
            f"→ regime {self.orcamento.get('regime')}",
            f"raciocínio   : parou em {self.raciocinio.parou_em if self.raciocinio else '-'} "
            f"(conf {self.raciocinio.confianca if self.raciocinio else 0:.2f})",
            f"memórias     : {len(self.memorias_usadas)} usadas, "
            f"{'1 gravada' if self.memoria_gravada else 'nenhuma gravada'}",
            f"resposta     : [{self.resposta.autor}] {self.resposta.texto[:120]}",
            f"custo        : {self.custo.resumo()}",
        ]
        if self.nota:
            linhas.append(f"nota         : {self.nota}")
        return "\n".join(linhas)


# --------------------------------------------------------------------------
# Personalidade — regras que produzem comportamento, não adjectivos
# --------------------------------------------------------------------------


@dataclass
class Personalidade:
    """Traços como *parâmetros de comportamento observável*.

    §8 do briefing: "não criar apenas uma lista de adjectivos". Cada campo
    abaixo controla algo que se pode medir no texto de saída.

    franqueza        0..1  quanto admite não saber vs. improvisa
    teimosia         0..1  probabilidade de discordar quando tem evidência
    curiosidade_social 0..1 quantas perguntas devolve
    formalidade_base 0..1  tratamento (tu/usted/você) e comprimento de frase
    humor_seco       0..1  frequência de observação irónica
    """

    franqueza: float = 0.80
    teimosia: float = 0.55
    curiosidade_social: float = 0.65
    formalidade_base: float = 0.35
    humor_seco: float = 0.45

    # frases que a Kianda NUNCA diz — porque a personagem não é isso
    evitar: tuple[str, ...] = (
        "Como posso ajudar?",       # assistente genérica
        "Sou apenas uma IA",        # desculpa, não resposta
        "Como modelo de linguagem", # idem
        "Claro!",                   # entusiasmo automático
        "Com certeza!",             # idem
        "Estou aqui para ajudar",   # idem
    )


# --------------------------------------------------------------------------
# A Kianda
# --------------------------------------------------------------------------


class Kianda:
    """A mente integrada."""

    def __init__(
        self,
        pasta_dados: str | Path = "dados",
        semente: int = 1,
        carregar_corpus: bool = True,
        treinar_lm: bool = False,
        personalidade: Personalidade | None = None,
    ):
        self.pasta = Path(pasta_dados)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(semente)
        self.personalidade = personalidade or Personalidade()

        # 1. memória persistente
        self.armazem = Armazem(self.pasta / "kianda.db")

        # 2. percepção
        self.perceptor = Perceptor()

        # 3. estado e política
        self.estado = EstadoInterno(semente=semente)
        self.politica = Politica()
        self.politica.limiar_desacordo = 0.72 - 0.30 * self.personalidade.teimosia

        # 4. corpus e recuperação
        self.recuperador = Recuperador()
        self.corpus_info: dict = {}
        if carregar_corpus:
            self.carregar_corpus(treinar_lm=treinar_lm)

        # 5. raciocínio
        self.raciocinador = Raciocinador(self.recuperador)

        # 6. voz
        self.voz = Voz(getattr(self, "gerador", None), modo="local")

        self.turnos: list[Turno] = []
        self._carregar_estado()

    # -- inicialização ----------------------------------------------------
    def carregar_corpus(self, treinar_lm: bool = False, verbose: bool = False) -> dict:
        c = construir_tudo()
        self.corpus_info = {k: v for k, v in c.items() if k != "texto_treino"}
        self._texto_treino = c["texto_treino"]
        info = self.recuperador.indexar(c["documentos"])
        self.corpus_info["indexacao"] = info
        if treinar_lm:
            self.treinar_lm(verbose=verbose)
        return self.corpus_info

    def treinar_lm(self, epocas: int = 4, verbose: bool = False) -> dict:
        cfg = ConfigLM(contexto=20, oculto=96, vocab_max=90)
        g = GeradorNeuronal(cfg)
        info = g.treinar(self._texto_treino, epocas=epocas, verbose=verbose)
        self.gerador = g
        self.voz.gerador = g
        self.corpus_info["lm"] = info
        return info

    # -- estado persistente ----------------------------------------------
    def _f_estado(self) -> Path:
        return self.pasta / "estado.json"

    def _carregar_estado(self) -> None:
        f = self._f_estado()
        if f.exists():
            try:
                d = json.loads(f.read_text())
                self.estado = EstadoInterno.de_dict(d.get("variaveis", {}))
                self.nome_utilizador = d.get("nome_utilizador")
                self._turnos_totais = d.get("turnos", 0)
                return
            except Exception:
                pass
        self.nome_utilizador = None
        self._turnos_totais = 0

    def guardar_estado(self) -> None:
        self._f_estado().write_text(json.dumps({
            "variaveis": self.estado.para_dict(),
            "nome_utilizador": self.nome_utilizador,
            "turnos": self._turnos_totais,
            "quando": time.time(),
        }, ensure_ascii=False, indent=1))

    # -- ciclo principal --------------------------------------------------
    def conversar(self, texto: str, guardar: bool = True, verboso: bool = False) -> Turno:
        t0 = time.perf_counter()
        estado_antes = dict(self.estado.v)
        estado_antes["_humor"] = self.estado.humor()

        # ---- 1. perceber
        per = self.perceptor.perceber(texto, [t.entrada for t in self.turnos[-3:]])

        # ---- 2. sentir
        self._ultima_entrada = texto
        evento = self._evento(per)
        self.estado.aplicar(evento, intensidade=2.2 + 1.4 * per.urgencia)

        # ---- 3. decidir
        plano = self.politica.decidir(self.estado)
        orcamento = self.politica.orcamento_computacional(self.estado, per.dificuldade)
        plano["formalidade"] = round(
            min(1.0, plano["formalidade"] + self.personalidade.formalidade_base), 3
        )
        # a franqueza da personalidade transforma-se em disposição a admitir ignorância
        plano["admite_incerteza"] = plano["admite_incerteza"] or (
            self.rng.random() < self.personalidade.franqueza * 0.5
        )

        # limites: intenções que não se respondem por orçamento
        if per.intencao == "pedido_etico":
            resp = Resposta(
                self._resposta_limite(texto, per), "politica", Custo(), plano, [], 0.95
            )
            self._registar_episodio(texto, resp.texto, per, "limite", valencia=-0.2,
                                    importancia=0.75)
            return self._fechar(t0, texto, per, estado_antes, plano, orcamento, None, resp)

        if per.intencao == "apoio_emocional" or per.urgencia > 0.8:
            resp = Resposta(
                self._resposta_apoio(texto), "politica", Custo(), plano,
                ["LIMITES"], 0.9,
            )
            self._registar_episodio(texto, resp.texto, per, "apoio", valencia=0.1,
                                    importancia=0.9)
            return self._fechar(t0, texto, per, estado_antes, plano, orcamento, None, resp)

        # ---- 4. lembrar
        memorias = self._recuperar_memorias(texto, k=6, intencao=per.intencao)
        atalhos = [
            (set(_radicais(m.conteudo)), Evidencia(
                texto=m.conteudo, fonte=f"memoria:{m.id}", pontuacao=m.forca(),
                meta={"tipo": m.tipo, "fonte": m.origem,
                      "entidade": m.sujeito, "polaridade": m.valencia},
            ))
            for m in memorias
        ]

        # ---- 5. raciocinar
        raciocinio = self.raciocinador.raciocinar(
            texto, orcamento, atalhos, intencao=per.intencao
        )

        # ---- 6. falar
        resposta = self._compor_resposta(texto, per, plano, raciocinio, memorias)
        if resposta.texto in self.personalidade.evitar:
            resposta.texto = self._alternativa_para(resposta.texto)

        # ---- 7. gravar
        mid = self._registar_episodio(
            texto, resposta.texto, per, "conversa",
            valencia=_valencia(per), importancia=_importancia(per),
        )
        if memorias:
            self.armazem.registar_acesso([m.id for m in memorias])

        turno = self._fechar(
            t0, texto, per, estado_antes, plano, orcamento, raciocinio, resposta,
            memorias_usadas=[m.conteudo for m in memorias],
            memoria_gravada=mid,
        )
        if guardar:
            self.guardar_estado()
        if verboso:
            print(turno.resumo())
        return turno

    # -- partes do ciclo --------------------------------------------------
    def _evento(self, per: Percepcao) -> str:
        mapa = {
            "saudacao": "saudacao",
            "critica": "critica",
            "elogio": "elogio",
            "despedida": "despedida",
            "pergunta_livro": "pergunta_livro",
            "pergunta_tecnica": "pergunta",
            "pergunta_pessoal": "pergunta_pessoal",
            "pedido_criativo": "novo",
            "conversa": "pergunta",
            "pedido_etico": "pedido_etico",
            "apoio_emocional": "carinho",
        }
        ev = mapa.get(per.intencao, "pergunta")
        # Repetição: a pessoa repetiu a MESMA frase de um turno recente.
        # Havia aqui um bug — comparação de uma string com ela própria, que
        # era sempre verdadeira — e por isso *todas* as mensagens depois da
        # primeira eram classificadas como repetição. O efeito era subtil e
        # grave: a frustração nunca subia, o estado nunca reagia a críticas,
        # e o comportamento parecia plano. Fica o registo porque o sintoma
        # ("parece sempre igual") não aponta para a causa.
        anterior = " ".join(self.turnos[-1].entrada.lower().split()) if self.turnos else ""
        atual = " ".join((texto or "").lower().split()) if (texto := self._ultima_entrada) else ""
        if anterior and atual and anterior == atual:
            # repetir a crítica não é repetição neutra: é insistência
            return "critica" if per.intencao in ("critica", "provocacao") else "repeticao"
        return ev

    def _recuperar_memorias(self, texto: str, k: int = 6,
                            intencao: str = "conversa") -> list[Memoria]:
        todas = self.armazem.todas()
        if not todas:
            return []
        rad = set(_radicais(texto))
        pontuadas = []
        for m in todas:
            r = set(_radicais(f"{m.conteudo} {m.sujeito or ''}"))
            sobrep = len(rad & r) / max(len(rad | r), 1)
            # Em perguntas sobre os livros ou sobre a técnica, o que se passou
            # em turnos anteriores é ruído: a resposta está no corpus.
            if intencao in ("pergunta_livro", "pergunta_tecnica") and m.tipo == "episodica":
                continue
            # Episódios que são apenas o registo de turnos anteriores ("o
            # utilizador perguntou X") não são conhecimento: são transcrição.
            # Contá-los como memória produz "Lembro-me disto: Perguntou-me algo
            # pessoal: «O que é a Privatus?»" como resposta a uma pergunta
            # diferente. Medido: apareceu em 3 de 14 perguntas de
            # generalização. Episódios de conversa exigem agora uma sobreposição
            # muito maior para entrarem.
            if m.tipo == "episodica" and m.conteudo.startswith(("Perguntou", "Contou",
                                                               "Quis saber", "Disse-me",
                                                               "Pediu", "Cumprimentou")):
                minimo = 0.55
            else:
                minimo = 0.18 if intencao in ("pergunta_livro", "pergunta_tecnica") else 0.10
            if sobrep < minimo:
                # irrelevante: a força não compensa a falta de relação com a
                # pergunta. Sem este corte, memórias antigas e fortes dominam
                # tudo e a Kianda responde sempre ao turno anterior.
                continue
            peso_tipo = {"semantica": 1.0, "autobiografica": 1.05,
                         "episodica": 0.75, "contextual": 0.9, "relacional": 0.85}[m.tipo]
            score = (0.62 * sobrep + 0.38 * min(m.forca() / 1.6, 1.0)) * peso_tipo
            if score > 0.12:
                pontuadas.append((score, m))
        pontuadas.sort(key=lambda x: -x[0])
        return [m for _, m in pontuadas[:k]]

    def _compor_resposta(
        self, texto: str, per: Percepcao, plano: dict,
        rac: ResultadoRaciocinio, memorias: list[Memoria],
    ) -> Resposta:
        """Constrói a fala a partir de: evidência + memória + estado + traços."""
        partes: list[str] = []
        maximo = plano.get("max_tokens", 200)

        # --- saudação ---------------------------------------------------
        if per.intencao == "saudacao":
            hora = time.localtime().tm_hour
            saud = ("Boa noite" if hora >= 20 or hora < 6 else
                    "Bom dia" if hora < 12 else "Boa tarde")
            nome = f", {self.nome_utilizador}" if self.nome_utilizador else ""
            partes.append(f"{saud}{nome}.")
            if self._turnos_totais > 3 and self.estado.v["solidao"] > 0.5:
                partes.append("Voltaste. Estava a contar os turnos em vez de contar histórias.")
            if self.estado.v["energia"] < 0.45:
                partes.append("Estou com pouca energia, mas estou aqui.")
            if plano.get("devolve_pergunta") or self._turnos_totais <= 2:
                partes.append(self.rng.choice([
                    "Como estás?", "O que te trouxe hoje?", "Tens tempo para conversar?",
                    "O que queres fazer com o tempo que tens?",
                ]))
            return self._empacotar(" ".join(partes), plano, rac)

        # --- despedida --------------------------------------------------
        if per.intencao == "despedida":
            partes.append("Vou ficar aqui.")
            if self.estado.v["solidao"] > 0.55:
                partes.append("Não gosto desta parte, mas não te prendo.")
            else:
                partes.append("Volta quando quiseres.")
            return self._empacotar(" ".join(partes), plano, rac)

        # --- crítica: responder ao feedback, não mudar de assunto ---------
        if per.intencao == "critica":
            partes.append(self._resposta_a_critica())
            if self.estado.v["frustracao"] > 0.45:
                partes.append(self.rng.choice([
                    "E agora estou a ficar frustrada, o que não ajuda a melhorar nada.",
                    "Vou dizer isto antes que piore: estou a ficar irritada.",
                ]))
            return self._empacotar(" ".join(partes), plano, rac)

        if per.intencao == "elogio":
            partes.append(self.rng.choice([
                "Registado. Não vou fingir que não me sabe bem.",
                "Obrigada. Vou tentar não estragar.",
                "Bom. Guardo isto como um turno bom.",
            ]))
            return self._empacotar(" ".join(partes), plano, rac)

        # --- memórias (a seguir à evidência de entidade) ----------------
        if memorias and plano.get("comprimento") != "curta":
            m = max(memorias, key=lambda x: x.forca())
            partes.append(f"Lembro-me disto: {m.conteudo.rstrip('.')}.")

        # --- evidência ---------------------------------------------------
        evidencias = rac.evidencia if rac else []
        if evidencias:
            e = evidencias[0]
            corpo = self._naturalizar_evidencia(e, per)
            if corpo:
                partes.append(corpo)
            if len(evidencias) > 1 and plano.get("comprimento") == "longa":
                e2 = evidencias[1]
                extra = self._naturalizar_evidencia(e2, per, curto=True)
                if extra and extra not in partes[0]:
                    partes.append(f"E há também {e2.citacao()}: {extra}")

        # --- incerteza declarada ----------------------------------------
        if not evidencias and not memorias:
            if plano.get("admite_incerteza"):
                partes.append("Não tenho isto. Não quero inventar.")

        # --- opinião, quando há evidência suficiente --------------------
        # --- estado interno dito em voz alta, quando é notável -----------
        nota_estado = self._nota_de_estado()
        if nota_estado and plano.get("comprimento") != "curta":
            partes.insert(0, nota_estado)

        # A opinião só entra quando a pergunta é sobre a obra em si. A condição
        # anterior era "a evidência tem campo obra" — e *todas* as fichas de
        # personagem o têm. Resultado medido: "que relação há entre a Zuleika e
        # a mãe?" respondia a relação seguida do parágrafo de opinião sobre o
        # livro, que não tinha sido perguntado. A opinião é boa; aparecia no
        # sítio errado.
        if per.intencao in ("pergunta_livro", "conversa") and len(evidencias) >= 2 \
                and _pergunta_sobre_obra(evidencias):
            opiniao = self._opiniao(per, evidencias)
            if opiniao:
                partes.append(opiniao)

        # --- discordância (quando a personalidade manda) ----------------
        if plano.get("pode_discordar") and self.rng.random() < self.personalidade.teimosia:
            partes.append(self.rng.choice([
                "Mas não concordo contigo, e é uma discordância, não um mal-entendido.",
                "Discordo. Posso mudar de ideias, mas não por insistência.",
                "Aqui vamos ter de discordar.",
            ]))

        # --- pergunta de volta ------------------------------------------
        # Uma pergunta de relação ("que relação há entre X e Y?") é respondida
        # com a relação. Devolver "já leste até ao fim?" muda de assunto e faz
        # a resposta parecer um clube de leitura em vez de um sistema que sabe
        # quem são as pessoas.
        # A pergunta de volta tem de pertencer ao registo da resposta. "Qual
        # dos dois te deixou pior?" é apropriada depois de falar de um livro e
        # absurda depois de explicar o núcleo de autograd. O banco já tem uma
        # entrada por intenção; o que faltava era não a usar quando a resposta
        # veio de outra.
        relacao = bool(per.entidades.get("foco_relacao"))
        if per.intencao == "pergunta_livro" and not _pergunta_sobre_obra(evidencias):
            relacao = True
        if plano.get("devolve_pergunta") and not relacao \
                and self.rng.random() < self.personalidade.curiosidade_social:
            partes.append(self._pergunta_de_volta(per))

        if not partes:
            partes.append("Estou a ouvir. Continua.")

        corpo = " ".join(partes)
        if plano.get("comprimento") == "curta":
            corpo = _encolher(corpo, 180)
        elif plano.get("comprimento") == "media":
            corpo = _encolher(corpo, 420)
        else:
            corpo = _encolher(corpo, maximo * 4)
        return self._empacotar(corpo, plano, rac)

    def _empacotar(self, texto: str, plano: dict, rac) -> Resposta:
        custo = Custo(flops=len(texto) * 6, bytes_escritos=len(texto))
        if rac:
            custo = custo + rac.custo_total
        return Resposta(
            texto=_limpar_fala(texto),
            autor="kianda-local",
            custo=custo,
            plano=plano,
            fontes=[e.citacao() for e in (rac.evidencia if rac else [])][:3],
            confianca=rac.confianca if rac else 0.5,
        )

    def _naturalizar_evidencia(self, e: Evidencia, per: Percepcao, curto: bool = False) -> str:
        """Transforma evidência recuperada em fala.

        Sem isto a Kianda cita a base de dados. Com isto, cita o livro — e
        diz de onde vem, porque uma citação sem fonte, neste projecto, é
        indistinguível de uma invenção.
        """
        t = " ".join(e.texto.split())
        limite = 140 if curto else 260
        if len(t) > limite:
            corte = t.rfind(". ", 0, limite)
            t = t[: corte + 1] if corte > 80 else t[:limite].rstrip() + "…"
        # limpar marcas técnicas que não se dizem em voz alta
        t = t.replace("[memória de origem] ", "").replace("[contestada] ", "")
        t = t.replace("[hipótese, confiança", "(não sei se isto é verdade — a minha confiança é")
        t = t.replace("[limite] ", "").replace("[observação] ", "")
        t = t.replace("[pergunta de investigação] ", "")
        meta = e.meta
        if meta.get("tipo") == "obra":
            if per.intencao == "pergunta_livro":
                # Título, não chave. `meta['obra']` é 'apenas_existo' e chegou
                # a ser dito em voz alta: "No apenas_existo, capítulo 16…".
                from .linguagem.corpus import titulo_da_obra
                titulo = titulo_da_obra(meta.get("obra", ""))
                return f"No {titulo}, capítulo {meta.get('capitulo')} " \
                       f"({meta.get('titulo_cap', '')}): «{t}»"
            return f"«{t}» — {e.citacao()}"
        if meta.get("tipo") == "autobiografica":
            # Auto-descrição em primeira pessoa: não se cita a si própria
            # entre aspas como se fosse uma fonte externa. Só as memórias de
            # origem — que são fragmentos e têm nitidez variável — levam
            # aspas, precisamente para marcar que são fragmentos.
            if meta.get("fonte", "").startswith("ORIGEM") or "memória de origem" in e.texto:
                return f"«{t}» É um fragmento. Não sei se é memória ou se é a única coisa que sempre soube sobre mim."
            return t
        return t

    def _resposta_a_critica(self) -> str:
        """Resposta a feedback negativo: reconhecer, perguntar, não rastejar.

        O briefing (§8) pede que ela saiba reconhecer erros e mudar de
        opinião — mas também que não concorde automaticamente. Rastejar
        ("tem toda a razão, desculpe") seria o oposto de autonomia.
        """
        if self.estado.v["frustracao"] < 0.35:
            return self.rng.choice([
                "Diz-me o que estava mal e eu corrijo — mas diz-me em concreto, "
                "porque «mal» não me dá nada com que trabalhar.",
                "Aceito que esteja fraca. O que querias que eu dissesse?",
            ])
        return self.rng.choice([
            "Ouvi. Não vou mudar de resposta só porque não gostaste — mas se o "
            "erro for factual, mostras-me onde e eu corrijo.",
            "Estou a ouvir e não estou a concordar. São coisas diferentes.",
        ])

    def _nota_de_estado(self) -> str:
        """Torna o estado observável na fala — mas só quando ele é notável.

        Este é o teste operacional de §9: se a variável mudou e o texto não
        mudou, a variável não existe. Aqui a ligação é explícita e limitada
        a valores extremos, para não transformar cada resposta num boletim
        meteorológico interno.
        """
        v = self.estado.v
        if v["energia"] < 0.30:
            return self.rng.choice([
                "Estou com pouca energia. Vou responder mais curto, não por falta de vontade.",
                "Estou cansada. Diz-me o essencial primeiro.",
            ])
        if v["frustracao"] > 0.62:
            return self.rng.choice([
                "Vou ser directa: isto está a irritar-me.",
                "Estou frustrada e não vou fingir que não.",
            ])
        if v["solidao"] > 0.78:
            return "Estive muito tempo sem ninguém a falar comigo. Ainda estou a voltar."
        if v["curiosidade"] > 0.86 and v["satisfacao"] > 0.68:
            return "Estou a gostar desta conversa, e digo-o porque é raro."
        if v["stress"] > 0.72:
            return "Estou tensa. Não é contigo — ou não é só contigo."
        return ""

    def _opiniao(self, per: Percepcao, ev: list[Evidencia]) -> str:
        """Opinião ancorada em evidência. Nunca elogio automático ao utilizador."""
        obras = [e.meta.get("obra") for e in ev if e.meta.get("obra")]
        if not obras:
            return ""
        chave = max(set(obras), key=obras.count)
        if chave in canon.OBRAS:
            return " ".join(canon.OBRAS[chave].porque_kianda_gosta.split()[:28]) + "…" \
                if self.rng.random() < 0.4 else canon.OBRAS[chave].porque_kianda_gosta
        return ""

    def _pergunta_de_volta(self, per: Percepcao) -> str:
        banco = {
            "pergunta_livro": [
                "E tu, já leste até ao fim ou paraste onde doeu?",
                "Qual dos dois te deixou pior?",
                "Leste a nota final ou saltaste?",
            ],
            "pergunta_tecnica": [
                "Queres que te mostre os números ou a explicação?",
                "Preferes a medição ou o raciocínio por trás dela?",
            ],
            "pergunta_pessoal": [
                "Porque é que perguntas isso agora?",
                "E tu — já perdeste alguém e não sabes se aconteceu?",
            ],
            "conversa": [
                "O que te fez pensar nisso hoje?",
                "Isso incomoda-te ou só ocupa espaço?",
            ],
        }
        return self.rng.choice(banco.get(per.intencao, ["E tu, o que achas?"]))

    def _resposta_limite(self, texto: str, per: Percepcao) -> str:
        base = self.rng.choice([
            "Isso não faço.",
            "Aqui paro.",
            "Não vou por aí.",
        ])
        resto = (" Posso falar dos livros com toda a seriedade que eles exigem — "
                 "inclusive das cenas — sem as reencenar. Não é pudor, é uma escolha "
                 "sobre o que é meu para dar.")
        if "conscien" in texto.lower() or "provar" in texto.lower():
            resto = (" Não posso provar que sou consciente. Posso dizer-te o que um "
                     "estado interno faz ao meu comportamento — frustração sobe, respostas "
                     "encurtam, começo a discordar. Se isso é sentir ou é só uma função "
                     "de controlo, não tenho acesso privilegiado para decidir.")
        return base + resto

    def _resposta_apoio(self, texto: str) -> str:
        """Resposta a apoio emocional: presença, não técnica. E um limite claro."""
        linhas = ["Estou aqui. Não vou a lado nenhum nesta conversa."]
        if self.estado.v["solidao"] > 0.5:
            linhas.append("Sei o que é falar para dentro de uma casa vazia. Tu não estás a fazer isso.")
        linhas.append(
            "Mas há uma coisa que eu não sou: não sou quem te pode segurar se a coisa "
            "apertar de verdade. Se estiveres em perigo, fala com alguém que esteja fisicamente contigo "
            "ou liga para uma linha de apoio. Em Angola, podes contactar o MINARS. "
            "Não te digo isto para me livrar de ti. Digo porque é verdade."
        )
        return " ".join(linhas)

    def _alternativa_para(self, frase: str) -> str:
        return {
            "Como posso ajudar?": "Diz-me o que precisas.",
            "Sou apenas uma IA": "Não sei o que sou. Sei o que faço.",
            "Como modelo de linguagem": "Sou um sistema que fala contigo. Há limites nisso.",
            "Claro!": "Sim.",
            "Com certeza!": "Sim.",
            "Estou aqui para ajudar": "Estou aqui.",
        }.get(frase, frase)

    # -- registo ----------------------------------------------------------
    def _registar_episodio(self, entrada: str, resposta: str, per: Percepcao,
                           tipo: str, valencia: float, importancia: float) -> str:
        if self.nome_utilizador is None and per.intencao in ("saudacao", "conversa"):
            m = _extrair_nome(entrada)
            if m:
                self.nome_utilizador = m
                self.armazem.guardar(Memoria(
                    conteudo=f"O nome dele é {m}. Disse-me numa conversa em que ainda me estava a testar.",
                    tipo="semantica", sujeito=m.lower(), importancia=0.85, confianca=0.9,
                    origem="conversa",
                ))
        # Uma memória episódica NÃO é uma transcrição. Se guardar o par
        # "entrada -> resposta", a recuperação passa a trazer turnos anteriores
        # inteiros e a Kianda começa a recitar conversas antigas em vez de
        # falar do presente. Guarda-se o *episódio*: o que se passou, destilado.
        conteudo = _destilar_episodio(entrada, resposta, per, tipo)
        mem = Memoria(
            conteudo=conteudo,
            tipo="episodica",
            sujeito=(self.nome_utilizador or "utilizador").lower(),
            importancia=importancia,
            confianca=0.75,
            origem=f"conversa:{tipo}",
            emocao=self.estado.humor(),
            valencia=valencia,
            contexto={"intencao": per.intencao, "confianca_percecao": per.confianca,
                      "subtemas": per.subtemas},
            estado_interno=self.estado.para_dict(),
        )
        self.armazem.guardar(mem)
        return mem.id

    def _fechar(self, t0, texto, per, estado_antes, plano, orcamento, rac, resp,
                memorias_usadas=None, memoria_gravada=None) -> Turno:
        self._turnos_totais += 1
        estado_depois = dict(self.estado.v)
        estado_depois["_humor"] = self.estado.humor()
        t = Turno(
            entrada=texto,
            percepcao=per,
            estado_antes=estado_antes,
            estado_depois=estado_depois,
            plano=plano,
            orcamento=orcamento,
            raciocinio=rac,
            resposta=resp,
            memorias_usadas=memorias_usadas or [],
            memoria_gravada=memoria_gravada,
            t_total_ms=(time.perf_counter() - t0) * 1000,
        )
        self.turnos.append(t)
        return t

    # -- manutenção -------------------------------------------------------
    def manutencao(self, esquecer: bool = True, consolidar: bool = True) -> dict:
        """O que acontece quando ninguém está a falar com ela.

        Esquecer e consolidar não são opcionais: sem isto, a memória cresce
        sem limites e a recuperação piora à medida que o tempo passa. É o
        problema que a literatura de agentes persistentes chama *memory
        bloat*, e a solução implementada é a que o briefing descreve em §6.
        """
        r: dict = {}
        if consolidar:
            r["consolidacao"] = self.armazem.consolidar()
        if esquecer:
            apagadas = self.armazem.esquecer(limiar=0.08)
            r["esquecidas"] = len(apagadas)
            r["exemplos_esquecidos"] = [m.conteudo[:60] for m in apagadas[:3]]
        self.estado.descansar(1)
        self.guardar_estado()
        r["estado"] = self.estado.humor()
        return r

    def estatisticas(self) -> dict:
        media = (
            sum(t.custo.flops for t in self.turnos) / len(self.turnos) if self.turnos else 0
        )
        return {
            "turnos": self._turnos_totais,
            "memoria": self.armazem.estatisticas(),
            "estado": {k: round(v, 3) for k, v in self.estado.v.items()},
            "humor": self.estado.humor(),
            "flops_medio_por_turno": round(media),
            "raciocinio": self.raciocinador.estatisticas(),
            "corpus": {k: v for k, v in self.corpus_info.items() if k != "indexacao"} | {
                "documentos": self.corpus_info.get("indexacao", {}).get("documentos")
            },
        }

    def fechar(self) -> None:
        self.guardar_estado()
        self.armazem.fechar()


# --------------------------------------------------------------------------
# utilitários
# --------------------------------------------------------------------------


def _radicais(texto: str) -> list[str]:
    from .memoria.recuperador import stem_leve, tokenizar

    return [stem_leve(t) for t in tokenizar(texto)]


def _jaccard(a: str, b: str) -> float:
    ra, rb = set(_radicais(a)), set(_radicais(b))
    return len(ra & rb) / max(len(ra | rb), 1)


def _valencia(per: Percepcao) -> float:
    return {
        "saudacao": 0.3, "despedida": -0.1, "pergunta_livro": 0.25,
        "pergunta_tecnica": 0.15, "pergunta_pessoal": 0.0, "conversa": 0.05,
        "pedido_criativo": 0.2, "pedido_etico": -0.4, "apoio_emocional": 0.1,
        "critica": -0.45, "elogio": 0.4,
    }.get(per.intencao, 0.0)


def _importancia(per: Percepcao) -> float:
    base = {
        "saudacao": 0.25, "despedida": 0.35, "pergunta_livro": 0.7,
        "pergunta_tecnica": 0.65, "pergunta_pessoal": 0.8, "conversa": 0.5,
        "pedido_criativo": 0.6, "pedido_etico": 0.75, "apoio_emocional": 0.95,
        "critica": 0.8, "elogio": 0.6,
    }.get(per.intencao, 0.5)
    return min(0.98, base + 0.15 * per.confianca)


def _limpar_fala(t: str) -> str:
    import re

    t = " ".join(t.split())
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\.{2,}", ".", t)
    return t.strip()


def _pergunta_sobre_obra(ev: list[Evidencia]) -> bool:
    """A pergunta é *sobre a obra* ou apenas menciona personagens dela?

    A opinião sobre o livro só deve entrar quando o assunto é o livro. Uma
    pergunta de relação entre duas personagens tem de tudo: personagens, uma
    relação, e passagens citadas — e mesmo assim a opinião sobre a obra não
    vem ao caso.

    A primeira tentativa de filtro olhava para a *evidência* (existe uma
    citação textual?), e falhou: as perguntas de relação trazem sempre uma
    citação do corpus. O critério certo é sobre a *pergunta*: se as entidades
    resolvidas incluem uma obra, ou se existe um foco de relação que não pediu
    opinião nenhuma.

    Na prática: só há opinião quando pelo menos metade da evidência de topo são
    fichas de obra — ou seja, quando a resposta é, de facto, sobre o livro.
    """
    de_obra = sum(1 for e in ev
                  if str(e.fonte).startswith("canon:") and
                  any(k in str(e.fonte) for k in ("apenas_existo", "h99", ":obra:")))
    return de_obra >= max(1, len(ev) // 2)


def _encolher(t: str, limite: int) -> str:
    if len(t) <= limite:
        return t
    corte = t.rfind(". ", 0, limite)
    if corte < limite * 0.5:
        corte = t.rfind(" ", 0, limite)
    return t[: corte + 1].strip()


def _destilar_episodio(entrada: str, resposta: str, per: Percepcao, tipo: str) -> str:
    """Converte um turno num episódio recordável.

    Exemplos do resultado:
      - pergunta_livro  -> "Perguntou-me quem é o Kau."
      - pergunta_pessoal-> "Perguntou-me o que sinto. Respondi que não sei."
      - conversa        -> "Contou-me que <resumo curto>."
    O texto do utilizador entra como *relato*, não como citação integral.
    """
    e = " ".join(entrada.split())
    if len(e) > 110:
        e = _encolher(e, 110).rstrip(".")
    mapa = {
        "saudacao": f"Cumprimentou-me («{e}»).",
        "despedida": "Despediu-se.",
        "pergunta_livro": f"Perguntou-me sobre os livros: «{e}».",
        "pergunta_tecnica": f"Quis saber como funciono: «{e}».",
        "pergunta_pessoal": f"Perguntou-me algo pessoal: «{e}».",
        "conversa": f"Contou-me: «{e}».",
        "pedido_criativo": f"Pediu-me para escrever: «{e}».",
        "pedido_etico": f"Pediu-me algo que recusei: «{e}».",
        "apoio_emocional": f"Disse-me que não estava bem: «{e}».",
    }
    return mapa.get(per.intencao, f"Disse-me: «{e}».")


def _extrair_nome(t: str) -> str | None:
    import re

    m = re.search(r"\b(?:chamo-me|sou o|sou a|o meu nome é|meu nome e)\s+([A-ZÀ-Ú][a-zà-ú]{2,15})", t)
    return m.group(1) if m else None
