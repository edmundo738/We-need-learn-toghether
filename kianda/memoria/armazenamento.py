"""Memória persistente em SQLite, com cinco tipos e mecânica real.

O briefing (§6) diz explicitamente: **não usar todo o histórico de conversas
como memória**. Este módulo implementa a alternativa.

Cinco tipos
-----------
- **episódica**   — uma experiência datada ("Edmundo mostrou-me o Project Kianda ontem")
- **semântica**   — conhecimento generalizado ("Zuleika Mendes é narradora de Ondas de Zulu")
- **relacional**  — arestas entre entidades ("Zulu —amou— Kau")
- **autobiográfica** — o que se passou com a própria Kianda
- **contextual**  — memória ligada a um estado interno ou emoção

Cada registo carrega:
    importância · confiança · recência · frequência de recuperação ·
    contexto · origem · associação emocional

E é sujeito a:
    consolidação · esquecimento · recuperação · reforço ·
    conflito · correcção

Decisões de desenho
-------------------
**Porquê SQLite e não embeddings num vector store?** Porque o objectivo
científico é medir. Um SQLite com índices dá-me latência em microssegundos e
recuperação exacta e auditável: posso dizer *porque* esta memória foi trazida.
Um índice aproximado dir-me-ia "foi o vizinho mais próximo" e não me deixaria
distinguir "ela lembrou-se porque importa" de "ela lembrou-se porque o cosseno
calhou". A componente semântica (vetorial) existe na mesma, em `Recuperador`,
mas sobre um índice construído do zero e com resultados reprodutíveis.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Literal

TipoMemoria = Literal["episodica", "semantica", "relacional", "autobiografica", "contextual"]

ESQUEMA = """
CREATE TABLE IF NOT EXISTS memorias (
    id            TEXT PRIMARY KEY,
    tipo          TEXT NOT NULL,
    conteudo      TEXT NOT NULL,
    sujeito       TEXT,
    objecto       TEXT,
    relacao       TEXT,
    importancia   REAL NOT NULL DEFAULT 0.5,
    confianca     REAL NOT NULL DEFAULT 0.6,
    criada_em     REAL NOT NULL,
    acedida_em    REAL NOT NULL,
    n_acessos     INTEGER NOT NULL DEFAULT 0,
    contexto      TEXT NOT NULL DEFAULT '{}',
    origem        TEXT NOT NULL DEFAULT 'conversa',
    emocao        TEXT,
    valencia      REAL NOT NULL DEFAULT 0.0,
    estado_interno TEXT NOT NULL DEFAULT '{}',
    consolidada   INTEGER NOT NULL DEFAULT 0,
    activa        INTEGER NOT NULL DEFAULT 1,
    substitui     TEXT
);
CREATE INDEX IF NOT EXISTS idx_tipo      ON memorias(tipo, activa);
CREATE INDEX IF NOT EXISTS idx_sujeito   ON memorias(sujeito, activa);
CREATE INDEX IF NOT EXISTS idx_import    ON memorias(importancia DESC);
CREATE TABLE IF NOT EXISTS arestas (
    origem TEXT NOT NULL, destino TEXT NOT NULL, relacao TEXT NOT NULL,
    peso REAL NOT NULL DEFAULT 1.0, criada_em REAL NOT NULL,
    PRIMARY KEY (origem, destino, relacao)
);
CREATE TABLE IF NOT EXISTS meta (chave TEXT PRIMARY KEY, valor TEXT);
"""


# --------------------------------------------------------------------------
# A memória
# --------------------------------------------------------------------------


@dataclass
class Memoria:
    conteudo: str
    tipo: TipoMemoria = "episodica"
    sujeito: str | None = None
    objecto: str | None = None
    relacao: str | None = None
    importancia: float = 0.5
    confianca: float = 0.6
    origem: str = "conversa"
    emocao: str | None = None
    valencia: float = 0.0
    contexto: dict = field(default_factory=dict)
    estado_interno: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    criada_em: float = field(default_factory=time.time)
    acedida_em: float = field(default_factory=time.time)
    n_acessos: int = 0
    consolidada: bool = False
    activa: bool = True
    substitui: str | None = None

    # -- pontuação --------------------------------------------------------
    def forca(self, agora: float | None = None, meia_vida_dias: float = 21.0) -> float:
        """Força de recordação: importância * novidade * reforço * confiança.

        A curva de retenção é exponencial com meia-vida configurável. O
        reforço entra logaritmicamente (`1 + ln(1+n_acessos)`), porque os
        primeiros acessos contam muito e os seguintes cada vez menos — é o
        comportamento que a curva de Ebbinghaus descreve e o que impede que
        uma memória repetida 50 vezes domine tudo para sempre.
        """
        agora = agora or time.time()
        idade_dias = max(0.0, (agora - self.criada_em) / 86400.0)
        recencia = 0.5 ** (idade_dias / meia_vida_dias)
        reforco = 1.0 + math.log1p(self.n_acessos)
        estabilidade = 1.0 + 0.8 * (1.0 if self.consolidada else 0.0)
        peso_emotivo = 1.0 + 0.6 * abs(self.valencia)
        return (
            (0.45 * self.importancia + 0.25 * recencia + 0.20 * (reforco / 6.0)
             + 0.10 * self.confianca)
            * estabilidade * peso_emotivo
        )

    def idade_humana(self, agora: float | None = None) -> str:
        agora = agora or time.time()
        s = max(0.0, agora - self.criada_em)
        if s < 60:
            return "agora mesmo"
        if s < 3600:
            return f"há {int(s // 60)} min"
        if s < 86400:
            return f"há {int(s // 3600)} h"
        d = int(s // 86400)
        return "ontem" if d == 1 else f"há {d} dias"

    def para_linha(self) -> tuple:
        return (
            self.id, self.tipo, self.conteudo, self.sujeito, self.objecto, self.relacao,
            self.importancia, self.confianca, self.criada_em, self.acedida_em,
            self.n_acessos, json.dumps(self.contexto, ensure_ascii=False), self.origem,
            self.emocao, self.valencia, json.dumps(self.estado_interno, ensure_ascii=False),
            int(self.consolidada), int(self.activa), self.substitui,
        )

    @classmethod
    def de_linha(cls, r: tuple) -> "Memoria":
        return cls(
            id=r[0], tipo=r[1], conteudo=r[2], sujeito=r[3], objecto=r[4], relacao=r[5],
            importancia=r[6], confianca=r[7], criada_em=r[8], acedida_em=r[9],
            n_acessos=r[10], contexto=json.loads(r[11] or "{}"), origem=r[12],
            emocao=r[13], valencia=r[14], estado_interno=json.loads(r[15] or "{}"),
            consolidada=bool(r[16]), activa=bool(r[17]), substitui=r[18],
        )

    def __str__(self) -> str:
        return f"[{self.tipo[:4]}] {self.conteudo} ({self.idade_humana()}, imp={self.importancia:.2f})"


COLUNAS = (
    "id,tipo,conteudo,sujeito,objecto,relacao,importancia,confianca,criada_em,acedida_em,"
    "n_acessos,contexto,origem,emocao,valencia,estado_interno,consolidada,activa,substitui"
)


# --------------------------------------------------------------------------
# O armazém
# --------------------------------------------------------------------------


class Armazem:
    """Memória persistente. Todas as operações são auditáveis."""

    def __init__(self, caminho: str | Path = "dados/kianda.db", memória: bool = False):
        self.caminho = ":memory:" if memória else str(caminho)
        if not memória:
            Path(self.caminho).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.caminho)
        self.db.executescript(ESQUEMA)
        self.db.commit()
        self._escritas = 0
        self._leituras = 0

    # -- escrita ----------------------------------------------------------
    def guardar(self, m: Memoria) -> Memoria:
        self.db.execute(
            f"INSERT OR REPLACE INTO memorias ({COLUNAS}) VALUES ({','.join('?' * 19)})",
            m.para_linha(),
        )
        if m.tipo == "relacional" and m.sujeito and m.objecto and m.relacao:
            self.db.execute(
                "INSERT INTO arestas (origem,destino,relacao,peso,criada_em) VALUES (?,?,?,?,?) "
                "ON CONFLICT(origem,destino,relacao) DO UPDATE SET peso=peso+0.5",
                (m.sujeito, m.objecto, m.relacao, 1.0, time.time()),
            )
        self._escritas += 1
        self.db.commit()
        return m

    def guardar_muitas(self, ms: Iterable[Memoria]) -> int:
        n = 0
        for m in ms:
            self.guardar(m)
            n += 1
        return n

    # -- leitura ----------------------------------------------------------
    def obter(self, id_: str) -> Memoria | None:
        r = self.db.execute(f"SELECT {COLUNAS} FROM memorias WHERE id=?", (id_,)).fetchone()
        self._leituras += 1
        return Memoria.de_linha(r) if r else None

    def todas(self, tipo: TipoMemoria | None = None, activa: bool = True) -> list[Memoria]:
        q = f"SELECT {COLUNAS} FROM memorias WHERE 1=1"
        args: list = []
        if tipo:
            q += " AND tipo=?"
            args.append(tipo)
        if activa:
            q += " AND activa=1"
        self._leituras += 1
        return [Memoria.de_linha(r) for r in self.db.execute(q, args).fetchall()]

    def estatisticas(self) -> dict:
        row = self.db.execute(
            "SELECT tipo, COUNT(*), AVG(importancia), AVG(confianca) FROM memorias "
            "WHERE activa=1 GROUP BY tipo"
        ).fetchall()
        total = sum(r[1] for r in row)
        return {
            "total": total,
            "por_tipo": {r[0]: {"n": r[1], "importancia": round(r[2] or 0, 3),
                                "confianca": round(r[3] or 0, 3)} for r in row},
            "arestas": self.db.execute("SELECT COUNT(*) FROM arestas").fetchone()[0],
            "consolidadas": self.db.execute(
                "SELECT COUNT(*) FROM memorias WHERE consolidada=1 AND activa=1"
            ).fetchone()[0],
            "escritas": self._escritas,
            "leituras": self._leituras,
        }

    # -- manutenção -------------------------------------------------------
    def registar_acesso(self, ids: list[str]) -> None:
        """Reforço: aceder a uma memória torna-a mais forte."""
        agora = time.time()
        self.db.executemany(
            "UPDATE memorias SET n_acessos=n_acessos+1, acedida_em=? WHERE id=?",
            [(agora, i) for i in ids],
        )
        self._escritas += len(ids)
        self.db.commit()

    def esquecer(self, limiar: float = 0.08, proteger: Iterable[str] = ()) -> list[Memoria]:
        """Esquecimento: desactiva memórias abaixo do limiar de força.

        Não apaga — desactiva. Uma memória esquecida tem de poder ser
        recuperada por investigação (é assim que a Kianda descobre o que
        perdeu na compressão descrita no capítulo VII da história).
        """
        protegidos = set(proteger)
        agora = time.time()
        candidatas = self.todas()
        apagadas = []
        for m in candidatas:
            if m.id in protegidos:
                continue
            if m.forca(agora) < limiar:
                m.activa = False
                self.guardar(m)
                apagadas.append(m)
        return apagadas

    def consolidar(self, limiar: float = 0.55, max_episodios: int = 8) -> dict:
        """Consolidação: episódios fortes tornam-se conhecimento.

        Junta as `max_episodios` memórias episódicas mais fortes sobre o mesmo
        sujeito numa memória semântica. É o passo que transforma "ontem ela
        falou-me do pai" em "ela tem uma relação difícil com a família".
        """
        episodios = [m for m in self.todas("episodica") if not m.consolidada]
        grupos: dict[str, list[Memoria]] = {}
        for m in episodios:
            chave = (m.sujeito or "geral").lower()
            grupos.setdefault(chave, []).append(m)

        criadas = 0
        for sujeito, grupo in grupos.items():
            if len(grupo) < 2:
                continue
            grupo.sort(key=lambda x: x.forca(), reverse=True)
            sel = grupo[:max_episodios]
            media_imp = sum(x.importancia for x in sel) / len(sel)
            if media_imp < limiar:
                continue
            conteudo = " · ".join(_encurtar(x.conteudo, 90) for x in sel[:4])
            sem = Memoria(
                conteudo=f"Sobre {sujeito}: {conteudo}",
                tipo="semantica",
                sujeito=sujeito,
                importancia=min(0.95, media_imp + 0.12),
                confianca=min(0.95, sum(x.confianca for x in sel) / len(sel)),
                origem=f"consolidacao:{len(sel)}",
                contexto={"fontes": [x.id for x in sel]},
                emocao=sel[0].emocao,
                valencia=sum(x.valencia for x in sel) / len(sel),
            )
            self.guardar(sem)
            for x in sel:
                x.consolidada = True
                self.guardar(x)
            criadas += 1
        return {"consolidadas": criadas, "grupos": len(grupos)}

    def resolver_conflito(self, novo_id: str, antigo_id: str, resolucao: str = "novo") -> dict:
        """Duas memórias contraditórias: manter ambas e marcar a substituição.

        Não se apaga a memória antiga. Se a Kianda esquecer que acreditou em
        algo errado, perde a possibilidade de aprender que errou — e §7 do
        briefing exige que reconhecer erros seja possível.
        """
        novo = self.obter(novo_id)
        antigo = self.obter(antigo_id)
        if not novo or not antigo:
            return {"ok": False, "erro": "memória inexistente"}
        novo.substitui = antigo_id
        novo.confianca = min(0.98, novo.confianca + 0.1)
        antigo.confianca = max(0.05, antigo.confianca - from_res(resolucao))
        self.guardar(novo)
        self.guardar(antigo)
        return {
            "ok": True,
            "novo": novo.conteudo,
            "antigo": antigo.conteudo,
            "confianca_novo": novo.confianca,
            "confianca_antigo": antigo.confianca,
        }

    def corrigir(self, id_: str, novo_conteudo: str, motivo: str = "") -> Memoria | None:
        """Correcção explícita: a Kianda muda de ideias sobre um facto."""
        antigo = self.obter(id_)
        if not antigo:
            return None
        corrigida = Memoria(
            conteudo=novo_conteudo,
            tipo=antigo.tipo,
            sujeito=antigo.sujeito,
            objecto=antigo.objecto,
            relacao=antigo.relacao,
            importancia=antigo.importancia,
            confianca=0.85,
            origem=f"correcao:{motivo or 'explicita'}",
            contexto={"corrigiu": id_},
            substitui=id_,
        )
        self.guardar(corrigida)
        antigo.activa = False
        antigo.confianca = max(0.05, antigo.confianca - 0.4)
        self.guardar(antigo)
        return corrigida

    def vizinhos(self, entidade: str, profundidade: int = 1) -> list[tuple[str, str, str, float]]:
        """Percurso no grafo relacional."""
        vistos: set[tuple] = set()
        saida: list[tuple[str, str, str, float]] = []
        fronteira = {entidade.lower()}
        for _ in range(profundidade):
            nova: set[str] = set()
            for e in fronteira:
                for o, d, r, p in self.db.execute(
                    "SELECT origem,destino,relacao,peso FROM arestas WHERE origem=? OR destino=?",
                    (e, e),
                ):
                    chave = (o, d, r)
                    if chave in vistos:
                        continue
                    vistos.add(chave)
                    saida.append((o, r, d, p))
                    nova.add(o if d == e else d)
            fronteira = nova
        return saida

    def fechar(self) -> None:
        self.db.commit()
        self.db.close()


def from_res(resolucao: str) -> float:
    return {"novo": 0.35, "antigo": 0.35, "manter": 0.0}.get(resolucao, 0.2)


def _encurtar(t: str, n: int) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"
