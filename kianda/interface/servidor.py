"""Servidor local: uma janela para dentro da Kianda.

Só biblioteca padrão — `http.server`. Sem Flask, sem FastAPI. A razão não é
purismo: é que a Kianda tem de arrancar numa máquina com Python e NumPy, mais
nada. Uma dependência a menos é uma forma de demo que não falha.

A API expõe o *interior*, não só a conversa:

    POST /api/falar     {texto}      -> resposta + estado + orçamento + passos
    GET  /api/estado                 -> variáveis internas, humor, memória
    GET  /api/memoria?q=             -> o que ela lembra sobre um assunto
    GET  /api/corpus                 -> estatísticas do corpus e do índice
    POST /api/manutencao             -> esquecer + consolidar
    GET  /api/benchmark              -> último resultado guardado

O painel do lado do cliente desenha as variáveis em tempo real. Isto é o
teste de §9 do briefing tornado visível: se mexer nos números não mudasse o
texto, ver-se-ia imediatamente.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RAIZ = Path(__file__).resolve().parents[2]
ESTATICOS = RAIZ / "kianda" / "interface" / "web"


def criar_servidor(kianda, porto: int = 8000, host: str = "0.0.0.0"):
    trava = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Kianda/0.1"

        # -- utilidades ---------------------------------------------------
        def _json(self, obj, codigo: int = 200) -> None:
            corpo = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def _ficheiro(self, caminho: Path) -> None:
            if not caminho.exists() or not caminho.is_file():
                self._json({"erro": "não encontrado"}, 404)
                return
            tipo = mimetypes.guess_type(str(caminho))[0] or "application/octet-stream"
            if caminho.suffix in (".html", ".css", ".js"):
                tipo += "; charset=utf-8"
            dados = caminho.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(dados)))
            self.end_headers()
            self.wfile.write(dados)

        def log_message(self, fmt, *args):  # silenciar o ruído do http.server
            return

        # -- rotas --------------------------------------------------------
        def do_GET(self) -> None:
            u = urlparse(self.path)
            r = u.path
            try:
                if r == "/api/estado":
                    self._json(_estado(kianda))
                elif r == "/api/corpus":
                    self._json(_corpus(kianda))
                elif r == "/api/memoria":
                    q = (parse_qs(u.query).get("q") or [""])[0]
                    self._json(_memoria(kianda, q))
                elif r == "/api/benchmark":
                    self._json(_benchmark())
                elif r == "/api/turnos":
                    self._json({"turnos": [
                        {"entrada": t.entrada, "resposta": t.resposta.texto,
                         "humor": t.estado_depois.get("_humor"),
                         "intencao": t.percepcao.intencao,
                         "parou_em": t.raciocinio.parou_em if t.raciocinio else "-"}
                        for t in kianda.turnos[-40:]
                    ]})
                elif r in ("/", "/index.html"):
                    self._ficheiro(ESTATICOS / "index.html")
                elif r.startswith("/"):
                    self._ficheiro(ESTATICOS / r.lstrip("/"))
                else:
                    self._json({"erro": "rota desconhecida"}, 404)
            except Exception:
                self._json({"erro": traceback.format_exc()}, 500)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                corpo = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                corpo = {}
            try:
                if u.path == "/api/falar":
                    texto = (corpo.get("texto") or "").strip()
                    if not texto:
                        self._json({"erro": "texto vazio"}, 400)
                        return
                    with trava:
                        turno = kianda.conversar(texto)
                    self._json(_turno_dict(turno))
                elif u.path == "/api/manutencao":
                    with trava:
                        self._json(kianda.manutencao())
                elif u.path == "/api/reiniciar":
                    with trava:
                        kianda.turnos.clear()
                        kianda.estado = type(kianda.estado)(semente=None)
                        kianda.guardar_estado()
                    self._json({"ok": True, "estado": _estado(kianda)})
                else:
                    self._json({"erro": "rota desconhecida"}, 404)
            except Exception:
                self._json({"erro": traceback.format_exc()}, 500)

    return ThreadingHTTPServer((host, porto), Handler)


# --------------------------------------------------------------------------
# Serialização
# --------------------------------------------------------------------------


def _turno_dict(t) -> dict:
    rac = t.raciocinio
    return {
        "resposta": t.resposta.texto,
        "autor": t.resposta.autor,
        "intencao": t.percepcao.intencao,
        "confianca_percecao": t.percepcao.confianca,
        "dificuldade": t.percepcao.dificuldade,
        "humor": t.estado_depois.get("_humor"),
        "humor_antes": t.estado_antes.get("_humor"),
        "estado": t.estado_depois,
        "estado_antes": t.estado_antes,
        "plano": t.plano,
        "orcamento": t.orcamento,
        "raciocinio": {
            "parou_em": rac.parou_em if rac else None,
            "confianca": rac.confianca if rac else 0,
            "passos": [
                {"nome": p.nome, "executado": p.executado, "motivo": p.motivo,
                 "mflop": round(p.custo.flops / 1e6, 4), "confianca": p.confianca}
                for p in (rac.passos if rac else [])
            ],
            "evidencias": [
                {"texto": e.texto[:400], "citacao": e.citacao()}
                for e in (rac.evidencia[:4] if rac else [])
            ],
        },
        "memorias_usadas": t.memorias_usadas[:4],
        "custo": {
            "mflop": round(t.custo.flops / 1e6, 4),
            "kib": round(t.custo.bytes / 1024, 1),
            "ms": round(t.custo.tempo_ms, 2),
        },
        "fontes": t.resposta.fontes,
    }


def _estado(k) -> dict:
    e = k.estado
    stats = k.armazem.estatisticas()
    return {
        "humor": e.humor(),
        "variaveis": {kk: round(v, 3) for kk, v in e.v.items()},
        "historico": {kk: [round(x, 3) for x in v[-40:]] for kk, v in e.historico.items()},
        "turnos": k._turnos_totais,
        "nome_utilizador": k.nome_utilizador,
        "memoria": stats,
        "raciocinio": k.raciocinador.estatisticas(),
    }


def _corpus(k) -> dict:
    c = k.corpus_info
    idx = c.get("indexacao", {})
    return {
        "obras": c.get("obras", []),
        "capitulos": c.get("capitulos", {}),
        "documentos": idx.get("documentos"),
        "termos": idx.get("termos"),
        "semantico": idx.get("semantico", {}),
        "n_canon": c.get("n_canon"),
    }


def _memoria(k, q: str) -> dict:
    todas = k.armazem.todas()
    if q:
        usadas = k._recuperar_memorias(q, k=10, intencao="conversa")
    else:
        usadas = sorted(todas, key=lambda m: -m.forca())[:10]
    return {
        "total": len(todas),
        "memorias": [
            {
                "id": m.id, "tipo": m.tipo, "conteudo": m.conteudo,
                "importancia": round(m.importancia, 2),
                "confianca": round(m.confianca, 2),
                "forca": round(m.forca(), 2),
                "quando": m.idade_humana(),
                "emocao": m.emocao, "valencia": round(m.valencia, 2),
                "acessos": m.n_acessos,
            }
            for m in usadas
        ],
    }


def _benchmark() -> dict:
    for f in (RAIZ / "dados" / "benchmark" / "benchmark.json",
              RAIZ / "experimentos" / "resultados" / "benchmark.json"):
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                continue
    return {"erro": "sem resultados. corre: python experimentos/01_benchmark.py"}


def servir(kianda, porto: int = 8000, host: str = "0.0.0.0") -> None:
    srv = criar_servidor(kianda, porto, host)
    print(f"\n  Kianda está a ouvir em http://{host}:{porto}")
    print(f"  memória: {kianda.armazem.estatisticas()['total']} registos "
          f"· turnos: {kianda._turnos_totais}")
    print("  Ctrl+C para parar\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  a fechar…")
    finally:
        srv.server_close()
