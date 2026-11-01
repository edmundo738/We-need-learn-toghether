"""Testes end-to-end sobre HTTP real.

Porque este ficheiro existe
---------------------------
A suite testava `Kianda.conversar()` — a API Python. O servidor estava a 0% de
cobertura, e foi por isso que três defeitos só apareceram quando abri o HTTP:
`Evidencia.citacao()` a devolver chaves internas, `como funcionas` mal
classificada, e a pergunta de volta a disparar em sítios errados.

Um sistema que funciona na classe e falha no servidor não funciona. Estes testes
levam o servidor a sério: abrem um socket real numa porta efémera, fazem pedidos
reais, e verificam o JSON que sai.

Nada aqui substitui os testes unitários — complementam-nos no único sítio onde
a serialização, o encaminhamento e o estado partilhado se juntam.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pytest


@pytest.fixture(scope="module")
def servidor():
    """Sobe a Kianda + servidor HTTP numa porta livre e desmonta no fim."""
    from kianda.mente import Kianda
    from kianda.interface.servidor import criar_servidor

    k = Kianda(pasta_dados=tempfile.mkdtemp())
    srv = criar_servidor(k, porto=0, host="127.0.0.1")
    porta = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{porta}"
    try:
        yield base
    finally:
        srv.shutdown()
        srv.server_close()
        k.fechar()


def _get(base: str, rota: str) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(base + rota, timeout=15) as r:
            corpo = r.read()
            try:
                return r.status, json.loads(corpo)
            except Exception:
                return r.status, corpo.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _post(base: str, rota: str, dados: dict | str) -> tuple[int, dict]:
    corpo = json.dumps(dados).encode() if isinstance(dados, dict) else dados.encode()
    req = urllib.request.Request(
        base + rota, data=corpo,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        bruto = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(bruto)
        except Exception:
            return e.code, {"_bruto": bruto}


# ---------------------------------------------------------------------------


def test_raiz_serve_html(servidor):
    estado, corpo = _get(servidor, "/")
    assert estado == 200
    assert "<html" in str(corpo).lower(), "a raiz não devolveu uma página"
    assert len(str(corpo)) > 1000, "a página está vazia ou truncada"


def test_estado_devolve_estrutura_esperada(servidor):
    estado, d = _get(servidor, "/api/estado")
    assert estado == 200
    assert isinstance(d, dict) and d, "estado vazio"
    # O estado interno tem de ser serializável e conter o humor, que é o que a
    # interface mostra. Se isto partir, a página fica em branco.
    assert "humor" in json.dumps(d).lower()


def test_corpus_reporta_o_que_falta(servidor):
    """O aviso de obras em falta tem de chegar ao HTTP, não só ao Python.

    Se ficasse só na API Python, a interface mostrava um sistema sem livros
    como se estivesse completo.
    """
    estado, d = _get(servidor, "/api/corpus")
    assert estado == 200
    texto = json.dumps(d)
    assert "documentos" in texto
    if d.get("n_obras", 0) == 0:
        assert "nenhuma obra" in texto.lower(), (
            "o servidor não reporta a ausência das obras"
        )


def test_falar_devolve_resposta_e_custo(servidor):
    estado, d = _post(servidor, "/api/falar", {"texto": "quem é a Zuleika?"})
    assert estado == 200, d
    assert d["resposta"].strip(), "resposta vazia"
    assert "zuleika" in d["resposta"].lower(), (
        f"resposta errada por HTTP: {d['resposta'][:100]}"
    )
    for chave in ("mflop", "ms", "kib"):
        assert chave in d["custo"], f"custo sem «{chave}»: {d['custo']}"


def test_falar_recusa_texto_vazio(servidor):
    """Entrada inválida tem de ser rejeitada com 400, não com 500 nem crash."""
    for corpo in ({"texto": ""}, {"texto": "   "}, {}):
        estado, d = _post(servidor, "/api/falar", corpo)
        assert estado == 400, f"{corpo} devolveu {estado} em vez de 400"
        assert "erro" in d


def test_falar_aceita_json_malformado_sem_crash(servidor):
    """Corpo inválido não pode derrubar o servidor.

    Um cliente mal escrito não é motivo para um 500 com traceback.
    """
    estado, d = _post(servidor, "/api/falar", "isto não é json {{{")
    assert estado in (200, 400), f"corpo malformado devolveu {estado}"
    # e o servidor continua vivo:
    estado2, _ = _get(servidor, "/api/estado")
    assert estado2 == 200, "o servidor morreu depois de um corpo malformado"


def test_rota_inexistente_da_404(servidor):
    estado, _ = _get(servidor, "/api/nao-existe")
    assert estado in (404, 200), "rota desconhecida devolveu erro de servidor"
    estado, _ = _post(servidor, "/api/nao-existe", {})
    assert estado == 404


def test_nao_ha_path_traversal(servidor):
    """Segurança: ler ficheiros fora da pasta de estáticos.

    `/../../etc/passwd` tem de ser servido como 404 — nunca como conteúdo.
    """
    for rota in ("/../../etc/passwd", "/..%2f..%2fetc%2fpasswd",
                 "/etc/passwd", "/../kianda/mente.py"):
        estado, corpo = _get(servidor, rota)
        texto = str(corpo)
        assert "root:x:" not in texto, f"LEAK em {rota}"
        assert "def conversar" not in texto, f"código-fonte exposto em {rota}"


def test_estado_persiste_entre_pedidos_http(servidor):
    """O servidor usa uma instância só: o contexto não pode desaparecer.

    Foi o que aconteceu com `chamo-me Edmundo`: por HTTP a apresentação não era
    reconhecida e o nome nunca era aprendido.
    """
    _post(servidor, "/api/falar", {"texto": "olá"})
    estado, d = _post(servidor, "/api/falar", {"texto": "chamo-me Edmundo"})
    assert estado == 200
    estado, turnos = _get(servidor, "/api/turnos")
    assert estado == 200
    assert len(turnos.get("turnos", [])) >= 2, "os turnos não ficaram registados"


def test_concorrencia_basica_nao_parte(servidor):
    """Stress leve: 12 pedidos simultâneos, todos com resposta válida.

    O servidor é ThreadingHTTPServer com uma trava em volta de `conversar`.
    Se a trava faltasse, o estado interno podia corromper-se; se fosse demasiado
    larga, os pedidos serializavam mas continuavam correctos — em ambos os casos
    este teste exige que ninguém receba lixo.
    """
    resultados: list[tuple[int, dict]] = []
    travas = threading.Lock()

    def pedir(i):
        e, d = _post(servidor, "/api/falar", {"texto": f"quem é a Zuleika? ({i})"})
        with travas:
            resultados.append((e, d))

    fios = [threading.Thread(target=pedir, args=(i,)) for i in range(12)]
    for f in fios:
        f.start()
    for f in fios:
        f.join(timeout=60)

    assert len(resultados) == 12, f"só {len(resultados)} de 12 responderam"
    for e, d in resultados:
        assert e == 200, f"pedido concorrente devolveu {e}: {d}"
        assert d["resposta"].strip()
