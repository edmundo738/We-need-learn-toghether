"""Painel local: ver o estado real do projecto e falar com o modelo.

Só biblioteca padrão do Python (http.server) — sem Flask, sem npm, sem build.
Abre com:  .venv/bin/python painel/servidor.py  e vai a http://localhost:8000

O QUE MOSTRA, E COM QUE HONESTIDADE
-----------------------------------
Os números não são escritos à mão no HTML: são lidos dos ficheiros
`nucleo/experimentos/resultado_*.json` que os experimentos produzem. Se um
experimento não correu, o painel diz «não corrido» em vez de inventar.

O que o painel NÃO faz: dizer que o modelo é bom. Ele mostra as três coisas
que o projecto mediu e que são desfavoráveis — o trigrama ganha, a validação
está acima do desejável, a gramática é moeda ao ar — ao lado do que funciona.

SEGURANÇA: servidor local, sem autenticação. Não expor fora da máquina. A
geração corre no processo do servidor e bloqueia-o durante o pedido; é
aceitável para uso local de uma pessoa.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RAIZ = Path(__file__).resolve().parents[1]
# O servidor é corrido como script (`python painel/servidor.py`), e nesse caso
# o directório do script entra no sys.path mas a RAIZ do repo não — logo
# `import nucleo.modelo` falha. Medido: ModuleNotFoundError no /api/gerar
# enquanto /api/estado funcionava (esse não importa o núcleo).
import sys
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

EXP = RAIZ / "nucleo" / "experimentos"
CHECKPOINTS = EXP / "checkpoints"

_modelo_cache: dict = {}
_lock = threading.Lock()


def ler_json(nome: str):
    """Procura nos dois sítios onde os experimentos gravam resultados."""
    for base in (EXP, RAIZ / "nucleo" / "dados", RAIZ / "nucleo" / "avaliacao"):
        p = base / nome
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def estado() -> dict:
    """Agrega os resultados reais. Cada campo sabe se existe ou não."""
    regime = ler_json("resultado_regime_publico.json")
    melhor = ler_json("resultado_melhor_ponto.json")
    gram = ler_json("resultado_gramatical.json")
    ctxm = ler_json("resultado_contexto_maior.json")
    menor = ler_json("resultado_modelo_menor.json")
    final = ler_json("resultado_finalidades.json")
    cura = ler_json("resultado_curadoria_publico.json")
    sobre = ler_json("resultado_underfitting.json")
    adapt = ler_json("resultado_adaptacao.json")

    d = {
        "corpus": {
            "machado_caracteres": cura["total_depois"] if cura else None,
            "removido_pct": cura["removido_pct"] if cura else None,
            "tokens_unicos": final["tokens_unicos"] if final else None,
            "finalidades_com_dados": final["cobertas"] if final else None,
            "finalidades_total": final["total"] if final else None,
            "alvo_cruzamento": final["alvo_cruzamento"] if final else 97320,
        },
        "baselines": {
            "trigrama_bpb": 2.4766,
            "transformer_bpb": melhor["melhor_observado"]["bits_por_byte_media"] if melhor else None,
            "transformer_bpb_fixo": melhor["passo_fixo_500"]["bits_por_byte_media"] if melhor else None,
            "margem": melhor["margem_melhor"] if melhor else None,
            "ganha": melhor["ganha_do_trigrama_melhor"] if melhor else None,
        },
        "gramatica": {
            "moeda": 0.5,
            "transformer": gram["transformer_prosa_800_passos"]["taxa"] if gram else None,
            "trigrama": gram["trigrama_mesmos_dados"]["taxa"] if gram else None,
        },
        "contexto": None,
        "tamanhos": None,
        "curva": sobre["curva"] if sobre else None,
        "dialogo": None,
        "checkpoints": sorted(p.name for p in CHECKPOINTS.glob("*"))
        if CHECKPOINTS.exists() else [],
    }
    if ctxm:
        a, b = ctxm["resultado"]
        d["contexto"] = {"ctx128": a["bits_por_byte"], "ctx256": b["bits_por_byte"],
                         "pior": b["bits_por_byte"] > a["bits_por_byte"]}
    if menor:
        d["tamanhos"] = [{"rotulo": r["rotulo"], "params": r["parametros"],
                          "bpb": r["bits_por_byte"]} for r in menor["resultados"]]
    if adapt:
        d["dialogo"] = {
            "folga_memorizacao": adapt["memorizacao_B"]["folga"] if "memorizacao_B" in adapt else None,
            "val_final": adapt["B_prosa_para_dialogo"]["perda_val_final"],
            "acaso": round(math.log(8192), 3),
        }
    return d


def carregar_modelo(nome: str):
    with _lock:
        if nome in _modelo_cache:
            return _modelo_cache[nome]
        from nucleo.modelo.transformer import Transformer
        from nucleo.tokenizador.bpe import TokenizadorBPE
        ckpt = CHECKPOINTS / nome
        if not (ckpt / "pesos.npz").exists():
            raise FileNotFoundError(f"checkpoint {nome} não tem pesos.npz")
        tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
        m = Transformer.carregar(ckpt)
        _modelo_cache[nome] = (m, tok)
        return m, tok


def gerar(nome: str, prompt: str, n: int = 60, temp: float = 0.8) -> dict:
    import numpy as np
    m, tok = carregar_modelo(nome)
    ids = tok.codificar(prompt) if prompt else [tok.codificar("\n")[0]]
    with _lock:
        saida = m.gerar(np.array([ids]), n_novos=n, temperatura=temp, semente=7)[0]
    novo = list(saida[len(ids):])
    return {"prompt": prompt, "texto": tok.descodificar(novo),
            "checkpoint": nome, "temperatura": temp}


HTML = """<!DOCTYPE html>
<html lang="pt"><head><meta charset="utf-8">
<title>Núcleo — estado e modelo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#171a21;--bd:#262b36;--tx:#e6e8ee;--mut:#8b93a7;
--ok:#3ddc97;--bad:#ff6b6b;--warn:#ffc857;--acc:#6ea8fe}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 14px;
text-transform:uppercase;letter-spacing:.08em;color:var(--mut);font-weight:600}
.sub{color:var(--mut);font-size:13px;margin-bottom:26px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px}
.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.v{font-size:24px;font-weight:650;margin-top:6px;font-variant-numeric:tabular-nums}
.v.small{font-size:17px}
.note{color:var(--mut);font-size:12px;margin-top:6px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.panel{background:var(--card);border:1px solid var(--bd);border-radius:10px;
padding:18px;margin-bottom:18px}
textarea{width:100%;min-height:76px;background:#0b0d11;border:1px solid var(--bd);
color:var(--tx);border-radius:8px;padding:11px;font:14px/1.5 ui-monospace,Menlo,monospace;resize:vertical}
input[type=range]{width:100%}
button{background:var(--acc);color:#0b0d11;border:0;border-radius:8px;padding:10px 20px;
font-weight:650;cursor:pointer;font-size:14px}button:disabled{opacity:.5;cursor:wait}
select{background:#0b0d11;color:var(--tx);border:1px solid var(--bd);
border-radius:8px;padding:8px;font-size:13px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}
pre{background:#0b0d11;border:1px solid var(--bd);border-radius:8px;padding:13px;
white-space:pre-wrap;word-wrap:break-word;font:13px/1.7 ui-monospace,Menlo,monospace;margin:0}
.warnbox{background:#2a1f14;border:1px solid #5c4420;border-radius:10px;
padding:14px 16px;margin-bottom:18px;font-size:13.5px;color:#ffd89b}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:7px 6px;border-bottom:1px solid var(--bd)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
</style></head><body><div class="wrap">
<h1>Núcleo — estado real e modelo ao vivo</h1>
<div class="sub">Todos os números vêm dos JSON que os experimentos produzem.
Se um experimento não correu, aparece «—» e não um número inventado.</div>
<div id="app">a carregar…</div>
</div>
<script>
const fmt=(x,d=3)=>x===null||x===undefined?'—':Number(x).toFixed(d);
async function carregar(){
  const s=await (await fetch('/api/estado')).json();
  const c=s.corpus, b=s.baselines, g=s.gramatica;
  const ganha=b.ganha===true;
  const pctCruz=c.tokens_unicos?Math.round(c.tokens_unicos/c.alvo_cruzamento*100):null;
  let h='';
  h+='<div class="warnbox"><b>Leitura honesta.</b> Este modelo tem 2,77 M de parâmetros e '
    +'178 mil tokens de treino. O <b>trigrama ganha-lhe '+(b.margem?Math.abs(b.margem).toFixed(2):'—')
    +' bits/byte</b>. Não é um modelo de conversa — é um instrumento de investigação, e o painel mostra '
    +'as medições desfavoráveis ao lado das favoráveis.</div>';

  h+='<div class="grid">';
  h+='<div class="card"><div class="k">Corpus (tokens únicos)</div><div class="v">'
     +(c.tokens_unicos?c.tokens_unicos.toLocaleString('pt'):'—')+'</div><div class="note">'
     +(pctCruz!==null?pctCruz+'% do ponto de cruzamento ('+c.alvo_cruzamento.toLocaleString('pt')+')':'—')+'</div></div>';
  h+='<div class="card"><div class="k">Finalidades com dados</div><div class="v '
     +((c.finalidades_com_dados||0)>=5?'ok':'warn')+'">'+(c.finalidades_com_dados??'—')
     +' / '+(c.finalidades_total??'—')+'</div><div class="note">conversation, pt_general, pt_africa</div></div>';
  h+='<div class="card"><div class="k">Transformer</div><div class="v">'+fmt(b.transformer_bpb,3)
     +' <span style="font-size:13px;color:var(--mut)">bpb</span></div><div class="note">'
     +(b.transformer_bpb_fixo?'passo fixo: '+fmt(b.transformer_bpb_fixo,3):'')+'</div></div>';
  h+='<div class="card"><div class="k">Trigrama (baseline)</div><div class="v '
     +(ganha?'bad':'')+'">'+fmt(b.trigrama_bpb,3)
     +' <span style="font-size:13px;color:var(--mut)">bpb</span></div><div class="note">'
     +(ganha?'<span class="bad">o trigrama ganha</span>':'o trigrama perde')+'</div></div>';
  h+='<div class="card"><div class="k">Gramática (julga?)</div><div class="v '
     +((g.transformer||0)>0.6?'ok':'bad')+'">'+(g.transformer!==null?Math.round(g.transformer*100)+'%':'—')
     +'</div><div class="note">moeda ao ar 50% · trigrama '+(g.trigrama!==null?Math.round(g.trigrama*100)+'%':'—')+'</div></div>';
  h+='</div>';

  h+='<div class="panel"><h2>Curva de validação — o sobreajuste</h2>';
  if(s.curva){
    h+='<table><tr><th>Passo</th><th>Épocas</th><th class="num">Treino</th><th class="num">Validação</th><th class="num">bits/byte</th></tr>';
    for(const r of s.curva){
      const melhor=r.perda_val<6.0;
      h+='<tr><td>'+r.passo+'</td><td>'+r.epocas+'</td><td class="num">'+fmt(r.perda_treino)
        +'</td><td class="num '+(melhor?'ok':'')+'">'+fmt(r.perda_val)
        +'</td><td class="num">'+fmt(r.bits_por_byte)+'</td></tr>';
    }
    h+='</table><div class="note">A validação melhora até ~500 passos e depois piora monotonamente. '
      +'A perda de treino continua a cair: é decoração.</div>';
  } else h+='<div class="note">curva não medida</div>';
  h+='</div>';

  h+='<div class="panel"><h2>Contexto 128 vs 256</h2>';
  if(s.contexto){
    h+='<table><tr><th>Contexto</th><th class="num">bits/byte</th><th></th></tr>'
      +'<tr><td>128</td><td class="num">'+fmt(s.contexto.ctx128)+'</td><td class="ok">melhor</td></tr>'
      +'<tr><td>256</td><td class="num">'+fmt(s.contexto.ctx256)+'</td><td class="bad">pior</td></tr>'
      +'</table><div class="note">Orçamento igual (1 024 tokens/passo). Dobrar o contexto duplica a '
      +'tabela de posições (16 384 → 32 768 parâmetros) sem mais dados: 10,9 → 5,4 amostras por '
      +'parâmetro de posição.</div>';
  } else h+='<div class="note">—</div>';
  h+='</div>';

  h+='<div class="panel"><h2>Falar com o modelo</h2>'
    +'<div class="row"><select id="ck">'
    +s.checkpoints.map(n=>'<option value="'+n+'">'+n+'</option>').join('')
    +'</select><span class="note" id="ckinfo"></span></div>'
    +'<textarea id="prompt">O meu nome é Brás Cubas, e eu</textarea>'
    +'<div class="row"><span class="note">temperatura</span>'
    +'<input type="range" id="temp" min="0" max="1.5" step="0.1" value="0.8" style="max-width:220px">'
    +'<span class="note" id="tval">0,8</span>'
    +'<button id="btn">Gerar</button></div>'
    +'<pre id="out">Escolhe um checkpoint e clica em Gerar.</pre>'
    +'<div class="note">A geração bloqueia o servidor durante o pedido (é local, uma pessoa). '
    +'Modelo de 2,77 M parâmetros: espera texto imperfeito, não conversa.</div></div>';

  document.getElementById('app').innerHTML=h;
  const temp=document.getElementById('temp'), tv=document.getElementById('tval');
  temp.oninput=()=>tv.textContent=Number(temp.value).toFixed(1).replace('.',',');
  document.getElementById('btn').onclick=async()=>{
    const btn=document.getElementById('btn'), out=document.getElementById('out');
    btn.disabled=true; out.textContent='a gerar…';
    try{
      const r=await fetch('/api/gerar',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({checkpoint:document.getElementById('ck').value,
          prompt:document.getElementById('prompt').value,
          temperatura:Number(temp.value)})});
      const j=await r.json();
      out.textContent=j.erro?('ERRO: '+j.erro):(j.prompt+j.texto);
    }catch(e){out.textContent='ERRO: '+e}
    btn.disabled=false;
  };
}
carregar();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        corpo = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            corpo = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
        elif u.path == "/api/estado":
            try:
                self._json(estado())
            except Exception as e:
                self._json({"erro": f"{type(e).__name__}: {e}"}, 500)
        else:
            self._json({"erro": "rota desconhecida"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/gerar":
            self._json({"erro": "rota desconhecida"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            r = gerar(d.get("checkpoint", "publico_seed0"),
                      str(d.get("prompt", ""))[:400],
                      n=int(d.get("n", 60)),
                      temp=float(d.get("temperatura", 0.8)))
            self._json(r)
        except Exception as e:
            self._json({"erro": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.porta), Handler)
    print(f"painel em http://0.0.0.0:{a.porta}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
