"""De onde vem a memória, medido tensor a tensor.

O relatório anterior afirmava que "99% da memória é grafo e activações" sem
o medir. A auditoria externa apontou isso e apontou também que a fórmula da
atenção omitia as 4 cabeças. Este script mede:

  1. parâmetros (exacto, é só contar);
  2. cada tensor guardado no grafo durante o forward, somando as formas
     reais que o autograd aloca;
  3. os scores, com e sem o factor das cabeças;
  4. o RSS do processo como controlo independente.

Nota de honestidade sobre o método: o autograd deste núcleo não expõe o
grafo. A contagem é feita somando, por operação, o que as primitivas
guardam: `meta["guardado"]` quando existe, e as cópias necessárias ao
backward de cada uma. É uma **reconstrução a partir do código**, não uma
introspecção do runtime. Chama-se-lhe INFERRED, não MEASURED.
"""

from __future__ import annotations

import json
import resource
from pathlib import Path

import numpy as np

from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada


def _mb(bytes_: float) -> float:
    return bytes_ / (1024 * 1024)


def medir(cfg: Config, lote: int) -> dict:
    m = Transformer(cfg, semente=0)
    B, T = lote, cfg.ctx

    params_mb = m.n_parametros() * 4 / 1024 / 1024

    # --- scores, por construção -------------------------------------------
    # Com a atenção por sequência: (B*h*T, T) por camada, guardada para o
    # backward das duas einsum (scores e softmax). A antiga era (h*B*T, B*T).
    por_sequencia = B * cfg.n_cabecas * T * T * 4
    global_antiga = cfg.n_cabecas * (B * T) * (B * T) * 4

    # --- activações por camada (reconstrução a partir das primitivas) -----
    # por token: x (d), q/k/v (3d), contexto (d), atencao intermédia,
    # ln1/ln2 guardadas (2*b*T*d cada), mlp up (d_ff)
    tok = B * T
    d, dff = cfg.d_model, cfg.d_ff
    por_token_bytes = (
        d          # x de entrada do bloco
        + 3 * d    # q, k, v
        + d        # contexto da atenção
        + d        # saída da atenção
        + dff      # up da MLP
        + d        # down da MLP
        + 2 * d    # ln1(x), ln2(h) guardados (xhat e inv por norma)
    ) * 4
    camadas_mb = _mb(por_token_bytes * tok * cfg.n_camadas)

    # --- o que o forward+backward realmente põe em memória ---------------
    antes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ids = np.zeros((lote, T + 1), dtype=np.int64)
    for t in m.parametros().values():
        t.zero_grad()
    lg = m.forward(ids[:, :-1])
    logits_mb = _mb(lg.data.nbytes)
    perda_entropia_cruzada(lg, ids[:, 1:].reshape(-1)).backward()
    depois = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "lote": lote, "ctx": T, "tokens": tok,
        "parametros_mb": round(params_mb, 2),
        "scores_por_sequencia_mb": round(_mb(por_sequencia * cfg.n_camadas), 2),
        "scores_global_antiga_mb": round(_mb(global_antiga * cfg.n_camadas), 2),
        "razao_global_vs_sequencia": round(global_antiga / por_sequencia, 2),
        "ativacoes_camadas_mb": round(camadas_mb, 2),
        "logits_mb": round(logits_mb, 2),
        "rss_pico_mb": round((depois - antes) / 1024, 1),
    }


def main() -> None:
    c = Config(vocab=8192, d_model=128, n_camadas=4, n_cabecas=4, ctx=128, d_ff=384)
    print(f"parâmetros: {c.n_parametros():,} · {c.n_parametros()*4/1024/1024:.2f} MB\n")
    print(f"{'lote':>5} {'tokens':>7} {'params':>8} {'scores':>8} {'ativ':>8} "
          f"{'logits':>8} {'RSS':>9} {'razão':>6}")
    linhas = []
    for lote in (1, 2, 4, 8):
        r = medir(c, lote)
        linhas.append(r)
        print(f"{r['lote']:>5} {r['tokens']:>7} {r['parametros_mb']:>7.1f}M "
              f"{r['scores_por_sequencia_mb']:>7.1f}M {r['ativacoes_camadas_mb']:>7.1f}M "
              f"{r['logits_mb']:>7.1f}M {r['rss_pico_mb']:>8.1f}M "
              f"{r['razao_global_vs_sequencia']:>6.1f}x")
    Path("nucleo/experimentos/resultado_memoria.json").write_text(
        json.dumps(linhas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
