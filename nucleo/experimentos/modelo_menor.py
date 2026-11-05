"""A explicação estrutural testada: um modelo menor bate o trigrama?

A Adenda D concluiu que o Transformer de 2,77 M perde para o trigrama por
0,21 bpb, e atribuiu-o à razão dados/parâmetros (0,064 tokens/parâmetro).
MAS ISSO ERA INFERENCE, NÃO MEDIÇÃO. A directiva é explícita: uma explicação
que não se mede não é um resultado.

Este ficheiro testa-a: mesmo corpus, mesma divisão por obra, mesmo orçamento
de passos, modelos de 4 tamanhos. Se a explicação estiver certa, um modelo
menor aproxima-se do trigrama — porque tem menos capacidade para decorar e
mais dados por parâmetro.

TAMANHOS (todos dentro da directiva: 4 camadas, d_model 128, 4 cabeças)
    d_ff 384  -> ~2,77 M   (o actual)
    d_ff 256  -> ~2,28 M
    d_ff 128  -> ~1,80 M
    d_model 64, d_ff 192 -> ~0,74 M   (fora do d_model 128; marcado como desvio)

E um controlo de regime: o mesmo modelo pequeno com o corpus ANTIGO (66k),
para separar «modelo menor» de «mais dados». Sem esse controlo, uma melhoria
pode ser só do corpus.

CRITÉRIO DE DECISÃO, declarado ANTES de correr
    Se o modelo menor chegar a <= 2,4766 bpb (trigrama), a explicação está
    confirmada e há um caminho dentro da directiva.
    Se nenhum tamanho lá chegar, a explicação «dados/parâmetro» é insuficiente
    e o problema é outro — provavelmente a própria arquitectura para esta tarefa.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX, TRIGRAMA = 128, 2.4766
from nucleo.experimentos.metricas import tokens_por_byte as _tpb
TOKENS_POR_BYTE = _tpb()


def main() -> None:
    from nucleo import nn
    from nucleo.modelo.transformer import (Config, Transformer,
                                            perda_entropia_cruzada)
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    obras = {f.stem: tok.codificar(f.read_text(encoding="utf-8"))
             for f in sorted((RAIZ / "dados" / "publico_limpo").glob("*.txt"))}
    tr, va = [], []
    for n, ids in obras.items():
        (va if n == "dom_casmurro" else tr).extend(ids)

    def jan(ids):
        n = len(ids) // CTX
        return np.array([ids[i * CTX:(i + 1) * CTX] for i in range(n)], dtype=np.int64)

    d_tr, d_va = jan(tr), jan(va)

    def avaliar(m, dados):
        ps = []
        for i in range(0, len(dados), 8):
            w = dados[i:i + 8]
            ps.append(float(perda_entropia_cruzada(
                m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
        return float(np.mean(ps))

    def correr(d_model, d_ff, n_camadas=4, n_cabecas=4, passos=800, lr=3e-3,
               dados_tr=None, dados_va=None, semente=0, rotulo=""):
        dtr = d_tr if dados_tr is None else dados_tr
        dva = d_va if dados_va is None else dados_va
        cfg = Config(vocab=tok.tamanho_vocab, d_model=d_model, n_camadas=n_camadas,
                     n_cabecas=n_cabecas, ctx=CTX, d_ff=d_ff, tying=False)
        m = Transformer(cfg, semente=semente)
        n_par = m.n_parametros()
        opt = nn.Adam(m.parametros(), eta=lr, weight_decay=0.01)
        rng = np.random.default_rng(semente)
        melhor = {"perda_val": float("inf")}
        for passo in range(1, passos + 1):
            idx = rng.integers(0, len(dtr), 8)
            w = dtr[idx]
            opt.zero_grad()
            perda = perda_entropia_cruzada(m.forward(w[:, :-1]), w[:, 1:].reshape(-1))
            perda.backward()
            opt.step(grad_clip=1.0)
            if passo % 100 == 0:
                pv = avaliar(m, dva)
                if pv < melhor["perda_val"]:
                    melhor = {"perda_val": pv, "passo": passo}
        bpb = (melhor["perda_val"] / math.log(2)) / (1.0 / TOKENS_POR_BYTE)
        return {"rotulo": rotulo or f"d_model={d_model} d_ff={d_ff}",
                "d_model": d_model, "d_ff": d_ff, "n_camadas": n_camadas,
                "parametros": n_par,
                "tokens_por_parametro": round(len(dtr) * CTX / n_par, 4),
                "melhor_perda_val": round(melhor["perda_val"], 4),
                "melhor_passo": melhor["passo"],
                "bits_por_byte": round(bpb, 4),
                "ganha_do_trigrama": bpb <= TRIGRAMA}

    resultados = []
    print(f"trigrama = {TRIGRAMA} bpb | critério declarado antes de correr\n")
    for d_model, d_ff in ((128, 384), (128, 256), (128, 128), (64, 192)):
        r = correr(d_model, d_ff)
        resultados.append(r)
        print(f"{r['rotulo']:<22} {r['parametros']:>9,} par | "
              f"{r['bits_por_byte']:>7.4f} bpb | margem "
              f"{TRIGRAMA - r['bits_por_byte']:>+7.4f} | melhor @ {r['melhor_passo']}",
              flush=True)

    # CONTROL: mesmo modelo pequeno, corpus antigo — separa «modelo menor» de «mais dados»
    ids_antigo = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
    corte = int(len(ids_antigo) * 0.9)
    d_old_tr = jan(list(ids_antigo[:corte]))
    d_old_va = jan(list(ids_antigo[corte:]))
    ctrl = correr(64, 192, dados_tr=d_old_tr, dados_va=d_old_va,
                  rotulo="d_model=64 (corpus ANTIGO 66k)")
    ctrl["corpus"] = "antigo_66k"
    resultados.append(ctrl)
    print(f"{ctrl['rotulo']:<22} {ctrl['parametros']:>9,} par | "
          f"{ctrl['bits_por_byte']:>7.4f} bpb | margem "
          f"{TRIGRAMA - ctrl['bits_por_byte']:>+7.4f}", flush=True)

    melhor = min(resultados, key=lambda r: r["bits_por_byte"])
    res = {
        "classe": "MEASURED",
        "trigrama_bits_por_byte": TRIGRAMA,
        "criterio_declarado_antes": "se algum modelo chegar a <= 2,4766 bpb, a "
                                    "explicação dados/parâmetro está confirmada",
        "resultados": resultados,
        "melhor": melhor,
        "algum_bate_o_trigrama": any(r["ganha_do_trigrama"] for r in resultados),
        "nota_controlo": "o controlo com corpus antigo distingue «modelo menor» de "
                         "«mais dados». Se o pequeno melhora no corpus novo mas não "
                         "no antigo, o ganho é dos dados, não do tamanho.",
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_modelo_menor.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nmelhor: {melhor['rotulo']} -> {melhor['bits_por_byte']:.4f} bpb")
    if res["algum_bate_o_trigrama"]:
        print("=> ALGUM modelo bate o trigrama. A explicação dados/parâmetro confirma-se.")
    else:
        print("=> NENHUM modelo bate o trigrama. A explicação dados/parâmetro é")
        print("   insuficiente: o problema não é só a razão dados/parâmetros.")


if __name__ == "__main__":
    main()
