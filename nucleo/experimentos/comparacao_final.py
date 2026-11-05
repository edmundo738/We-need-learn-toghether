"""A comparação que interessa: o melhor Transformer contra o trigrama.

Todas as corridas anteriores treinavam o Transformer e comparavam-no com um
número de trigrama calculado noutro ficheiro. Aqui os dois são avaliados no
MESMO conjunto (Dom Casmurro), com o MESMO texto, na mesma chamada.

Trigramas: `baselines.Interpolado` — desconto absoluto (d=0,75) com backoff
recursivo verdadeiro, exactamente o mesmo objecto que `regime_publico.py` usa
para produzir o 2,4766.

ERRO CORRIGIDO NESTE FICHEIRO: a primeira versão comparava contra um trigrama
com suavização de Laplace e salto para uniforme quando o contexto era novo.
Deu 3,9094 bpb e "o Transformer ganha por 1,23 bits" — número inválido. Era
precisamente o defeito que `baselines.py` documenta como apanhado pela
auditoria externa: um baseline sabotado faz o modelo parecer melhor do que é.
Não se aceita um número novo só porque é favorável.

O Transformer é o `machado_melhor` (passo 550).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CTX = 128


def main() -> None:
    from nucleo.experimentos.metricas import nats_para_bits_por_byte
    from nucleo.modelo.transformer import Transformer, perda_entropia_cruzada
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    obras = {f.stem: tok.codificar(f.read_text(encoding="utf-8"))
             for f in sorted((RAIZ / "dados" / "publico_limpo").glob("*.txt"))}
    treino, valid = [], []
    for n, ids in obras.items():
        (valid if n == "dom_casmurro" else treino).extend(ids)
    V = tok.tamanho_vocab

    # --- trigrama com o baseline verdadeiro (desconto absoluto + backoff) ---
    from nucleo.experimentos.baselines import Interpolado

    vocab = max(max(i) for i in obras.values()) + 1
    tri = Interpolado(3, vocab, 0.75)
    tri.treinar(np.array(treino, dtype=np.int64))
    nll_tri = 0.0
    n_tok = 0
    for i in range(2, len(valid)):
        ctx = (int(valid[i - 2]), int(valid[i - 1]))
        nll_tri += -tri.logp(ctx, int(valid[i]))
        n_tok += 1
    perda_tri = nll_tri / n_tok

    # --- Transformer no mesmo texto ---
    m = Transformer.carregar(RAIZ / "nucleo" / "experimentos" / "checkpoints" / "machado_melhor")
    janelas = np.array([valid[i * CTX:(i + 1) * CTX]
                        for i in range(len(valid) // CTX)], dtype=np.int64)
    ps = []
    for i in range(0, len(janelas), 8):
        w = janelas[i:i + 8]
        ps.append(float(perda_entropia_cruzada(
            m.forward(w[:, :-1]), w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    perda_tr = float(np.mean(ps))

    res = {
        "classe": "MEASURED",
        "validacao": "Dom Casmurro (obra inteira, nunca vista no treino)",
        "tokens_validacao": int(n_tok),
        "janelas_validacao": int(len(janelas)),
        "trigrama": {"perda_nats": round(perda_tri, 4),
                     "bits_por_byte": round(nats_para_bits_por_byte(perda_tri), 4),
                     "suavizacao": "desconto absoluto d=0,75 + backoff recursivo",
                     "vocabulario": vocab},
        "transformer": {"checkpoint": "machado_melhor", "passo": 550,
                        "perda_nats": round(perda_tr, 4),
                        "bits_por_byte": round(nats_para_bits_por_byte(perda_tr), 4)},
    }
    d = res["transformer"]["bits_por_byte"] - res["trigrama"]["bits_por_byte"]
    res["diferenca_bpb"] = round(d, 4)
    res["vencedor"] = "trigrama" if d > 0 else "transformer"
    res["nota"] = ("Avaliação justa: ambos no mesmo texto, mesma tokenização, "
                   "mesma divisão. O trigrama não tem parâmetros treinados; "
                   "o Transformer tem 2,77 M. Se o Transformer não ganha, "
                   "mais capacidade não está a ser convertida em melhor previsão.")
    (RAIZ / "nucleo" / "experimentos" / "resultado_comparacao_final.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"trigrama     {res['trigrama']['perda_nats']:.4f} nats -> "
          f"{res['trigrama']['bits_por_byte']:.4f} bpb")
    print(f"transformer  {res['transformer']['perda_nats']:.4f} nats -> "
          f"{res['transformer']['bits_por_byte']:.4f} bpb")
    print(f"\nvencedor: {res['vencedor'].upper()} por {abs(d):.4f} bits/byte")
    print(f"({res['tokens_validacao']} tokens de validação)")


if __name__ == "__main__":
    main()
