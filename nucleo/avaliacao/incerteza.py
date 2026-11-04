"""O modelo sabe que não sabe? — medível com o que existe.

A perplexidade média diz que o modelo está incerto EM MÉDIA. Não diz se
ele distingue o que sabe do que não sabe. A pergunta operacional:

    a perda em texto do domínio é menor que a perda em texto fora do domínio?

Se sim, o modelo tem alguma noção de familiaridade — é o pré-requisito para
dizer «não sei». Se não, confunde igualmente tudo e nenhuma política de
factualidade funcionaria.

COMO SE MEDE SEM CORPUS NOVO
----------------------------
Em vez de inventar texto fora do domínio, usa-se texto que o modelo NUNCA
viu de duas naturezas:
  - DENTRO do domínio: a validação do próprio corpus (52 janelas, mesmo autor);
  - FORA do domínio: texto sintético que não é prosa portuguesa (sequências
    de tokens do vocabulário escolhidas ao acaso — não é "inglês", é "nada").

O segundo caso é um limite inferior grosseiro, e está declarado como tal:
um modelo treinado em 66 mil tokens de uma obra não tem como reconhecer
"fora do domínio" com subtileza. O que se mede é o caso extremo.

O BASELINE É OBRIGATÓRIO: o trigrama tem de ser submetido ao mesmo teste.
Se o trigrama também separa os dois, a separação é da estatística do texto,
não do modelo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]


def perda_por_janela_transformer(modelo, janelas: np.ndarray) -> float:
    from nucleo.modelo.transformer import perda_entropia_cruzada
    perdas = []
    for i in range(0, len(janelas), 8):
        w = janelas[i:i + 8]
        lg = modelo.forward(w[:, :-1])
        perdas.append(float(perda_entropia_cruzada(lg, w[:, 1:].reshape(-1)).data.reshape(-1)[0]))
    return float(np.mean(perdas))


def perda_ngrama(flat_treino: np.ndarray, janelas: np.ndarray, vocab: int,
                 ordem: int = 3, desconto: float = 0.75) -> float:
    from nucleo.experimentos.baselines import Interpolado
    m = Interpolado(ordem, vocab, desconto)
    m.treinar(flat_treino)
    total, n = 0.0, 0
    for w in janelas:
        for i in range(ordem - 1, len(w)):
            ctx = tuple(int(x) for x in w[i - ordem + 1:i])
            total += -m.logp(ctx, int(w[i]))
            n += 1
    return total / max(n, 1)


def main() -> None:
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada
    from nucleo.tokenizador.bpe import TokenizadorBPE

    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    ids = np.array(json.loads(
        (RAIZ / "nucleo" / "experimentos" / "corpus_ids_v1.json").read_text()), dtype=np.int64)
    ctx = 128
    total = len(ids) // (ctx + 1)
    dados = np.array([ids[i * (ctx + 1):(i + 1) * (ctx + 1)] for i in range(total)])
    corte = int(total * 0.9)
    treino, dentro = dados[:corte], dados[corte:]

    # Fora do domínio: tokens uniformes do vocabulário. Não é língua nenhuma.
    rng = np.random.default_rng(0)
    fora = rng.integers(0, tok.tamanho_vocab, dentro.shape)

    # CONTROL NEGATIVO: modelo de inicialização aleatória. Se ele separar os
    # dois conjuntos, a métrica mede o formato do texto e não conhecimento.
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=ctx, d_ff=384, tying=False)
    m_aleatorio = Transformer(cfg, semente=0)
    p_dentro_ale = perda_por_janela_transformer(m_aleatorio, dentro)
    p_fora_ale = perda_por_janela_transformer(m_aleatorio, fora)

    # O MODELO TREINADO: é este que responde à pergunta. Sem ele, o ficheiro
    # anterior media apenas o controlo — e um controlo não é um resultado.
    p_dentro_t = p_fora_t = None
    ckpt = RAIZ / "nucleo" / "experimentos" / "checkpoints" / "semblying_seed0"
    if (ckpt / "pesos.npz").exists():
        m_treinado = Transformer.carregar(ckpt)
        p_dentro_t = perda_por_janela_transformer(m_treinado, dentro)
        p_fora_t = perda_por_janela_transformer(m_treinado, fora)

    p_dentro_n = perda_ngrama(treino.reshape(-1), dentro, tok.tamanho_vocab)
    p_fora_n = perda_ngrama(treino.reshape(-1), fora, tok.tamanho_vocab)

    res = {
        "classe": "MEASURED",
        "nota": "checkpoint semblying_seed0 treinado 1000 passos (val 5,7156); "
                "aleatorio = controlo negativo; trigrama = baseline com os dados",
        "transformer_aleatorio": {"dentro": round(p_dentro_ale, 4),
                                   "fora": round(p_fora_ale, 4),
                                   "razao": round(p_fora_ale / p_dentro_ale, 4)},
        "transformer_treinado": None if p_dentro_t is None else {
            "dentro": round(p_dentro_t, 4), "fora": round(p_fora_t, 4),
            "razao": round(p_fora_t / p_dentro_t, 4),
            "checkpoint": str(ckpt)},
        "trigrama": {"dentro": round(p_dentro_n, 4), "fora": round(p_fora_n, 4),
                      "razao": round(p_fora_n / p_dentro_n, 4)},
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_incerteza.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"{'modelo':<26} {'dentro':>9} {'fora':>9} {'fora/dentro':>13}")
    print("-" * 62)
    print(f"{'Transformer aleatório':<26} {p_dentro_ale:>9.4f} {p_fora_ale:>9.4f} "
          f"{p_fora_ale/p_dentro_ale:>12.2f}x")
    if p_dentro_t is not None:
        print(f"{'Transformer treinado':<26} {p_dentro_t:>9.4f} {p_fora_t:>9.4f} "
              f"{p_fora_t/p_dentro_t:>12.2f}x")
    else:
        print(f"{'Transformer treinado':<26} {'—':>9} {'—':>9} {'sem ckpt':>13}")
    print(f"{'trigrama (treinado)':<26} {p_dentro_n:>9.4f} {p_fora_n:>9.4f} "
          f"{p_fora_n/p_dentro_n:>12.2f}x")
    print()
    print("LEITURA: se o Transformer TREINADO separar mais que o aleatório (1,00x),")
    print("tem alguma noção de familiaridade — pré-requisito para dizer «não sei».")
    print("Se ficar em ~1,00x como o aleatório, não distingue saber de não saber,")
    print("e nenhuma política de factualidade funcionaria neste modelo.")


if __name__ == "__main__":
    main()
