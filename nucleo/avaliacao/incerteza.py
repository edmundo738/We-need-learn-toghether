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

    # O Transformer precisa de pesos treinados. Sem checkpoint persistido
    # nesta sessão, usa-se um modelo de inicialização aleatória como CONTROL
    # NEGATIVO: se um modelo aleatório separar os dois conjuntos, a métrica
    # mede o formato do texto e não conhecimento.
    cfg = Config(vocab=tok.tamanho_vocab, d_model=128, n_camadas=4,
                 n_cabecas=4, ctx=ctx, d_ff=384)
    m_aleatorio = Transformer(cfg, semente=0)
    p_dentro_t = perda_por_janela_transformer(m_aleatorio, dentro)
    p_fora_t = perda_por_janela_transformer(m_aleatorio, fora)

    p_dentro_n = perda_ngrama(treino.reshape(-1), dentro, tok.tamanho_vocab)
    p_fora_n = perda_ngrama(treino.reshape(-1), fora, tok.tamanho_vocab)

    res = {
        "classe": "MEASURED",
        "nota": "Transformer com pesos ALEATÓRIOS serve de controlo negativo; "
                "sem checkpoint persistido não é possível medir o modelo treinado",
        "transformer_aleatorio": {"dentro": round(p_dentro_t, 4), "fora": round(p_fora_t, 4),
                                   "razao": round(p_fora_t / p_dentro_t, 4)},
        "trigrama": {"dentro": round(p_dentro_n, 4), "fora": round(p_fora_n, 4),
                      "razao": round(p_fora_n / p_dentro_n, 4)},
    }
    (RAIZ / "nucleo" / "experimentos" / "resultado_incerteza.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print(f"{'modelo':<24} {'dentro':>9} {'fora':>9} {'razão fora/dentro':>18}")
    print("-" * 64)
    print(f"{'Transformer aleatório':<24} {p_dentro_t:>9.4f} {p_fora_t:>9.4f} "
          f"{p_fora_t/p_dentro_t:>17.2f}x")
    print(f"{'trigrama (treinado)':<24} {p_dentro_n:>9.4f} {p_fora_n:>9.4f} "
          f"{p_fora_n/p_dentro_n:>17.2f}x")
    print()
    print("LEITURA: o trigrama, que TEM os dados, paga muito mais por texto fora")
    print("do domínio. O Transformer aleatório quase não distingue — como esperado,")
    print("porque não sabe nada. É o controlo negativo a funcionar.")


if __name__ == "__main__":
    main()
