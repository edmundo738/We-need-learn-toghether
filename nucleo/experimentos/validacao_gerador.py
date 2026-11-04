"""Validação do gerador neuronal de caracteres existente.

Condição 5 da directiva: "apresentar os resultados de validação do gerador actual".

O que este script responde, e porque cada pergunta importa
----------------------------------------------------------
A Fase 0 reportou «perplexidade 7,41». Isso não é evidência de qualidade
linguística por três razões, e este script corrige as três:

1. **Era perplexidade de TREINO.** Um modelo pode decorar o corpus e ter
   perplexidade de treino baixa. Aqui há um conjunto de validação que o
   modelo nunca vê.

2. **Não havia conjunto de validação.** O `treinar()` de `voz.py` recebe o
   texto inteiro. Este script faz a separação por fora, com corte contíguo
   (o fim do texto), para não haver fugas entre janelas adjacentes.

3. **Não se media generalização.** Aqui mede-se também quanto do texto gerado
   é cópia literal do treino — a diferença entre aprender a *forma* da língua
   e recitar o corpus.

Metodologia
-----------
- Corte contíguo: 90% treino, 10% validação. Contíguo e não aleatório porque
  janelas sobrepostas de um split aleatório partilhariam caracteres entre
  treino e validação, inflacionando o resultado.
- Perplexidade = exp(perda média por caracter) no conjunto de validação.
- Memorização = maior substring comum entre uma amostra gerada e o corpus de
  treino, normalizada pelo comprimento da amostra. Se o modelo decorou, esta
  fracção aproxima-se de 1.
- O gerador é treinado do zero aqui, com semente fixa, para ser reprodutível.

Uso:
    python -m nucleo.experimentos.validacao_gerador
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]


def carregar_texto() -> str:
    """Texto de treino do corpus, exactamente como a Kianda o vê."""
    import sys
    sys.path.insert(0, str(RAIZ))
    from kianda.linguagem import corpus  # noqa: E402

    return corpus.construir_tudo()["texto_treino"]


def dividir(texto: str, fracao_val: float = 0.10) -> tuple[str, str]:
    """Corte contíguo no fim do texto.

    Contíguo de propósito: um split aleatório de caracteres deixaria janelas
    de contexto com metade treino e metade validação, e o modelo veria os
    alvos antes de os prever.
    """
    corte = int(len(texto) * (1.0 - fracao_val))
    return texto[:corte], texto[corte:]


def perda_no_conjunto(gerador, texto: str, n_lotes: int = 40) -> tuple[float, int]:
    """Perda média por caracter num texto arbitrário, com os pesos congelados.

    Não usa `treinar()`: reconstrói o forward e mede entropia cruzada sem
    chamar `backward()` nem `opt.step()`. Uma validação que actualiza pesos
    não é validação.
    """
    from nucleo.autograd import Tensor, entropia_cruzada, matmul

    cfg = gerador.cfg
    ids = gerador._codificar(texto)
    ctx = cfg.contexto
    rng = np.random.default_rng(7)

    if len(ids) <= ctx + 1:
        return float("nan"), 0

    perdas, vistos = [], 0
    for _ in range(n_lotes):
        i = int(rng.integers(0, len(ids) - ctx - 1))
        janela = ids[i:i + ctx]
        alvo = np.array([ids[i + ctx]], dtype=np.int64)
        x = Tensor(gerador._um_quente(np.array([janela])), False)
        h = (matmul(x, gerador.W1) + gerador.b1).relu()
        h = (matmul(h, gerador.W2) + gerador.b2).relu()
        logits = matmul(h, gerador.W3) + gerador.b3
        perda = entropia_cruzada(logits, alvo)
        perdas.append(float(perda.data.item()))
        vistos += 1
    return float(np.mean(perdas)), vistos


def maior_substring_comum(a: str, b: str, minimo: int = 20) -> int:
    """Comprimento da maior substring de `a` presente em `b`.

    Usa janelas de tamanho decrescente a partir de `minimo`; serve para
    responder a «o modelo copiou ou aprendeu?». Não é o algoritmo óptimo —
    é o suficiente e é explícito sobre o que mede.
    """
    melhor = 0
    n = len(a)
    tamanho = minimo
    # Testa janelas de `a` de comprimento crecente; para na primeira falha
    # longa o suficiente para não compensar.
    while tamanho <= n:
        achou = False
        for i in range(0, n - tamanho + 1, max(1, tamanho // 4)):
            if a[i:i + tamanho] in b:
                melhor = tamanho
                achou = True
                break
        if not achou and tamanho > melhor + 40:
            break
        tamanho += 10
    return melhor


def main() -> int:
    from kianda.linguagem.voz import ConfigLM, GeradorNeuronal

    texto = carregar_texto()
    treino, val = dividir(texto)
    print(f"corpus: {len(texto):,} caracteres")
    print(f"treino: {len(treino):,} | validação: {len(val):,} (corte contíguo 90/10)")

    gerador = GeradorNeuronal(ConfigLM(contexto=20, oculto=96, vocab_max=90))
    t0 = time.perf_counter()
    info = gerador.treinar(treino, epocas=6)
    tempo_treino = time.perf_counter() - t0

    perda_val, n_val = perda_no_conjunto(gerador, val)
    perda_treino, n_tr = perda_no_conjunto(gerador, treino)

    print(f"\ntreino: {tempo_treino:.1f}s para 6 épocas")
    print(f"perda treino    : {perda_treino:.4f}  (perplexidade {math.exp(perda_treino):.2f})")
    print(f"perda validação : {perda_val:.4f}  (perplexidade {math.exp(perda_val):.2f})")
    print(f"lacuna treino-validação: {perda_val - perda_treino:+.4f}")

    # Amostras geradas e quanto delas é cópia do treino.
    print("\namostras geradas (semente «O mar»):")
    memorizacao = []
    for temp in (0.5, 0.8, 1.0):
        s, _ = gerador.gerar(semente="O mar", n=120, temperatura=temp)
        m = maior_substring_comum(s, treino)
        memorizacao.append(m / max(len(s), 1))
        print(f"  T={temp}: {s[:110]!r}")
        print(f"          maior cópia contígua do treino: {m} caracteres "
              f"({100 * m / max(len(s), 1):.0f}% da amostra)")

    resultado = {
        "corpus_caracteres": len(texto),
        "treino_caracteres": len(treino),
        "validacao_caracteres": len(val),
        "epocas": 6,
        "parametros": info.get("parametros"),
        "vocab": info.get("vocab"),
        "perda_treino": round(perda_treino, 4),
        "perda_validacao": round(perda_val, 4),
        "perplexidade_treino": round(math.exp(perda_treino), 3),
        "perplexidade_validacao": round(math.exp(perda_val), 3),
        "lacuna_val_menos_treino": round(perda_val - perda_treino, 4),
        "tempo_treino_s": round(tempo_treino, 1),
        "memorizacao_media": round(float(np.mean(memorizacao)), 4),
        "nota": ("perplexidade de caracteres — NÃO comparável com perplexidade "
                 "de modelo BPE; vocabulários de tamanhos diferentes"),
    }
    saida = RAIZ / "nucleo" / "experimentos" / "resultado_validacao.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"\nresultado guardado em {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
