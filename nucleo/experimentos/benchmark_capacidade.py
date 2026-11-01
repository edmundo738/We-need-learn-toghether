"""Benchmark de capacidade de treino — com definições explícitas.

Porque este ficheiro existe
---------------------------
A Fase 0 reportou «29 849 tokens/s». O Edmundo pediu, e com razão, que se
esclarecesse o que isso mede. A resposta honesta é: **não é uma medição de
Transformer**. Era uma medição de throughput de multiplicação de matrizes
através do autograd NumPy do projecto, que foi ainda por cima contaminada por
outro processo a correr em paralelo (numa re-execução isolada deu 121 780, e
uma segunda deu 168 452 — três números para o mesmo código).

Este script substitui aquilo por uma medição com método:

1. **Aquecimento** antes de medir (as primeira chamadas pagam alocação).
2. **Repetições** e mediana, não uma só corrida.
3. **Definição por escrito** do que está a ser medido em cada caso.
4. **Comparação honesta** entre o autograd próprio e o PyTorch — para que a
   decisão de framework assente em números medidos, não em preferência.

O que NÃO é medido aqui
-----------------------
O custo de um Transformer real. Sem atenção, sem LayerNorm, sem embeddings,
sem softmax sobre a sequência. Um bloco verdadeiro custa substancialmente mais
do que os matmuls aqui testados, e o número real será medido na Fase 2, com o
modelo construído. Estimar o Transformer a partir destes números seria
repetir o erro da Fase 0 com mais decimais.

Uso:
    python -m nucleo.experimentos.benchmark_capacidade
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))


def _ambiente() -> dict:
    import numpy as np

    info = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "cpu": platform.processor() or "desconhecido",
        "numpy": np.__version__,
    }
    info["numpy_threads"] = __import__("os").environ.get("OMP_NUM_THREADS", "auto")
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["gpu"] = torch.cuda.is_available()
        # Em 2 vCPU o default pode ser 1 fio. Isto mede o efeito, em vez de o
        # assumir: o benchmark seguinte corre com o valor que o PyTorch usa.
        info["torch_cpu_count"] = __import__("os").cpu_count()
    except ImportError:
        info["torch"] = "ausente"
    return info


def benchmark_autograd(repeticoes: int = 15, passos_por_rep: int = 5) -> dict:
    """Throughput do autograd próprio em FWD+BWD+Adam sobre 4 matmuls.

    MEDE: `tokens/s` = (passos × lote × seq) / tempo, onde cada passo faz
    forward de 4 matmuls d=128, backward, e um passo de Adam.
    NÃO MEDE: um Transformer.
    """
    import numpy as np

    from kianda.nucleo.autograd import Tensor, matmul
    from kianda.nucleo.nn import Adam

    B, S, d = 16, 256, 128
    rng = np.random.default_rng(0)
    pesos = [Tensor(rng.standard_normal((d, d)).astype(np.float32) * 0.09, True)
             for _ in range(4)]
    x = Tensor(rng.standard_normal((B * S, d)).astype(np.float32), False)
    opt = Adam({f"w{i}": w for i, w in enumerate(pesos)}, eta=1e-3)

    def passo() -> None:
        h = x
        for w in pesos:
            h = matmul(h, w).relu()
        h.sum().backward()
        opt.step()

    for _ in range(5):          # aquecimento
        passo()

    amostras = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        for _ in range(passos_por_rep):
            passo()
        dt = time.perf_counter() - t0
        amostras.append(passos_por_rep * B * S / dt)

    return {
        "caso": "autograd_numPy: 4 matmuls d=128, FWD+BWD+Adam",
        "tokens_s_mediana": int(statistics.median(amostras)),
        "tokens_s_min": int(min(amostras)),
        "tokens_s_max": int(max(amostras)),
        "dispersao_pct": round(100 * (max(amostras) - min(amostras)) / statistics.median(amostras), 1),
        "mede_transformer": False,
    }


def benchmark_torch(repeticoes: int = 15, passos_por_rep: int = 5) -> dict | None:
    """Mesma operação, em PyTorch, para comparação directa.

    Igual ao anterior em formas e em trabalho: 4 matmuls d=128, forward,
    backward, passo de Adam. Só muda a biblioteca. É a única forma justa de
    comparar os dois.
    """
    try:
        import torch
    except ImportError:
        return None

    B, S, d = 16, 256, 128
    torch.manual_seed(0)
    pesos = [torch.nn.Parameter(torch.randn(d, d) * 0.09) for _ in range(4)]
    x = torch.randn(B * S, d)
    opt = torch.optim.Adam(pesos, lr=1e-3)

    def passo() -> None:
        h = x
        for w in pesos:
            h = torch.relu(h @ w)
        perda = h.sum()
        opt.zero_grad()
        perda.backward()
        opt.step()

    for _ in range(5):
        passo()

    amostras = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        for _ in range(passos_por_rep):
            passo()
        dt = time.perf_counter() - t0
        amostras.append(passos_por_rep * B * S / dt)

    return {
        "caso": "pytorch: 4 matmuls d=128, FWD+BWD+Adam",
        "torch": torch.__version__,
        "tokens_s_mediana": int(statistics.median(amostras)),
        "tokens_s_min": int(min(amostras)),
        "tokens_s_max": int(max(amostras)),
        "dispersao_pct": round(100 * (max(amostras) - min(amostras)) / statistics.median(amostras), 1),
        "mede_transformer": False,
    }


def benchmark_bloco_aproximado(repeticoes: int = 10) -> dict | None:
    """Custo de uma operação com a ORDEM de grandeza de atenção causal.

    MEDE: Q@K^T + máscara + softmax + @V para (lote, cabeças, seq, d_cabeca),
    só em PyTorch. Continua a NÃO ser um Transformer completo — falta a MLP,
    as normais e o resto — mas já inclui o termo quadrático na sequência, que
    é o que domina em contexto 256.

    Serve para responder à pergunta certa: «quanto custa a atenção nesta
    máquina?», em vez de «quanto custam quatro matmuls?».
    """
    try:
        import torch
    except ImportError:
        return None

    B, H, S, dc = 8, 4, 256, 32
    torch.manual_seed(0)
    q = torch.randn(B, H, S, dc, requires_grad=True)
    k = torch.randn(B, H, S, dc, requires_grad=True)
    v = torch.randn(B, H, S, dc, requires_grad=True)
    mascara = torch.triu(torch.ones(S, S), diagonal=1).bool()

    def passo() -> None:
        pontos = q @ k.transpose(-2, -1) / (dc ** 0.5)
        pontos = pontos.masked_fill(mascara, float("-inf"))
        pesos = torch.softmax(pontos, dim=-1)
        saida = pesos @ v
        saida.sum().backward()

    for _ in range(3):
        passo()
    q.grad = k.grad = v.grad = None

    amostras = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        passo()
        q.grad = k.grad = v.grad = None
        dt = time.perf_counter() - t0
        amostras.append(B * S / dt)

    return {
        "caso": "pytorch: atencao causal B=8 H=4 S=256 d=32, FWD+BWD",
        "tokens_s_mediana": int(statistics.median(amostras)),
        "mede_transformer": False,
        "mede_atencao": True,
    }


def main() -> int:
    print("=" * 74)
    print("CAPACIDADE DE TREINO — medição com definições explícitas")
    print("=" * 74)

    amb = _ambiente()
    print("\nambiente:")
    for k, v in amb.items():
        print(f"  {k:16s} {v}")

    print("\nA medir (cada caso isolado, aquecido, mediana de 15 repetições)…")
    print("  NOTA: nada aqui é um Transformer. Ver a docstring do módulo.\n")

    resultados = {"ambiente": amb, "casos": []}

    r1 = benchmark_autograd()
    resultados["casos"].append(r1)
    print(f"  {r1['caso']}")
    print(f"    {r1['tokens_s_mediana']:,} tokens/s "
          f"[{r1['tokens_s_min']:,}–{r1['tokens_s_max']:,}, dispersão {r1['dispersao_pct']}%]")

    r2 = benchmark_torch()
    if r2:
        resultados["casos"].append(r2)
        print(f"  {r2['caso']}")
        print(f"    {r2['tokens_s_mediana']:,} tokens/s "
              f"[{r2['tokens_s_min']:,}–{r2['tokens_s_max']:,}, dispersão {r2['dispersao_pct']}%]")
        rap = r2["tokens_s_mediana"] / max(r1["tokens_s_mediana"], 1)
        print(f"    → PyTorch é {rap:.1f}× o autograd próprio nesta operação")

    r3 = benchmark_bloco_aproximado()
    if r3:
        resultados["casos"].append(r3)
        print(f"  {r3['caso']}")
        print(f"    {r3['tokens_s_mediana']:,} tokens/s")
        if r2:
            print(f"    → a atenção custa "
                  f"{r2['tokens_s_mediana'] / max(r3['tokens_s_mediana'], 1):.1f}× "
                  f"mais do que os matmuls equivalentes")

    saida = RAIZ / "nucleo" / "experimentos" / "resultado_benchmark.json"
    saida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"\nresultado guardado em {saida}")
    print("\nO que estes números NÃO dizem:")
    print("  · nada sobre o custo de um Transformer completo")
    print("  · nada sobre a qualidade do que o modelo aprende")
    print("  · nada sobre esta máquina ser representativa da sua")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
