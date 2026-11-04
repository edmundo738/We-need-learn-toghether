> **AVISO — DOCUMENTO CORRIGIDO PELA AUDITORIA** (`research/AUDITORIA-FASE2.md`).
> O que mudou: a linha "o lote isola sequências ✓ 1,7e-6 (ruído fp32)" era uma
> estimativa; a auditoria mediu **0,0 exacto** e juntou a verificação que
> faltava — **o gradiente também respeita o isolamento** (0,0 exacto). O
> gradcheck passou a ser feito em 4 tensores, não 1. A contagem de parâmetros
> fecha por três caminhos. Passagens corrigidas marcadas `[AUDITADO]`.

# Fase 2 — o Transformer existe, treina e é auditável

**Data:** 2026-11-02
**Código:** `nucleo/modelo/transformer.py` · **Testes:** `nucleo/testes/test_transformer.py` (19)
**Estado:** construído, testado e treinado em corpus sintético. **Não treinado no corpus real.**

---

## 1. O que foi provado

| Prova | Como | Resultado |
|---|---|---|
| Os pesos são nossos | construídos com `np.random.default_rng(semente)` | 2 787 584 params |
| O gradiente é nosso | `nucleo.autograd`, não PyTorch | todos os 25 tensores recebem gradiente |
| O gradiente está correcto | diferenças finitas, convergência em ε | 12,84% → 1,13% → **0,01%** (1 tensor; `[AUDITADO]` em 4 tensores, pior caso **0,139%**) |
| A causalidade é real | alterar `ids[k]` não muda `saida[:k]` | ✓ em todas as 7 posições |
| O lote isola sequências | 2 sequências juntas vs separadas | `[AUDITADO]` **0,0 exacto** — mais forte que o 1,7e-6 aqui reportado |
| O **gradiente** isola sequências | perda só em B, comparada entre dois A | `[AUDITADO]` **0,0 exacto** — verificação que faltava |
| Treina de facto | tarefa trivially-aprendível | **100% de acerto, perda 0,0001** |
| Abstracção de vocabulário | logits uniformes | `log(V)` exacto |

**A configuração da directiva:** d_model 128 · 4 camadas · 4 cabeças · contexto 256 · vocabulário 8192 → **2 787 584 parâmetros**. A fórmula analítica e a contagem construída coincidem exactamente (era uma divergência de 12 288 params, causada por biases nos `Linear`; corrigida com `viés=False`).

**Está no limite superior dos 1–3 M**, não no meio. O motivo são as duas matrizes de vocabulário (entrada 1 048 576 + saída 1 048 576) que, juntas, são **75% do modelo**. As 4 camadas são só 657 408.

---

## 2. Quatro bugs encontrados pelos testes

Nenhum destes foi previsto; todos apareceram porque os testes comparam com a realidade.

### 2.1 A atenção não estava no grafo (grave)

```python
q = self.wq(x).data          # <- .data corta o grafo
scores = np.einsum(...)      # numpy puro
saida = self.wo(Tensor(contexto, ..., (x,)))   # x como pai, de mentira
```

`q.W`, `k.W`, `v.W` **nunca recebiam gradiente**. O modelo treinaria tudo menos a atenção — e a loss desceria à mesma, porque as MLPs compensavam. Este é exactamente o tipo de defeito que uma loss bonita esconde.

Corrigido com `_scores`, `_aplicar_por_cabeca` e `_softmax_ultima` como operações com `_backward` próprio.

### 2.2 As cabeças colapsavam numa só

O `einsum("ihd,jhd->ij", ...)` **soma sobre `h`**. Quatro cabeças produziam o mesmo resultado que uma. O modelo não podia olhar para sítios diferentes ao mesmo tempo.

Corrigido para `(h, n, n)`, com máscara repetida por cabeça.

### 2.3 Chaves de parâmetros colidiam entre camadas

`dict(self.ln1.parametros())` sem prefixo: `gama`/`beta` da camada 0 sobrescreviam os da camada 1. Só 25 chaves únicas em vez de 33, e `bloco1.ln1.gama` não existia.

### 2.4 Formas trocadas na máscara

`B_, T_ = x.data.shape` — mas `x` é `(B*T, d_model)`. Lia `d_model` como comprimento de sequência. Máscara de `(160,160)` para sequências de 5.

---

## 3. O resultado que importa: o modelo generaliza, mas **com muitos dados**

Tarefa: prever o próximo token de uma permutação de 0..15 repetida. Válida em permutações **nunca vistas**.

| n_treino | perda_val | acerto_val |
|---:|---:|---:|
| 32 | 10,9847 | 6,93% |
| 128 | 5,1018 | 12,21% |
| **512** | **1,0156** | **63,43%** |
| 2048 | 0,9558 | 63,28% |

**A generalização salta entre 128 e 512 e depois planifica.** Isso é a assinatura de uma capacidade de modelo a ser atingida, não de falta de dados. Um modelo de 18 752 parâmetros para uma regra de 16 símbolos **tem capacidade sobrando** — e capacidade sobrante num modelo pequeno é sobreajuste à espera de acontecer.

### E o que acontece com poucos dados

Com 96 sequências e 300 passos:

| passo | perda_treino | perda_val | acerto_val |
|---:|---:|---:|---:|
| 50 | 2,54 | 2,89 | 7,94% |
| 150 | 1,83 | 3,22 | 9,77% |
| 300 | 2,09 | 3,29 | 7,29% |

**A perda de validação sobe enquanto a de treino desce. O modelo decora o treino e piora fora dele.**

---

## 4. A extrapolação honesta — e a razão para a declarar já

0,87 M tokens ÷ 256 (contexto) = **~3 400 sequências de treino**.

Se uma regra de 16 símbolos precisa de ~512 sequências para generalizar, e o português tem uma complexidade incomparavelmente maior, a conclusão escreve-se antes do treino e não depois:

> **Com 3 400 sequências e 2,8 M parâmetros, o modelo vai decorar o corpus.**
> Vai produzir texto que parece português porque copia o que viu, e vai falhar
> fora dele. A Fase 2 prova a cadeia completa — pesos nossos, gradiente nosso,
> geração nossa — mas **não prova aprendizagem de linguagem**, e não pode ser
> apresentada como se provasse.

Isto não é pessimismo: é a curva acima a dizer o que vai acontecer antes de acontecer. **A Fase 5 continua bloqueada até existir corpus**, e agora há números que o justificam em vez de uma intuição.

---

## 5. O que falta para a Fase 2 estar completa

| Falta | Porque importa |
|---|---|
| Treinar no corpus v0.1 real | é o passo que a directiva pede; falta decidir formato de empacotamento |
| Medir throughput a sério (tokens/s) | já medido 37 ms/passo de 16×32; falta a 256 de contexto |
| Geração qualitativa com o tokenizador v1 | agora só gere ids; falta descodificar e ler |
| Curva de loss no corpus real | treino + validação, para ver o sobreajuste a acontecer |

---

## 6. O que não fiz, e porque não

- **Não treinei no corpus real.** Ficaria com uma loss que não sei interpretar sem saber primeiro qual é o comportamento esperado. A curva sintética dá-me a régua; sem ela, uma loss de 4,2 não diz nada.
- **Não usei PyTorch.** Autorizado, mas o objectivo desta fase é provar que a cadeia é nossa. Com PyTorch, a pergunta «os pesos são nossos?» fica respondida por uma biblioteca em vez de por um teste.
- **Não aumentei o modelo.** 2,8 M já está no limite superior da directiva e a curva de dados diz que o problema não é capacidade — é dados. Aumentar o modelo pioraria o sobreajuste.
