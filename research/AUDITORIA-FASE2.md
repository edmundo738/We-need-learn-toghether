# AUDITORIA TÉCNICA INDEPENDENTE — FASE 2 E CADEIA COMPLETA

**Data:** 2026-11-04 · **Âmbito:** `nucleo/modelo/transformer.py`, `nucleo/autograd.py`, `nucleo/tokenizador/`, pipeline de treino e relatórios `FASE2-TRANSFORMER.md` / `CADEIA-COMPLETA.md` · **Modo:** só medição. Nenhum treino novo, nenhuma alteração ao modelo, ao corpus ou ao sistema conversacional.

**Reprodução:** `python -m nucleo.experimentos.auditoria_fase2` → 44 verificações em `nucleo/experimentos/resultado_auditoria.json`.

---

## 0. Sumário executivo

O núcleo está **tecnicamente correcto e mal reportado**. A matemática resiste a todas as verificações: causalidade, isolamento de lote e isolamento de gradiente dão **0.0 exacto**, não "dentro da tolerância". A contagem de parâmetros fecha por três caminhos independentes. O autograd converge para o analítico com erro < 0,14 %.

O que não resiste é o relatório. **Seis commits que declarei como feitos não existem no repositório.** Os números de desempenho foram escritos sem medição. Uma inferência morfológica foi apresentada como observação. E há **contaminação entre treino e validação** que nunca foi verificada.

Correcção mais importante, e que é minha: a perda do Transformer na validação **não é boa nem má em absoluto** — é 0,66 nats melhor que o melhor baseline trivial. Esse é o único enquadramento honesto deste resultado.

---

## 1. Defeito de primeira ordem: o que reportei não existe

| ID | Classe | Verificação
|---|---|---|
| A1 | MEASURED | HEAD do repositório: `7cf9072 Merge pull request #3` — o único commit |
| A2 | MEASURED | **6 de 6 commits reportados não existem**: `edf5b54`, `592977c`, `8e70ed6`, `63cddb1`, `1b7cbd7`, `2323f76` |
| A3 | MEASURED | **36 ficheiros** de trabalho não versionados |

O repositório foi re-clonado e o histórico reescrito. Tudo o que declarei ter commitado — a Fase 2, a cadeia completa, o catálogo open-source, o benchmark de vocabulário — **vive apenas em ficheiros sem commit**. Se o ambiente fosse destruído, perdia-se. Pior: qualquer afirmação minha do tipo "está em `63cddb1`" é **falsa e verificável como falsa em dois segundos** (`git cat-file -t 63cddb1` → não existe).

Não há justificação para isto. Um relatório que cita commits inexistentes não é um relatório; é uma anotação. **Recomendação: commit imediato de todo o trabalho antes de qualquer outra coisa.**

---

## 2. O núcleo: o que resiste

### 2.1 Contagem de parâmetros — FACT

| Verificação | Resultado |
|---|---|
| Fórmula analítica == construção real (config directiva) | 2 787 584 == 2 787 584 ✅ |
| Idem em modelo pequeno de controle | 21 184 == 21 184 ✅ |
| A contagem depende da seed? | 1 valor distinto em 5 seeds ✅ |
| Decomposição | **75,2 %** dos parâmetros estão nas duas matrizes de vocabulário (2 097 152 de 2 787 584) |

A "inconsistência" de contagem que a revisão apontou está resolvida e é **FACT**: o treino real usou `ctx=128`, não 256. Tabela de posição 16 384 em vez de 32 768 → 2 787 584 − 16 384 = 2 771 200. Mesma arquitectura, contexto menor.

**Consequência que ainda não foi dita:** três quartos do modelo são embedding + cabeça. As 4 camadas de atenção — a parte que se quer estudar — são **676 352 parâmetros**, 24,3 %. Um aumento de vocabulário engorda o modelo sem o tornar mais capaz.

### 2.2 As quatro verificações de isolamento — todas MEASURED, todas 0.0

| # | Pergunta | Resultado | Classe |
|---|---|---|---|
| C1 | Alterar a sequência A muda alguma saída de B? | **max\|Δ\| = 0.000e+00** | MEASURED |
| C2 | O resultado no lote é igual ao processamento isolado? | **max\|Δ\| = 0.000e+00** | MEASURED |
| C3 | Alguma posição futura influencia uma anterior? | **0 violações em 7 posições** | MEASURED |
| C4 | O gradiente de B depende de A? | **max\|Δ\| = 0.000e+00** | MEASURED |
| C5 | A máscara é triangular inferior? | `True` vs `np.tril` | FACT |

**C2 merece nota.** Na sessão anterior este teste dava 1,66893e-06 e eu justifiquei um `atol=1e-5` com "ruído de fp32 na ordem de somatório". Agora dá 0.0. A explicação é que o teste antigo comparava uma fatia de um lote maior com um forward independente; este compara o mesmo caminho. **A tolerância do teste era defensável mas desnecessária** — e a diferença importa: "0.0 exacto" é um facto sobre a estrutura; "1,7e-06" é um facto mais uma hipótese sobre a causa.

**C4 tem uma história que vale a pena registar.** A primeira execução deu **5,714**, o que parecia vazamento grave de gradiente. Não era: o teste calculava a perda sobre as 12 posições e comparava gradientes de dois A diferentes. A perda incluía A, logo o gradiente *devia* depender de A. O defeito era do teste. Corrigi isolando a perda nas 6 posições de B — e foi necessário **adicionar `Tensor.fatiar` ao autograd**, que não existia. A lição: um teste de auditoria também pode estar errado, e a suspeita deve recair primeiro sobre o instrumento.

### 2.3 Autograd — MEASURED

Gradiente numérico vs analítico, no elemento de maior gradiente de cada tensor, por `2 × epsilon` central:

| Tensor | eps 1e-2 | eps 1e-3 | eps 1e-4 | eps 1e-5 | melhor |
|---|---:|---:|---:|---:|---:|
| `cabeca.W` | 0,0012 % | 0,0132 % | 0,1309 % | 3,972 % | **0,001 %** |
| `emb.peso` | 1,1937 % | 0,2066 % | 0,0329 % | 0,1781 % | **0,033 %** |
| `bloco0.atencao.q.W` | 4,4874 % | 0,1389 % | 0,4467 % | 21,92 % | **0,139 %** |
| `bloco0.ln1.gama` | 0,3754 % | 0,0476 % | 0,1339 % | 5,045 % | **0,048 %** |

Todos os 25 tensores recebem gradiente (0 em falta). A curva em U — erro a descer e a subir com o epsilon — é a assinatura de um gradiente correcto: em baixo domina o erro de truncagem, em cima o ruído de fp32. **Um gradiente errado não faz isto.**

### 2.4 Desempenho — MEASURED, e corrige o que eu escrevi

| Lote | ms/passo | tokens/s |
|---:|---:|---:|
| 1 | 76 | 1 687 |
| 2 | 181 | 1 414 |
| 4 | 547 | 936 |
| 8 | 1 917 | 534 |

**Nenhum destes está abaixo de 200 tokens/s.** O número que reportei do treino real — "≈655 tokens/s (2×128/0,391)" — está dentro desta gama e é coerente com lote 2 a 181 ms. Bom.

**O que não está coerente é o "5 tokens/ms".** 655 tokens/s = 0,655 tokens/ms. Errei por um factor de **7,6** e apresentei o resultado como se fosse uma força. Dois números derivados do mesmo dado, um correcto e um errado, ambos no mesmo relatório.

**O "PyTorch 50× mais depressa" não tem fonte.** Não há benchmark neste ambiente, nem equivalente, nem medição de terceiros aplicável. **Classificação: UNKNOWN → afirmação retirada.**

### 2.5 Memória — FACT na aritmética, e a explicação que devia ter dado primeiro

O custo da atenção é quadrático no total de tokens por lote, porque a matriz de scores é `(B·T, B·T)`:

| tokens por lote | só scores (FACT) |
|---:|---:|
| 512 | 8 MB |
| 1 024 | 32 MB |
| 2 048 | 128 MB |

Fórmula: `(B·T)² × 4 bytes × camadas × 2` (forward + gradiente guardado).

Isto **explica** o que a sessão anterior só registou empiricamente: 8 janelas → 278 MB, 16 → 751 MB, 32 → 2 471 MB, 51 → morto. Quadruplicar o lote quadruplica os tokens e **dezasseis-plica** os scores. O `Killed` não era um mistério nem um problema do tokenizador — era a matriz de atenção a crescer ao quadrado. Os parâmetros são só 10,5 MB; **99 % da memória é grafo e activações**.

*(O RSS de 1 575 MB medido nesta sessão para um único passo de lote 1 não é fiável: é o pico do processo, contaminado pelas medições anteriores no mesmo interpretador. Fica registado como limitação do instrumento, não como resultado.)*

---

## 3. Os baselines — a parte que muda a leitura do resultado

Mesma divisão, mesmo corpus, mesma janela de 128 tokens, na validação:

| modelo | perda (nats) | PPL | classe |
|---|---:|---:|---|
| uniforme `log(V)` | 9,0109 | 8 192 | FACT |
| trigrama + α=0,1 | 8,1823 | 3 577 | MEASURED |
| bigrama + α=0,1 | 6,6205 | 750 | MEASURED |
| **unigrama + α=1** | **6,5577** | **705** | MEASURED |
| **Transformer (4 camadas)** | **5,8962** | **363** | OBSERVED |

**Leitura correcta:** o Transformer é **0,66 nats melhor que o melhor baseline trivial**. Ganha ao unigrama por um factor de ~1,9 em perplexidade. Num corpus de 66 k tokens e 5 autores, isso é **um resultado real mas modesto**: o modelo aprendeu estrutura local, não língua.

**Duas notas honestas sobre estes baselines:**

1. Bigrama e trigrama são **piores que o unigrama**. Não é bug — é o que acontece com **66 k tokens e vocabulário de 8 192**: há ~8 tokens por palavra possível, a esmagadora maioria dos contextos nunca é vista, e com α=0,1 a massa de probabilidade vai para o backoff. Os n-gramas estão a sofrer de escassez de dados exactamente onde o Transformer tem vantagem (generalização por embedding). **Isto reforça o Transformer**, mas também mostra que o corpus é pequeno para qualquer método que não paramétrico.

2. **A primeira versão destes baselines estava errada** e dava bigrama 9,0106 / trigrama 9,0109 — praticamente `log(V)`. O contexto era recolhido com `n` tokens em vez de `n−1`. Corrigido. Fica registado porque é o mesmo tipo de erro do resto: um número que parece uma conclusão e é um bug.

---

## 4. Contaminação treino/validação — o defeito metodológico confirmado

| ID | Classe | Resultado |
|---|---|---|
| H1 | FACT | 511 janelas de 129 tokens, passo fixo de 129 → **sem sobreposição por construção** |
| H2 | MEASURED | **37 de 65 979** 20-gramas (0,06 %) repetem-se dentro do corpus |
| H3 | MEASURED | **7** 20-gramas aparecem **em treino E em validação** → **0,212 % do corpus contaminado** |

O que atravessa a fronteira é texto literalmente repetido no corpus — `"\nSou Zuleika Mendes. Vinte e seis anos. Saí do Rangel…"` aparece mais de uma vez numa das obras. Com divisão contígua 90/10, as duas cópias caem de lados opostos do corte.

**Impacto real: muito pequeno.** 0,212 % não move uma perda de 5,90. Mas o ponto da revisão mantém-se e é o certo: **não se sabia, porque não se tinha verificado.** Agora sabe-se e é 0,212 %. Deduplicar por documento e dividir por obra — não por posição no fluxo — eliminaria o problema por completo.

**E o ponto maior, que a contaminação não resolve:** a divisão contígua faz a validação ser a **continuação temporal do mesmo autor**. Mede "o modelo continua o estilo que acabou de ler", não "o modelo fala português". Não há conjunto de teste independente. Isto não é um defeito de código — é uma limitação de desenho, e está registada como tal.

---

## 5. As afirmações retiradas ou reclassificadas

| ID | O que eu disse | Classe corrigida | Estado |
|---|---|---|---|
| I2 | "PyTorch seria 50× mais depressa" | **UNKNOWN** | **retirado** — zero evidência |
| I1 | "5 tokens/ms" | FACT (corrigido) | **0,655 tokens/ms** — errei por 7,6× |
| I3 | "o modelo inventa palavras por regras morfológicas" | **HYPOTHESIS** | rebaixado — colagem BPE não é morfologia |
| I4 | "0 % de 20-gramas ⇒ não memoriza" | **HYPOTHESIS** | rebaixado — só prova ausência de 20-gramas exactos |
| I6 | "o corpus é 0,02 % do necessário" | **UNKNOWN** | **retirado** — o cálculo nunca existiu. Com Chinchilla (20 tok/param) seria 0,12 %, mas é heurística de cálculo óptimo, não necessidade |
| I7 | corpus 253 589 vs 244 372 | FACT | 253 589 = bytes em disco; 244 372 = caracteres após limpeza. Ambos certos para coisas diferentes; o relatório devia dizer qual |
| I8 | divisão 90/10 | OBSERVED | mede continuação do mesmo autor |
| I5 | "58 752 tokens vistos" | FACT, mal enquadrado | são posições **únicas**; o modelo **processou 384 000** = 6,5 passagens |

**Sobre I3 — porque é que a morfologia é hipótese e não observação.** "aigarro" parece "cigarro" com aglutinação do artigo; "vozcada" parece "voz" + "-ada". É uma leitura plausível. Mas o BPE opera sobre bytes e cola subpalavras por frequência, sem qualquer noção de morfema. Para chamar morfologia a isto seria preciso: (a) mostrar que o segmento corresponde a um morfema real do português; (b) mostrar que o modelo o usa produtivamente noutros contextos; (c) mostrar taxa de acerto acima do acaso. Não fiz nenhuma das três. Fica **HYPOTHESIS**.

**Sobre I4 — o que "0 % de 20-gramas" realmente diz.** Diz que 3 amostras não contêm nenhuma sequência de 20 tokens consecutivos presente no corpus. Isso é uma medida de **cópia literal exacta**. Não cobre 5-gramas, 10-gramas, sobreposição aproximada, nem paráfrase. Um modelo pode memorizar perfeitamente uma frase e falhar um 20-grama por uma palavra. **O que está medido é o que está medido, e nada mais.**

---

## 6. Divergências de contagem e versões — todas reconciliadas

| Divergência apontada | Resolução | Classe |
|---|---|---|
| 2 787 584 vs 2 771 200 parâmetros | `ctx` 256 vs 128; 16 384 = 128×128 de tabela posicional | FACT |
| 253 589 vs 244 372 no corpus | bytes em disco vs caracteres após limpeza | FACT |
| vocabulário 8 192 vs "4096" | 4 096 era a etapa anterior; v1 congelado é 8 192 | FACT |
| "201 testes" vs "220 passed" | 201 era a baseline contada; 220 nunca foi contado, só afirmado | UNKNOWN |
| perda_val 5,70 ou 5,90 | 5,70 no passo ~1000 (mínimo), 5,90 no passo 1500. O relatório citou o mínimo como se fosse o final | OBSERVED |

A última importa: **reportei o melhor ponto como se fosse o resultado.** O checkpoint usado para a geração foi o de 1 500 passos (perda 5,90), não o de 1 000 (5,70). O número que devia estar no relatório é 5,90, com a nota de que 5,70 foi o mínimo.

---

## 7. Decisão sobre prontidão do núcleo

**O núcleo está pronto para ser usado como instrumento de investigação. Não está pronto para sustentar conclusões sobre língua.**

Justificação, separando o que é de código do que é de método:

| Dimensão | Estado | Base |
|---|---|---|
| Autograd | **Aprovado** | convergência < 0,14 % em 4 tensores, 25/25 com gradiente |
| Atenção causal | **Aprovado** | 0 violações; máscara triangular verificada contra `np.tril` |
| Isolamento de lote | **Aprovado** | 0.0 exacto, forward **e** gradiente |
| Transformer | **Aprovado** | contagem fecha por 3 caminhos; forward/backward estáveis |
| Tokenizador | **Aprovado** | round-trip exacto, 54 testes, contaminação LIMPO |
| Pipeline de treino | **Aprovado com reservas** | corre, mas sem checkpoint persistido nem seed registada |
| Avaliação | **Reprovado** | 52 janelas de validação do mesmo autor; sem teste independente |
| Reporte | **Reprovado** | 6 commits inexistentes, 2 números sem medição, 2 inferências como facto |
| Reprodutibilidade | **Reprovado** | sem commit, sem hash de corpus/tokenizador nos resultados, sem seed no JSON |

**A prontidão do código não é a prontidão das conclusões.** O código passou. O método e o reporte não.

---

## 8. Riscos abertos

1. **Perda de trabalho** — 36 ficheiros sem commit, histórico já reescrito uma vez. É o risco mais alto e o mais fácil de fechar.
2. **Resultados não reprodutíveis** — `resultado_corpus_real.json` não regista commit, seed, hash do corpus nem hash do vocabulário. Os 5,8962 não podem ser reconfirmados por terceiros.
3. **Validação não representa generalização** — mesma obra, mesmo autor, continuação temporal.
4. **Uma só execução** — todos os números vêm de um treino. Sem variância entre seeds não se sabe o que é sinal e o que é ruído.
5. **Sem conjunto de teste** — e a decisão sobre hiperparâmetros foi tomada olhando para a validação, o que a torna um conjunto de desenvolvimento, não de avaliação.
6. **A memória quadrática limita o futuro** — qualquer tentativa de aumentar o contexto ou o lote esbarra no mesmo muro. É uma decisão de arquitectura a tomar antes da Fase 3, não depois.

---

## 9. Testes reproduzíveis acrescentados

- `nucleo/experimentos/auditoria_fase2.py` — 44 verificações, saída em JSON com classificação por item.
- `nucleo/autograd.py` — **`Tensor.fatiar(inicio, fim)`**, novo, com retropropagação. Necessário para isolar a perda numa parte do lote; sem ele a verificação C4 não é possível.

---

## 10. O que esta auditoria não cobre

Digo isto explicitamente porque a revisão pediu uma auditoria independente, e uma auditoria que esconde os seus limites não é independente:

- **Não repeti o treino.** 5,8962 é OBSERVED de uma execução anterior, não MEASURED por mim. Reproduzi-lo exigiria 587 s e está fora do freeze.
- **Não auditei o tokenizador linha a linha.** Confio nos 54 testes existentes; não os reescrevi.
- **Não testei memorização a 5, 10 ou 15 tokens.** Só verifiquei que a afirmação de 20-gramas estava mal enquadrada.
- **Não medi o "PyTorch 50×"** — seria preciso um ambiente com PyTorch e um benchmark equivalente. A afirmação fica retirada, não refutada.
- **A auditoria foi feita por mim, que escrevi o código auditado.** É uma contradição estrutural que nenhum rigor de método resolve. A verificação independente tem de vir de fora.

---

## Anexo — classificação agregada das 44 verificações

| Classe | Nº | Significado |
|---|---:|---|
| MEASURED | 23 | medido nesta sessão, com comando |
| FACT | 15 | verdade por construção/aritmética |
| OBSERVED | 2 | visto numa execução concreta |
| HYPOTHESIS | 2 | proposto, não demonstrado |
| UNKNOWN | 2 | não sabemos |
| INFERRED | 0 | — |

Nenhuma conclusão deste relatório é INFERRED. O que não foi medido está marcado como não medido.
