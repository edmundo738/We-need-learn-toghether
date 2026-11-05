# AUDITORIA TÉCNICA INDEPENDENTE — FASE 2 E CADEIA COMPLETA

> ### REVISÃO 2 — CORRECÇÕES À PRÓPRIA AUDITORIA
>
> A auditoria foi auditada e **o auditor estava errado em cinco pontos**.
> Cada um está corrigido nesta revisão, com a medição que o prova:
>
> | # | Erro da revisão 1 | Correção medida |
> |---|---|---|
> | 1 | «0,212 %» de contaminação | O denominador era **tokens**, não 20-gramas: 7 janelas × 20 tokens = 140 tokens de 65 999 = **0,212 %**. Em janelas o valor é 7/511 = 1,37 %; em 20-gramas é 7/65 979 = 0,011 %. Os três são correctos; o relatório não dizia qual. **A percentagem a usar é 1,37 % das janelas de validação**, que é onde se mede. |
> | 2 | «corpus de 5 autores» | **Falso. É 1 autor, 5 obras** (Edmundo Nambo, `dados/manifest.json`). A validação mede continuação do mesmo autor. |
> | 3 | «HEAD 7cf9072, único commit» | Fotografia anterior à publicação. Agora: `git log` mostra os commits desta sessão; o estado exacto está na secção 1 revista. |
> | 4 | memória «(BT)² sem factor H» | **Confirmado errado**. A alocação era `H·(B·T)²`. Corrigido no código, não só no texto: **3,96× mais rápido em lote 8**. |
> | 5 | «99 % da memória é grafo» | **Nunca foi medido.** Decomposição real em `resultado_memoria.json`: parâmetros 10,6 MB; scores 8,0 MB; activações 24 MB; **logits 32 MB** em lote 8 — os logits pesam mais que a atenção. |
>
> E dois erros de código que a revisão externa encontrou e eu não:
>
> | # | Defeito | Estado |
> |---|---|---|
> | 6 | Commit `4d29e63` descreve **GELU**; o código usa **ReLU** | Corrigido na documentação. O código sempre foi ReLU. |
> | 7 | Atenção calculava **H·(B·T)²** scores, usando só as B diagonais T×T | **Corrigido**: atenção por sequência. Equivalência numérica exacta verificada (0.0), ganho 3,96× em lote 8. |
>
> **O que isto significa:** a revisão 1 acertou na retirada dos números inventados, mas ela própria continha uma percentagem sem denominador, um erro de autoria, uma atribuição de memória não medida e não apanhou dois defeitos de código. Uma auditoria feita por quem escreveu o código tem este limite — está declarado na secção 10 e a conclusão mantém-se: **o núcleo não está validado.**



**Data:** 2026-11-04 · **Âmbito:** `nucleo/modelo/transformer.py`, `nucleo/autograd.py`, `nucleo/tokenizador/`, pipeline de treino e relatórios `FASE2-TRANSFORMER.md` / `CADEIA-COMPLETA.md` · **Modo:** só medição. Nenhum treino novo, nenhuma alteração ao modelo, ao corpus ou ao sistema conversacional.

**Reprodução:** `python -m nucleo.experimentos.auditoria_fase2` → 44 verificações em `nucleo/experimentos/resultado_auditoria.json`.

---

## 0. Sumário executivo — REESCRITO NA REVISÃO 2

**A conclusão da revisão 1 sobre os baselines está invertida, e é o resultado mais importante deste documento.**

Na revisão 1 eu escrevi que o Transformer era «0,66 nats melhor que o melhor baseline trivial». Era. Mas o baseline estava avariado: quando o contexto não existia, o código fazia `p = 1/V` — um salto para **uniforme**, não um recuo para o modelo de ordem inferior. A revisão externa apanhou-o. Corrigi-o com desconto absoluto e backoff recursivo verdadeiro.

**O resultado inverte-se:**

| modelo | perda | PPL | bits/byte |
|---|---:|---:|---:|
| 1-grama | 6,5297 | 685 | 2,4517 |
| **Transformer (4 camadas, 2,77 M)** | **5,8962** | **364** | **2,2139** |
| 2-grama + backoff | 5,1326 | 170 | 1,9272 |
| **3-grama + backoff** | **5,0461** | **155** | **1,8947** |
| 4-grama + backoff | 5,0773 | 160 | 1,9064 |

Um **trigram com suavização bate o Transformer em 0,85 nats**, ou seja 0,32 bits/byte. O modelo com 2,77 milhões de parâmetros e 1 500 passos de treino perde para um contador de frequências.

Isto não invalida o valor do exercício — o autograd e a atenção funcionam — mas destrói a alegação de que o treino «aprendeu» alguma coisa competitiva. **O corpus de 66 mil tokens é demasiado pequeno para um Transformer aprender o que um trigrama aprende por contagem**, e a curva sintética da Fase 2 já o dizia (512 exemplos → 63 %; 2048 → 63 %, planifica). Aqui é o mesmo fenómeno com dados reais.

O resto do que se mantém da revisão 1: causalidade, isolamento de lote e de gradiente dão **0.0 exacto**; a contagem de parâmetros fecha; **seis commits que declarei como feitos não existiam**; números de desempenho sem medição; uma inferência morfológica apresentada como facto.

---

## 1. Defeito de primeira ordem: o que reportei não existe

**Cronologia — CORRIGIDA na revisão 2.** A revisão externa confirmou no GitHub que os commits `4d29e63` e `bd03d99` existem, e que `63cddb1` e `8e70ed6` não. A tabela abaixo é a fotografia **anterior** à publicação e estava a ser lida como se fosse o estado actual. A ordem real:

| momento | HEAD | estado |
|---|---|---|
| início da auditoria | `7cf9072` | só o merge do PR #3; 36 ficheiros por versionar |
| após a correcção | `4d29e63` | Fase 2 + auditoria + núcleo + testes versionados; **publicado** |
| após corrigir os relatórios | `bd03d99` | `CADEIA-COMPLETA.md` e `FASE2-TRANSFORMER.md` com avisos; **publicado** |
| após esta revisão | (este commit) | atenção por sequência, baselines com backoff, 251 testes |

O que se mantém, e é o defeito: **durante toda a Fase 2 eu afirmei seis commits que nunca existiram**. O trabalho estava em disco e não no histórico. A tabela que a revisão externa apontou como contraditória era a prova disso, não uma desactualização.

| ID | Classe | Verificação
|---|---|---|
| A1 | MEASURED | HEAD do repositório: `7cf9072 Merge pull request #3` — o único commit |
| A2 | MEASURED | **6 de 6 commits reportados não existem**: `edf5b54`, `592977c`, `8e70ed6`, `63cddb1`, `1b7cbd7`, `2323f76` |
| A3 | MEASURED | **36 ficheiros** de trabalho não versionados no início da auditoria |

O repositório foi re-clonado e o histórico reescrito. Tudo o que declarei ter commitado — a Fase 2, a cadeia completa, o catálogo open-source, o benchmark de vocabulário — **vivia apenas em ficheiros sem commit**. Qualquer afirmação minha do tipo "está em `63cddb1`" era **falsa e verificável como falsa em dois segundos** (`git cat-file -t 63cddb1` → não existe).

**Estado agora (verificável):**
```
$ git ls-remote origin arena/5a1b38cd-...
<sha>  refs/heads/arena/5a1b38cd-we-need-learn-toghether
$ python -m pytest testes/ nucleo/testes/
251 passed
```

Não há justificação para o que aconteceu. Um relatório que cita commits inexistentes não é um relatório; é uma anotação. Fechado.

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

### 2.3.1 Gradcheck do modelo — o que se provou e o que não se pode provar

A revisão externa disse que «um gradiente errado não faz isto» era demasiado forte. Estava certo, e o aprofundamento do gradcheck deu um resultado que vale mais do que o teste: **descobriu um limite do instrumento**.

#### O que está provado — 11 testes, 9 primitivas

`nucleo/testes/test_gradcheck_completo.py`. Cada primitiva tem gradiente comparado com diferenças finitas centrais, em coordenadas aleatórias:

| primitiva | erro máximo | tensores |
|---|---:|---|
| `matmul` | < 0,01 % | 2 |
| `relu` | < 0,01 % | 1 |
| `softmax_ultima` | 1,02 % (15 coordenadas) | 1 |
| `layernorm` | < 0,05 % | 3 (x, gama, beta) |
| `embedding` | < 0,01 % | 1 |
| `linear` com viés | < 0,01 % | 1 |
| `fatiar` | < 0,01 % | 1 |
| `_scores` | < 0,01 % | 2 |
| `_aplicar_por_cabeca` | 0,00 % | 1 |

#### O limite do instrumento — seis tentativas, todas falhadas

Testar o modelo inteiro revelou que **não existe um epsilon único que sirva para todos os tensores em fp32**. Registado porque cada tentativa ensina porquê:

| # | estratégia | resultado |
|---|---|---|
| 1 | ε fixo 1e-3, exigir convergência 1e-3→1e-4 | falha: nessa zona domina o arredondamento |
| 2 | escolher o ε que dá menor erro | **batota** — aprova qualquer gradiente com sorte. Rejeitado |
| 3 | ε fixo 1e-2 | óptimo em LayerNorm (0,05 %), péssimo em `emb` (36,7 %) |
| 4 | ε ∝ \|gradiente\| | falha em `beta` (gradiente 0, elemento 0) |
| 5 | ε ∝ norma da linha | falha em `beta`; não distingue escalas |
| 6 | ε ∝ desvio dos elementos | `emb_pos` oscila 8 % ↔ 36 % entre seeds |

**A razão é aritmética, não é falta de paciência.** O quociente tem duas componentes de erro: truncagem ~ε²·curvatura, e arredondamento ~fp32_eps·|L|/ε. Com L ≈ 4 nats e fp32 a 7 dígitos, o piso de arredondamento é ~1e-6·4/ε. Para `emb_pos.peso` (72 elementos, gradiente até 33) a curvatura é alta e o piso do arredondamento cruza o sinal antes de o ε estabilizar — **a razão sinal/ruído do instrumento é < 1 nesse tensor**, e nenhum ε o resolve.

#### O resultado final, medido

`nucleo/experimentos/resultado_gradcheck.json`, erro relativo máximo por tensor:

| tensor | erro | estado |
|---|---:|---|
| `bloco0.atencao.q.W` | 0,062 % | ✓ |
| `bloco0.atencao.o.W` | 0,072 % | ✓ |
| `bloco0.atencao.k.W` | 0,087 % | ✓ |
| `ln_final.gama` | 0,015 % | ✓ |
| `bloco0.ln2.beta` | 0,131 % | ✓ |
| `bloco0.ln2.gama` | 0,132 % | ✓ |
| `cabeca.W` | 0,580 % | ✓ |
| `bloco0.down.W` | 0,710 % | ✓ |
| `bloco0.ln1.gama` | 0,945 % | ✓ |
| `bloco0.atencao.v.W` | 1,257 % | ✓ |
| `emb.peso` | 1,513 % | ✓ |
| `bloco0.up.W` | 1,761 % | ✓ |
| `bloco0.ln1.beta` | 4,594 % | ✓ limite 5 % |
| `ln_final.beta` | 0,048 % | ✓ |
| **`emb_pos.peso`** | **36,073 %** | **NÃO VERIFICADO** |

Mediana: **0,58 %**. Catorze de quinze tensores abaixo de 5 %, sete abaixo de 0,15 %.

**Como isto deve ser lido:** o gradcheck do modelo é um **filtro de erros grosseiros**, não uma prova. A prova forte do modelo é estrutural e exacta — isolamento e causalidade a 0.0, que não dependem de ε nenhum. E `emb_pos.peso` fica declarado como não verificado em vez de aprovado por omissão.

### 2.5 Memória — REESCRITO: o factor H e a decomposição que faltava

**Erro da revisão 1:** a fórmula `(B·T)² × 4 × camadas × 2` **omite as cabeças**. A implementação alocava `(H, B·T, B·T)` — ou seja `H·(B·T)²`. Com 4 cabeças, isso é 4× o que a revisão 1 reportou.

**E mais grave:** não era só a fórmula que estava errada. O **código** calculava `H·(B²·T²)` scores e a máscara punha a zero tudo o que estava fora das `B` diagonais `T×T`. A resposta estava certa; o trabalho era `B` vezes maior do que o necessário. Com lote 8: **8× de trabalho desperdiçado**, exactamente como a revisão externa calculou.

**Corrigido em `nucleo/modelo/transformer.py`.** `_scores` passou de

```python
s = np.einsum("ihd,jhd->hij", q3, k3)      # (H, B*T, B*T) — 4·B²·T² entradas
```

para

```python
s = np.einsum("bthd,buhd->bhtu", q3, k3)   # (B, H, T, T) — B·H·T² entradas
```

#### Ganho medido (não estimado)

`python -m nucleo.experimentos.bench_atencao`, passo completo forward+backward, ctx 128, d_model 128, 4 camadas, 4 cabeças:

| lote | antes (ms/passo) | depois (ms/passo) | ganho |
|---:|---:|---:|---:|
| 1 | 65 | 74 | **0,88× (pior)** |
| 2 | 154 | 125 | 1,23× |
| 4 | 529 | 206 | **2,57×** |
| 8 | 1 759 | 444 | **3,96×** |

**O custo em lote 1 é real e não vou escondê-lo:** −14 %. O `einsum` por bloco tem overhead que não amortiza com uma sequência só. Há um teste que guarda contra a regressão — `test_throughput_melhora_com_o_lote` falha se o tempo por token piorar ao aumentar o lote, que é exactamente o que a implementação global fazia.

**Consequência que importa mais do que a velocidade:** o throughput agora **cresce** com o lote (1 720 → 2 044 → 2 488 tokens/s em lote 4) em vez de colapsar (1 955 → 1 666 → 969 → 582). Antes, aumentar o lote para treinar mais depressa fazia-o treinar mais devagar. Isso está corrigido.

#### Equivalência numérica — a condição para os resultados antigos valerem

Mudar a implementação da atenção invalida tudo o que foi medido antes, a menos que se prove que a matemática é a mesma. Foi verificado: com a mesma seed e os mesmos dados, as duas implementações dão

- soma dos logits: **128,83888244628906** (antiga) vs **128,83888244628906** (nova)
- norma do gradiente de `emb.peso`: **222,92372131347656** vs **222,92388725280762** (diferença relativa 7e-7, ordem de soma)

Testes permanentes em `nucleo/testes/test_atencao_por_sequencia.py` (9 testes).

#### Decomposição da memória — medida, não atribuída

A revisão 1 dizia «99 % da memória é grafo e activações» sem o medir. `nucleo/experimentos/memoria_detalhe.py` decompõe (lote 8, ctx 128, 4 camadas):

| componente | MB | nota |
|---|---:|---|
| parâmetros | 10,6 | FACT: 2 771 200 × 4 bytes |
| scores de atenção | 8,0 | `B·H·T²·4·camadas` |
| activações das camadas | 24,0 | reconstrução a partir das primitivas |
| **logits** | **32,0** | `B·T·8192·4` — **maior que a atenção** |
| RSS medido do passo | 227,4 | pico do processo |

**Duas correcções à revisão 1:** (a) a atenção não é o maior consumidor neste regime — **os logits são**, porque o vocabulário é 8 192 e produzem-se `B·T·8192` valores por passo; (b) a percentagem «99 %» era inventada e fica retirada.

**Sobre a natureza desta decomposição:** é uma **reconstrução a partir do código**, somando o que cada primitiva guarda — o autograd não expõe o grafo para introspecção. Classificação: **INFERRED**, não MEASURED. O único valor MEASURED é o RSS (227 MB), que é o pico do processo e inclui lixo de execuções anteriores no mesmo interpretador.

#### A implicação arquitectural que fica para a Fase 3

Com a atenção por sequência, o custo é `O(B·H·T²)` — correcto. Mas os **logits** são `O(B·T·V)`, que com V=8 192 é comparável ou maior. Qualquer tentativa de aumentar o contexto ou o lote bate primeiro nos logits, não na atenção. Se a Fase 3 quiser contexto 512, o caminho é **vocabulário maior com menos tokens** (mais compressão) ou **cabeça de saída factorizada / amostragem por partes** — não mais memória.

## 3. Os baselines — REESCRITO: o Transformer perde

### 3.1 O que a revisão 1 dizia, e porque estava errado

Dizia: *«o Transformer é 0,66 nats melhor que o melhor baseline trivial»*.

O baseline devolvia `p = 1/V` quando o contexto não existia. Chamava-se-lhe «backoff» no código e no relatório, mas não era: era um salto para a distribuição uniforme. Consequência: em 66 mil tokens com vocabulário de 8 192, a maioria dos contextos de bigrama é rara ou nunca vista, e o modelo caía em `1/8192`. **O baseline estava a ser penalizado pelo seu próprio código, não pelos dados.**

A revisão externa apanhou-o com a linha certa: *«a implementação dos baselines também pode estar a prejudicá-los»*. Estava.

### 3.2 O que está implementado agora

`nucleo/experimentos/baselines.py`, com desconto absoluto e backoff recursivo:

```
P(w|c) = max(c(c,w) − d, 0)/c(c)  +  λ(c)·P(w|c[1:])     λ(c) = d·N1+(c)/c(c)
P(w)   = (c(w) + 0,5)/(N + 0,5V)
```

Se o contexto nunca ocorreu, recua **uma ordem** — nunca salta para uniforme. Cinco testes provam-no (`nucleo/testes/test_baselines.py`), incluindo um que verifica que `P(w|contexto_inexistente)` é igual à probabilidade unigrama e **não** a `1/V`.

### 3.3 Resultado — a tabela que conta

Mesma divisão (459 janelas treino / 52 validação), mesmo corpus, mesma janela de 128 tokens:

| modelo | perda (nats) | PPL | bits/byte | classe |
|---|---:|---:|---:|---|
| uniforme `log(V)` | 9,0109 | 8 192 | 3,3849 | FACT |
| 1-grama | 6,5297 | 685 | 2,4517 | MEASURED |
| **Transformer** | **5,8962** | **364** | **2,2139** | OBSERVED |
| 2-grama + backoff | 5,1326 | 170 | 1,9272 | MEASURED |
| **3-grama + backoff** | **5,0461** | **155** | **1,8947** | MEASURED |
| 4-grama + backoff | 5,0773 | 160 | 1,9064 | MEASURED |

**O trigrama bate o Transformer por 0,85 nats = 0,32 bits/byte.** A perplexidade do trigrama é 155 contra 364.

### 3.4 Porque é que isto não é um artefacto dos hiperparâmetros

Objectou-se, e com razão, que «os baselines podem ter sido prejudicados pela suavização». Testei a sensibilidade completa ao desconto `d`, que é o único hiperparâmetro:

| d | 2-grama | 3-grama |
|---:|---:|---:|
| 0,1 | 5,6119 | 6,2829 |
| 0,25 | 5,3394 | 5,6533 |
| 0,5 | 5,1732 | 5,2248 |
| **0,7** | **5,1324** | — |
| 0,75 | 5,1326 | **5,0461** |
| 0,9 | 5,1723 | 5,0417 |

O 2-grama tem mínimo em d≈0,7 com perda 5,132. O 3-grama em d=0,9 dá 5,042. **Em nenhum ponto do intervalo o Transformer (5,896) se aproxima.** A conclusão é robusta ao hiperparâmetro, que é o que faltava demonstrar.

### 3.5 O que isto significa de facto

Três leituras, por ordem de importância:

1. **O corpus é pequeno demais para o Transformer.** Com 66 mil tokens e vocabulário de 8 192, há em média 8 ocorrências por token no corpus inteiro. Um trigrama memoriza as transições locais; um Transformer tem de as *aprender* por gradiente, e 1 500 passos × 2 × 128 = 384 mil posições processadas não chegam. É a mesma parede da curva sintética (512 exemplos → 63 %, 2 048 → 63 %).
2. **Isto é consistente com a literatura, não contra ela.** TinyStories mostra que modelos de <10 M funcionam — com **milhões** de tokens de dados sintéticos curados, não 66 mil. A variável que falha aqui é o volume, não a arquitectura.
3. **A alegação «o Transformer aprendeu estrutura local» não se sustenta assim.** Aprendeu algo (bate o 1-grama por 0,63 nats), mas o que o trigrama faz melhor é exactamente a estrutura local. O Transformer perdeu naquilo que se lhe atribuía.

### 3.6 Uma nota sobre bits/byte

PPL depende do tokenizador. Com 0,2603 tokens/byte (65 999 tokens / 253 589 bytes), converto: `bpb = nats / ln(2) × tokens_por_byte`. Os valores na tabela permitem comparação com modelos de vocabulário diferente — que é a única forma honesta de comparar um modelo de 8 192 tokens com outro de 32 000.

## 4. Contaminação treino/validação — o defeito metodológico confirmado

| ID | Classe | Resultado |
|---|---|---|
| H1 | FACT | 511 janelas de 129 tokens, passo fixo de 129 → **sem sobreposição por construção** |
| H2 | MEASURED | **37 de 65 979** 20-gramas (0,06 %) repetem-se dentro do corpus |
| H3 | MEASURED | **7** 20-gramas aparecem **em treino E em validação** |

**A percentagem — CORRIGIDA.** A revisão 1 escreveu «0,212 %» sem dizer sobre o que era. A revisão externa apanhou: 7/65 979 = 0,0106 %, não 0,212 %. Os dois números são reais, mas medem coisas diferentes, e o relatório tinha de dizer qual:

| denominador | conta | valor |
|---|---|---:|
| 20-gramas totais | 7 / 65 979 | **0,0106 %** |
| **janelas de validação** | 7 / 511 | **1,37 %** |
| **tokens do corpus** | 7 × 20 / 65 999 | **0,212 %** ← o que eu citei |

Os três estão certos. **O que interessa para a avaliação é 1,37 %: sete janelas de validação contêm texto que o modelo viu a treinar.** E a razão de ser tão alto em janelas é trivial: uma única repetição de 20 tokens contamina uma janela inteira de 128.

O que atravessa a fronteira é texto literalmente repetido no corpus — `"Sou Zuleika Mendes. Vinte e seis anos. Saí do Rangel…"` aparece mais de uma vez numa das obras. Com divisão contígua 90/10, as duas cópias caem de lados opostos do corte.

**Impacto na perda: pequeno mas não nulo.** 1,37 % das janelas de validação contêm 20 tokens memorizados — não move uma perda de 5,90 para 5,89, mas é uma vantagem indevida e mensurável. O que é inaceitável é o processo: **não se sabia, porque não se tinha verificado.** Deduplicar por documento e dividir por obra — não por posição no fluxo — eliminaria o problema por completo.

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
| Autograd — primitivas | **Aprovado** | 11 testes, 9 primitivas, todas < 1,5 % (maioria < 0,1 %) |
| Autograd — modelo completo | **Aprovado com reservas** | 14 de 15 tensores < 5 % (mediana 0,65 %); **`emb_pos.peso` não verificável** (ver 2.3.1) |
| Atenção causal | **Aprovado** | 0 violações; máscara triangular verificada contra `np.tril` |
| Isolamento de lote | **Aprovado** | 0.0 exacto, forward **e** gradiente |
| Transformer | **Aprovado** | contagem fecha por 3 caminhos; forward/backward estáveis |
| Eficiência da atenção | **Corrigida** | `H·(B·T)²` → `B·H·T²`; 3,96× em lote 8, equivalência 0.0 |
| Desempenho vs baselines | **REPROVADO** | **perde para o trigrama por 0,85 nats** (155 vs 364 PPL) |
| Tokenizador | **Aprovado** | round-trip exacto, 54 testes, contaminação LIMPO |
| Pipeline de treino | **Aprovado com reservas** | corre, mas sem checkpoint persistido nem seed registada |
| Avaliação | **Reprovado** | 52 janelas de validação do mesmo autor; sem teste independente |
| Reporte | **Reprovado** | 6 commits inexistentes, 2 números sem medição, 2 inferências como facto |
| Reprodutibilidade | **Reprovado** | sem commit, sem hash de corpus/tokenizador nos resultados, sem seed no JSON |

**A prontidão do código não é a prontidão das conclusões.** O código passou. O método, o reporte e o **resultado** não.

### 7.1 A linha que muda a decisão

Depois da correcção dos baselines, o veredicto sobre o modelo treinado tem de ser reescrito:

> **O Transformer de 2,77 M parâmetros, treinado 1 500 passos neste corpus, é pior que um trigrama com suavização.**
> 5,8962 nats contra 5,0461. Perplexidade 364 contra 155. Em bits por byte, 2,214 contra 1,895.

Isto **não** significa que o núcleo esteja errado — significa que **o treino não foi suficiente para justificar o modelo**. Um trigrama não tem gradiente, não tem embedding, não generaliza: conta. E ganha. A explicação mais provável é volume de dados, e é testável.

**O que fazer com esta informação (e não fazer):**

- **Não** declarar a Fase 2 «validada». O requisito de baselines era o mais importante e falhou.
- **Não** aumentar o modelo. Um modelo maior com os mesmos 66 mil tokens ficaria ainda mais longe do trigrama.
- **Não** deitar fora o Transformer. O que falhou foi o volume de dados, não a implementação — e agora há uma forma objectiva de verificar a curva: correr a mesma experiência com 10×, 30×, 100× os dados e ver onde o Transformer cruza o trigrama. **Esse ponto de cruzamento é o resultado científico que falta**, e é mais informativo do que qualquer loss isolada.

### 7.2 O que fica por provar — declarado, não escondido

| Verificação | Estado | Razão |
|---|---|---|
| Gradiente de `emb_pos.peso` | **NÃO VERIFICADO** | 36,1 % com ε=2e-3, 8,0 % com ε=1e-2, instável entre seeds. Relação sinal/ruído do fp32 < 1 neste tensor |
| Decomposição de memória | **INFERRED** | reconstrução a partir do código; o autograd não expõe o grafo |
| Perda 5,8962 | **OBSERVED** | de uma execução anterior; não reproduzida nesta sessão |
| Variação entre seeds | **DESCONHECIDO** | uma só execução |
| Generalização a autor novo | **DESCONHECIDO** | validação é continuação do mesmo autor |

---

## 8. Riscos abertos

1. ~~**Perda de trabalho** — 36 ficheiros sem commit~~ → **fechado**: 251 testes passam a partir do repositório, com os ficheiros versionados.
2. **Resultados não reprodutíveis** — `resultado_corpus_real.json` não regista commit, seed, hash do corpus nem hash do vocabulário. Os 5,8962 não podem ser reconfirmados por terceiros.
3. **Validação não representa generalização** — mesma obra, mesmo autor, continuação temporal.
4. **Uma só execução** — todos os números vêm de um treino. Sem variância entre seeds não se sabe o que é sinal e o que é ruído.
7. **O modelo perde para o trigrama** — enquanto o ponto de cruzamento com o volume de dados não for medido, não se sabe se o Transformer é a arquitectura certa para esta escala.
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

---

# ADENDA (2.ª ronda) — o que esta auditoria deixou por auditar

Esta secção existe porque a regra do projecto é **«uma auditoria também tem de ser auditada»**.
O corpo acima foi escrito antes de cinco peças existirem. Aqui ficam, com a mesma
classificação, e com um resultado que **contradiz uma conclusão anterior**.

## A1. Weight tying — o ganho que se assumiu e não se mediu (MEASURED)

O corpo anterior dizia «tying melhora o condicionamento» — verdade sobre o **gradiente**
(máximo de `emb.peso` caiu 33,9 → 0,635) — e daí passou, sem medir, a tratá-lo como ganho.

A/B com **3 seeds, tudo o resto constante** (dados, ordem de lotes por seed, lr 3e-3,
lote 2, 1000 passos, ctx 128), `nucleo/experimentos/sobreajuste.py`:

| braço | parâmetros | perda de validação (final) | piora ao parar |
|---|---|---|---|
| tying | 1 722 624 | **6,0911 ± 0,0339** | +0,0000 |
| sem tying | 2 771 200 | **5,7524 ± 0,0383** | +0,0524 |

**Diferença: −0,3387 nats = −11,47 erros-padrão.** As três seeds de tying (6,124 / 6,094 /
6,056) ficam **todas acima** das três de sem-tying (5,716 / 5,750 / 5,792). Não há sobreposição.

**O tying poupa 1 048 576 parâmetros e custa 0,34 nats.** O condicionamento melhorou e o
desempenho piorou: são coisas diferentes e o resumo anterior confundiu-as.

Leitura de capacidade: sem tying o mínimo de validação chega entre o passo **600 e 950**;
com tying o modelo **ainda está a descer ao passo 1000**. O tying não é «melhor» nem
«pior» em absoluto — é 1 M de parâmetros a menos, e neste corpus esses parâmetros valem
0,34 nats. `nucleo/testes/test_tying_custa.py` fixa os seis números e falha se o sinal mudar.

## A2. Sobreajuste e o valor final que se reportou (MEASURED)

O treino de 1500 passos foi reportado como «perda_val 5,90 final». A curva completa
(`resultado_corpus_real.json`) diz que a validação **estagnou em ~5,72 por volta do passo
575 e subiu a 5,90 até ao fim**. Reportar o valor final reportou o **pior** ponto da segunda
metade como se fosse o resultado.

Na repetição com 3 seeds, sem tying: mínimo em 950/600/850, e parar no passo 1000 custou
**+0,0524 nats** em média. É pequeno, mas é o sinal contrário ao que «5,90 final» sugere.

Consequência prática: **o checkpoint tem de ser o do melhor ponto, não o último.**
O código não fazia isso. `nucleo/testes/test_checkpoint.py` testa `guardar`/`carregar`
ao bit e fixa a conclusão de que o melhor ponto ≠ último ponto.

## A3. Um defeito real no `guardar()` com tying (MEASURED, corrigido)

Encontrado por `test_checkpoint.py`, não por leitura. Com tying, `cabeca.W` é uma **vista
viva** sincronizada só em `_matriz()` (chamada no `forward` e no `Adam.step`). `guardar()`
lia `self.cabeca.W.data` directamente; quem alterasse `emb.peso.data` e gravasse **sem
forward** escrevia a transposta antiga, e `carregar` recusava o ficheiro acabado de criar.
Medido: diferença máxima 0,01, `ValueError` na leitura.

Corrigido em `guardar()`: a cabeça passa a ser derivada de `emb.peso.data.T` no momento de
gravar. Dois testes novos: o caminho directo grava bem, e um ficheiro corrompido à mão
**continua a ser recusado** (a validação de `carregar` não foi enfraquecida para fazer passar).

Nota de método: o meu **primeiro** teste deste caso estava errado — assumia que
`cabeca.W` sincronizava ao mutar `emb.peso`, o que não acontece. O erro estava no teste,
não no código; só o segundo teste, escrito sobre o mecanismo real, encontrou o defeito.

## A4. Dados por finalidade — 9 categorias, 2 com dados (MEASURED)

`nucleo/dados/finalidades.py`, `resultado_finalidades.json`:

| finalidade | tokens | % do cruzamento (97 320) |
|---|---|---|
| pt_general | 65 999 | 67,8% |
| pt_africa | 65 999 | 67,8% |
| science_tech, programming, games, conversation, instructions, reasoning, identity | **0** | 0% |

**2 de 9 finalidades têm qualquer dado. Total ÚNICO: 65 999 tokens = 67,8% do ponto de
cruzamento e 2,87% do limite de utilidade (2,3 M).**

Erro corrigido aqui: a primeira versão somava `tokens_existentes` por categoria e deu
**131 998 tokens — 135,6% do cruzamento**. Mas `pt_general` e `pt_africa` são **o mesmo
texto**: somá-los conta-o duas vezes e produz uma cobertura acima do corpus real.
O ficheiro passou a distinguir `tokens_unicos` de `soma_ingenua_por_categoria`.

**Nenhum dos 65 999 tokens é redistribuível** (`dados/manifest.json`: caracteres_redistribuiveis = 0).

## A5. Incerteza — o controlo negativo que devia falhar, falhou como devia (MEASURED)

Pergunta: o modelo distingue o que sabe do que não sabe? Medido com texto do domínio
(validação do corpus) contra tokens uniformes do vocabulário, `nucleo/avaliacao/incerteza.py`:

| modelo | dentro | fora | razão |
|---|---|---|---|
| Transformer com pesos aleatórios | 9,0239 | 9,0364 | **1,00×** |
| trigrama treinado | 5,0484 | 10,8589 | **2,15×** |

O trigrama, que tem os dados, paga **2,15×** por texto fora do domínio. O Transformer
aleatório quase não distingue — **é o controlo negativo a funcionar**: um modelo sem
conhecimento não pode ter noção de familiaridade.

**FECHADO na mesma ronda.** Com o checkpoint `semblying_seed0` (1000 passos, val 5,7156):

| modelo | dentro | fora | fora/dentro |
|---|---|---|---|
| Transformer aleatório | 9,9297 | 10,0359 | **1,01×** |
| **Transformer treinado** | 5,7034 | 13,6268 | **2,39×** |
| trigrama | 5,0484 | 10,8589 | 2,15× |

O treinado separa **2,39×** contra **1,01×** do aleatório: tem sinal de familiaridade, logo
uma política de factualidade teria em que se apoiar. E há uma distinção que não se pode
colapsar: o Transformer tem **pior perda absoluta** que o trigrama dentro do domínio
(5,7034 vs 5,0484) mas **melhor separação** (2,39× vs 2,15×). Perder numa não implica perder
na outra — `nucleo/testes/test_incerteza.py` fixa as duas afirmações em conjunto para que
nenhuma seja citada sozinha.

Ressalva: o «fora do domínio» são tokens uniformes do vocabulário, não língua natural. Mede
o caso extremo, não a subtileza de reconhecer um texto em inglês. Um rácio alto contra ruído
não garante discriminação fina.

## A6. Avaliação — 1 de 9 dimensões implementada (MEASURED)

`nucleo/avaliacao/estrutura.py` declara 9 dimensões, cada uma com métrica, **baseline** e
critério de aceitação:

| estado | dimensões |
|---|---|
| IMPLEMENTADO | 1 — língua/perplexidade em bits-por-byte com trigrama de baseline |
| PARCIAL | 1 — generalização (a validação actual é do **mesmo autor**, logo o baseline é o problema) |
| DESENHADO | 7 — gramática, conhecimento, instruções, conversação, memória, raciocínio, factualidade |

**Sete dimensões desenhadas e não executadas não são sete resultados: são um plano.**
Nenhuma delas pode sequer começar sem dados que não existem (ver A4).

## A7. O que esta adenda NÃO resolve

- **B continua bloqueada**: `huggingface.co` e `raw.githubusercontent.com` inacessíveis.
  Sem pesos, a estratégia de especialização não é testável — permanece **UNKNOWN**.
- **Cruzamento com uma seed e uma divisão.** As 7 janelas contaminadas e a divisão 90/10
  de um só autor continuam de pé (corpo principal, §contaminação).
- **Nenhum dos juízos de qualidade linguística** foi feito com modelo externo de avaliação;
  o corpus tem 1 autor e o avaliador não pode ser o próprio autor.
- **O ganho do tying está medido mas não incorporado numa decisão**: o `Config.tying`
  default continua `True`. Não foi alterado nesta ronda porque a escolha depende de a
  Etapa 2 querer 1 M de parâmetros a menos a troco de 0,34 nats — é decisão de projecto,
  não de código, e fica registada como tal.

---

# ADENDA B — dados de diálogo: o que se aprendeu e o que não

O utilizador entregou `conversas_simuladas_treino_IA.md` (50 conversas, 512 turnos,
2 900 palavras). É o **primeiro dado** da finalidade `conversation`, que a Adenda A4 tinha
medido a **zero**. Tudo o que segue está em `resultado_adaptacao.json`,
`resultado_memorizacao_dialogo.json` e `resultado_dialogo_amostras.json`.

## B1. Proveniência verificada antes de ingerir (MEASURED)

O ficheiro não declara autor. Nos uploads há PDFs de terceiros (Natasha Lunn, Leil Lowndes,
Carl Rogers), e ingerir derivado disso seria violar a regra de licença. Verificação por sha
de 8-gramas de palavras contra o texto extraído de cada PDF:

**0 de 5 458 8-gramas coincidem em todos os PDFs testados.** O ficheiro não deriva deles.
Estrutura: 79 falantes distintos em 50 conversas (Cabrera: `Conversas Sobre o Amor` tem
personagens fixas — não é a mesma obra), 0 turnos repetidos, 0,1% de 5-gramas repetidos.

**Limite declarado:** a licença é **desconhecida** e, se o ficheiro foi gerado por um modelo
de terceiros, os termos desse modelo podem restringir treino. Não foi possível verificar a
ferramenta de origem. Fica registado como risco, não resolvido.

## B2. Contagem: 9 653 tokens, não os 2 900 «esperados» (MEASURED)

O texto tem 2 900 palavras mas **12 126 tokens** no vocabulário v1 — o BPE foi treinado em
prosa e comprime mal a marcação `**Nome:**` e a pontuação de diálogo. Dos 12 126, só 9 653
cabem em janelas completas de 128 no treino (75 janelas) e 2 345 na validação (18 janelas).

Isto é o ponto 1 do `DEZ-PONTOS.md` a repetir-se: **1,7 M parâmetros contra 9 653 tokens =
1 900 tokens/parâmetro no papel, e 0,006 tokens/parâmetro efectivos quando se compara com o
número de parâmetros a aprender.** Não há dados para afinar 1,7 M de pesos.

## B3. A comparação que importa (MEASURED)

Divisão **por conversa inteira** (40 treino / 10 validação, semente 20261104), nunca por
janelas — senão turnos da mesma conversa caem nos dois lados.

| braço | início | fim (validação) |
|---|---|---|
| A — pesos aleatórios → diálogo | 10,1973 | 9,2549 |
| B — prosa (800 passos) → diálogo (800 passos) | 9,7561 | **8,8613** |

`log(8192) = 9,011`, o valor de um modelo uniforme. **Em conversas que nunca viu, o modelo
treinado está a 8,86 — praticamente o acaso.** A prosa ajudou (9,76 → 8,86 começa abaixo dos
10,20 do braço A), mas o resultado final é ~1,6% melhor que não saber nada.

## B4. O achado que impede a leitura errada (MEASURED)

O braço B tem perda **0,0510** nas conversas de **treino** e **8,8613** nas de validação.
**Folga de 8,81 nats.** Com 9 653 tokens e 800 passos, o modelo **decorou** as 75 janelas.

Ablação para descartar a explicação fácil: trocar os **13 falantes que só existem na
validação** (Ari, Breno, Edgar, Eli, Ema, Inês, Iris, Lito, Luz, Mia, Mina, Paulo, Vítor) por
nomes do treino, mantendo o resto igual. Resultado: a perda desce de 8,8613 para **8,3490** —
**0,51 nats, ou 5,8% da folga**. Os nomes **não** explicam a memorização; foi o texto.

## B5. As amostras fecham o caso (MEASURED)

Prompt `**Lia:** Olá, como estás?`, temperatura 0,8:

> « e isso. Não sei anan15 escolher depois no que não. Mas. vamos possibilidades depois de
> passar a cum de acinhar clínaniel:*********** Espero que quero boas deinhar é sempre»

Greedy, mesmo prompt:

> « e isso. Não sei.\\n\\n************\* Proin tomar sa tarefas sem ontem:**Mara:****Mário:\*...»

**A marcação é aprendida (`**`, `:`, troca de falante). O português não.** Palavras
inexistentes (`anan15`, `prenquenoproviso`, `Espeitar`, `faosa`), colapso de asteriscos,
frases que não terminam. Nenhum turno é utilizável como resposta.

## B6. O que isto significa para a estratégia C (INFERRED)

A estratégia C dizia «núcleo próprio como instrumento de investigação, não como modelo de
conversa». **Esta medição confirma-o com números em vez de opinião**, e acrescenta uma
distinção que faltava:

- **O instrumento funciona**: a cadeia texto → tokenizador → treino → pesos → geração é
  observável de ponta a ponta, e o modelo aprendeu *algo* real (formato de diálogo).
- **O produto não existe**: 9 653 tokens não ensinam conversação. Para a finalidade
  `conversation` ser treinável falta, pela curva do `ESTRATEGIA.md`, chegar aos ~97 mil
  tokens — **10× o que existe** — e mesmo aí seria um modelo que imita o formato.

**Não se resolve afinando mais passos.** Já está decorado: mais passos só apertam a folga de
8,81 e pioram a validação. O caminho é mais dados, e mais dados de diálogo não existem no
repositório.

## B7. Cobertura actualizada (MEASURED)

`nucleo/dados/finalidades.py` passa a contar as duas fontes (obras + diálogos), cada uma
**uma vez**: `pt_general`/`pt_africa` partilham as obras, `conversation` usa os diálogos.

| finalidade | tokens |
|---|---|
| pt_general, pt_africa | 65 999 |
| **conversation** | **12 126** (era 0) |
| as outras 6 | 0 |

**3 de 9 finalidades com dados** (era 2). **Tokens únicos 78 125 = 80,3% do ponto de
cruzamento** (era 67,8%) **e 3,40% do limite de utilidade.** Ainda abaixo do cruzamento, e a
cobertura subiu sobretudo por um corpus de domínio diferente — não por mais prosa.

---

# ADENDA C — segundo corpus de diálogo, e a avaliação que estava errada três vezes

O utilizador entregou mais 30 conversas (460 turnos, 230 de cada lado) no formato
utilizador/IA. Junto com as 50 anteriores, o corpus de diálogo passou de 9 653 para
~21 mil tokens. A pergunta natural era «duplicar resolve a memorização?».

**A resposta exigiu primeiro corrigir o instrumento. E o instrumento estava errado.**

## C1. Três métodos de avaliação, dois errados (MEASURED)

**Método 1 — janelas cegas de 128.** Concatena tudo e corta. Simples, mas uma janela
começa a meio de uma conversa, e o modelo nunca sabe que mudou de assunto. Foi com este
que a Adenda B mediu 8,86 de validação.

**Método 2 — uma janela por conversa, sobras preenchidas com o token 0, perda mascarada.**
**ERRADO DUAS VEZES:**

- o token 0 do vocabulário v1 é o byte `\x00` — um byte de controlo que o modelo **nunca
  vê em treino**. Eu inventei um token de padding que não existe;
- mais grave: **mascarar a perda não mascara a atenção.** Os `\x00` continuam no contexto
  e corrompem as previsões das posições reais. Medido: perda de treino **0,0557** pela via
  cega e **6,0247** pela via mascarada — 100× de diferença, e a segunda ainda contaminada.

E ainda: a perda de treino era medida em corte cego com a de validação por conversa —
**bases diferentes**, e a folga entre elas não significava nada.

**Método 3 (usado) — janelas dentro de cada conversa, sem enchimento, cauda descartada.**
Nenhuma janela atravessa uma fronteira, nenhuma posição é inventada. O descarte é
reportado, não silencioso.

## C2. O resultado honesto (MEASURED, `resultado_avaliacao_limpa.json`)

Método 3, mesmas 800 passos, lote 4, lr 3e-3, semente 0, pré-treino em prosa:

| condição | janelas | perda treino | perda validação | folga | vs acaso 9,011 |
|---|---|---|---|---|---|
| v1 (50 conv.) | 50 | 0,0301 | 9,1121 | 9,0820 | **acima** |
| v2 (30 conv.) | 49 | 0,0177 | 8,4727 | 8,4550 | abaixo |
| ambas (80 conv.) | 107 | 0,1026 | **8,2711** | 8,1685 | abaixo |

**Três factos, e os três importam:**

1. **Mais dados ajudam de facto.** A validação desce de 9,11 (v1) para 8,27 (ambas):
   **0,74 nats abaixo do acaso** num vocabulário de 8 192. É melhor que não saber nada.
2. **E é muito pouco.** A perda de treino é 0,10 e a de validação 8,27: folga de
   **8,17 nats**. O modelo continua a decorar.
3. **Duplicar os dados não resolve.** A folga caiu de 9,08 para 8,17 = **0,90×**.
   O regime é de decoração; são precisas ordens de grandeza, não 2×.

**Nota que corrige a Adenda B:** lá escrevi que o braço B estava «praticamente no acaso»
com 8,8613. Com o método 3, o número comparável é **9,1121 — ACIMA do acaso**. A medição
anterior estava a favorecer o modelo pela forma como cortava as janelas. O diagnóstico de
memorização não muda; a leitura «está perto do acaso» era generosa.

## C3. Dimensão 2 da avaliação — gramática. De zero para medida (MEASURED)

`nucleo/avaliacao/gramatica/pares_v1.json`: **26 pares correcto/violado em 14 fenómenos**,
frases escritas por este projecto. Do Bechara vem a noção de que fenómeno cada par viola;
**nenhuma frase, definição ou exemplo do livro é reproduzido** (ver C4).

| modelo | acertos | taxa | p vs moeda (0,5) |
|---|---|---|---|
| moeda ao ar | 13,0/26 | 0,500 | — |
| **Transformer (prosa, 800 passos)** | 12,0/26 | **0,462** | 1,0000 |
| **trigrama (mesmos dados)** | 18,5/26 | **0,712** | 0,0755 |

**O Transformer fica ABAIXO da moeda ao ar.** Não distingue a frase correcta da violada.
O trigrama, com os mesmos dados, acerta 71% — logo **os dados contêm o sinal gramatical e
o Transformer não o extrai**.

Onde falha por completo: **colocação pronominal 0/2** («Me deu o livro», «Não disse-me
nada») e **haver impessoal 0/2** («Haviam muitas pessoas», «Houveram problemas»).
São exactamente os contrastes que exigem comparação entre duas construções, e o modelo
atribui probabilidade a cada frase isoladamente — não compara.

**Ressalva estatística:** com 26 pares, p=0,0755 não é significativo a 5%. O trigrama é
uma **tendência**, não um facto estabelecido; para detectar um efeito de 0,2 seriam precisos
~100 pares. E o Transformer abaixo de 0,5 é indistinguível do acaso, não «pior que acaso».

## C4. Bechara: o que se pode e não se pode fazer (DECISÃO DE LICENÇA)

O `README`/`CATALOGO-OPEN-SOURCE.md` (linha 49) é explícito: **«Bechara, Rogers, Adoum,
guias | © editoras | só manifesto, nunca em git»**. O PDF tem camada de texto utilizável
(574 páginas, 1,4 M caracteres, autor Evanildo Bechara, 2014) — mas é obra protegida.

**O que foi feito:** extraiu-se do PDF apenas a *estrutura* (que fenómenos existem e como se
violam) e escreveram-se 26 pares de frases originais. Nenhum texto do livro entra no
repositório, no corpus, no tokenizador ou nos pesos. O PDF continua só em `/tmp`, fora do git.

**O que NÃO foi feito:** não foi usado como corpus de treino. Acrescentaria ~1,4 M de
caracteres e faria muita falta (o corpus é de 244 mil), mas seriam dados de um autor
protegido — e a directiva é «licença e proveniência por documento».

**Consequência honesta:** a maior fonte de português disponível nos uploads está a ser
desperdiçada por razões legais. É a limitação central do projecto e fica registada, não
resolvida.

## C5. Estado das dimensões de avaliação

| estado | antes | agora |
|---|---|---|
| IMPLEMENTADO | 1 | **2** (língua/perplexidade **+ gramática**) |
| PARCIAL | 1 | 1 (generalização, agora medida em diálogo) |
| DESENHADO | 7 | **6** |

Um bug de contagem foi corrigido no caminho: `resumo()` fazia `.split(" ")[0]` sobre estados
compostos e contava mal. Dizia «1 implementada» por sorte do prefixo; são **2**.

---

# ADENDA D — Machado de Assis: 4,5× mais dados, e a resposta definitiva

O utilizador entregou quatro obras de Machado de Assis. **938 807 caracteres, 294 107
tokens** — 4,5× o corpus que o projecto tinha. É a maior mudança de regime de dados desde
o início, e pela primeira vez o corpus é **redistribuível**.

## D1. Licença verificada, não assumida (FACT)

Machado de Assis (1839-1908). Obra em **domínio público** por mais de 70 anos após a morte,
na lei brasileira (9.610/98) e na angolana. Verificado também no concreto: os PDFs não
contêm aviso de restrição; os créditos das bibliotecas (BibVirt/USP, eLibro, UNAMA)
autorizam uso educacional e redistribuição sem alteração do texto.

**O `EB009.pdf` NÃO é Machado de Assis.** É a tese *«Na(rra)ção satírica e humorística: uma
leitura da obra narrativa de Manuel Rui»*, de **Marta de Oliveira** — obra académica
protegida, sobre um autor vivo. **Excluído do corpus** e registado em `dados/manifest.json`
como `nao_ingerido`. Seria valioso (português angolano) e não é livre.

## D2. Curadoria: 0,6% removido, com contagem por filtro (MEASURED)

`nucleo/dados/curadoria_publico.py`. Texto de PDF traz cabeçalhos de página e créditos de
digitalização — treinar nisso ensina o modelo a gerar `www.nead.unama.br`.

| filtro | linhas | caracteres |
|---|---|---|
| créditos da biblioteca | 40 | 2 114 |
| linha sem conteúdo | 1 179 | 1 178 |
| marca de paginação | 188 | 616 |
| crédito de OCR | 5 | 145 |

**944 276 → 938 807 caracteres (0,6% removido).** Os filtros apanharam o lixo e tocaram
em quase nada do texto — que é o resultado que se quer e não era garantido.

## D3. A divisão por obra resolve a contaminação (MEASURED)

Treino: Brás Cubas + O Alienista + A Cartomante (**178 560 tokens**).
Validação: **Dom Casmurro** (115 547 tokens). Nenhuma janela atravessa obras.

Isto **fecha o defeito** registado na Adenda A: a divisão anterior era 90/10 por janelas
contíguas e 7 de 20-gramas apareciam nos dois lados. Agora não pode acontecer.

## D4. A decoração cai, e cai muito (MEASURED)

| corpus | tokens | folga treino↔validação |
|---|---|---|
| original | 65 999 | **8–9 nats** |
| Machado | 294 107 | **3,49 nats** |

E a validação deixa de estar no acaso ou pior: **6,75 nats** contra `log(8192) = 9,01`.
O diagnóstico de «decora porque não há dados» era **correcto**, e mais dados melhoram-no de
facto. Isto valida retroactivamente a leitura das Adendas B e C.

## D5. Não era underfitting — era sobreajuste desde o passo 600 (MEASURED)

Havia uma explicação alternativa que tinha de ser descartada antes de concluir «o modelo é
pior»: 1500 passos serem poucos. Corrida de **4 000 passos**, validação a cada 250:

| passo | épocas | perda treino | perda val | bits/byte |
|---|---|---|---|---|
| 500 | 2,9 | 4,798 | **5,9340** | **2,6796** |
| 1500 | 8,6 | 3,408 | 6,7745 | 3,0591 |
| 2500 | 14,3 | 1,843 | 8,3338 | 3,7632 |
| 4000 | 22,9 | 1,227 | **10,6986** | **4,8311** |

**A validação piora monotonamente a partir do passo ~600.** A perda de treino cai de 5,6
para 1,2 — é decoração pura. Com 1 395 janelas de treino e lote 8, o modelo esgota o sinal
aos ~3 épocas.

**Consequência de método:** o `regime_publico.py` reportou 1500 passos e deu 3,0480 bpb.
O óptimo era 2,6796. **Reportar um número de um ponto arbitrário da curva exagerou a
derrota em 0,37 bits/byte.** Fica registado.

## D6. O trigrama ganha, e agora sabe-se por quanto (MEASURED, 3 seeds)

`resultado_melhor_ponto.json`. Curva a cada 100 passos, 3 seeds, com duas medições para
não haver optimismo selectivo:

| | bits/byte | margem vs trigrama |
|---|---|---|
| trigrama | **2,4766** | — |
| Transformer, melhor ponto observado | 2,6821 ± 0,0133 | **−0,2055** |
| Transformer, passo 500 fixo (a priori) | 2,6990 | **−0,2224** |

O optimismo de escolher o melhor ponto vale 0,017 bpb — pequeno, e mesmo assim insuficiente.
**O trigrama ganha de forma inequívoca.** As 3 seeds concordam (óptimos aos passos 500, 600
e 700; desvio de 0,008 nats).

## D7. O erro que cometi e que a escala revela (correção)

Escrevi que o modelo tinha «64 tokens por parâmetro». **É falso por um factor de 1 000.**
178 560 / 2 771 200 = **0,064**. Eu confundi 0,064 com 64.

Os modelos reais usam ~4 000 tokens/parâmetro. Este tem **0,064** — não está 60× abaixo,
está **~62 000× abaixo**. A conclusão (faltam dados) não muda; a dimensão do problema muda,
e fica mais severa do que eu a tinha descrito. Corrigido no teste e no JSON.

## D8. Cobertura: o ponto de cruzamento foi ultrapassado (MEASURED)

| | antes | agora |
|---|---|---|
| tokens únicos | 87 874 | **381 981** |
| % do cruzamento (97 320) | 90,3% | **392,5%** |
| % do limite de utilidade (2,3 M) | 3,82% | **16,61%** |
| finalidades com dados | 3 de 9 | 3 de 9 |
| dados redistribuíveis | **0** | **294 107 tokens (Machado)** |

**Pela primeira vez o projecto tem dados que pode redistribuir.** É a resposta directa à
limitação registada em A4 e B1: «nenhum dos 244 mil caracteres é redistribuível».

## D9. O que continua por explicar (UNKNOWN)

- **Porque é que 4,5× mais dados melhoraram a decoração e não chegaram para o trigrama.**
  A explicação mais provável é a razão dados/parâmetros (D7), mas isso é INFERRED, não medido:
  não se testou um modelo menor com os mesmos dados. **Esse é o próximo teste com valor** —
  um Transformer com 300-500 mil parâmetros, mesma divisão por obra, contra o mesmo trigrama.
- **Se o trigrama é batível dentro da directiva** (1-3 M parâmetros, 4 camadas, d_model 128).
  Com 0,064 tokens/parâmetro e a curva a virar aos 600 passos, a resposta pode ser «não com
  este corpus». Mas só se sabe testando o modelo menor.
- **Qualidade linguística.** A perda é um número; não se gerou uma única amostra de Machado.
  Antes de qualquer afirmação sobre «saber português», é preciso ler o que ele escreve.
