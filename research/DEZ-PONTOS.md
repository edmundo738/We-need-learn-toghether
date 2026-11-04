# Dez pontos: como este projecto se compara e o que fazer a seguir

Documento pedido como entregável da directiva de investigação (nanochat, llm.c, litgpt,
TinyStories, OLMo, dolma, Tucano/GigaVerbo). Cada ponto tem: o que os outros fazem, o que
este projecto faz, a **medição** que sustenta a comparação, e a consequência prática.
Classificação em cada afirmação: FACT / MEASURED / OBSERVED / INFERRED / HYPOTHESIS / UNKNOWN.

O que este documento **não** faz: recomendar copiar qualquer um deles. A conclusão do
`ESTRATEGIA.md` mantém-se — o núcleo próprio avalia-se como **instrumento de investigação**,
não como futuro modelo de conversa.

---

## 1. Escala: 24 tokens por parâmetro, não 4 000 (MEASURED)

A linha que explica tudo o resto:

| | tokens de treino | parâmetros | tokens/parâmetro |
|---|---|---|---|
| este projecto | 65 999 | 2 771 200 | **24** |
| OLMo-2 1B (stage 1) | 4×10¹² (máx.) | 1,2×10⁹ | **~3 300** |
| Tucano-2b4 | >2×10¹¹ | 2,44×10⁹ | **~82** |

Duas ordens de grandeza de diferença no primeiro caso, uma no segundo. Qualquer comparação
de qualidade linguística com estes modelos **não é uma comparação** — é uma categoria diferente.

FACT (do `OLMo2-1B-stage1.yaml` lido): `max_duration: 4e12 tokens`, warmup `8,389e9`.
FACT (do GigaVerbo): >200 mil milhões de tokens PT, Tucano-2b4 2,44 B parâmetros, 16×A100 ~845 h.

**Consequência:** o alvo do projecto não pode ser «aproximar-se do Tucano». O alvo é
medir o que um modelo desta escala faz e não faz, com rigor. É o que os pontos 4-10 medem.

## 2. O que a escala pequena já consegue: bater o trigrama (MEASURED)

A única comparação com benchmark equivalente que este projecto tem:

| modelo | nats | bits/byte |
|---|---|---|
| 3-grama com backoff | 5,0461 | 1,8947 |
| **Transformer 2,77 M** | **5,8962** | **2,2139** |

**O Transformer perde para o trigrama por 0,85 nats / 0,32 bits/byte.** Não é um detalhe:
o modelo com 2,77 M parâmetros é pior do que contar trigramas, com o mesmo corpus.

O ponto de cruzamento ajustado (`resultado_ajuste.json`, R² 0,9366) é **97 320 tokens =
1,47× o corpus** — a 39 segundos de treino. Ou seja: **é recuperável**, e esse é o alvo
concreto da Etapa 2. Não são 2,3 M tokens (esse número foi um erro meu, corrigido).

## 3. Weight tying: medição contra suposição (MEASURED)

nanoGPT faz tying (`wte.weight = lm_head.weight`). Este projecto implementou tying e depois
**assumiu** que era ganho. Medido com 3 seeds:

| braço | parâmetros | validação |
|---|---|---|
| tying | 1 722 624 | 6,0911 ± 0,0339 |
| sem tying | 2 771 200 | **5,7524 ± 0,0383** |

**Custa 0,3387 nats (−11,47 erros-padrão), e as 3 seeds de tying ficam todas acima das 3
de sem-tying.** O tying poupa 1 M de parâmetros e, neste corpus, esses parâmetros valem
0,34 nats. **O `nanoGPT` usa tying num regime de dados onde 1 M de parâmetros é ruído;
aqui não é.** Copiar uma decisão sem medir o seu custo é o que este ponto corrige.

## 4. A curva de dados é logarítmica e o resíduo é grande (MEASURED)

Ajuste de `perda_diferença ~ log(tokens)`, 4 pontos, `np.linalg.lstsq`:
declive **−2,4340**, intercepto **+27,9565**, **R² 0,9366**, **resíduo máximo 0,8920 nats**.

O resíduo máximo é quase 1 nat num total de ~5-7 nats. **A previsão é de ordem de grandeza,
não de precisão.** Faixa honesta do cruzamento: **0,5× a 5× o corpus**, não um número.

Comparação ajustado vs medido: 5 760→6,88 vs 6,36 · 14 592→4,62 vs 5,51 · 29 312→2,92 vs
2,89 · 58 752→1,23 vs 0,89. O segundo ponto erra por 0,89 nats.

## 5. O corpora reais não são 244 mil caracteres (OBSERVED)

Dolma (gopher taggers) e GigaVerbo filtram **datasets** de centenas de GB. Este projecto tem
244 372 caracteres, 1 autor, 5 obras — 0,0002% do GigaVerbo em caracteres.

Dolma tem `taggers/gopher.py` com `REQUIRED_ENGLISH_WORDS`: filtros específicos de inglês,
**não transponíveis para português**. O que é transponível é o método (median_word_length,
symbol_to_word_ratio, duplicate_ngrams), e foi o que se fez em `curadoria.py`:
5 obras aceites com 0,1158-0,1442; spam rejeitado a **0,996**. (MEASURED)

## 6. A divisão por obra não é a que está em uso (MEASURED — defeito)

**7 de 20-gramas aparecem entre treino e validação: 1,37% das 511 janelas**, com a fronteira
no token 59 211. A divisão é 90/10 por janelas contíguas, não por obra.

`nanochat` faz `tok_train`/`tok_eval` separados desde o início; `OLMo` tem divisão de treino
declarada no config. **Este projecto não tem.** Está registado no corpo da auditoria e
continua aberto. As medições desta adenda **herdam esse defeito**.

## 7. Optimização: o que se usa vs o que os outros usam (FACT, do código lido)

| | nanoGPT | OLMo-2 | este projecto |
|---|---|---|---|
| optimizer | AdamW | AdamW | Adam (wd 0,01) |
| weight decay | só params 2D | só params 2D | **todos** |
| warmup | 100 iter | 8,389e9 tokens | **nenhum** |
| schedule | cosine | cosine | **constante** |
| beta2 | 0,99 | — | 0,999 (padrão) |
| grad clip | 1,0 | 1,0 | 1,0 |

Três ausências reais: **sem warmup, sem cosine, weight decay em tensores 1D.** Nenhuma
delas está medida neste projecto como ganho ou perda — são **UNKNOWN**, e a Etapa 2 tem de
as medir uma a uma antes de as adoptar. Adoptar por imitação é o erro do ponto 3.

## 8. O que outros têm e este projecto não vai ter (HYPOTHESIS → confirmado como limite)

- Flash attention / `scaled_dot_product_attention`: nanoGPT usa-a; aqui há atenção O(T²) própria.
  `resultado_bench_atencao.json`: lote 8 → 517,2 ms, **1 980 tok/s** (pior que lote 4: 2 522).
- bf16/fp8: OLMo usa bf16, nanochat fp8. Aqui **FP32 por directiva**.
- RoPE θ=500 000, RMSNorm, gate QK: OLMo; aqui posicionais aprendidas, LayerNorm, atenção padrão.
- Vocabulário 32 768 (nanochat) vs **8 192** (aqui). Aumentá-lo está congelado por directiva.

Nenhuma destas ausências é um defeito a corrigir às cegas — a directiva congela a arquitectura
até as correcções fecharem. São o mapa do que **não** se pode reivindicar.

## 9. A inferência é dominada pela memória, não pelo cálculo (MEASURED)

`resultado_memoria.json`, lote 8, ctx 128: parâmetros 10,6 MB · scores 8,0 MB ·
activações 24,0 MB · **logits 32,0 MB** · RSS 227,4 MB.

Os logits (32 MB) são a maior parcela — 3× os parâmetros. Um vocabulário de 32 768
quadruplicaria isso. É um FACT medido que sustenta a decisão congelada de não aumentar o
vocabulário por razões de memória, independentemente da directiva.

## 10. O que fazer a seguir, em ordem, com critério de aceitação (INFERRED)

1. **Fechar o cruzamento**: treinar a 97 320 tokens (39 s), 3 seeds, divisão por obra.
   *Aceitação:* validação por obra, com trigrama medido no mesmo conjunto.
2. **Medir warmup + cosine + wd só-2D**, uma variável de cada vez.
   *Aceitação:* reportar os três como ganho, perda ou neutro, com desvio entre seeds.
3. **Re-treinar com o melhor checkpoint guardado** (não o último).
   *Aceitação:* o checkpoint em disco reproduz a perda de validação ao bit.
4. **Fechar a incerteza com o Transformer treinado** (A5 da adenda).
   *Aceitação:* razão fora/dentro > 1,0 com o modelo treinado, contra o controlo negativo 1,00×.
5. **Só depois** decidir sobre tying, arquitectura e escala.

**Nada disto é um modelo de conversa.** É medir um instrumento de investigação com um corpus
que não é redistribuível, 1 autor e 24 tokens por parâmetro. As estratégias A e B do
`ESTRATEGIA.md` continuam: A viável até ao cruzamento, B bloqueada, **C recomendada com a
correcção de que o núcleo próprio é instrumento, não produto**.

---

## Tabela de classificação

| classe | pontos |
|---|---|
| MEASURED | 1, 2, 3, 4, 6, 9 |
| FACT (de fonte lida) | 7 (colunas dos outros), 1 (OLMo/Tucano) |
| OBSERVED | 5 |
| INFERRED | 10 |
| HYPOTHESIS | 8 |
| UNKNOWN | optimizações ausentes do ponto 7 não medidas |
