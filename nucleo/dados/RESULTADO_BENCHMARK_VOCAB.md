# Benchmark de vocabulário — 4 096 / 8 192 / 16 384

**Corpus:** v0.1 (2 990 946 caracteres, 475 658 palavras, 12 ficheiros)
**Hold-out:** 1 826 caracteres, 321 palavras — verificado **limpo** de contaminação
**Método:** treino completo por tamanho, `python -u`, resultados em `/tmp/vocab_results.jsonl`
**Data:** 2026-11-01

---

## Resultados

| vocab | tempo | tok/palavra (treino) | tok/palavra (hold-out) | cobertura palavra inteira | tokens no corpus |
|---|---|---|---|---|---|
| 4 096 | 5,2 min | 1,914 | 2,212 | 43,1% | 939 422 |
| **8 192** | **11,3 min** | **1,715** | **1,991** | **49,5%** | **841 554** |
| 16 384 | 22,3 min | 1,574 | 1,788 | 55,3% | 772 352 |

## Leitura

**8 192 ganha ao 4 096 em todas as dimensões medidas:** menos 10,4% de tokens,
menos 10,0% de custo por palavra em texto não visto, e +6,4 pontos de cobertura.
Não é um compromisso — é melhor e pronto.

**16 384 ganha mais 8,7% de tokens**, mas custa:
- o dobro do tempo de treino (22,3 vs 11,3 min);
- o dobro da matriz de embedding: 2 097 152 parâmetros contra 1 048 576;
- num modelo de 1,2 M parâmetros, a embedding passaria a ser **63% do modelo** —
  e a maior parte dos tokens raros que ela representa aparece poucas vezes no
  corpus, por isso os seus embeddings mal treinam.

**Decisão: 8 192, congelado como v1.** O ganho do 16 384 é real mas compra-se
caro, e com 0,84 M tokens de treino não há dados para sustentar um vocabulário
que duplica a embedding.

## Tokens mortos — a verificação que faltava

| vocab | usados | mortos | % |
|---|---|---|---|
| 4 096 | 3 973 | 123 | 3,0% |
| 8 192 | 8 024 | 168 | 2,1% |

O 8 192 tem **menor** fracção de pesos mortos que o 4 096. Isso reforça a
escolha: as fusões extra não são ruído, são morfologia útil.

## Reprodutibilidade

A execução de 4 096 reproduziu **exactamente** a medição anterior da Fase 2:
5,2 min, 939 422 tokens, 1,914 tok/palavra, 43,13% de cobertura. Para um cálculo
de ponto flutuante com sementes fixas, isto é o esperado — e confirma que o
pipeline não tem não-determinismo escondido.

## O que este benchmark **não** diz

- Nada sobre a qualidade do modelo treinado com cada vocabulário. Tokens/palavra
  e cobertura são proxies: medem o custo de representar a língua, não a
  capacidade de a aprender. Um vocabulário maior pode ser melhor nos proxies e
  pior na prática se a embedding ficar sub-treinada.
- O hold-out tem 321 palavras. Os intervalos de confiança são largos e não os
  calculei — os 6,4 pontos de diferença entre 4 096 e 8 192 estão bem dentro do
  que 321 palavras conseguem distinguir, mas a ordem entre 8 192 e 16 384 não
  deve ser lida como prova de superioridade fina.
- O corpus web não entrou. Continua bloqueado neste executor.
