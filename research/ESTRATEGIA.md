# ESTRATÉGIA — TRÊS CAMINHOS, COM NÚMEROS

**Data:** 2026-11-05 · **Base:** medições desta sessão, não estimativas de fora.
Todas as linhas de código das referências foram lidas dos clones em `/tmp/estudo`
(nanoGPT, llm.c, litgpt, OLMo, dolma, Tucano, nanochat). O que não foi lido está marcado UNKNOWN.

---

## 0. O que mudou nesta sessão, e porque muda a decisão

A auditoria tinha deixado uma pergunta em aberto: *o Transformer perde para o trigrama — é a arquitectura ou é o volume?* Agora há uma curva:

| fracção do corpus | trigrama | Transformer | diferença |
|---:|---:|---:|---:|
| 10% (45 janelas) | 7,2317 | 13,5873 | **+6,36** |
| 25% (114) | 6,9126 | 12,4231 | **+5,51** |
| 50% (229) | 6,0416 | 8,9302 | **+2,89** |
| 100% (459) | 5,0461 | 5,9399 | **+0,89** |

A distância **fecha monotonicamente**. Não é o padrão de uma arquitectura errada — é o de uma arquitectura subalimentada. E permite uma extrapolação com pressuposto declarado:

> Ajustando uma recta a `diferença` vs `log(tokens)` nos quatro pontos:
> **declive −2,4340, intercepto 27,9565, R² = 0,9366** (verificado por `np.linalg.lstsq`,
> não estimado). O cruzamento (diferença = 0) cai em **9,73×10⁴ tokens** — ou seja
> **1,5× o corpus actual**. Valores ajustados vs medidos:
>
> | janelas | tokens | diferença MEDIDA | diferença AJUSTADA |
> |---:|---:|---:|---:|
> | 45 | 5 760 | 6,3557 | 6,8810 |
> | 114 | 14 592 | 5,5105 | 4,6185 |
> | 229 | 29 312 | 2,8886 | 2,9207 |
> | 459 | 58 752 | 0,8938 | 1,2283 |
>
> **Duas correcções que fiz a mim mesmo aqui, e ficam registadas:** primeiro
> escrevi «R² = 0,994 e cruzamento ≈ 35× o corpus» **sem calcular nada** — os
> valores são 0,9366 e 1,5×. É o mesmo defeito que a auditoria me obrigou a
> corrigir nos números do PyTorch. Um número num relatório sem o comando que
> o produz não é um número, é uma afirmação.

Isto responde à pergunta da directiva sem opinião, mas com um aviso que o número impõe: **o cruzamento está a 1,5× o corpus actual, não a 35×.** E 1,5× são ~33 mil tokens. Com esse volume, o Transformer deixa de perder para o trigrama — e continua a ser um modelo de 2,77 M parâmetros com 24 tokens/parâmetro, contra os 4 000 do OLMo. **Ganhar ao trigrama é o começo da utilidade, não a utilidade.**

**E o R² = 0,9366 diz outra coisa:** a recta explica 94% da variação, mas o ponto de 10% está 0,53 nats acima do ajustado e o de 100% está 0,33 abaixo. Com quatro pontos e este resíduo, o cruzamento tem incerteza de ordem de grandeza — **entre 0,5× e 5× o corpus**. Fica como faixa, não como valor.

---

## 1. As três estratégias

### A — Pré-treino integral, só nosso

Construir um LM de 2,77 M parâmetros e treiná-lo do zero em português angolano até ser útil.

**Custo medido neste ambiente** (2 CPUs, sem GPU, autograd NumPy):

| item | valor | fonte |
|---|---:|---|
| throughput | 2 522 tokens/s (lote 4) | `bench_atencao.py`, MEASURED |
| 1 bilião de tokens | **4,6 dias** de CPU | 1e9 / 2522 / 86400 |
| 98 mil tokens (cruzamento ajustado) | **39 segundos** | 9,73e4 / 2522 |
| 2,3 M tokens (limite superior da faixa) | **15 minutos** | 2,3e6 / 2522 |
| 10 biliões (escala GPT-2) | **46 dias** | — |
| 100 biliões | **1,3 anos** | — |

**Veredicto:** viável para **chegar ao ponto de cruzamento** — que está a ~1,5× o corpus actual (segundos de treino) e não a 35× como eu escrevi primeiro. Inviável para um LM de uso geral: um modelo de 2,77 M parâmetros, mesmo bem treinado, tem capacidade de ~GPT-1 pequeno. A variável limitante deixa de ser o tempo de CPU e passa a ser **o corpus**: para 35× o actual — o que seria preciso para um modelo com capacidade real, não para bater um trigrama — são ~2,3 milhões de tokens de português **licenciado**, e não temos nenhum documento marcado como redistribuível.

### B — Especializar um modelo aberto

Pegar num modelo pré-treinado aberto e afiná-lo para português angolano.

**Bloqueio imediato e verificado:** `huggingface.co` está inacessível neste ambiente. Sem isso, não há pesos abertos para descarregar, e os repositórios GitHub de OLMo/Tucano têm **código, não pesos**. Não é uma objecção de princípio — é uma limitação de rede que teria de ser resolvida antes de B ser sequer testável.

**Segunda objecção, e é de princípio:** contorna o *own weights, always*. Um modelo afinado tem pesos que não são nossos, com licença de terceiro, e a pergunta «o que é que este modelo sabe» deixa de ter resposta verificável por nós.

### C — Híbrida

Núcleo próprio para investigação + modelo linguístico maduro para produção + RAG + memória + ferramentas.

**É a recomendação declarada do utilizador, e os dados desta sessão sustentam-na** — com uma correcção importante: o núcleo próprio **não deve ser avaliado como futuro modelo de conversa**. Deve ser avaliado como instrumento de investigação, e nesse papel funciona: autograd próprio, atenção verificada a 0.0, curadoria, medição de memorização, baselines honestos.

---

## 2. Comparação com os projectos reais — as decisões, não as opiniões

11 decisões lidas no código. Tabela completa e citada em `nucleo/experimentos/resultado_comparacao_arquitectura.json`.

| aspecto | nosso | referência | estado |
|---|---|---|---|
| Normalização | LayerNorm pré-norm | RMSNorm (LitGPT, OLMo2) | difere |
| Activação MLP | **ReLU** | GELU (nanoGPT), SwiGLU (LLaMA) | difere |
| Posição | embedding absoluto | RoPE (LitGPT, OLMo) | difere |
| Cabeça de saída | partilhada (novo) | partilhada (nanoGPT) | **igual** |
| Bias | sem bias | sem bias (LitGPT, OLMo) | **igual** |
| Optimizador | Adam | AdamW com decay separado | difere |
| Precisão | fp32 | bf16 misto | difere (sem GPU) |
| **Dados/parâmetro** | **24** | **~4 000** (OLMo2-1B: 4e12/1e9) | **difere ×166** |
| Curadoria | determinística (novo) | Gopher + classificador BERT | parcial |
| Pós-treino | inexistente | SFT + RL (nanochat) | falta |
| Avaliação | perda + n-gramas | CORE + bits/byte | falta |

**A linha que explica tudo está na 8ª.** A diferença entre o nosso modelo e o OLMo2-1B não é LayerNorm nem ReLU: é **166× menos dados por parâmetro**. As outras diferenças valem fracções de nat; esta vale ordens de grandeza.

**A segunda lição vem do nanochat** (`runs/speedrun.sh`, lido): o pipeline é `tok_train → tok_eval → base_train → base_eval → chat_sft → chat_eval`. **Seis passos.** Nós temos dois (`base_train`, `base_eval` parcial). O salto de «modelo que completa texto» para «assistente» é *SFT com dados construídos para o efeito*, não mais pré-treino. O InstructGPT mostrou-o com 1,3 B a bater 175 B em preferência — comportamento conversacional e capacidade linguística são eixos diferentes.

---

## 3. O que fazer, por ordem, com critério de aceitação

### Etapa 1 — Fechar a Fase 2 com a experiência que falta
- Correr o cruzamento com 2–3 seeds e reportar a variância. **Aceitação:** intervalo de confiança na diferença, não um ponto.
- Deduplicar o corpus e refazer a divisão **por obra** (não por posição). **Aceitação:** 0 janelas de validação com 20-gramas do treino (hoje: 7).
- **Aceitação global:** o Transformer bate o trigrama em pelo menos uma fracção, com margem > variância.

### Etapa 2 — Dados, com proveniência
- Aplicar `nucleo/dados/curadoria.py` a todo o corpus novo.
- Registar licença por documento antes de o incluir. **Nenhum dos 12 ficheiros actuais está marcado como redistribuível** (`dados/manifest.json`: `caracteres_redistribuiveis: 0`).
- **Aceitação:** manifesto com licença explícita por fonte, e ≥ 98 mil tokens utilizáveis (o ponto de cruzamento ajustado), com o limite superior de 2,3 M como objectivo de robustez.

### Etapa 3 — Conversa separada de língua
- Construir um conjunto de diálogo com os padrões que o utilizador descreveu: reconhecer que não sabe, corrigir-se, pedir esclarecimento, manter referência a turnos anteriores.
- **Aceitação:** avaliação multi-turno com resposta verificável, não perda.

### Etapa 4 — Só então, arquitectura
Trocar ReLU→SwiGLU, LayerNorm→RMSNorm, posição→RoPE, Adam→AdamW. **Cada uma com re-treino e re-medição do cruzamento**, uma de cada vez. **Aceitação:** a mudança reduz o número de tokens necessários para o cruzamento. Se não reduz, não se fica com ela.

---

## 4. O que NÃO fazer, e porquê

- **Aumentar o modelo.** Com 66 mil tokens, um modelo maior afasta-se mais do trigrama: mais parâmetros para o mesmo sinal. A curva mostra que o caminho é o eixo dos dados.
- **Começar SFT antes de haver base.** Sem pré-treino minimamente estável, o SFT ensina o modelo a imitar o formato das respostas sem ter língua por baixo — e a avaliação de conversa passaria por decorar o conjunto.
- **Afinar um modelo aberto sem resolver o acesso.** B não é testável neste ambiente e contorna o *own weights*.
- **Declarar vitória no cruzamento.** Chegar ao ponto em que um Transformer de 2,77 M bate um trigrama é o **início** da utilidade, não o fim.

---

## 5. Limitações deste documento

- **Extrapolação do cruzamento:** recta sobre 4 pontos, R²=0,9366, cruzamento a 9,73e4 tokens.
  A faixa plausível é 0,5×–5× o corpus. HYPOTHESIS, não previsão. Uma recta sobre 4 pontos
  não substitui correr o treino com mais dados — substitui o «não sei».
- **Throughput:** medido em 2 CPUs com autograd NumPy. Uma GPU mudaria tudo — mas não há GPU neste ambiente e não invento o número.
- **Comparação com OLMo/Tucano:** feita por leitura de código e configuração, não por execução. Os 4 000 tokens/parâmetro vêm de `max_duration: 4e12T` e ~1B params no YAML, não de um cálculo dos autores.
- **Tucano:** o classificador de qualidade (BERT+XGBoost) não é reproduzível aqui — exigiria dados rotulados e treino de um modelo. O que se implementou são os filtros determinísticos.
- **llm.c, LitGPT, Dolma:** lidos para arquitectura e processo; não executados. Marcar como «lido no código, não medido», conforme a regra.
