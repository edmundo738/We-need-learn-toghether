# Hipótese, desenho experimental e o que conta como refutação

## A hipótese

> **H1.** Uma arquitectura modular, adaptativa, com memória persistente,
> computação condicional, modelos especializados e mecanismos de aprendizagem
> atinge determinadas capacidades cognitivas com menor custo computacional do
> que uma arquitectura que depende exclusivamente do aumento de escala de um
> único modelo.

Esta é a hipótese de §20 do briefing. Abaixo, o que a tornaria falsa.

## O que conta como refutação

A hipótese cai se, para uma capacidade igual:

1. **Custo igual ou maior.** A Kianda não gasta menos FLOPs por resposta
   correcta do que a alternativa monolítica.
2. **Capacidade igual apenas na memorização.** Se a Kianda só iguala a
   baseline no conjunto visto e nunca a supera em formulações novas, então a
   vantagem não é arquitectural — é um artefacto do conjunto de testes.
3. **Modularidade sem efeito.** Desligar módulos não altera a exactidão.
   Nesse caso os módulos são decoração e devem ser removidos.

## O que foi de facto testado (e o que não foi)

### Testado

| pergunta | instrumento | resultado |
|---|---|---|
| Custo por resposta? | contagem de FLOPs no grafo | 3,57 vs 49,74 MFLOP |
| Capacidade em formulações novas? | conjunto de paráfrases | 42,9 % vs 28,6 % |
| Que módulo contribui? | ablação individual | entidades: −35 p.p. |
| O estado muda o comportamento? | `experimentos/02_estado.py` | 5 humores, 50 % de discordância sob crítica |
| Quanto custa a memória externa? | bytes do índice | 522 048 parâmetros ≈ 2 MB |

### Não testado — e é importante dizê-lo

- **Fluência.** Nenhum juiz humano avaliou o texto. A métrica é acerto
  factual, não qualidade de escrita.
- **Verdade.** O conjunto de avaliação (20 + 14 perguntas) foi escrito por
  mim. É pequeno e é enviesado pelo que eu sei responder.
- **Energia.** Sem contadores de hardware, qualquer número em joules seria
  inventado. Reporta-se FLOPs e bytes, que são contáveis.
- **Generalização para outros domínios.** Tudo aqui é sobre duas novelas
  angolanas. Não há razão para supor que os resultados transfiram.
- **Percepção humana de personalidade** (§21, perguntas 8–10). Exigiria
  protocolo com participantes e consentimento informado. Não foi feito.

## Desenho

### Conjunto A — visto
20 perguntas. A baseline monolítica é treinada **nelas**. Serve só para
confirmar que ambos os sistemas conseguem, em princípio, responder.

**Resultado: 100 % nos dois.** Este conjunto não distingue nada.

### Conjunto B — paráfrases
14 perguntas com as mesmas entidades e formulações diferentes. Nenhum
sistema as viu.

**Resultado: Kianda 42,9 %, baseline 28,6 %.** É aqui que se mede capacidade.

### Porque é que a baseline obteve 100 % no Conjunto A

Porque decorou. Perda final 0.0000 em 150 épocas sobre 420 exemplos repetidos.
Um modelo com 1,24 M de parâmetros e 20 classes memoriza trivialmente. O
resultado de 100 % é o sinal de que o conjunto A é inútil, e foi por isso que
o Conjunto B passou a existir.

### Ablações

Cada peça é desligada individualmente e o sistema é reavaliado. O objectivo
não é provar que tudo serve: é descobrir o que **não** serve.

| peça desligada | Δ exactidão | Δ FLOPs |
|---|---|---|
| ligação de entidades | **−35 p.p.** | +0,01 M |
| computação condicional | 0 p.p. | +0,03 M |
| recuperação externa | 0 p.p. | −0,002 M |

A leitura honesta: neste conjunto, a computação condicional e a recuperação
não provam o seu valor. A primeira poupa 0,03 MFLOP — irrelevante à escala de
milissegundos. A segunda é redundante porque as fichas de entidade já contêm
a resposta. **Ambas ficam porque são necessárias noutros regimes** (perguntas
sem entidade conhecida, corpora grandes) — mas isso é uma expectativa, não um
resultado, e está marcado como tal.

## Ameaças à validade

1. **Circularidade.** A camada de entidades foi escrita por alguém que já
   sabia que respostas seriam avaliadas. Uma paráfrase que não contenha
   nenhum dos apelidos falha por construção.
2. **Tamanho.** 14 perguntas no Conjunto B. Um único erro move a exactidão
   em 7 pontos percentuais.
3. **Baseline fraca.** A baseline é um MLP de uma camada oculta treinado em
   20 exemplos por classe. Uma baseline mais forte (um modelo de linguagem
   real, com prompt) teria muito mais capacidade — e muito mais custo. A
   comparação é entre *arquitecturas pequenas*, não entre a Kianda e o estado
   da arte.
4. **Métrica binária.** Acerto/no acerto ignora respostas parcialmente certas
   e ignora totalmente a coerência.

## Próximo experimento que valeria a pena

**E3 — escalar o corpus.** Repetir os dois conjuntos com 10× mais documentos,
mantendo o orçamento de FLOPs fixo. A previsão da hipótese é que a vantagem
da Kianda **cresce**, porque o custo do índice é sublinear (BM25 é O(log n)
com índice invertido) enquanto o custo de memorizar no peso é linear em n.

Se a vantagem não crescer com o corpus, a hipótese está em apuros.

---

*Documento vivo. Se um resultado for contrariado, actualiza-se aqui e não no README.*
