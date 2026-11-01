# Auditoria adversarial — 2026-11-01

**Veredicto: o sistema não passa.** Não é uma questão de polimento. Em conversa
real, a Kianda responde a coisas que não lhe foram perguntadas, ignora
afirmações emocionais, e não distingue «não gosto de ti» de «já perdeste
alguém?». O utilizador disse *«min ada esta a funcionar nada»* e, reproduzindo a
conversa dele, tem razão.

Este documento existe porque os 96 testes verdes anteriores estavam a medir o
que o sistema sabe fazer, não o que ele falha a fazer.

---

## 0. Reprodução da conversa do utilizador

16 turnos, mesma instância, mesma ordem. **Zero respostas coerentes.**

| turno | intenção | resposta |
|---|---|---|
| «a última memória do mar» | pergunta_tecnica 0.60 | *A memória tem cinco tipos: episódica…* |
| «tensa» | pergunta_pessoal 0.70 | *A memória tem cinco tipos: episódica…* |
| «como vai, kianda?» | pergunta_pessoal 0.62 | *A Kianda não afirma ser consciente…* |
| «eu apenas quero saber como você está» | conversa 0.52 | *Relação entre Zuleika e Bisa…* |
| «você está mal» | pergunta_pessoal 0.18 | *Porque é que perguntas isso agora?* |
| «estou a falar com o nada pois não funciona» | pergunta_pessoal 0.55 | *Se me perguntas do medo…* |

O classificador acerta em várias (a intenção está certa e é irrelevante) e a
**fala** falha sempre. O problema não é entender a pergunta: é não ter nada
para dizer que responda a ela.

---

## 1. Defeitos confirmados com medição

### A. `crítica` é um saco único que engole feedback, irritação e desabafo

```
«você é burra»                           -> pergunta_pessoal
«não gosto de ti»                        -> pergunta_pessoal
«não quero estas a me irritar»           -> pergunta_pessoal
«estou a falar com o nada pois não funciona» -> pergunta_pessoal
```
Quatro insultos/desabafos diferentes, nenhum reconhecido como tal. E quando
acerta em `critica`, a resposta é sempre a mesma frase defensiva:

```
«Diz-me o que estava mal e eu corrijo — mas diz-me em concreto…»
```
Repetida literalmente a «nao perguntei afirmei» e a «o erro é o seguinte…».

**Gravidade: alta.** Esta é a intenção em que a Kianda mais se afasta de uma
pessoa, e é justamente onde o briefing pede autonomia sem rastejar.

### B. Não há resolução anafórica

```
«o que sentes?»            -> [pergunta_pessoal]
«o que isso significa?»    -> «Estou a ouvir. Continua.»
                           -> «E tu — já perdeste alguém e não sabes se aconteceu?»
```
«isso» não está ligado a nada. Numa conversa de 16 turnos, **não há um único
ponto em que ela retome o que foi dito antes**. O histórico existe (`turnos`),
a memória episódica existe, e nenhum dos dois é consultado para interpretar um
pronome.

### C. Afirmações de uma palavra não são afirmações

```
«tensa»   -> «A memória tem cinco tipos: episódica…»
«cansada» -> «Não tenho isto. Não quero inventar.»
«triste»  -> «Estou a ouvir. Continua.»
```
Não há caminho para «estou triste» ser tratado como estado do utilizador em vez
de pergunta sobre o corpus. A Kianda não tem modelo de *quem fala*.

### D. O passo `VERIFICAR` nunca corre

```
onde o raciocínio parou, em 39 turnos:
    COMPOR   36
    -         3
    VERIFICAR 0
```
O passo existe, tem orçamento e código, e **nunca é autorizado**. A Kianda nunca
verifica uma resposta antes de a dizer. A UI mostra «orçamento não o justifica»
em todos os turnos — não é um caso, é a totalidade.

### E. O mesmo parágrafo servido a perguntas diferentes

15 perguntas diferentes de conversa corrente → 12 respostas distintas, e
`estás bem?` / `estou triste` / `estou cansado` recebem **a mesma frase**. As
respostas vêm de ~14 documentos de identidade fixos. Não há composição: há
selecção de um bloco.

### F. Sobreposição lexical pergunta↔resposta

4 de 10 respostas partilham **menos de 15%** do vocabulário com a pergunta.
Numa delas, 0%. Isto não é estilo: é ausência de ligação.

---

## 2. O problema de fundo que ninguém quer escrever

A arquitectura tem **três gerações de capacidade**: recuperar (BM25 + índice
semântico), classificar (Naive Bayes), e falar (templates escolhidos por regras).
Há uma **quarta que não existe**: compreender.

Toda a conversa é um pipeline `texto → classe → template`. Não há nenhum
momento em que o sistema construa uma representação do que o utilizador disse e
a use para produzir uma resposta nova. As respostas são blocos pré-escritos,
bem escritos, escolhidos por uma tabela de palavras.

Isso limita o tecto de forma dura:
- qualquer pergunta fora do vocabulário cai num bloco genérico;
- qualquer conversa longa é uma sequência de blocos sem fio;
- melhorar exige escrever mais blocos, o que não escala nem demonstra capacidade.

**A pergunta de investigação do projecto — capacidade sem custo proporcional —
continua legítima. Mas o que está a ser medido não é capacidade: é cobertura de
templates.** Um sistema de templates com 300 frases acerta nas perguntas para que
foi escrito e falha tudo o resto; o custo é baixo porque não pensa. Apresentar
isso como eficiência cognitiva é o erro central do projecto neste momento.

---

## 3. Onde os números publicados enganam

| publicado | problema |
|---|---|
| 100% nas paráfrases | o conjunto foi usado para corrigir 11 defeitos. É treino. |
| 93,3% no reservado | corrigi 7 das 15 perguntas depois de ver os resultados |
| 0,53 MFLOP/resposta | **não inclui compreender**. A "compreensão" é contar palavras: custo ~0. O número mede o custo de *não* compreender. |
| ablação `sem_recuperacao` = 95% | prova que a recuperação quase não importa — e ninguém quis ler isso |
| exactidão vs baseline | a baseline é um NB treinado em 14 exemplos. Ganhar-lhe não é um resultado. |
| 45/45 depois 96/96 testes | nenhum testava conversa real. O utilizador encontrou em 3 minutos o que 96 testes não viram. |

---

## 4. O que faria este projecto seguir em frente

Não é mais um módulo. É escolher entre duas coisas honestas:

**(a) Assumir que é um sistema de recuperação com interface de personagem** e
parar de o apresentar como arquitectura cognitiva. Aí os números servem, o
âmbito fecha, e o trabalho é bom no que é.

**(b) Construir a peça que falta:** um mecanismo que produza uma resposta a
partir do que foi dito, não um bloco da lista. O caminho mais barato e
verificável:
1. **Representação do turno** — o que o utilizador afirmou/perguntou/pediu,
   com sujeito (ele/ela/mundo) e tópico. Sem isto, C e B são impossíveis.
2. **Resolução de referências** contra os últimos N turnos.
3. **Composição mínima**: responder com as palavras do interlocutor +
   a evidência certa, em vez de escolher um bloco.
4. **Ligar o `VERIFICAR`** — sem verificação, (b) não é verificável.

Ordem obrigatória: (b) só conta se cada passo tiver um teste que **falhe hoje**.
Cada um dos defeitos A–F acima é um teste que hoje falha.

---

## 5. Recomendação

Não continuar a acrescentar intenções ao classificador. O classificador não é o
problema — está a responder com 0,6–0,9 de confiança a frases que a fala
transforma em não-respostas. **O problema está entre a classificação e a fala,** e
é aí que a próxima semana de trabalho deve ser gasta.

Se em duas semanas a conversa do utilizador (a mesma, os mesmos 16 turnos) não
melhorar de forma mensurável, a resposta honesta é a opção (a).
