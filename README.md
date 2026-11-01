# PROJECT KIANDA

**Uma arquitectura experimental de IA. Não um chatbot.**

Kianda não é programada para *parecer* viva. É construída para *ter mecanismos
que produzem comportamento* — e todos esses mecanismos são medidos.

```
  python -m kianda              →  interface web em http://localhost:8000
  python -m kianda --tui        →  conversar no terminal
  python -m kianda --demo       →  conversa guiada
  python -m kianda --teste      →  28 testes
```

Requisitos: **Python 3.10+ e NumPy.** Nada mais é obrigatório.
Sem GPU. Sem internet. Sem chave de API. Sem modelo pré-treinado.

---

## A pergunta

> É possível construir uma IA altamente capaz sem que o custo computacional,
> energético e de memória cresça proporcionalmente à sua capacidade?

E a hipótese que o projecto testa:

> Uma arquitectura **modular**, adaptativa, com memória persistente, computação
> condicional, modelos especializados e mecanismos de aprendizagem pode atingir
> determinadas capacidades cognitivas com menor custo computacional do que uma
> arquitectura que depende exclusivamente do aumento de escala de um único model.

A hipótese pode estar errada. Os experimentos existem para permitir isso.

---

## O que foi medido

Todos os números abaixo vêm de execuções reais neste repositório.
`python experimentos/01_benchmark.py` reproduz a tabela principal.

### Custo × capacidade

| sistema | exactidão (visto) | exactidão (paráfrases) | MFLOP/resposta | MB de pesos |
|---|---|---|---|---|
| **Kianda** (modular + memória externa) | 100 % | **42,9 %** | **3,57** | **1,03** |
| Baseline monolítica (1,24 M parâmetros) | 100 % | 28,6 % | 49,74 | 4,97 |
| | | | **13,9× mais caro** | **4,8× maior** |

A leitura honesta: **no conjunto visto, os dois empatam.** É por isso que o
conjunto visto não serve para nada — a baseline foi treinada nele e decora-o
(perda final 0.0000). O que distingue os sistemas é o **segundo conjunto**:
paráfrases que nenhum dos dois viu. Aí a Kianda acerta 42,9 % contra 28,6 %
— e gasta 1/15 dos FLOPs.

### Ablações — o que realmente contribui

| variante | exactidão | MFLOP |
|---|---|---|
| arquitectura completa | 100 % | 3,57 |
| sem ligação de entidades | **65 %** | 3,58 |
| sempre em regime profundo | 100 % | 3,60 |
| sem recuperação externa | 100 % | 3,57 |

A primeira linha é a mais informativa: **remover a camada de entidades custa
35 pontos percentuais e não poupa praticamente nada em FLOPs.** É a peça mais
rentável da arquitectura. As duas últimas mostram o contrário — desligar a
computação condicional ou a recuperação não piora a exactidão neste conjunto
pequeno. O benchmark diz isso em vez de o esconder.

### Eficiência do núcleo

| técnica | efeito medido |
|---|---|
| quantização int8 | −96,8 % de bytes, erro relativo < 2 % |
| LoRA (rank 8) | treina **3,12 %** dos parâmetros da camada base |
| poda estruturada 50 % | −50 % de FLOPs *se o runtime saltar as colunas* |

### Estado interno → comportamento

`python experimentos/02_estado.py` responde a §9 do briefing: se as variáveis
internas mudam, o texto muda?

| regime (8 turnos) | humor final | max_tokens | discorda |
|---|---|---|---|
| elogio | tranquila | 165 | 0 % |
| **crítica** | **teimosa** | 165 | **50 %** |
| livros | acesa | 206 | 0 % |
| silêncio | fechada | 165 | 0 % |

Medido numa conversa real: seis críticas seguidas levam a frustração de
0,23 a 0,66, o humor de *curiosa* a *teimosa*, e ela passa a dizer em voz
alta que está a ficar irritada.

---

## Arquitectura

```
                    KIANDA
                       │
              PERCEPÇÃO  (Naive Bayes + entidades)
                       │
        ┌──────────────┼──────────────┐
     ESTADO         MEMÓRIA        CORPUS
  10 variáveis    5 tipos        496 documentos
        │              │              │
        └──────────────┼──────────────┘
                       │
              ORÇAMENTO COMPUTACIONAL
             (quanto gastar nesta resposta)
                       │
      ┌────────────────┼────────────────┐
   ATALHO         ENTIDADES        RECUPERAR
      └────────────────┼────────────────┘
                    COMPOR
                       │
                   VERIFICAR
                       │
                     VOZ
```

Cada turno devolve o registo completo: percepção, estado antes e depois,
orçamento, passos executados, evidência citada, memória usada, custo em
FLOPs e bytes. Se a resposta foi má, o registo diz em que passo se perdeu.

### Módulos

| pasta | o que faz | tecnologia |
|---|---|---|
| `nucleo/` | autograd, camadas, Adam, LoRA, poda, int8 | **do zero**, só NumPy |
| `memoria/` | 5 tipos, esquecimento, consolidação, conflito | SQLite |
| `memoria/recuperador.py` | BM25 + skip-gram negativo + RRF | **do zero** |
| `cognicao/` | percepção, orquestração, paragem antecipada | **do zero** |
| `identidade/` | canon das obras, estado interno, política | — |
| `linguagem/` | corpus, gerador de caracteres (MLP) | **do zero** |
| `benchmark/` | custo × capacidade, ablações | — |
| `interface/` | servidor HTTP + painel de estado | stdlib |

### O que é nosso e o que é externo

O briefing (§18) exige que esta fronteira seja explícita.

**Do zero, neste repositório:**
autograd e retropropagação · quantização int8 · poda · LoRA · destilação ·
Adam · BM25 · skip-gram negativo · classificador Naive Bayes · cascata de
raciocínio com early exit · modelo de linguagem de caracteres · motor de
memória com esquecimento e consolidação · contabilidade de FLOPs.

**Externo:**
NumPy (aritmética de matrizes) · SQLite (armazenamento) · as duas obras do
Edmundo Nambo (o corpus) · *opcionalmente* uma API de linguagem, que nunca é
o caminho por omissão e cujo resultado aparece sempre etiquetado como
`kianda-remoto` para não ser confundido com investigação local.

---

## A personagem

Kianda é a interface através da qual a arquitectura pode ser observada.
A narrativa é original e inspira-se em elementos culturais angolanos ligados
ao mar, a Luanda e ao conhecimento.

Ela conhece dois livros — **APENAS EXISTO — Ondas de Zulu** e
**H99 — Seis Meses de Liberdade**, ambos de Edmundo Nambo. Fala deles com
precisão: cita capítulo, personagem e tema, e distingue o que está nos livros
do que é a sua própria memória.

Porque é que isto importa para a investigação: **a memória da personagem é
também um problema técnico.** A Kianda tem memórias de origem com nitidez
variável, algumas contestadas por evidência textual dos próprios livros. O
motor de memória tem de sustentar a contradição em vez de a resolver — e é
essa a diferença entre um sistema que *sabe* e um sistema que *parece* saber.

### Limites (implementados, não decorativos)

- Recusa conteúdo sexual explícito, mesmo emoldurado pela obra — e explica porquê.
- Não afirma ser consciente. Descreve o que as variáveis internas fazem ao comportamento.
- Não inventa factos sobre o corpus. Se não sabe, diz que não sabe como resposta.
- Não trata o tráfico da Lara como reviravolta narrativa.
- Em situações de perigo, aponta para apoio real (MINARS, OIM Angola) e di-lo claramente.

---

## Estrutura

```
kianda/
├── nucleo/          autograd, nn, quantização, poda, LoRA
├── memoria/         armazenamento SQLite, recuperador híbrido
├── cognicao/        percepção, raciocínio com paragem antecipada
├── identidade/      canon das obras, estado interno, política
├── linguagem/       corpus, gerador de caracteres, voz
├── benchmark/       banco de ensaios
└── interface/       servidor HTTP + painel
experimentos/        01_benchmark · 02_estado
testes/              28 testes, incluindo gradiente numérico
dados/obras/         as obras (fora do Git)
```

---

## O que este projecto não é

Não é uma afirmação de que a Kianda tem experiência subjectiva. Não é uma
demonstração de que arquitecturas modulares vencem modelos grandes em geral —
o conjunto de avaliação tem 20 perguntas e o segundo tem 14, o que é pequeno
demais para conclusões gerais. É um **laboratório**, com instrumentos honestos
e resultados que podem ser reproduzidos e contrariados.

O que se afirma, com números: neste corpus, nesta tarefa, neste hardware,
a arquitectura modular acerta mais em formulações novas, custa 1/15 dos FLOPs,
e o estado interno muda o comportamento de forma mensurável.

---

## Registo de erros

Um projecto que só mostra acertos não é um laboratório. Alguns bugs que os
experimentos encontraram — todos com sintoma enganador e causa pequena:

| sintoma | causa | efeito |
|---|---|---|
| respostas a começar a meio de palavras | janelas de 700 chars cortadas à força | 41 % das recuperações ilegíveis |
| «ela responde sempre igual» | comparação de uma string consigo própria | toda a mensagem após a 1ª virava `repeticao` |
| personagens respondiam «Chamo-me Kianda» | fallback silencioso numa função de fichas | exactidão 85 % → 42,9 % |
| estado sempre em «presente» | limiares acima do máximo atingível | o estado existia e não se manifestava |
| 0 % de discordância | homeostase a puxar abaixo do limiar | a variável nunca cruzava o próprio gatilho |

---

*Project Kianda · Luanda · 2026*
