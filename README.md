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
| **Kianda** (modular + memória externa) | 95 % | **100 %** | **0,53** | **0,10** |
| Baseline monolítica (188 k parâmetros) | 90 % | 14,3 % | 7,15 | 0,71 |
| | | | **13,6× mais caro** | **7,1× maior** |

A leitura honesta: **no conjunto visto, os dois quase empatam** (95 % contra
90 %). É por isso que o conjunto visto não serve para nada — a baseline foi
treinada nele e decora-o. O que distingue os sistemas é o **segundo conjunto**:
paráfrases que nenhum dos dois viu.

> ### O 100 % não é prova de nada
>
> Corrigi 11 defeitos **olhando para o conjunto de paráfrases**. A partir desse
> momento ele deixou de ser um conjunto de avaliação e passou a ser um conjunto
> de treino com outro nome. O número honesto para uma versão futura é o
> **conjunto reservado**: escrito de uma vez, medido antes de qualquer
> correcção.
>
> | conjunto | 1.ª medição | depois de corrigir |
> |---|---|---|
> | paráfrases (usado para corrigir) | 78,6 % | 100 % |
> | reservado v1 | **46,7 %** | 93,3 % |
>
> As 93,3 % **já não valem como generalização**: corrigi 7 das 15 perguntas
> depois de ver os resultados. É necessária uma quarta reserva.
>
> Medido também: com o critério que o banco tinha *antes* deste ciclo e o código
> de hoje, o resultado é 92,9 %, não 100 %. A diferença vem das minhas alterações
> ao critério, não de melhorias na Kianda. Ver
> `research/RELATORIO-CICLO-2026-11-01.md`.

> A baseline estava em 28,6 % antes de o stemmer ser corrigido. A mesma
> correcção que ajudou a Kianda ajudou-a a ela, porque ambas recuperam do mesmo
> índice. A comparação foi refeita em vez de se guardar o número antigo, que
> era mais favorável à Kianda. A margem é 50 pontos percentuais.

### Ablações — o que realmente contribui

| variante | exactidão | MFLOP |
|---|---|---|
| arquitectura completa | 100 % | 3,62 |
| sem ligação de entidades | **75 %** | 3,61 |
| sempre em regime profundo | 100 % | 3,64 |
| sem recuperação externa | 100 % | 3,62 |

A primeira linha é a mais informativa: **remover a camada de entidades custa
25 pontos percentuais e não poupa praticamente nada em FLOPs.** É a peça mais
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
| «Não te vou escrever isso» a quem pediu uma personagem | `fala-me de X` apanhado pela lista de verbos de escrita | 1 pergunta em 14 |
| «chamo-me Edmundo» respondido com um capítulo | o apelido do autor vencia; faltava a intenção `apresentacao` | o nome do utilizador deixou de ser aprendido |
| «preciso de placa gráfica?» → passagem da novela | a palavra não existia em documento nenhum | nenhum recuperador inventa vocabulário ausente |
| respostas a começar a meio de palavras | janelas de 700 chars cortadas à força | 41 % das recuperações ilegíveis |
| «Bom dia» a uma pergunta sobre a mãe da Zuleika | `mae_zulu` sobrevivia ao tokenizador (3 tokens = saudação) | 1 pergunta em 14 |
| «no apenas_existo, capítulo 16» | chave interna dita em voz alta | visível só quando lido |
| «Já perdeste algo?» → números de apoio | NB confundia dúvida com crise | falso positivo perigoso |
| «ela responde sempre igual» | comparação de uma string consigo própria | toda a mensagem após a 1ª virava `repeticao` |
| personagens respondiam «Chamo-me Kianda» | fallback silencioso numa função de fichas | exactidão 85 % → 42,9 % |
| estado sempre em «presente» | limiares acima do máximo atingível | o estado existia e não se manifestava |
| 0 % de discordância | homeostase a puxar abaixo do limiar | a variável nunca cruzava o próprio gatilho |
| `consultar("asdkjhasd qqwe")` → 3 resultados | o RRF dava um posto a todos os documentos, mesmo com similaridade zero | a recuperação nunca devolvia "não sei" |
| benchmark a 100 % no conjunto visto | `guardar=False` não impedia a escrita, e a instância era partilhada | 100 % contaminado contra 90 % limpo |
| resposta a «que técnicas foram testadas» com metade | `_acertou` usava `any()` sobre requisitos conjuntos | aprovaria uma resposta incompleta |
| «o que acontece quando esqueceres algo» sem esquecimento | a evidência era cortada aos 260 caracteres e a frase começava aos 262 | resposta certa no conteúdo, vazia na pergunta |
| 66 documentos em vez de 400, sem aviso | `dados/obras/` ausente e `carregar_tudo()` devolvia `{}` | o sistema parecia completo sem os livros |
| aviso de corpus visível no Python e não no HTTP | `_corpus()` copiava só as chaves que conhecia | a interface mostrava um sistema completo |
| «mano, estás tranquilo?» → pergunta do livro | palavras desconhecidas votavam com peso que dependia da classe | 0,61 nats por palavra a favor das classes pequenas |
| «Qual é o teu nome?» → «Registado. Guardo o teu nome» | o fragmento nu `"o meu nome"` dava 2,11 nats à palavra `nom` | pedia-se o nome e ela agradecia |
| «De quem é filho o Edmilson?» → ficha do Edmilson | a ficha valia 0,95 e a relação pedida 0,028 | a resposta certa era atropelada |

---

## Como se verifica

```bash
pytest testes/ -q                    # 96 testes
python testes/validar_mutacoes.py    # 9 defeitos reintroduzidos, 9 detectados
python testes/conjunto_reservado.py  # barreira fora do conjunto de treino
```

O segundo comando é o que impede o ciclo vicioso: apagar um teste para ficar
verde faz com que o mutante correspondente sobreviva e o comando falhe. Está em
CI (`.github/workflows/qualidade.yml`), em duas versões de Python, mais um
trabalho separado que corre a suite num clone **sem** `dados/obras`.

O relatório do último ciclo — com os defeitos ainda em aberto — está em
`research/RELATORIO-CICLO-2026-11-01.md`.

---

*Project Kianda · Luanda · 2026*
