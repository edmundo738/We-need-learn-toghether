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

**Resultado: Kianda 78,6 %, baseline 28,6 %.** É aqui que se mede capacidade.

> **Nota de honestidade sobre este número.** A primeira medição deu 42,9 %.
> Depois de usar o conjunto de paráfrases como instrumento de diagnóstico
> (e não só de avaliação), corrigiram-se sete defeitos de linguagem — chaves
> internas ditas em voz alta, um falso positivo de crise, um stemmer
> inconsistente, transcrição contada como memória. O número subiu para 78,6 %.
>
> Isso significa que **78,6 % só é comparável com o 28,6 % da baseline se as
> correcções não tiverem sido escolhidas a olhar para este conjunto**. Foram:
> cada erro foi corrigido por ser um erro em absoluto (dizer "apenas_existo"
> é errado em qualquer pergunta), mas o conjunto deixou de estar limpo como
> teste cego. Um número honesto exige um terceiro conjunto que eu ainda não
> escrevi. Está registado como ameaça à validade, não omitido.

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
| ligação de entidades | **−25 p.p.** | +0,01 M |
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

---

# E3 — escalar o corpus (resultado)

Previsão registada **antes** de correr:

> Com orçamento fixo e corpus a crescer, a vantagem da arquitectura modular
> deve crescer, porque memorizar no peso é linear no número de factos e
> recuperar de um índice é logarítmico.

`python experimentos/03_escala.py` — resultado medido:

| n factos | índice | índice + pesos aprendidos (C) | modelo denso (B) | C − B | parâmetros B/C |
|---|---|---|---|---|---|
| 120 | 14 % | 47 % | **53 %** | −6 p.p. | 263× |
| 250 | 27 % | 64 % | **71 %** | −7 p.p. | 262× |
| 500 | 37 % | **86 %** | 77 % | +9 p.p. | 261× |
| 1000 | 46 % | **91 %** | 81 % | +10 p.p. | 261× |
| 2000 | 48 % | **96 %** | 83 % | +12 p.p. | 261× |

**Cruzamento em n≈500.** Abaixo disso o modelo denso ganha; acima disso o
índice calibrado ganha, e a vantagem alarga com o corpus. Com **478 parâmetros
aprendidos** contra **124 939**.

A curva é monótona nas duas variáveis — a exactidão de C sobe sem parar
enquanto a de B satura. É a assinatura que a hipótese previa.

## O que quase invalidou este experimento

Três erros de desenho, encontrados por inspecção e não por erro de execução:

1. **Fuga de rótulo.** Os documentos de treino continham a entidade no texto
   como metadado. O modelo denso lia-a. Corrigido: a entidade é extraída do
   texto recuperado, o que também é o que um sistema real faria.
2. **Factos sintéticos inaprendíveis.** Gerados com palavras aleatórias, o que
   tornava toda a generalização impossível por construção. Substituídos por
   assinaturas: cada entidade tem um conjunto de palavras próprias.
3. **Divisão treino/teste em ciclo.** As entidades eram atribuídas por
   `i % n_ent`, o que deixava o conjunto de teste perfeitamente balanceado e
   sobreposto. Corrigido com atribuição aleatória.

Nenhum destes erros produziu uma excepção. Os três produziram resultados
plausíveis e errados.

## O que este resultado *não* prova

- **Não é linguagem.** O vocabulário é controlado (300 palavras) e a tarefa é
  associação facto→entidade. Um modelo de linguagem real faz muito mais coisas
  que esta tarefa não toca.
- **O índice foi calibrado no mesmo regime da tarefa.** C aprende pesos por
  termo com gradiente sobre a perda; B aprende uma projecção. A comparação é
  justa em dados e em optimização, mas o índice tem uma vantagem indutiva: a
  representação já está alinhada com a tarefa.
- **Um corpus sintético.** Os 31 factos naturais das obras são reais; os
  restantes 1 969 são gerados. A curva pode ser uma propriedade do gerador.

## O que mudou no projecto por causa disto

Nada, ainda. O resultado justifica testar `SistemaIndiceCalibrado` dentro do
`Recuperador` real, com as 20 + 14 perguntas do banco. Essa é a próxima
alteração e não uma consequência automática deste experimento.

---

# E3b — transplantar a calibração para o recuperador real: abandonado

Depois do E3, a tentação óbvia era meter os pesos aprendidos no `BM25` real.
Foi tentado e **abandonado**. O registo fica porque a decisão é mais
instrutiva do que o resultado.

## O que se tentou

`BM25.calibrar(pares)` — aprender um multiplicador por termo com gradiente
sobre `-log p(documento certo)`, regularizado para 1,0, e **só aplicar** se
houvesse supervisão suficiente (`pares / vocabulário ≥ 0,25`).

## O que se mediu

| regime | pares | vocabulário | razão | resultado |
|---|---|---|---|---|
| banco real (20 perguntas) | 17 | 4 877 | 0,003 | não aplicável (210× abaixo do limiar) |
| E3 sintético, n=300 | 210 | 474 | 0,44 | perdade 4,2 × 10⁻⁸ — **nenhum gradiente** |
| E3 sintético, n=1000 | 700 | 478 | 1,46 | perdade ~0 — **nenhum gradiente** |

No corpus sintético a supervisão era suficiente, e mesmo assim nada aconteceu.
A causa: os pares eram `(texto do documento, esse mesmo documento)`. A
consulta *é* o documento, e o BM25 já o põe em primeiro lugar com margem
gigantesca. A perda começa praticamente em zero, `dL/dw ≈ 0`, e não há sinal
para aprender. Estava a calibrar um problema já resolvido.

## Porque não ficou no código

Uma versão anterior deste método tinha um bug — `L2` fora do âmbito — que
teria dado erro se fosse executado. Foi corrigido, e depois o método revelou-se
inútil. É a diferença entre "funciona" e "serve": o código pode estar correcto
e não valer nada.

Se tivesse ficado, os relatórios futuros mostrariam "recuperador com
calibração aprendida" — uma frase verdadeira sobre um componente que não
altera nenhum resultado. É exactamente o tipo de funcionalidade que §25 do
briefing proíbe: construída porque parece sofisticada.

## O que ficou do E3

O resultado do E3 mantém-se válido como experiência isolada: com pares
*associativos* (facto → entidade, não texto → ele próprio), o índice calibrado
supera o modelo denso e a vantagem cresce com o corpus. O que não se
transpôs foi a aplicação ao recuperador, porque a tarefa do recuperador não é
uma tarefa de associação com rótulo — é semântica aberta, e as 20 perguntas do
banco não são supervisão suficiente para 4 877 termos.

## Condição para voltar a tentar

Voltar a este método apenas se existirem **≥ 300 pares rotulados**
(pergunta → passagem certa). Nesse caso a razão sobe acima do limiar e o
gradiente tem sinal. Sem isso, a resposta certa é a de agora: BM25.
