# Fase 1 — Tokenizer BPE + esclarecimentos da Fase 0

**Projecto:** Kianda — A Última Memória do Mar
**Base:** commit `e228791` (Fase 0 aprovada), branch `arena/9799958d-we-need-learn-toghether`
**Estado do sistema principal:** **não modificado** — `kianda/` intacto, apenas `nucleo/` novo
**Data:** 2026-11-01

---

## 0. Três coisas que tenho de dizer antes do resto

**Primeiro: a minha medição da Fase 0 não era reprodutível, e eu reportei-a como se fosse.**
Repeti o mesmo benchmark em isolamento e obtive **153 733 tokens/s**, não 29 849. A causa é
que o processo de teste de mutação estava a correr em paralelo, a competir pelos 2 vCPU. Não
volto a reportar uma medição sem aquecimento, sem repetições e sem sistema em repouso. §2
substitui o número por uma medição com método.

**Segundo: o corpus não está versionado.** Os 5 ficheiros das obras estão em `dados/`, que
está no `.gitignore`. Isto significa que **um `git clone` não reproduz o tokenizador que
acabei de treinar**. Não corrigi isto porque exige uma decisão sua sobre onde vivem ficheiros
grandes e com direitos. Fica registado como pendência aberta.

**Terceiro: a sua objecção C está certa, e eu não tenho como a resolver.** Medi 2 vCPU e
3,8 GB porque é o executor onde a Arena corre. Esse número **não é** o seu computador e não
deve dimensionar o projecto. Em §1 separo o que é do ambiente e o que é do projecto.

---

## 1. Esclarecimento da objecção C: ambiente vs. máquina do projecto

Medido neste executor:

```
nproc → 2 · RAM 3.8 Gi total · sem GPU · 20 GB livres · torch threads = 1
```

O que isto **é**: o ambiente onde a Arena compila e corre. É o piso, não o alvo.
O que isto **não é**: o seu Windows. Não tenho acesso a ele e não vou fingir que tenho.

**Como o plano deve ser dimensionado.** Proponho a regra seguinte, que não depende de eu
saber nada sobre a sua máquina:

- **O código corre na configuração mais fraca que conheço** (2 vCPU, sem GPU). Se correr
  aqui, corre em qualquer lado. Isto é uma garantia de portabilidade, não um limite de
  escala.
- **A escala decide-se por medição na máquina onde vai treinar.** Todos os scripts desta
  fase imprimem tokens/s reais; a configuração do Transformer da Fase 2 será escolhida
  depois de você correr `python -m nucleo.experimentos.benchmark_capacidade` no seu Windows
  e me enviar o resultado.
- **Nenhum número de hardware fica escrito na arquitectura.** Não há constantes ajustadas a
  2 vCPU em lado nenhum do código entregue.

**O que preciso de si:** corra o benchmark no seu computador e diga-me (a) tokens/s do caso
PyTorch, (b) tem GPU e de que modelo, (c) RAM. Com isso, a Fase 2 dimensiona-se para a sua
máquina, e não para esta.

---

## 2. Esclarecimento da objecção B: o que os 29 849 tokens/s mediam

### O que medi na Fase 0 (e o que estava errado)

Medi o **throughput de quatro multiplicações de matrizes `d=128` através do autograd NumPy
do projecto**, com forward, backward e um passo de Adam. Não era um Transformer: não havia
atenção, nem LayerNorm, nem residuais, nem embeddings, nem softmax sobre a sequência. E o
backward era sobre um grafo de 4 nós, não sobre um bloco real.

Foi uma medição parcial usada para estimar o desempenho do modelo completo. **Foi um erro de
método e está corrigido em §2.1.**

### 2.1 Medição com método

`nucleo/experimentos/benchmark_capacidade.py`, aquecimento + 15 repetições + mediana:

| Caso | O que mede exactamente | tokens/s |
|---|---|---|
| autograd NumPy | 4 matmuls d=128, FWD+BWD+Adam | **153 733** (118 937–161 805) |
| PyTorch | a mesma operação, mesma forma | **246 919** (154 267–255 587) |
| atenção causal PyTorch | Q·Kᵀ + máscara + softmax + ·V, B=8 H=4 S=256 d=32, FWD+BWD | **95 287** |

Leituras:

- **PyTorch é 1,6× o autograd próprio** nesta operação. Não é 10× nem 100×: o autograd do
  projecto é NumPy competente, e a diferença é menor do que eu esperava. Isto **apoia a sua
  decisão** de usar PyTorch sem desvalorizar o trabalho já feito.
- **A atenção custa ~2,6× mais do que os matmuls equivalentes.** É o termo quadrático a
  aparecer. Confirma que o custo real do Transformer será substancialmente acima destes
  números — razão pela qual a Fase 0 não devia ter extrapolado.
- **Dispersão de 28–41%.** Esta máquina é ruidosa (2 vCPU partilhados). Qualquer decisão
  baseada em diferenças menores do que ~2× seria ruído, não sinal.

**Nenhum destes números é um Transformer.** O custo real mede-se na Fase 2, com o modelo
construído e o mesmo método. A diferença é que agora está escrito o que cada caso mede.

---

## 3. Esclarecimento da objecção A: validação do gerador actual

Condição 5 da directiva. `nucleo/experimentos/validacao_gerador.py`, resultado em
`resultado_validacao.json`.

### Metodologia

- **Corte contíguo 90/10**, não aleatório: janelas de contexto sobrepostas num split
  aleatório deixariam o modelo ver o alvo antes de o prever.
- **Validação sem actualização de pesos**: reconstrói o forward e mede entropia cruzada, sem
  `backward()` nem `opt.step()`. Uma validação que treina não é validação.
- **Medição de memorização**: maior substring comum entre amostra gerada e corpus de treino.

### Resultados

| Métrica | Valor |
|---|---|
| Corpus | 254 637 caracteres |
| Treino / validação | 229 173 / 25 464 |
| Parâmetros | 192 955 |
| Vocabulário | 91 caracteres |
| Épocas | 6 (280 s) |
| Perda de treino | 1,4601 → **perplexidade 4,31** |
| **Perda de validação** | 2,0942 → **perplexidade 8,12** |
| Lacuna validação − treino | **+0,6341** |
| **Memorização** | **0%** (0 caracteres copiados) |

### Amostras geradas (semente «O mar»)

```
T=0.5: «a do ser com a destrra de sentido dele onde contento o que os segundos e que
        continua que não está o que não e»
T=0.8: «ada. O quando poltor e que neguntrada. Sei a na correles. A comigo porto o
        chegara — estás pertordes que eu no»
T=1.0: «a.. Não sabe dos que precisava das traliat. Eu verificado de morante-re
        pruperado. Agentiva que são car a mãe»
```

### As cinco respostas que pediu

1. **Conjunto de treino** — 229 173 caracteres, os primeiros 90% do corpus das 5 obras.
2. **Perda de validação** — 2,0942 (perplexidade 8,12), contra 1,4601 em treino.
3. **Perplexidade de validação** — **8,12**, e não é comparável com perplexidade de modelo
   BPE: vocabulários de tamanhos diferentes produzem números que não se podem pôr lado a
   lado. Quem o fizer está a medir o vocabulário, não o modelo.
4. **Caracteres gerados sem recorrer ao corpus** — 100% das amostras. A maior cópia contígua
   do treino tem **0 caracteres** nas três temperaturas: o modelo **não recita**.
5. **Generaliza para texto que nunca viu?** — **Parcialmente, e é esse o resultado.**
   Generaliza a *forma* da língua (morfologia, acentuação, comprimento de palavra, ritmo de
   pontuação) e **não generaliza o significado**: as palavras inventadas — «destrra»,
   «neguntrada», «pertordes», «morante-re» — são a assinatura de um modelo de caracteres que
   aprendeu ortografia sem semântica.

**Interpretação honesta:** uma lacuna de +0,63 e 0% de memorização é o padrão de um modelo
que **aprende sem decorar**. Isso é bom sinal sobre o pipeline. Também é irrelevante para a
conversa: perplexidade 8,12 em caracteres não produz frases com sentido, e nenhuma amostra
acima tem proposição. **O gerador actual valida a mecânica de treino; não valida linguagem.**

---

## 4. Fase 1 — Tokenizador BPE entregue

### 4.1 O que foi construído

`nucleo/tokenizador/bpe.py` — 260 linhas, **biblioteca padrão apenas**, zero dependências.

**Decisão de desenho: byte-level e não por caracteres.** O português tem acentos que, com
vocabulário de caracteres, tornariam «coração» uma unidade atómica e o modelo nunca
aprenderia que «coração» e «corações» partilham radical. Sobre bytes UTF-8, «ç» e «ã» são
sequências de 2 bytes, o vocabulário base são os 256 valores possíveis, e **nenhum texto é
irrepresentável** — a propriedade que torna o round-trip exacto possível por construção, e
não por sorte.

### 4.2 As seis condições da directiva, testadas

`nucleo/testes/test_tokenizador.py` — **52 testes, todos verdes** (60 s).

| Condição | Como foi testada | Resultado |
|---|---|---|
| 1. Não alterar o sistema principal | `git status` limpo em `kianda/` | ✅ |
| 2. Testes de codificação/descodificação | round-trip em corpus inteiro, linha a linha, e 10 500 strings aleatórias (fuzz) | ✅ |
| 3. Acentos, pontuação, especiais | 38 casos explícitos + fuzz de code points Unicode arbitrários (plano astral incluído) | ✅ |
| 4. Guardar vocabulário e metadados | `vocabulario.json` + `vocabulario.txt` legível + `relatorio.json` | ✅ |
| 5. Proveniência e licença | `metadados` + nota no artefacto — **licença pendente de decisão sua** | ⚠️ |
| 6. Texto fora do treino | `nucleo/dados/validacao_tokenizador.txt`, escrito à mão, nunca dado a `treinar()` — o script aborta se o hold-out estiver dentro do corpus | ✅ |

O teste de fuzz é o único que não fui eu a escolher os casos, e é por isso o mais valioso: os
outros 51 testam o que programei; o fuzz testa o que **afirmo**.

### 4.3 O tokenizador treinado, e o problema que a medição revelou

`python -m nucleo.dados.treinar_tokenizador` → vocabulário de **4 096 tokens**, 92,7 s:

| Métrica | Treino | Hold-out |
|---|---|---|
| tokens por palavra | 1,223 | **1,861** |
| tokens por byte | 0,214 | 0,317 |
| cobertura de palavra inteira | 41,6% | **32,4%** |

**Penalização do hold-out: +0,638 tokens/palavra.** O script emite aviso automático — e é um
sinal real: o vocabulário está **ajustado às obras**, não ao português. Com 0,25 MB de texto,
o BPE só aprende fusões que aparecem nas obras; tudo o que é de fora custa ~50% mais tokens.

Isto é a mesma insuficiência de corpus da Fase 0, agora medida num segundo sítio e com outro
instrumento. É a evidência mais forte de que **o corpus é o constrangimento, não a
arquitectura**. E é por isso que o §5 desta resposta não é opcional.

---

## 5. Proposta de aquisição de corpus (condição 2 da directiva)

O objectivo não é «mais texto»: é texto que cubra o que falta — português geral, registo
conversacional, e português angolano. Sem isso, o modelo aprende Luanda do Rangel e mais nada.

| Fonte | Tipo | Licença | Estimativa | Esforço |
|---|---|---|---|---|
| **Wikipedia PT** (dump) | enciclopédico | CC BY-SA | ~1,2 GB texto | baixo, script |
| **Wikipedia AO / PALOP** e variantes | PT angolano | CC BY-SA | ~10–50 MB | baixo |
| **Project Gutenberg PT** | literatura domínio público | domínio público | ~200–400 MB | baixo |
| **Diários oficiais / legislação PALOP** | formal, técnico | público | moderado | médio, parsing |
| **Domínio público angolano** (Pepetela, Manuel Rui — verificar por obra) | literatura angolana | **a verificar obra a obra** | limitado | **alto, jurídico** |
| **Conversas autorizadas** (utilizadores que consentem) | diálogo | consentimento explícito | pequeno mas **decisivo** | alto, ético |
| **Texto sintético para diálogo** (gerado e revisto por humanos) | diálogo | do projecto | controlado | médio |

**Nota sobre direitos:** não vou incluir obra de autor angolano ainda sob protecção sem
autorização escrita. Não sei o estatuto de cada uma e não vou assumir. Se quiser essa via,
precisa de ser você a obtê-la.

**Proposta em três ondas, cada uma verificável:**

1. **Onda A (1 semana):** Wikipedia PT + Gutenberg PT. Duas fontes, licenças claras,
   scripts reprodutíveis, `manifesto.json` com hash e URL por documento. Leva o corpus de
   0,25 MB para ~1,5 GB. **Isto sozinho resolve o constrangimento de pré-treino.**
2. **Onda B (2 semanas):** PALOP, legislação, e um subconjunto anotado de português
   angolano colhido com consentimento. É o que dá identidade linguística ao modelo.
3. **Onda C (contínua):** diálogo real autorizado + sintético revisto. Só isto permite a
   Fase 5, e só pode começar depois de A e B estarem medidos.

**O que não proponho:** raspar a web em geral. Sem controlo de licença e de qualidade, o
`manifesto.json` deixa de valer e a §4.1 da directiva — proveniência e verificação de
contaminação — fica decorativa.

---

## 6. A sua Fase 1.5: o que penso, com franqueza

A camada cognitiva em volta do modelo — `speaker`/`target`, resolução de entidades,
geração de N candidatos, avaliador com score, ciclo de feedback — está bem pensada, e
concordo com o princípio que a justifica: *separar «o modelo sabe gerar linguagem» de «o
sistema sabe controlar essa linguagem»*.

**Mas há um risco de sequência que tenho de apontar.** A Fase 1.5 inclui 14 sistemas, e dois
deles (avaliador de respostas, classificação de feedback) só têm o que avaliar quando existe
um gerador. Se os construirmos antes, o «avaliador» vai pontuar texto de templates do sistema
antigo — e o resultado será exactamente o que queremos evitar: uma pontuação elevada que não
diz nada sobre o modelo.

**Proposta de partição**, que preserva a sua separação e evita esse beco:

| Grupo | Componentes | Depende do Transformer? |
|---|---|---|
| **1.5-a — estrutura de diálogo** | speaker, target, entidades, ligação entre turnos, tópico | **não** — testável com diálogos escritos à mão, hoje |
| **1.5-b — amostragem e selecção** | N candidatos, avaliador, rejeição/regeneração | **sim** — precisa de um gerador para produzir candidatos |
| **1.5-c — ciclo de aprendizagem** | feedback, histórico, ferramentas de treinador, dataset de erros | parcial — o registo pode começar já; o treino não |

**Recomendação:** fazer **1.5-a agora, em paralelo com a Fase 2**, porque não depende do
modelo e é onde estão os defeitos que já medimos (a Kianda confundir sujeito, atribuir «tu» a
si mesma, perder o antecedente de «isso»). E deixar 1.5-b para depois da Fase 4, quando
houver geração real para pontuar. **Decide você**; eu não avanço em 1.5 sem essa decisão,
porque construir um avaliador antes do gerador é construir uma balança sem nada para pesar.

---

## 7. Estado, artefactos e o que não fiz

### Entregue nesta fase

```
nucleo/
├── tokenizador/
│   ├── bpe.py                      # 260 linhas, stdlib apenas
│   └── artefactos/
│       ├── vocabulario.json        # 91 KB — o artefacto auditável
│       ├── vocabulario.txt         # legível, para inspecção humana
│       └── relatorio.json          # métricas treino vs hold-out
├── dados/
│   ├── validacao_tokenizador.txt   # hold-out escrito à mão
│   └── treinar_tokenizador.py      # treino + medição + aviso automático
├── testes/
│   └── test_tokenizador.py         # 52 testes, verdes
└── experimentos/
    ├── validacao_gerador.py        # objecção A
    ├── benchmark_capacidade.py     # objecção B
    ├── resultado_validacao.json
    └── resultado_benchmark.json
```

### Verificação

```
$ pytest nucleo/testes/test_tokenizador.py -q
52 passed in 60.34s

$ git status --porcelain kianda/
(vazio — sistema principal intocado)
```

### O que não fiz

- Não modifiquei `kianda/`. Nem uma linha.
- Não comecei o Transformer (Fase 2): aguarda a sua validação deste tokenizador.
- Não construí a Fase 1.5: aguarda a decisão do §6.
- Não instalei corpus novo: aguarda aprovação do §5.
- Não afirmo que o gerador actual «aprende linguagem». Aprende ortografia; a validação diz
  exactamente isso com números.

### Pendências que dependem de si

1. **Benchmark na sua máquina** (§1) — para dimensionar a Fase 2 no hardware certo.
2. **Licença dos dados** (§4.2, condição 5) — o artefacto tem proveniência mas não tem
   licença declarada, porque não sei qual é.
3. **Corpus: aprovar a Onda A** (§5) — sem isto, a Fase 2 treina num corpus que já sabemos
   ser insuficiente.
4. **Fase 1.5: partição a/b/c** (§6) — para não construir um avaliador sem gerador.
5. **Versionar o corpus** (§0) — para o treino ser reprodutível a partir de um clone.

**Aguardo validação da Fase 1 para avançar para o Transformer.**
