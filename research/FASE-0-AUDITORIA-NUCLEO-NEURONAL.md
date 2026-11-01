# FASE 0 — Auditoria e isolamento

**Projecto:** Kianda — A Última Memória do Mar
**Natureza:** Relatório de auditoria exigido pela Directiva de Reforma Arquitectural (ponto 12)
**Estado do sistema principal:** **não modificado** por este relatório
**Base de código auditada:** branch `arena/9799958d-we-need-learn-toghether`, commit `772d5d6`
**Data:** 2026-11-01
**Hardware auditado:** o ambiente de desenvolvimento onde este relatório foi produzido

> Este documento responde aos 8 pontos do ponto 12 da directiva. Não implementa nada.
> Todos os números foram obtidos por execução, e cada um indica o comando ou o ficheiro
> de onde saiu. Onde não medi, digo que não medi.

---

## 0. Sumário executivo — quatro factos que mudam o plano

Antes dos oito pontos, os quatro factos que a auditoria encontrou e que a directiva
pressupunha de outra maneira:

**Facto 1 — O projecto já tem um núcleo neuronal próprio, e ele está fora do caminho.**
`kianda/nucleo/autograd.py` (540 linhas) e `kianda/nucleo/nn.py` (432 linhas) implementam
autograd com retropropagação, `Linear`, `MLP`, `Adam`, LoRA, poda e quantização int8, só com
NumPy. **Não é preciso começar do zero: é preciso ligá-lo.** A directiva assumia um sistema
"predominantemente heurístico"; na verdade existe um núcleo diferencial funcional que
**nenhuma resposta usa**.

**Facto 2 — `mente.py` nunca chama a voz.** O caminho real de produção é
`Mente.conversar` → `_compor_resposta` → `_empacotar(texto)`. O texto é composto por regras
em Python e etiquetado `autor="kianda-local"`. A classe `Voz`, que sabe usar o gerador
neuronal, **tem zero chamadas a partir de `mente.py`** (só `self.voz.gerador = g` na linha
236). Isto é o mesmo problema de honestidade que o projecto já corrigiu antes com o defeito
do fallback silencioso: a etiqueta diz "local" e sugere geração própria, mas o texto é
escrito à mão.

**Facto 3 — O gerador existente é de caracteres e é o único que já treinou de facto.**
`GeradorNeuronal` (MLP 2 camadas, contexto 20, 192 955 parâmetros) treina em 64,5 s por
época neste hardware, atinge perplexidade 7,41 e produz
`«a a sulêncio. A Bisa menter tinhas da vire da des completa da com dia a a mãe te»`.
Isto demonstra **que a cadeia treino→pesos→geração funciona de ponta a ponta**. É a base
mais valiosa que existe no repositório e a directiva não a menciona.

**Facto 4 — O hardware real não é o da directiva.** A directiva descreve "12 GB de RAM e
gráficos integrados". O ambiente medido tem **2 vCPU e 3,8 GB de RAM, sem GPU, sem PyTorch**
(ver §5). A configuração de 1–3 M parâmetros continua viável, mas o orçamento de tempo é
**entre 20 e 60 vezes maior** do que num portátil com GPU integrada recente. O plano da §7
está dimensionado para o hardware que existe, não para o descrito.

**Consequência para a decisão:** a Fase 0 não recomenda "começar a construir um Transformer
do zero". Recomenda **ligar o núcleo que já existe ao caminho de resposta**, e só depois
substituir o MLP por atenção. A ordem importa porque a primeira entrega é verificável em
dias e a segunda em semanas — e porque uma arquitectura que ninguém chama não é uma
arquitectura, é uma biblioteca.

---

## 1. Estrutura actual do repositório

Árvore de primeiro nível:

```
README.md
cerebro-mosca-1.0.0.zip      # artefacto antigo, 0 uso no código
dados/                       # gitignored — ver aviso abaixo
experimentos/                # 3 scripts (benchmark, estado, escala)
kianda/                      # 23 ficheiros .py, 7 962 linhas
kianda.egg-info/
pyproject.toml
research/                    # auditorias e relatórios
testes/                      # 10 ficheiros, 2 846 linhas
```

Distribuição de `kianda/` por subsistema:

| Módulo | Ficheiros | Peso | Função real |
|---|---|---|---|
| `mente.py` | 1 | 68K / 1 393 linhas | Orquestrador. É aqui que **toda** a resposta é composta |
| `identidade/` | 2 | 120K | Canon, personagens, fichas, estado interno |
| `cognicao/` | 3 | 164K | Percepção (NB), raciocínio, representação de turno |
| `memoria/` | 2 | 120K | BM25, armazenamento, consolidação |
| `linguagem/` | 2 | 112K | Corpus (497 docs), voz (gerador neuronal) |
| `nucleo/` | 2 | 104K | **Autograd e nn — o núcleo diferencial** |
| `benchmark/` | 1 | 72K | Banco de avaliação |
| `interface/` | 1 | 56K | Servidor HTTP stdlib |

Ficheiros maiores: `mente.py` 1 393 · `canon.py` 703 · `percepcao.py` 658 ·
`recuperador.py` 621 · `corpus.py` 590 · `banco.py` 585 · `autograd.py` 540 ·
`raciocinio.py` 505 · `nn.py` 432 · `turno.py` 406.

**Observação estrutural:** `mente.py` com 1 393 linhas concentra orquestração, composição
de fala, personalidade, estado e ~20 métodos de resposta. É o ponto de acoplamento que
impede qualquer substituição incremental do gerador. Qualquer plano que troque o gerador
tem de resolver isto primeiro.

### Aviso: o corpus não está versionado

```
$ git check-ignore -v dados/obras/apenas_existo_parte1.txt
.gitignore:10:dados/    dados/obras/apenas_existo_parte1.txt
$ git ls-files dados/
(vazio)
```

Os 5 ficheiros de obra dos quais dependem 431 documentos (`obra`) existem **apenas no
disco**. Um `git clone` produz uma Kianda que responde sem os livros e — como o próprio
código avisa em `mente.py:215` — *falha sem dizer que falhou*. Para um pipeline de treino
isto é inaceitável: o corpus tem de ter origem registada e reprodutível (§4.1 da directiva).
**Isto é o item de risco nº 1.** Não resolvi neste relatório porque exige uma decisão sobre
licenciamento e sobre onde vivem ficheiros grandes.

---

## 2. Componentes que podem ser preservados

Classifico por **evidência de uso**, não por qualidade percebida.

| Componente | Linhas | Uso real | Veredicto |
|---|---|---|---|
| `nucleo/autograd.py` | 540 | Usado por `voz.py` e `test_autograd.py` | **Preservar — é o activo central.** Já faz backward, Adam, int8 |
| `nucleo/nn.py` | 432 | Idem | Preservar (`Linear`, `MLP`, `Adam`, `LoRA`, `podar`, `destilar`) |
| `linguagem/corpus.py` | 590 | Produz os 497 documentos | Preservar. É a pipeline de dados que a Fase 3 precisa |
| `memoria/recuperador.py` | 621 | BM25, usado em cada turno | Preservar como **recuperação auxiliar** (§6.C da directiva) |
| `memoria/armazenamento.py` | 413 | Persistência + consolidação | Preservar como memória conversacional (§6.B) |
| `identidade/canon.py` | 703 | 22 docs canónicos | Preservar como **conhecimento verificado**, não como personalidade |
| `identidade/estado.py` | 342 | 8 variáveis internas | Preservar como telemetria, **não** como emoção (§7 da directiva) |
| `cognicao/turno.py` | 406 | Representação de turno (novo) | Preservar em `experimentos/` — ver §3 |
| `cognicao/percepcao.py` | 658 | Classificador NB | **Isolar** — ver §3 |
| `cognicao/raciocinio.py` | 505 | Escolha de evidência | Isolar como auxiliar |
| `linguagem/voz.py` | 371 | Gerador neuronal + templates | **Dividir**: gerador preserva-se, templates isolam-se |
| `mente.py` | 1 393 | Orquestra tudo | **Refactor obrigatório** antes de qualquer integração |
| `benchmark/banco.py` | 585 | Avaliação | Preservar e estender |

---

## 3. Componentes que devem ser isolados

A directiva (§7) diz: preservar como ferramentas auxiliares, **nunca como núcleo**. Proponho
uma regra operacional única que resolve a ambiguidade de uma vez:

> **Regra do núcleo único.** Nenhum componente heurístico pode produzir directamente o texto
> final. Só o modelo pode. Os heurísticos podem preparar entrada, escolher contexto,
> seleccionar evidência e validar saída.

Com essa regra, o que fica isolado (movido para `experimentos/legado/`, importável mas fora
do caminho de produção):

1. **`percepcao.py`** — classificador Naive Bayes por palavras. Não é compreensão: decide
   *tipo* de resposta. A directiva (§5) exige que a coerência venha do contexto e não de
   "uma lista crescente de excepções", e é exactamente isso que este ficheiro é.
2. **A tabela `_AUTO_FICHAS` / bancos de frases de `mente.py`** — respostas pré-escritas por
   intenção. É o objecto directo da §7 ("criação de centenas de templates").
3. **`_estruturar()` de `voz.py`** — esqueleto discursivo por intenção. Mesma categoria.
4. **A pergunta automática de retorno** — a directiva proíbe-a explicitamente ("perguntas
   automáticas no final de quase todas as respostas"). Hoje aparece em
   `_pergunta_de_volta`, no ramo de saudação e em `_acolher_estado`.

**Não proponho apagar nada.** Todos estes ficheiros passam a ser *baseline de comparação*:
o benchmark deve continuar a medir o sistema heurístico, porque sem esse número não há como
demonstrar que o modelo neuronal é melhor. Apagar o baseline destruiria a única medida que
temos.

---

## 4. Dependências instaladas

`pyproject.toml` declara **uma** dependência obrigatória:

```toml
dependencies = ["numpy>=1.24"]
```

Estado verificado no ambiente:

| Pacote | Versão | Necessário para |
|---|---|---|
| Python | 3.11.2 | — |
| numpy | 2.4.6 | núcleo, corpus, tudo |
| pytest | instalado | testes |
| pypdf | instalado | extracção dos PDFs → `.txt` (feito, resultado em `dados/obras/`) |
| **torch** | **ausente** | a directiva §4.1 o permite; o núcleo actual não precisa dele |

**Decisão que exige aprovação (§12, ponto 4):** a directiva diz "PyTorch pode ser utilizado".
Instalar `torch` neste ambiente custa ~2 GB de disco (há 20 GB livres) e ~200 MB de RAM
residente. **Recomendo NÃO instalar**, por três razões concretas:

1. O autograd próprio já faz forward, backward e Adam, e está **testado** (`test_autograd.py`,
   227 linhas).
2. Os 192 955 parâmetros do gerador actual treinam em 64 s/época **sem** PyTorch. A
   configuração de 1–3 M parâmetros da directiva é ~10× maior e continua no orçamento.
3. Um núcleo próprio é verificável linha a linha — que é precisamente o que a §9 exige
   ("não afirmar que existe um Transformer sem demonstrar o seu código").

Se a aprovação for para usar PyTorch, o plano da §7 muda de "semanas" para "dias" na Fase 2,
mas perde-se a propriedade de "compreender, modificar e executar localmente sem dependências"
que a §11 nomeia como ambição. **É uma escolha sua, não minha.** Não avancei em nenhuma das
direcções.

---

## 5. Capacidade real de execução no hardware disponível

Medido, não estimado:

```
$ nproc                    → 2
$ grep "model name" /proc/cpuinfo → Intel(R) Xeon(R) @ 2.60GHz
$ free -h                  → total 3.8Gi, disponível 3.5Gi
$ lspci                    → indisponível (sem GPU)
$ df -h /                  → 20G livres
$ python -c "import torch" → ModuleNotFoundError
```

**Divergência com a directiva:** o documento descreve "12 GB de RAM e gráficos integrados".
Este ambiente tem 3,8 GB e **zero** GPU. Não sei qual dos dois é o alvo de desenvolvimento;
assumo este, porque é onde o código corre e onde medi.

### Taxa de treino medida

Benchmark do custo dominante de um bloco Transformer (4 matmuls `d=128`, lote 16, seq 256,
forward+backward+Adam), com o autograd do projecto:

```
29 849 tokens/s   →   1 época sobre os 254 637 caracteres actuais: 0,1 min
```

Isto significa que **o corpus, e não o hardware, é o limite da Fase 3**. Com 254 KB de texto,
o modelo de 1–3 M parâmetros vê o corpus inteiro em poucos minutos e decora-o (overfit) em
poucas épocas. A §3.3 da directiva já antecipa isto ao exigir perda de validação; a
consequência prática é que **a Fase 3 não pode ser avaliada com este corpus**. Ver §8.

### Memória

O gerador actual usa ~193 k parâmetros em FP32 ≈ 0,8 MB de pesos. Um Transformer de 3 M
parâmetros ≈ 12 MB de pesos, mais activações e gradientes (≈4×) ≈ 60 MB. **Cabe folgadamente
em 3,8 GB.** O limite não é memória: é tempo de CPU por época quando o corpus crescer para
dezenas de MB.

---

## 6. Proposta de estrutura de pastas para o novo núcleo

A regra de ouro da directiva §8 é "área experimental separada" e §10 "não misturar
experiências com o sistema principal". Proponho que o núcleo novo seja **um pacote
independente que não importa nada de `kianda/`**:

```
nucleo/                          # NOVO — independente, sem imports de kianda/
├── __init__.py
├── dados/
│   ├── corpus.py                # limpeza, normalização, dedup, splits
│   ├── origem.py                # registo de proveniência por documento
│   └── manifesto.json           # hash + fonte + licença de cada ficheiro
├── tokenizador/
│   ├── bpe.py                   # treino BPE, encode/decode
│   └── teste_roundtrip.py       # critério de aceitação da Fase 1
├── modelo/
│   ├── atencao.py               # causal self-attention, multi-head
│   ├── bloco.py                 # residual, LayerNorm, MLP
│   ├── transformer.py           # decoder-only, logits
│   └── teste_formas.py          # máscara causal, dims, gradientes
├── treino/
│   ├── laco.py                  # batches, cross-entropy, AdamW, checkpoints
│   └── avaliar.py               # perda treino/validação, perplexidade
├── geracao/
│   ├── amostragem.py            # greedy, temperature, top-k, top-p
│   └── motor.py                 # inferência autoregressiva
└── avaliacao/
    ├── tarefas.py               # conversacionais, hold-out
    └── relatorio.py             # gera as evidências da §9
```

**Porquê uma pasta de topo e não dentro de `kianda/`:** porque a directiva exige provar que
"o texto foi produzido pelo modelo e não recuperado de documentos" (§9.3). Se o núcleo não
puder importar `kianda.mente`, essa prova deixa de depender de disciplina e passa a ser
**estrutural**. É a única forma de a tornar verificável.

**Integração (Fase 6), só depois de validado:** um adaptador fino
`kianda/linguagem/gerador.py` que expõe `gerar(prompt) -> str`, e `mente.py` passa a
chamar `self.voz.falar(...)` — a chamada que hoje não existe.

---

## 7. Plano de implementação: tokenizer e Transformer

Reordenado por dependência real e dimensionado para 2 vCPU. Cada fase com o critério de
aceitação da directiva, tornado executável.

### Fase 1 — Tokenizer BPE (estimativa: 1 dia)

- Vocabulário de 4 000 tokens, treinado sobre o corpus com `manifesto.json` assinado.
- `encode`/`decode` com **round-trip exacto** sobre 100% do corpus.
- Testes obrigatórios: acentos (`ã, ç, ê, õ`), `«»`, números, `H99`, `Z.M.`, palavras
  angolanas (`kixi`, `musseque`, `capulana`).
- **Critério:** `decode(encode(t)) == t` para todas as linhas do corpus. Este teste é
  binário e não admite "quase".

### Fase 2 — Transformer decoder-only (estimativa: 3–5 dias)

- 4 camadas, `d_model=128`, 4 cabeças, contexto 256, ≈1,2 M parâmetros (contados, não
  estimados).
- Implementado sobre `kianda.nucleo.autograd` — que já tem `softmax` (`autograd.py:305`) e
  falta apenas a multiplicação q·kᵀ mascarada e o `LayerNorm`.
- **Critério:** (a) a máscara causal é verificada por prova directa — alterar o token `t+1`
  não muda o logit em `t`; (b) todos os parâmetros mudam após um passo; (c) o número de
  parâmetros impresso bate com a soma das formas.

### Fase 3 — Treino inicial (estimativa: 2 dias de código + tempo de máquina)

- AdamW, grad-clip, checkpoints recuperáveis com hash.
- **Critério:** perda de validação desce; perda de treino **não** é aceita como prova.
- **Bloqueio conhecido:** com 254 KB de texto o modelo decora. Ver §8.

### Fase 4 — Geração (estimativa: 1 dia)

- Greedy, temperature, top-k; top-p depois.
- **Critério:** medir tokens/s reais e apresentar 20 gerações, incluindo as más.

### Fase 5 — Treino conversacional (estimativa: bloqueada até §8)

- Depende de um dataset de diálogo que **não existe**. Ver §8.

### Fase 6 — Runtime próprio (estimativa: 3 dias)

- Carregamento de checkpoints, gestão de contexto, KV-cache (só quando medido).

**Total estimado até à Fase 4: 7 a 9 dias de trabalho efectivo**, assumindo que o corpus de
§8 é resolvido em paralelo e que não se instala PyTorch. Com PyTorch, Fases 1–2 descem para
~1 dia no total.

---

## 8. Riscos técnicos e limitações de recursos

Ordenados por probabilidade de matar o projecto. Os três primeiros são **bloqueantes** e
nenhum deles se resolve com código.

### Risco 1 — O corpus é insuficiente, e isto é fatal para a Fase 5 (CRÍTICO)

Números reais do corpus hoje:

| Fonte | Documentos | Palavras |
|---|---|---|
| Obras (5 ficheiros) | 431 | ~42 400 |
| Canon, projecto, identidade, etc. | 66 | ~8 800 |
| **Total indexado** | **497** | **~51 200** |

A directiva (§4.2) propõe 1–3 M parâmetros. Regra prática: um modelo precisa de ~20 tokens
por parâmetro para não decorar. **1,2 M parâmetros × 20 ≈ 24 M tokens.** Temos **0,25 M
caracteres** — duas ordens de grandeza abaixo.

Com 254 KB de texto, o modelo de 1,2 M parâmetros atinge perda de treino baixa e **perda de
validação má** em poucas épocas. Cumpriria o critério literal da Fase 3 ("redução consistente
da perda de treino") e falharia o objectivo real ("aprender padrões"). A §3.3 da directiva já
proíbe aceitar só a perda de treino — mas a proibição não cria corpus.

**Isto resolve-se com dados, não com arquitectura.** As opções, todas com consequências:

1. Reduzir a escala para ~150 k parâmetros — cabe nos dados, mas a qualidade fica abaixo do
   gerador de caracteres actual.
2. Adquirir corpus externo licenciado em português (a §4.1 o permite: "textos em português
   com licenças adequadas"). Exige decisão sua sobre licenças e sobre download.
3. Aceitar que as Fases 1–4 são uma **demonstração de engenharia** (provar que a cadeia
   funciona) e que a Fase 5 fica bloqueada até existir corpus.

**Não escolhi por si.** Recomendo a opção 3 para as Fases 1–4 e a opção 2 em paralelo, por
ser a única que respeita "não afirmar que houve aprendizagem sem a demonstrar".

### Risco 2 — O caminho de resposta tem de ser refactorizado antes de qualquer integração (ALTO)

`mente.py:1393` linhas concentram a composição de todas as respostas. Enquanto
`_compor_resposta` decidir o texto, ligar o modelo novo não muda nada de visível. O
refactor necessário é: `_compor_resposta` deixa de devolver texto e passa a devolver
**contexto + intenção + evidência**, que alimentam o modelo como prompt. É a Fase 6, e é
mais delicada do que a Fase 2.

### Risco 3 — 2 vCPU muda a economia do projecto (MÉDIO)

29 849 tokens/s é suficiente para 1–3 M parâmetros e insuficiente para qualquer coisa acima
de ~20 M. Isto não é um problema agora; é um tecto que deve estar escrito antes de alguém
propor escalar. **Não aumentar a dimensão sem medição** (§10) — a medição está em §5.

### Risco 4 — O corpus não está versionado (MÉDIO, mas trivial de corrigir)

Ver §1. Um clone limpo não reproduz o treino. O `manifesto.json` da §6 resolve isto por
desenho, mas exige decisão sobre licenciamento dos livros.

### Risco 5 — Regressão de qualidade conversacional durante a transição (MÉDIO)

O sistema actual, com todos os seus defeitos, responde a 139 testes verdes, incluindo 16
turnos de um transcript real. O modelo novo, com 0,25 M caracteres, **vai ser pior em
conversa** durante semanas. A §10 proíbe substituir componentes em falta por templates; a
consequência honesta é que **o sistema antigo tem de continuar a servir até o novo o bater
numa medida justa**. Isso exige manter as duas vias e compará-las — não é opcional.

### Limitação declarada deste relatório

Não medi: latência de inferência do Transformer (não existe), consumo de RAM do treino a
1,2 M parâmetros (estimado por regra, não medido), nem tempo de Fase 5 (bloqueada). Estes
números entram no relatório da fase respectiva, não neste.

---

## 9. Resposta directa ao ponto 12 da directiva

| Pedido | Onde |
|---|---|
| 1. Estrutura actual | §1 |
| 2. Componentes preserváveis | §2 |
| 3. Componentes a isolar | §3 |
| 4. Dependências instaladas | §4 — **decisão pendente: PyTorch ou núcleo próprio** |
| 5. Capacidade real de execução | §5 — **2 vCPU / 3,8 GB, sem GPU** |
| 6. Estrutura de pastas | §6 |
| 7. Plano tokenizer + Transformer | §7 |
| 8. Riscos e limitações | §8 — **Risco 1 bloqueia a Fase 5** |

### Três decisões que preciso de si antes de escrever código

1. **PyTorch ou núcleo próprio?** (§4) Muda o calendário de semanas para dias, e custa a
   independência de dependências que a §11 nomeia como ambição.
2. **De onde vem o corpus?** (§8) Sem isto, a Fase 5 não pode começar com honestidade.
3. **O sistema antigo continua a responder durante a transição?** (§8, risco 5) Recomendo que
   sim, com comparação medida — mas isso significa que a Kianda terá duas vozes durante
   algum tempo, e ambas têm de ser rotuladas.

---

## 10. O que este relatório não fez

- Não modifiquei o sistema principal. `git status` na base auditada: limpo, commit `772d5d6`.
- Não instalei nada, não criei `nucleo/`, não escrevi tokenizer nem Transformer.
- Não apaguei nem movi os heurísticos: a §7 manda preservá-los como auxiliares e o
  benchmark precisa deles como baseline.
- Não afirmo que o sistema actual é "só heurístico". É heurístico no **texto** e tem um
  núcleo diferencial real **desligado** — a distinção muda o plano e está no §0.

**Aguardando aprovação para a Fase 1.**
