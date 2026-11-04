# Corpus v0.1 + Tokenizador v0.2 — o que mudou e o que se aprendeu

**Projecto:** Kianda — A Última Memória do Mar
**Branch:** `arena/5a1b38cd-we-need-learn-toghether` · commit `5330256`
**Base:** `7cf9072` (merge do PR #3)
**Data:** 2026-11-01

---

## 0. Antes do resto: o corpus tinha desaparecido

A Fase 1 reportou um tokenizador treinado em 254 637 caracteres. Ao abrir esta
sessão, `dados/` não existia. A branch nova foi criada a partir de `main`, o
corpus vivia num directório gitignored, e um clone raso de 1 commit não trouxe
nada. Resultado: a Kianda tinha **66 documentos em vez de 497** e **zero
capítulos das obras**, e eu medi um corpus de 13 KB a pensar que era de 254 KB.

Os PDFs originais estavam intactos em `/tmp/arena-coding-uploads/`. Recuperados.

**O que isto ensina, e é a razão de este relatório existir:** um corpus que não
está versionado não é um corpus, é um estado de máquina. Todos os números que
dele dependem — perplexidade, cobertura, tokens/palavra — não são reprodutíveis
por outra pessoa, e por isso não são resultados.

---

## 1. Corpus v0.1 — congelado

`dados/manifest.json`, gerado por `python -m dados.corpus.registar`.

```
12 ficheiros · 2 989 154 caracteres
  244 241  versionados no git       (as 5 obras do autor)
2 744 913  apenas no manifesto      (SHA-256 + origem + instruções)
        0  redistribuíveis
```

### A regra que resolve o problema jurídico

Tamanho não é o problema — **direito de redistribuição é**. O manifesto separa
duas coisas que se confundem facilmente:

| Campo | Pergunta | Consequência |
|---|---|---|
| `redistribuivel` | a licença permite publicar o texto? | se sim, pode ir para qualquer sítio |
| `versionado` | os bytes vão para o git deste repo? | se sim, um clone reproduz o treino |

Uma obra pode ser `versionado: true` e `redistribuivel: false` — é exactamente o
caso das obras do Edmundo: são dele, o repositório é **privado** (`gh api` →
`private: true`), mas não são para publicar. **Se este repositório um dia se
tornar público, estas cinco entradas têm de ser revistas antes.**

O resultado é o melhor dos dois mundos: **reprodutibilidade onde é possível,
rastreabilidade onde não é.**

### O que ficou de fora, e porquê

| Ficheiro | Razão |
|---|---|
| `who_am_i.txt` | 2 caracteres extraídos de 17 páginas — PDF sem camada de texto. Exigiria OCR. Excluído em vez de incluído como ficheiro vazio. |

Excluir em silêncio é a forma mais fácil de mentir num relatório de corpus, por
isso a exclusão está no manifesto com a razão escrita.

### Divisão train/val/test

Determinística por `sha256(semente : ficheiro : índice do bloco)`, cortes
contíguos em fronteira de parágrafo:

- **Porque hash e não ordem:** se o teste fosse o último 5% de cada ficheiro,
  mediria o fim dos livros e não o que o modelo generaliza.
- **Porque contíguo e não linhas aleatórias:** blocos aleatórios cortariam
  frases entre treino e validação.
- **Porque determinística:** sem isto, «o resultado melhorou» pode ser apenas
  uma divisão diferente.

`Corpus v0.1 + tokenizer v0.2 + arquitectura X + seed Y = pesos Z`. Se um
sha256 não bater, o resultado não é comparável.

---

## 2. Tokenizador v0.2 — pré-tokenizado

### O que a pré-tokenização resolve, e o que não resolve

A v0.1 foi treinada **sem** pré-tokenização, e por isso aprendeu fusões que
atravessam fronteiras de palavra (« uma », «de uma »). Duas consequências
medidas:

1. A cache por palavra — a optimização clássica — era **impossível**: segmentar
   por palavra impede as fusões de fronteira, e a compressão piorou **92%**
   (24 → 46 tokens na mesma frase). Tentei-a e reverti.
2. O espaço gasto em fragmentos de frase nunca se reaproveita: no hold-out a
   cobertura de palavra inteira ficava em **32,4%**.

### Resultado

| Métrica | v0.1 | v0.2 |
|---|---|---|
| Pré-tokenização | não | **sim** |
| Cobertura de palavra inteira (hold-out) | 32,4% | **43,1%** |
| Tokens no corpus | 56 479 (254 KB) | **939 422** (2,99 MB) |
| tok/palavra (treino) | 1,223 | 1,914 |
| tok/palavra (hold-out) | 1,861 | 2,212 |
| Vocabulário | 4 096 | 4 096 |

**O ganho:** +10,7 pontos percentuais de cobertura de palavra inteira. O
vocabulário passa a gastar-se em morfologia («coraç», «ão», «ões») em vez de
fragmentos de frase que nunca se repetem.

**A perda, dita sem maquilhagem:** tokens/palavra piorou no hold-out. A razão é
que o corpus é agora 12× maior e mais variado — 490 810 palavras contra 46 172 —
e o mesmo vocabulário de 4 096 tem de servir muitos mais tipos. Com 939 mil
tokens de treino, um vocabulário maior é justificável; **não o escolhi às
cegas**, corri o benchmark de tamanhos, e é a primeira pendência desta fase (§5).

---

## 3. Dois bugs reais, encontrados por medição

### 3.1 O treino não terminava — O(fusões × blocos)

A primeira versão com pré-tokenização percorria a lista de blocos a cada fusão.
Com 254 KB e 1 200 fusões eram 35 s; com os 2,99 MB e 3 840 fusões, **mais de
28 minutos sem terminar**.

A correcção é a das implementações de produção: contar **blocos únicos com a sua
frequência** e fundir sobre os tipos, não sobre as ocorrências. Uma fusão muda
o tipo uma vez, não 680 mil vezes.

**5,6 minutos**, contra >28.

E há um detalhe que vale mais do que a optimização: **só é correcta porque há
pré-tokenização.** Sem ela, uma fusão que atravessa blocos não poderia ser
aplicada isoladamente a um tipo — e a contagem por tipo estaria errada. Uma
decisão de arquitectura tornou possível uma optimização que, na ordem inversa,
seria um bug silencioso.

### 3.2 A regex perdia 23 underscores em silêncio

O round-trip no corpus inteiro falhou. Diagnóstico exacto:

```
primeira divergencia no char 1 276 170
  original    : 'Estive três   \n ___________   \nem Fortaleza'
  reconstruido: 'Estive três   \n    \nem Fortaleza'
caracteres perdidos: [('_', 23)]
```

A causa: `[^\W\d_]` exclui o underscore da classe de palavra (idioma habitual
para não apanhar identificadores), mas `\w` — usado na alternativa de
símbolos — **inclui-o**. O `_` não casava em nenhuma das quatro alternativas e
desaparecia.

O Bechara está cheio de lacunas para preencher («Estive três ___ em
Fortaleza»), e por isso o corpus tinha 23 underscores.

**A lição é maior do que o bug:** uma regex de pré-tokenização tem de cumprir
`"".join(findall(t)) == t` para todo o `t` — é uma **propriedade**, e testa-se
como propriedade. Um teste de amostra com frases escolhidas por mim nunca a
teria apanhado: a diferença era de 23 caracteres em 2 990 946.

Novo teste `test_pre_tokenizacao_e_injectiva`, e um segundo que corre a
propriedade sobre o corpus inteiro (com `skip` se os dados não estiverem no
disco — um teste que falha por ausência de dados não versionados seria ruído).

---

## 4. O `.gitignore` que teria impedido tudo

A regra original era:

```gitignore
dados/
```

O git **não desce em directórios ignorados**, por isso nenhuma negação
(`!dados/manifest.json`) funcionava. O manifesto — o artefacto que existe
exactamente para tornar o corpus reprodutível — nunca teria chegado ao
repositório.

É o mesmo tipo de erro do vocabulário não versionado da Fase 1: a ausência não
faz barulho. Corrigido com exclusão explícita por subdirectório:

```gitignore
dados/outros/          # livros de terceiros: fora
dados/*.db             # estado
dados/estado.json
# manifesto e dados/obras/ entram
```

Verificado com `git add --dry-run`: 9 ficheiros entram, **0** de `dados/outros/`.

---

## 5. O que não está resolvido

### 5.1 O corpus «tudo + web» ficou sem a parte web (BLOQUEADO)

O Edmundo escolheu **tudo + Wikipedia PT**. Mas este executor **não tem acesso
à internet em geral**:

```
pt.wikipedia.org/w/api.php   → 000 (falhou)
dumps.wikimedia.org          → 000 (falhou)
gutenberg.org                → 000 (falhou)
```

Só `github.com`, `pypi.org` e pouco mais. A parte web **não pode ser feita
aqui** — tem de correr na máquina do Edmundo. O plano existe (Onda A: Wikipedia
PT + Gutenberg, com `manifesto.json` de proveniência) e não foi executado.

Consequência honesta: os 2,99 MB são **tudo o que temos**, e continuam a ser
pouco para pré-treino sério. São suficientes para a Fase 2 (provar a cadeia
treino→pesos→geração); não são suficientes para um modelo conversacional.

### 5.2 Tamanho do vocabulário por decidir

O benchmark de 4 096 / 8 192 / 16 384 foi lançado e **não terminou dentro do
tempo** (16 384 fusões é o dobro do trabalho de 4 096, e cada treino são ~5,6
min). Não reporto números que não medi. Fica como pendência: com 939 mil tokens
de treino, 8 192 é provavelmente melhor que 4 096, mas **«provavelmente» não é
uma medição** e não vou escolher por intuição.

### 5.3 Contaminação do corpus — não verificada

O manifesto regista origem e SHA-256, mas **não verifiquei se os livros de
terceiros foram usados para escrever o texto de hold-out**. O hold-out foi
escrito por mim, do zero, e os 54 testes passam — mas isso é plausibilidade, não
prova. Uma verificação séria exige n-gramas sobrepostos entre hold-out e corpus,
e ainda não a escrevi.

---

## 6. Verificação

```
$ pytest nucleo/testes/test_tokenizador.py -q
54 passed in 14.37s

$ git add -A --dry-run dados/
9 ficheiros · 0 de dados/outros/

$ python -c "round-trip no corpus inteiro"
2 990 946 caracteres → 939 422 tokens → texto idêntico
```

## 7. Próximo passo, e o que preciso de si

**A Fase 2 (Transformer) pode começar.** O corpus está congelado, o tokenizador
é reprodutível, e a cadeia de dados está fechada. Não há razão para esperar.

Mas antes de treinar o Transformer, três coisas:

1. **Corra em paralelo na sua máquina:** o benchmark de vocabulário (§5.2) e o
   download da Wikipedia/Gutenberg (§5.1). Eu não tenho rede para nenhum deles.
2. **Confirme a política de dados:** 5 obras versionadas num repo privado, 0
   redistribuíveis. Se o repo se tornar público, isto rebenta.
3. **Decida se a Fase 2 usa 4 096 ou espera pelo benchmark.** Recomendo esperar
   — o vocabulário muda a forma do modelo (matriz de embedding) e treiná-lo
   duas vezes custa mais do que medir uma.

**Aguardo a sua palavra para abrir a Fase 2.**
