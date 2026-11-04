# Como resolver a falta de Wikipedia e Gutenberg

**Pergunta:** sem Wikipedia, sem Gutenberg — como resolver isto?
**Data:** 2026-11-01
**Método:** cada via testada com um pedido real, não assumida.

---

## 1. O que está realmente bloqueado

Testei 12 hosts com `curl`:

| Host | Estado | Consequência |
|---|---|---|
| `api.github.com` | **200** | busca de repositórios funciona |
| `github.com` | **200** | — |
| `codeload.github.com` | **200** | **tarballs e `git clone` funcionam** |
| `pypi.org` / `files.pythonhosted.org` | **200** | **instalar pacotes funciona** |
| `raw.githubusercontent.com` | 000 | ficheiros soltos do GitHub: **não** |
| `objects.githubusercontent.com` | 000 | clones completos: não |
| `huggingface.co` | 000 | datasets HF: **não** |
| `dumps.wikimedia.org` | 000 | dump da Wikipedia: **não** |
| `pt.wikipedia.org` | 000 | API da Wikipedia: **não** |
| `gutenberg.org` | 000 | Project Gutenberg: **não** |
| `archive.org` | 000 | Internet Archive: **não** |

**Descoberta importante que corrigi:** `raw.githubusercontent.com` e
`objects.githubusercontent.com` estão bloqueados, mas `codeload.github.com` e
`github.com` **não**. Testei `git clone` de verdade e funcionou. Isto abre uma
via que eu tinha descartado na resposta anterior — **repositórios GitHub são
alcançáveis**, e é por aí que se chega a corpora publicados como repositório.

---

## 2. As vias, testadas uma a uma

### Via A — Pacotes PyPI com corpus embutido (funciona, mas é pequeno)

O PyPI está aberto, por isso qualquer pacote que **distribua texto** serve.
Testei os candidatos:

| Pacote | O que traz | Texto PT | Licença |
|---|---|---|---|
| `chatterbot-corpus` | 12 ficheiros de diálogo PT, 321 conversas | **28 322 chars** | BSD-3 |
| `nltk` | código; os corpora descarregam de `raw.githubusercontent.com` → **bloqueado** | 0 | — |
| `wikipedia` | cliente HTTP; precisa da API → **bloqueada** | 0 | — |
| `wikiextractor` | só o extrator; precisa do dump → bloqueado | 0 | — |

**Resultado:** 28 322 caracteres de **diálogo real em português**. É o registo
conversacional que o corpus actual quase não tem (as obras são literárias, a
gramática é normativa, Rogers é ensaio). Para treino conversacional vale mais do
que os seus 28 KB sugerem, porque é pergunta-resposta.

**Mas é 1% do que precisamos.** Esta via sozinha não resolve.

### Via B — Repositórios GitHub (funciona; o problema é encontrar os bons)

`git clone` e tarballs funcionam. Busquei corpora PT com `api.github.com`:

| Repositório | ★ | Licença | Tamanho | Texto utilizável? |
|---|---|---|---|---|
| `Edresson/TTS-Portuguese-Corpus` | 180 | CC-BY-4.0 | 378 MB | **não** — é áudio; o texto (71 358 palavras) está no HF, bloqueado |
| `fabiocmazzo/corpusportugues` | 26 | MIT | 0 MB | **não** — é código, não dados |
| `eberlitz/pt-br-corpus` | 27 | GPL-3.0 | 0,1 MB | **não** — é scripts que descarregam de outros sítios |
| `erickrf/ptwiki2text` | 14 | sem licença | 0,2 MB | não — extrator |

**O padrão é claro e é importante:** os repositórios de "corpus português" quase
todos contêm **código para descarregar dados**, não os dados. Os dados ficam em
HuggingFace ou em dumps, que estão bloqueados. E os que têm dados usam áudio.

**Onde esta via vale a pena:** repositórios que embutem texto literário em
domínio público. Uma busca por `dominio publico portugues literatura` e afins
devolve pouca coisa com licença clara. Não encontrei nenhum corpus PT grande e
legítimo hospedado como repositório GitHub.

### Via C — Instalar `datasets` e contornar (FALHOU)

`pip install datasets` funciona (o pacote está no PyPI), mas os **dados** vêm de
`huggingface.co`, que devolve `TLS/SSL connection has been closed` em qualquer
pedido. Medi com o `huggingface_hub`: a API falha na ligação, não na
autenticação. **Sem HF não há OSCAR, C4, CulturaX nem nenhum corpus grande PT.**

### Via D — Wikipedia por tabela de hashes (teórico, não testado)

É possível pedir à API do GitHub para descarregar ficheiros por hash de blob
(que usa `api.github.com`, aberto). Se existir um repositório com o dump PT
fragmentado, isto serviria. **Não encontrei tal repositório** e não vou
recomendar um caminho que não verifiquei.

---

## 3. A conclusão honesta

**Sem Wikipedia e sem Gutenberg, e sem HuggingFace, o texto português legítimo
alcançável a partir deste executor é:**

| Fonte | Caracteres | Estado |
|---|---|---|
| As 5 obras do autor | 244 241 | já no corpus |
| Bechara, Rogers, Adoum, conversação | 2 744 913 | já no corpus |
| `chatterbot-corpus` (diálogo PT) | **28 322** | **novo, BSD-3** |
| **Total** | **≈ 3,02 M** | |

Ou seja: **a via PyPI acrescenta 28 KB a 3 MB. É 0,9%.** Não resolve o
problema, melhora-o marginalmente — e o que acrescenta é do tipo certo (diálogo).

**Não há solução neste executor.** O bloqueio é de rede, não de engenharia:
nenhuma quantidade de código contorna uma ligação TLS que o ambiente recusa.

---

## 4. As três saídas reais

### Saída 1 — Você descarrega, eu construo a pipeline (recomendada)

O corpus tem de entrar por onde a rede permite. Você tem acesso normal à
internet. Uma destas serve:

```
# Wikipedia PT (a via da directiva) — dump de artigos
wget https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2
# ~1,2 GB comprimido → ~3,5 GB de texto

# Gutenberg PT (domínio público)
# catálogo: https://www.gutenberg.org/browse/languages/pt
# são ~1 500 obras; um script de download corre em minutos
```

Eu construo e testo o script de ingestão (`manifesto.json`, SHA-256, dedupe,
splits, verificação de contaminação) e você corre-o. **Isto respeita a sua regra
de proveniência e não depende de eu ter rede.**

### Saída 2 — O corpus cresce por acumulação

Há duas fontes de texto que já existem e não estão a ser usadas:

1. **Os 28 KB de diálogo do `chatterbot-corpus`** — incluir já. É pouco, mas é o
   registo certo e a licença é limpa.
2. **Conversas reais autorizadas** (§4.1 da directiva). Cada conversa que você
   autorizar a guardar é um exemplo de diálogo com português angolano, que
   nenhuma Wikipedia tem. **Pequeno por unidade, único por natureza.**

### Saída 3 — Aceitar o âmbito e declarar

Com 0,87 M tokens, o modelo de 1,2 M parâmetros decora o corpus em poucas
épocas. A Fase 2 continua a valer a pena — prova a cadeia treino→pesos→geração,
que é o objectivo declarado — mas **a Fase 5 (conversa) fica bloqueada até
existir corpus**, e isso deve estar escrito no relatório da Fase 2, não
descoberto na Fase 5.

---

## 5. O que eu recomendo

1. **Incluir já o `chatterbot-corpus`** (28 KB de diálogo, BSD-3). Custa uma
   tarde, melhora o registo e é legítimo. Já está instalado neste ambiente.
2. **Preparar o ingest de Wikipedia e Gutenberg** e correr na sua máquina. É a
   única via para os milhões de tokens que a directiva assume.
3. **Construir o Transformer de qualquer maneira.** O objectivo da Fase 2 é
   provar que os pesos são nossos e que a geração é real. Isso não precisa de
   3,5 GB — precisa de um corpus que o modelo possa aprender, e 0,87 M tokens
   chegam para demonstrar a cadeia e medir sobreajuste honestamente.
4. **Não fingir que resolve.** Se a Fase 2 correr com 0,87 M tokens, o relatório
   tem de dizer «este modelo decora o corpus; é uma demonstração de engenharia,
   não de aprendizagem de linguagem» — e a Fase 5 fica BLOQUEADA com essa razão.

---

## 6. Nota sobre o que não fiz

- Não incluí o `chatterbot-corpus` no corpus v0.2 sem a sua palavra: muda o
  `manifest.json`, o que obriga a retreinar o tokenizador. **Decide você se
  vale 28 KB de retrabalho.** A minha opinião é que sim, mas é opinião.
- Não tentei contornar bloqueios de rede (proxies, mirrors, DNS alternativo).
  Seria violar a política de acesso do ambiente e não é engenharia.
- Não contei com o texto do TTS-Corpus (71 k palavras): o README diz que vem de
  Wikipedia e está no HF. Sem HF, inalcançável.
