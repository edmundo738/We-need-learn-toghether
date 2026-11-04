# Mapa do repositório Kianda — auditoria de estrutura

**Base:** branch `arena/5a1b38cd-we-need-learn-toghether`, commit `249b33e`
**Data:** 2026-11-01
**Natureza:** levantamento pedido pelo Edmundo antes da Fase 2
**Método:** import grafo, contagem de chamadas, `git ls-files`, `grep`. Nada por memória.

---

## 1. Árvore real

```
We-need-learn-toghether/
├── README.md
├── pyproject.toml
├── cerebro-mosca-1.0.0.zip            ← 5,9 MB · 0 referências no código
│
├── dados/
│   ├── manifest.json                  ← corpus v0.1 congelado
│   ├── corpus/registar.py
│   ├── obras/                         ← 5 .txt · 244 241 chars · VERSIONADO
│   └── outros/                        ← 7 .txt · 2 744 913 chars · NÃO versionado
│
├── kianda/                            ← SISTEMA ANTIGO (8 059 linhas)
│   ├── mente.py                       ← 1 393 · orquestrador + composição de fala
│   ├── nucleo/                        ← 972 · autograd + nn (DESLIGADO)
│   │   ├── autograd.py                (540)
│   │   └── nn.py                      (432)
│   ├── cognicao/                      ← 1 569
│   │   ├── percepcao.py               (658) NB · classificador de intenção
│   │   ├── raciocinio.py              (505) escolha de evidência
│   │   └── turno.py                   (406) representação de turno
│   ├── identidade/                    ← 1 045
│   │   ├── canon.py                   (703) factos + fichas
│   │   ├── estado.py                  (342) 8 variáveis internas
│   │   └── voz_da_personagem.md
│   ├── memoria/                       ← 1 034
│   │   ├── recuperador.py             (621) BM25 + stemming
│   │   └── armazenamento.py           (413) SQLite + consolidação
│   ├── linguagem/                     ← 961
│   │   ├── corpus.py                  (590) 497 docs indexados
│   │   └── voz.py                     (371) gerador de chars + Voz
│   ├── benchmark/banco.py             ← 585
│   └── interface/servidor.py          ← 255 · web/index.html
│
├── nucleo/                            ← NÚCLEO NOVO (1 020 linhas)
│   ├── tokenizador/bpe.py             (344) BPE byte-level v0.2
│   ├── tokenizador/vocabulario/       artefacto versionado
│   ├── dados/validacao_tokenizador.txt
│   ├── testes/test_tokenizador.py     (262) 54 testes
│   └── experimentos/                  (448) validação + benchmark
│
├── testes/                            ← 2 846 linhas, 10 ficheiros
├── experimentos/                      ← 3 scripts · 759 linhas · órfãos
└── research/                          ← 6 relatórios
```

**Total de código:** 12 947 linhas em 34 ficheiros `.py` (fora testes: 10 101).

---

## 2. O que existe e funciona — por evidência de uso

Classificação por **quem chama o quê**, não por qualidade percebida.

### Funciona e é chamado

| Componente | Linhas | Chamado por | Papel real |
|---|---|---|---|
| `mente.py` | 1 393 | `__main__`, 9 testes | **orquestra tudo** |
| `memoria/recuperador.py` | 621 | 11 ficheiros | BM25 — recuperação |
| `linguagem/corpus.py` | 590 | 20 ficheiros | 497 documentos indexados |
| `identidade/canon.py` | 703 | 11 ficheiros | factos verificados |
| `cognicao/percepcao.py` | 658 | 8 ficheiros | classificador NB |
| `identidade/estado.py` | 342 | 17 ficheiros | variáveis internas |
| `memoria/armazenamento.py` | 413 | 6 ficheiros | persistência |
| `nucleo/tokenizador/bpe.py` | 344 | 54 testes | BPE v0.2 |

### Existe, funciona em isolamento, mas **está fora do caminho**

| Componente | Linhas | Situação medida |
|---|---|---|
| `kianda/nucleo/autograd.py` + `nn.py` | **972** | Testado (`test_autograd.py`, 227 linhas). Usado só por `voz.py` e `benchmark/banco.py`. **Não toca em nenhuma resposta.** |
| `linguagem/voz.py` → `Voz.falar` | 371 | **`grep -rn "\.falar("` devolve vazio.** A classe existe, sabe gerar, e ninguém a chama. |
| `nucleo/` (novo) | 1 020 | **0 importadores fora de si próprio.** É o estado correcto: área experimental isolada. |

### Código órfão confirmado

| Item | Tamanho | Evidência |
|---|---|---|
| `cerebro-mosca-1.0.0.zip` | 5,9 MB | 0 referências em código, test ou config |
| `experimentos/*.py` | 759 linhas | Nenhum import. Só o `servidor.py` lê `experimentos/resultados/benchmark.json` |

---

## 3. O que está duplicado

### 3.1 Limpeza de texto: **4 implementações**

| Ficheiro | Função | O que faz |
|---|---|---|
| `canon.py:495` | `_sem_acento` | NFKD + remove diacríticos |
| `corpus.py:80` | `limpar_artefactos` | remove cabeçalhos de página PDF |
| `voz.py:235` | `_limpar` | colapsa espaços e linhas |
| `mente.py:1277` | `_limpar_fala` | normaliza pontuação da fala |

**Veredicto:** parcialmente justificável — cada uma serve um estádio diferente
(ingestão, geração, saída). **Mas não têm contrato comum nem testes próprios**, e
foi exactamente nesta zona que apareceu o bug dos 23 underscores. Uma delas
(`_sem_acento`) muda o texto semanticamente; as outras não. Essa distinção não
está escrita em lado nenhum.

### 3.2 Dois pacotes com o mesmo nome

`kianda/nucleo/` (autograd) e `nucleo/` (tokenizador) **não são duplicados** —
fazem coisas diferentes. Mas o nome é o mesmo e a confusão é garantida. O novo
`nucleo/` também **não é totalmente independente**: três dos seus ficheiros de
experimento importam `kianda.nucleo.autograd` e `kianda.linguagem.corpus`.

Para a prova estrutural que a directiva exige («este texto veio do modelo, não
dos documentos»), isto tem de ser resolvido antes da Fase 2.

### 3.3 Duas fontes de conhecimento por personagem

`canon.py` tem `PERSONAGENS` (factos) **e** `_FICHA_PARTES` (fichas de
auto-descrição) **e** `ficha_personagem()` **e** `ficha_auto()`. Quatro caminhos
para responder «quem é X». Já produziu um bug conhecido: o fallback silencioso
que devolvia «Chamo-me Kianda» para qualquer pergunta (*registado em
`_zuleika_gosta_de_ler`*).

### 3.4 `mente.py`: o verdadeiro foco de duplicação

1 393 linhas fazem, ao mesmo tempo:

- orquestração do ciclo (perceber → sentir → raciocinar → compor)
- composição de fala por regras (**~20 métodos `_compor_*`, `_resposta_*`, `_acolher_*`**)
- personalidade (frases escolhidas por `rng`)
- estado e persistência

**Nenhuma destas quatro responsabilidades pode ser substituída sem mexer nas
outras três.** É o acoplamento que impede a Fase 2 de integrar um gerador.

---

## 4. O que pertence ao modelo linguístico

Segundo a separação que a directiva exige (modelo / memória / conhecimento /
personalidade):

| Componente | Pertence a | Estado |
|---|---|---|
| `nucleo/tokenizador/bpe.py` | **modelo** | pronto (v0.2) |
| `kianda/nucleo/autograd.py`, `nn.py` | **modelo** (infra) | funciona, desligado |
| `linguagem/voz.py` → `GeradorNeuronal` | **modelo** (protótipo) | treina, ppl 8,12 validação |
| `linguagem/voz.py` → `Voz._estruturar` | **personalidade** | misturado no mesmo ficheiro |
| `memoria/recuperador.py` | **conhecimento** | funciona |
| `memoria/armazenamento.py` | **memória** | funciona |
| `linguagem/corpus.py` | **conhecimento** (dados) | duplica `dados/manifest.json` |
| `identidade/canon.py` | **conhecimento** (factos) | funciona, 4 caminhos |
| `identidade/estado.py` | **personalidade** (telemetria) | funciona |
| `cognicao/percepcao.py` | **cognição** | é o que a Fase 1.5-a deve substituir |
| `cognicao/turno.py` | **cognição** | funciona, 8 chamadas |
| `cognicao/raciocinio.py` | **cognição** | 1 teste apenas |

---

## 5. O que deve ser reconstruído ou integrado

Proposta em quatro movimentos, ordenados por dependência. **Nenhum começa sem a
sua aprovação.**

### Movimento 1 — Fechar a Fase 1 (antes da Fase 2)

1. Correr o benchmark de vocabulário **4 096 / 8 192 / 16 384** (não terminou
   aqui; §5.2 do relatório anterior). Congelar v0.2 só depois.
2. Verificar **contaminação** hold-out ↔ corpus por n-gramas. É a única
   afirmação do relatório anterior que não está provada.
3. Resolver a independência do `nucleo/`: mover `kianda/nucleo/{autograd,nn}.py`
   para `nucleo/autograd/` e deixar `kianda/` importar de lá. **É a mudança mais
   pequena com o maior efeito** — torna a prova estrutural verdadeira e elimina
   a ambiguidade dos dois «núcleos».

### Movimento 2 — Construir o Transformer (Fase 2)

`nucleo/modelo/`: atenção causal, multi-head, MLP, LayerNorm, residuais,
projecção de saída. Critério de aceitação já escrito e não negociável:
**alterar o token `t+1` não pode mudar o logit em `t`**.

### Movimento 3 — Quebrar `mente.py`

Só depois de o Transformer gerar. Extrair `_compor_resposta` para
`kianda/composicao/`, deixando `mente.py` como orquestrador puro. **Não fazer
antes** — sem um gerador para comparar, o refactor não tem bitola.

### Movimento 4 — Limpeza

`cerebro-mosca-1.0.0.zip` (5,9 MB, 0 referências) sai do repositório. Os
`experimentos/` são promovidos a `research/` ou removidos.

---

## 6. Riscos estruturais

| Risco | Evidência | Gravidade |
|---|---|---|
| `mente.py` acopla 4 responsabilidades | 1 393 linhas, 20 métodos de composição | **ALTO** — bloqueia a Fase 2 |
| `kianda/nucleo` desligado do caminho | `Voz.falar` nunca chamada | **ALTO** — a capacidade existe e não é usada |
| Dois pacotes «nucleo» | nomes colidem, 3 imports cruzados | MÉDIO — quebra a prova estrutural |
| 4 limpezas de texto sem contrato | 4 implementações | MÉDIO — origem do bug dos underscores |
| Órfãos (zip 5,9 MB, experimentos) | 0 referências | BAIXO — mas polui o mapa |
| `raciocinio.py` com 1 teste | 505 linhas, 1 ficheiro de teste | MÉDIO — zona não medida |

---

## 7. O que este mapa **não** diz

- Não medi a qualidade de nenhum componente que não tenha teste. «Funciona» em
  §2 significa **é chamado**, não «faz bem».
- Não avaliei `benchmark/banco.py` (585 linhas) em detalhe.
- Não sei se `interface/servidor.py` está em uso activo; tem 2 testes.
- Não verifiquei o conteúdo do zip; só que nada o referencia.
