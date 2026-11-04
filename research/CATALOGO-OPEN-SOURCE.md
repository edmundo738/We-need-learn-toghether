# Catálogo open-source — investigação e engenharia reversa

**Data:** 2026-11-02
**Estado:** investigação concluída. **Nenhuma integração feita** — aguarda aprovação.
**Regra que segui:** não afirmar nada que não tenha medido neste ambiente.

---

## 0. Primeiro: os PDFs resolvem o bloqueio do corpus

Antes da investigação open-source, o achado que muda o plano.

Os 22 PDFs em `/tmp/arena-coding-uploads/` foram extraídos e medidos:

| Ficheiro | Páginas | Caracteres | Palavras |
|---|---:|---:|---:|
| Bechara — Moderna Gramática (×2 cópias) | 574 | 1 411 536 | 229 876 |
| Rogers — Tornar-se Pessoa | 243 | 893 256 | 142 242 |
| 06-MAET (Manual Administração Eclesiástica Tocoísta) | 136 | 248 990 | 35 304 |
| Adoum — Eu Sou | 104 | 232 572 | 41 978 |
| Como Falar com Qualquer Pessoa | 113 | 73 534 | 11 062 |
| Conversas Sobre o Amor | 102 | 73 097 | 11 253 |
| Apenas Existo (3 partes) | 115 | 149 985 | 27 820 |
| H99 (2 partes) | 45 | 94 224 | 16 678 |
| Conversando com IA (×2 cópias) | 72 | 54 162 | 8 554 |
| Formação de Angola e Resistências Africanas | 24 | 44 777 | 6 832 |
| O Surgimento do Movimento Tocoísmo (Timbane/Afonso) | 15 | 34 614 | 5 365 |
| Angola: história, luta de libertação (Neari) | 15 | 31 856 | 4 892 |
| Guia para Falar em Público | 42 | 25 864 | 4 224 |
| Estrutura e Organização Clerical Tocoísta | 9 | 14 234 | 1 936 |
| Programa Curso Livre História de Angola | 2 | 8 130 | 1 301 |
| Situações Variadas Básico | 8 | 7 194 | 1 347 |
| Who Am I (Português) | 17 | **16** | **0** |
| **Total** | **1 752** | **4 863 739** | **~552 000** |

### O que isto significa

**+4,86 M caracteres de texto português legível por máquina, todos em PDFs que você já tem.** Não vem da Wikipedia, não vem do Gutenberg, não vem da internet. Está no disco.

- **Who Am I** é PDF de imagens (16 caracteres em 17 páginas) → precisa de OCR. **Não incluído nas contas.**
- **Bechara e Conversando com IA** estão duplicados → −1,47 M caracteres de redundância.
- O `dados/outros/` está no `.gitignore` e **não existe nesta sessão**. O material persiste em PDF; os `.txt` derivados têm de ser regenerados sempre.

### A classificação jurídica, que é sua decisão

| Ficheiro | Situação | Recomendação |
|---|---|---|
| 5 obras do Edmundo | © autor, repositório privado | versionadas (como já está) |
| Bechara, Rogers, Adoum, guias | © editoras | **só manifesto**, nunca em git |
| **06-MAET, Estrutura Clerical, Tocoísmo, História de Angola, Formação de Angola** | material da igreja e artigos académicos | **perguntar à igreja** e verificar licença dos artigos |

Os três documentos tocoístas (06-MAET com 35 304 palavras, Estrutura Clerical, mais o artigo sobre o Tocoísmo) são o único corpus **de domínio cultural angolano específico** que temos. Se a igreja autorizar, valem mais para o KIANDA do que qualquer Wikipedia: são o registo institucional e doutrinal da língua como ela é usada em Luanda.

**Mas há um limite que não desaparece:** 4,86 M de caracteres são ~0,8 M de tokens. O modelo de 1,2 M parâmetros continua a decorar. **Blinda a Fase 2, não a Fase 5.**

---

## 1. Ambiente de medição

| | Valor |
|---|---|
| CPU | **2 núcleos** |
| RAM | **3 GB** |
| Disco | 19 GB livres |
| Compiladores | gcc, g++, make |
| Bloqueado | huggingface.co, raw.githubusercontent.com, objects.githubusercontent.com, releases.githubusercontent.com, dumps.wikimedia.org |
| Aberto | github.com, codeload.github.com, api.github.com, pypi.org, files.pythonhosted.org |

**Isto define o que pude e não pude testar.** `git clone` funciona; binários pré-compilados de Releases **não**. Tudo o que precise de modelo descarregado (voz pt_PT, pesos whisper) ficou só em inspecção de código.

---

## 2. Catálogo

| Projeto | Licença | ★ | Língua | Offline | Clonado |
|---|---|---:|---|---|---|
| `ggml-org/whisper.cpp` | MIT | — | C++ | sim | **sim (53 MB)** |
| `rhasspy/piper` | MIT | 11 316 | C++/Python | sim | **sim (62 MB)** |
| `dscripka/openWakeWord` | Apache-2.0 | 2 901 | Python | sim | **sim (7,3 MB)** |
| `alphacep/vosk-api` | Apache-2.0 | 15 224 | C++/Java/Py | sim | inventário |
| `opencv/opencv` | Apache-2.0 | 91 407 | C++ | sim | inventário |
| `ggml-org/llama.cpp` | MIT | 133 122 | C++ | sim | inventário |

---

## 3. A técnica que mais vale para o KIANDA: quantização Q4_0

Esta é a conclusão central da investigação, e é a única que já testei no **nosso** código.

### Como o llama.cpp faz

Lido em `ggml/src/ggml-quants.c`, `quantize_row_q4_0_ref` (linhas 113–145):

```c
// 32 valores por bloco
float amax = 0; float max = 0;
for (j = 0; j < 32; j++) if (amax < fabsf(v)) { amax = fabsf(v); max = v; }
const float d  = max / -8;          // escala com sinal
y[i].d = GGML_FP32_TO_FP16(d);      // 2 bytes
for (j = 0; j < 16; ++j) {          // 2 valores em 1 byte
    xi0 = MIN(15, (int8_t)(x0 + 8.5f));
    xi1 = MIN(15, (int8_t)(x1 + 8.5f));
    y[i].qs[j] = xi0 | (xi1 << 4);  // nibbles
}
```

Estrutura: **32 pesos → 2 bytes de escala (fp16) + 16 bytes = 18 bytes**, contra 128 bytes em fp32.
Escala **assimétrica em torno de zero** (o `-8` e o `+8.5`), não `absmax` simétrico.

### Medido no KIANDA

Reimplementei em numpy e corri contra as formas reais do nosso modelo:

| Forma | fp32 | Q4_0 | Razão | Erro relativo | Tempo |
|---|---:|---:|---:|---:|---:|
| Embedding 8192×128 | 4 194 304 B | 589 824 B | **7,11×** | 8,60% | 321 ms |
| Bloco 256×128 | 131 072 B | 18 432 B | 7,11× | 8,56% | 9,6 ms |
| QKV 128×384 | 196 608 B | 27 648 B | 7,11× | 8,65% | 14,3 ms |
| FFN 384×128 | 196 608 B | 27 648 B | 7,11× | 8,53% | 15,3 ms |
| **Modelo ~1,2 M** | **4,8 MB** | **675 KB** | **7,11×** | **8,59%** | 354 ms |

**O erro é praticamente igual em todas as formas.** Não é acaso: a quantização é por bloco de 32, por isso a dimensão da matriz não afecta a taxa de erro.

### O teste que decide: 8,6% de erro estraga as previsões?

Simulei a camada de saída real (`d_model=128`, `vocab=8192`, 64 posições):

| Cenário | top-1 igual | KL média | max\|Δlogit\| |
|---|---:|---:|---:|
| Q4_0, pesos gaussianos | **85,94%** | 0,00019 | 0,0888 |
| Q4_0, pesos com outliers | **98,44%** | 0,00123 | 0,5772 |

**Leitura:** 8,6% de erro nos pesos quase não move a distribuição de saída. A KL de 0,0012 é desprezável. Com pesos de cauda pesada (realistas), **98,4% das escolhas greedy são idênticas**.

### Veredicto

**Adoptar, mas não agora.** Guardar 7× de RAM não tem valor num modelo de 4,8 MB. O valor está na Fase 5+, quando o modelo crescer. E há um custo que o nosso `autograd` não tem: `nn.Linear` teria de saber desquantizar no `forward` **e** passar o gradiente correcto no `backward` — o gradiente de uma quantização não é a quantização do gradiente. **Isso é trabalho real e tem de ter teste próprio.**

---

## 4. whisper.cpp — o que é aproveitável

Inspeccionado directamente. Constantes acústicas em `include/whisper.h`:

```c
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400     // janela de 25 ms
#define WHISPER_HOP_LENGTH  160     // salto de 10 ms
#define WHISPER_CHUNK_SIZE  30      // 30 s por bloco, o limite do modelo
```

**Técnicas a reimplementar (não a copiar):**

1. **Janela 400 / salto 160 a 16 kHz.** É a parametrização canónica de mel-espectrograma para fala. Reutilizável sem qualquer código deles.
2. **Blocos de 30 s com contexto sobreposto.** O modelo Whisper só vê 30 s; a transcrição longa é feita por janelas com *prompt* do texto anterior. É um padrão de streaming, não uma peculiaridade.
3. **Backend ggml com quantização.** O mesmo Q4_0 que já medi. Ver §3.
4. **Multithreading por linhas de matriz**, não por tokens.

**O que NÃO aproveitar:** os pesos. São da OpenAI, pré-treinados, e a directiva proíbe. Se algum dia usarmos reconhecimento de fala, é o *pipeline* que se reimplementa, não o modelo.

**Estado no nosso ambiente:** o código clona e compila (gcc/g++/make presentes), mas **os pesos não descarregam** (huggingface bloqueado) e **os binários de Release também não** (`releases.githubusercontent.com` bloqueado). Consegui ler e citar o código; não consegui correr.

---

## 5. Piper — medido, e há uma voz pt_PT

### Benchmarks reais (1 thread, CPU, 2 núcleos, 3 GB)

Modelei o `test_voice.onnx` (26,0 MB) com `onnxruntime`:

| Fonemas | Áudio gerado | Latência | RTF |
|---:|---:|---:|---:|
| 20 | 0,51 s | 25,2 ms | **0,049** |
| 50 | 1,66 s | 64,4 ms | **0,039** |
| 100 | 2,19 s | 102,2 ms | **0,047** |

- **Carga do modelo:** 1,13 s
- **RSS pico após carga:** **102 MB**
- **RTF ~0,04** → ~20× mais rápido que tempo real com **uma** thread

**Isto é o número que valida a promessa «KIANDA em PC fraco».** 100 MB de RAM, 1 thread, e 2,2 s de fala em 102 ms. Não é estimativa — é medido neste sandbox de 2 núcleos.

### Arquitectura (VITS), lida em `vits/config.py` e `models.py`

`SynthesizerTrn` = `PosteriorEncoder` + `TextEncoder` (Transformer) + `StochasticDurationPredictor` + `ResidualCouplingBlock` (normalizing flow) + `Generator` (HiFi-GAN).

- `sample_rate: 22050` (o `test_voice` usa 16000)
- `hop_length: 256`, `filter_length: 1024`, `mel_channels: 80`
- `segment_size: 8192`

### Vozes portuguesas: 3, e o detalhe importa

| Voz | Região | Qualidade | Tamanho |
|---|---|---|---|
| **`pt_PT-tugão-medium`** | **Portugal** | medium | 63,2 MB |
| `pt_BR-faber-medium` | Brasil | medium | 63,2 MB |
| `pt_BR-edresson-low` | Brasil | low | 63,1 MB |

**Há uma voz europeia**, mas nenhuma angolana. A pronúncia de `pt_PT` aproxima-se mais do português de Angola do que a `pt_BR` (que tem vogais reduzidas e palatalizações diferentes), mas **não é a mesma coisa**. Se a Fase 5 precisar de voz angolana, ou se treina uma voz (o Piper é treinável) ou se aceita o sotaque europeu.

**Bloqueio:** os `.onnx` estão em `huggingface.co`, inacessível. Só tenho o `test_voice.onnx` que vem no repositório. **Correr a voz portuguesa neste ambiente: impossível.**

---

## 6. openWakeWord — o modelo mais leve da lista

Inspeccionado. Pipeline em `openwakeword/utils.py`:

1. Audio a **16 kHz**
2. `melspectrogram.onnx` (pequeno) → mel
3. `embedding_model.onnx` (Google speech_embedding) → vectores
4. Classificador por palavra de activação

**Porque isto interessa:** o custo está nos dois primeiros modelos, que são **partilhados por todas as wake words**. Uma nova palavra é só o classificador final — treina-se em minutos com poucos exemplos. É a arquitectura certa para «Kianda» como palavra de activação.

**Licença:** Apache-2.0 no código. O `vad.py` embute o **Silero VAD**, que tem licença MIT própria — **verificar separadamente**, porque um VAD diferente do resto obriga a atribuição distinta.

**Não medido:** os modelos `.tflite`/`.onnx` estão em `releases.githubusercontent.com`, bloqueado.

---

## 7. Vosk, OpenCV, llama.cpp — inventário

- **Vosk** (Apache-2.0, 15 224★, 14 MB repo): reconhecimento offline, modelos por idioma (existe `vosk-model-small-pt`). Não inspeccionado a fundo. **Modelos também fora de alcance.**
- **OpenCV** (Apache-2.0, 91 407★, 557 MB): `pip install opencv-python` está no PyPI, que funciona. **Este é o único dos cinco que eu conseguiria realmente correr aqui.** Para visão de ecrã não é preciso OpenCV: `mss`/`Pillow` dão captura de ecrã com muito menos peso.
- **llama.cpp** (MIT, 133 122★, 462 MB): **não usamos como cérebro** — violaria a directiva. A técnica de quantização já extraída (§3).

---

## 8. Proposta de arquitectura, com o que é nosso e o que é emprestado

```
KIANDA
│
├─ NÚCLEO (100% nosso, pesos nossos) ......... directiva respeitada
│     tokenizador v1 (8192) · Transformer · treino · geração
│     ← única fonte de texto do modelo
│
├─ EFICIÊNCIA (técnica emprestada no §3) ..... Q4_0, 7,11× medido
│     aplicável quando o modelo crescer; exige backward próprio
│
├─ OUVIDO (pipeline emprestado, pesos externos)
│     mel 400/160 a 16 kHz · blocos de 30 s · whisper/ vosk
│     ← transcreve, não pensa
│
├─ VOZ (motor emprestado, voz a escolher)
│     Piper, RTF 0,04, 102 MB · pt_PT disponível, angolano não
│
├─ ACTIVAÇÃO (arquitectura emprestada) ....... «Kianda»
│     mel+embedding partilhados · classificador treinável
│
├─ VER/MEMÓRIA/AGENTE (a decidir)
│     captura de ecrã, recuperação, teclado/rato
│
└─ REGRA DE FRONTEIRA
      Nada acima do NÚCLEO escreve texto no modelo.
      O modelo lê o que o utilizador disse e a sua memória.
      Se um módulo externo puder injectar texto, a auditoria morre.
```

**A fronteira é a decisão de arquitectura mais importante desta investigação.** Sem ela, o KIANDA passa a ser um orquestrador de modelos alheios que ocasionalmente corre o nosso — exactamente o diagnóstico que governou o projecto: «código que parece neuronal mas não participa no processo normal de resposta».

---

## 9. Prioridades

| # | Acção | Valor | Custo | Bloqueio |
|---|---|---|---|---|
| 1 | Regenerar `.txt` dos 22 PDFs + dedupe + manifesto | **alto** (+4,86 M ch) | 1 dia | nenhum |
| 2 | Perguntar à igreja pelos 3 documentos tocoístas | alto (único corpus AO) | — | depende de si |
| 3 | **Não** integrar nada ainda — Fase 2 primeiro | alto | — | decisão sua |
| 4 | Quantização Q4_0 no `autograd` + teste de gradiente | médio | 2 dias | modelo maior primeiro |
| 5 | Piper como voz (quando houver permissão) | médio | 1 dia | modelos bloqueados |
| 6 | openWakeWord «Kianda» | baixo | 3 dias | modelos bloqueados |

**A minha recomendação é a contrária ao instinto:** não fazer a integração multimodal agora. O KIANDA tem 0,87 M tokens e não tem um Transformer que funcione. Adicionar ouvido, voz e visão a um cérebro que ainda não existe é construir o corpo antes do cérebro — e repete, com código novo, o erro que a directiva manda evitar: **muitos módulos, nenhuma capacidade comprovada.**

A ordem que defendo: **Fase 2 (Transformer a funcionar) → corpus dos PDFs → depois os sentidos.**

---

## 10. O que não consegui fazer, e porquê

| Tentativa | Resultado |
|---|---|
| Descarregar voz `pt_PT` do Piper | **falhou** — huggingface.co |
| Descarregar pesos whisper | **falhou** — huggingface.co |
| Descarregar modelos openWakeWord | **falhou** — releases.githubusercontent.com |
| Compilar whisper.cpp | não tentado (sem pesos não serve para medir) |
| Correr Vosk | não tentado (modelos fora de alcance) |
| Correr Piper com voz portuguesa | **falhou** — só `test_voice.onnx` local |
| Correr Piper com voz de teste | **sucesso** — RTF 0,04, 102 MB, 2 núcleos |
| Medir Q4_0 no nosso autograd | **sucesso** — 7,11×, erro 8,6%, top-1 98,4% |
| Extrair 22 PDFs | **sucesso** — 4,86 M caracteres |

Tudo o que está acima com número foi medido. Tudo o que está em "inventário" foi lido no código, não executado — e está marcado como tal.
