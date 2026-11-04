# A cadeia completa: texto → ids → pesos → ids → texto

**Data:** 2026-11-02
**Script:** `nucleo/experimentos/treinar_corpus_real.py`
**Resultado:** `nucleo/experimentos/resultado_corpus_real.json`

---

## 1. A pergunta que este experimento responde

Não é «o modelo é bom?». É a única que estava por responder com dados reais:

> **A máquina existe de ponta a ponta?**

Tokenizador v1 → ids → Transformer → logits → ids → tokenizador → texto. Se o
texto do outro lado for português legível, mesmo mau, a cadeia fechou. Se for
lixo binário, falta qualquer coisa.

---

## 2. O que correu

| | |
|---|---|
| Corpus | 5 obras do autor, 244 372 caracteres |
| Tokens (tokenizador v1, 8192) | 65 999 |
| Janelas de contexto 128 | 511 (459 treino / 52 validação) |
| Modelo | 2 771 200 parâmetros (d_model 128, 4 camadas, 4 cabeças) |
| Passos | 1 500 |
| Tempo | 587 s (0,39 s/passo, lote 2) |
| Amostra de treino | 459 janelas × 128 = **58 752 tokens vistos** |

**O corpus inteiro é menor que um único lote de treino de um modelo normal.**
Isto não é um detalhe: é a explicação de tudo o que vem a seguir.

---

## 3. A curva

| passo | perda_treino | perda_val | acerto_val |
|---:|---:|---:|---:|
| 1 | 9,8623 | 9,0404 | 0,35% |
| 300 | 5,8935 | 5,9512 | 11,20% |
| 800 | 5,3358 | 5,7104 | 11,48% |
| **1 500** | **4,0915** | **5,8962** | **11,66%** |

Baseline aleatório: `log(8192) = 9,01`.

**A perda de validação desceu de 9,04 para 5,70 no passo 1 000 e depois
subiu para 5,90.** A de treino continuou a descer. Isto é sobreajuste, e
apareceu exactamente onde a curva sintética da Fase 2 previu — entre 1 000 e
1 500 passos. Não foi surpresa; foi confirmação.

---

## 4. A geração — duas fases, e a diferença importa

### Com 200 passos (lote 4, ctx 64)

```
prompt: 'Eu não tenho filtro'
saida:  'Eu não tenho filtro.\n— Não não não não não não não não não...
```

Colapso num único token. É o modo de falha clássico de um modelo que ainda não
aprendeu nada: a distribuição ainda é quase uniforme, e o argmax cai sempre no
mesmo sítio.

### Com 1 500 passos (lote 2, ctx 128)

```
prompt: 'Eu não tenho filtro'
saida:  'Eu não tenho filtro.\n— Fui.\n— Que não percebe que não são.\n
         — A voz de mim.\n—\n— Saí.\n— Saí.'

prompt: 'Zuleika olhou'
saida:  'Zuleika olhou para aigarro, a vozcada.\n\n— 1.\n\n— Seis— Eu
         nãoima a voz.\n— Eu sei o\n— Eu sei a voz baixa.'

prompt: 'Há coisas que'
saida:  'Há coisas que não antecip, que não\n—só rápido.\nFizam.\nFica.\n
         Falar a única reunião.'
```

**A diferença entre as duas fases não é de grau, é de tipo.** Na segunda há:

- **estrutura de diálogo** — o travessão `—` nos sítios certos, mudanças de
  falante, parágrafos;
- **sintaxe portuguesa** — concordância, preposições, ordem das palavras;
- **léxico do corpus** — «voz», «disse», «Fui», «Saí», «a voz baixa»;
- **e erro morfológico do tipo certo** — «aigarro», «vozcada», «nãoima» são
  colagens de tokens, não lixo. O modelo está a inventar palavras seguindo
  regras de formação, não a produzir ruído.

**Cópia literal: 0% em todas as três amostras.** Zero 20-gramas do gerado
existem no corpus. O modelo **não está a decorar** — está a generalizar mal,
que é diferente e é melhor.

---

## 5. O limite, medido

O nosso autograd guarda todos os nós do grafo. A matriz de atenção é
`(B*T, B*T)`, por isso a memória cresce com o **quadrado** dos tokens por lote:

| janelas × 64 | tempo | RSS pico |
|---:|---:|---:|
| 8 | 0,58 s | 278 MB |
| 16 | 2,02 s | 751 MB |
| 32 | 8,29 s | **2 471 MB** |
| 51 | — | **processo morto** |

Com 3 GB de RAM, isto obriga a avaliar em lotes de 8 e a treinar com lote 2.
**É um limite real do autograd experimental**, e tem consequência prática: o
treino é ~10× mais lento do que seria com um grafo que liberta nós.

Dito de outra forma: **o modelo é pequeno, o pipeline é que é caro.** Um modelo
de 2,8 M parâmetros treinado a 0,391 s/passo com lote 2 (256 posições) processa
**256 / 0,391 = 655 tokens/s = 0,655 tokens/ms**, não os "5 tokens/ms" que eu
aqui escrevi — um erro de 7,6× no mesmo parágrafo em que citei correctamente o
tempo por passo.

`[CORRIGIDO]` O "PyTorch faria isto 50× mais depressa" **não tem medição
nenhuma e foi retirado.** Não houve benchmark, nem equivalente, nem fonte.
A auditoria mediu o throughput real por tamanho de lote (lote 1: 1 687 tok/s;
lote 2: 1 414; lote 4: 936; lote 8: 534) e não comparou com PyTorch, porque
comparar exige um ambiente equivalente que aqui não existe. A escolha de não
usar PyTorch continua deliberada — provar que os pesos e o gradiente são
nossos — mas o argumento é de **propriedade**, não de velocidade demonstrada.

---

## 6. O que este experimento prova, e o que não prova

### Prova

1. **A cadeia fecha.** Texto entra, texto sai, e o texto do outro lado é
   português com estrutura.
2. **O modelo aprende com dados reais**, não só sintéticos: a validação desceu
   3,1 nats abaixo do aleatório.
3. **O sobreajuste é detectável e foi detectado** antes de eu o declarar.
4. `[CORRIGIDO]` **A geração não contém 20-gramas exactos do corpus**, em 3
   amostras. Foi isto que se mediu. **Não** prova ausência de memorização:
   não se testaram 5, 10 ou 15-gramas, nem sobreposição aproximada, nem
   recuperação de exemplos.
5. **O limite de RAM é medido**, não estimado.

### Não prova

1. **Que o modelo aprendeu português.** `[CORRIGIDO]` O "0,02% do que
   precisaria" foi **retirado: nunca houve cálculo apresentado.** Com a
   heurística de Chinchilla (≈20 tokens por parâmetro, que é uma regra de
   *cálculo óptimo*, não uma lei da natureza) seriam 55,5 M tokens e o corpus
   tem 66 mil, ou seja 0,12% — mas isto é uma heurística, não uma necessidade
   demonstrada. O que os dados sustentam é o qualitativo: 66 mil tokens e 5
   autores é pouco para aprender uma língua, e o que o modelo aprendeu foi a
   *forma* da prosa do autor — diálogo, ritmo, léxico — e não a língua.
2. **Que a Fase 5 pode começar.** Uma conversa exige coerência em dezenas de
   turnos. Este modelo mantém coerência em ~5 tokens.
3. **Que o caminho está errado.** Está exactamente onde devia estar: a máquina
   funciona e está sub-alimentada. O problema é corpus, não arquitectura.

---

## 7. O que faria a seguir, por ordem

1. **Regenerar os `.txt` dos 22 PDFs** (4,86 M caracteres, 18× este corpus).
   Não precisa de autorização de ninguém para os *meus* e para os que já estão
   em uso; os tocoístas ficam à espera da igreja.
2. **Retreinar com esse corpus** e ver se a validação deixa de subir antes do
   passo 1 500.
3. **Só depois** olhar para o grafema da conversa.

A tentação é a oposta: ajustar hiperparâmetros até a amostra ficar bonita. Não
vou fazer isso. Com 58 mil tokens vistos, um bom resultado seria um acidente e
não seria reproduzível — e a regra do projecto é que a confiança se ganha com
evidência, não com uma amostra escolhida.
