# Correcção — o que estava partido e o que foi verificado

Data: 2026-11-05. Classe de tudo aqui: **MEASURED**, salvo indicação em contrário.
Este documento existe porque o trabalho anterior foi rejeitado com razão. Não é um
relatório de progresso: é a lista de defeitos, a prova de que existiam, e a prova
de que estão corrigidos.

---

## 1. O painel servia o pior modelo que temos

**Defeito.** `regime_publico.py` gravava o checkpoint no fim da corrida. O fim era o
passo 1500. Medido: val 6,7498 nesse passo, val 5,9340 no passo 500. A curva piora
depois do 500. O painel apontava para `publico_seed0`, ou seja, **o utilizador
estava a ver, e a julgar, o pior ponto da curva.**

Isto explica a percepção de "pega recorta várias mensagens aleatórias": o texto
mostrado era de um modelo em sobreajuste terminal, não do melhor que o projecto
conseguiu.

**Correcção.** `nucleo/experimentos/treinar_checkpoint_bom.py` avalia a cada 50
passos, guarda o estado do melhor, treina até 700. Resultado:

| | valor |
|---|---|
| melhor passo | 550 |
| val (nats/token) | **5,9231** |
| bits/byte | **2,6770** |

Melhor do que qualquer valor anterior do projecto (2,6796). `CHECKPOINT_OMISSAO =
"machado_melhor"` no servidor. Os `publico_seed*` ficam no selector para comparação.

**Teste.** `test_checkpoint_por_omissao_e_o_do_melhor_ponto`.

---

## 2. A geração colapsava e não havia como o evitar

**Defeito.** `Transformer.gerar` não tinha penalização de repetição. Com prompts
curtos produzia 15 travessões seguidos e pares como "que lhe lhe".

**Correcção.** Parâmetro `repeticao` (penalidade de presença, aplicada uma vez por
token distinto na janela). Valor por omissão no painel: 1,2.

**Honestidade sobre o alcance.** A penalização **não torna o modelo bom**. Remove a
degenerescência total; não cria competência que o modelo não tem. Continua a haver
palavras inexistentes ("eraza", "bocrava") e concordâncias quebradas. O valor dela é
permitir ver o que o modelo aprendeu em vez de ver o colapso a esconder tudo.

**Teste.** `test_repeticao_chega_ao_gerador` (o parâmetro chegava ao handler e era
descartado — o teste fixa a ligação).

---

## 3. Cinco constantes copiadas à mão

**Defeito.** `TOKENS_POR_BYTE = 0.313` em `contexto_maior`, `melhor_ponto`,
`modelo_menor`, `regime_publico`, `underfitting`. O valor medido é 0,313277.

**Correcção.** Os cinco importam `metricas.tokens_por_byte()`, que lê o valor medido
de `medida_tokens_por_byte.json`. Verificado por importação: os cinco dão 0,313277.

---

## 4. Números medidos escritos à mão no HTML do painel

**Defeito.** O HTML dizia `2,77 M parâmetros`, `10,9 → 5,4 amostras por parâmetro de
posição` e (no servidor) `"trigrama_bpb": 2.4766`. Se um experimento fosse refeito, o
painel mentia sem ninguém dar por isso.

**Correcção.** Os parâmetros são contados do `pesos.npz` (2 771 200); as amostras vêm
de `resultado_diagnostico_contexto.json`; o trigrama vem de
`resultado_comparacao_final.json`.

**Teste.** `test_html_nao_tem_numeros_medidos_a_mao` — proíbe qualquer decimal no
HTML fora de uma lista branca de valores de interface (mínimos/máximos de sliders).
Este teste falhou duas vezes durante a correcção e apanhou duas ocorrências que eu
tinha deixado.

---

## 5. Um erro meu que quase virou resultado

**O que aconteceu.** Escrevi `comparacao_final.py` para pôr o Transformer e o
trigrama no mesmo texto. Usei suavização de Laplace com salto para uniforme. Deu
**trigrama 3,9094 bpb** e "o Transformer ganha por 1,23 bits/byte" — a primeira vez
que o modelo ganharia ao baseline.

**Porque é que era falso.** `baselines.py` já documentava que a auditoria externa
tinha apanhado exactamente este erro: saltar para uniforme quando o contexto é novo
**não é backoff**, é desistir. Sabotar o baseline faz o modelo parecer melhor.

**Correcção.** Passou a usar `baselines.Interpolado` (desconto absoluto d=0,75,
backoff recursivo) — o mesmo objecto que produz o 2,4766 histórico.

| | nats/token | bits/byte |
|---|---|---|
| trigrama (desconto absoluto) | 5,4845 | **2,4788** |
| Transformer (passo 550) | 5,9231 | 2,6770 |

**O trigrama ganha por 0,1982 bits/byte.** A conclusão original mantém-se: o modelo
não bate um n-grama. Consistente com o 2,4766 anterior (diferença 0,0022, explicada
por o Transformer ser aqui avaliado em janelas de 128 e o trigrama em tokens
contíguos).

**Teste.** `test_baseline_usa_desconto_absoluto_nao_laplace` — num contexto nunca
visto, a probabilidade tem de ser **maior** que 1/V (o backoff redistribui massa);
com salto para uniforme seria exactamente 1/V.

---

## 6. O núcleo matemático, verificado por fora

Todos os testes existentes comparavam o núcleo **consigo próprio**: se houvesse um
erro sistemático de formulação, todos passavam. A auditoria externa já tinha apanhado
dois casos (ReLU em vez de GELU, o pseudo-backoff).

`nucleo/testes/test_verificacao_independente.py` reimplementa em numpy puro, sem usar
`nucleo`, e compara:

- atenção causal escalada (causalidade verificada posição a posição)
- entropia cruzada contra softmax escrito à mão — coincidem a 1e-6
- caso limite com valor conhecido: logits iguais ⇒ perda = log(V)
- LayerNorm contra a fórmula
- GELU contra a **aproximação tanh** (e prova de que não é ReLU)
- máscara causal do modelo completo: mudar o último token não altera as previsões
  anteriores, e altera a última (prova de que o teste não é vazio)

**6 testes, todos passam.** Durante a escrita, dois falharam por erros meus, não do
núcleo: comparei a GELU com a fórmula exacta em vez da aproximação (diferença 2e-4,
esperada por construção), e usei `transpose(0,2,1)` num array de 4 eixos. Ficam
registados para não se repetirem.

---

## 7. Hiperparâmetros: hipótese refutada

Cinco braços, 1200 passos, semente 0.

| braço | bits/byte |
|---|---|
| **lr 3e-3 constante (actual)** | **2,6819** |
| lr 1e-3 constante | 2,8254 |
| lr 1e-3 cosine+warmup | 2,8438 |
| lr 4e-4 constante | 2,9967 |
| lr 4e-4 cosine+warmup | 3,0256 |

**Baixar o lr piora. Cosine também não ajuda.** O lr actual é o melhor dos cinco. Os
hiperparâmetros não são a causa da derrota — deixa de fazer sentido continuar a
mexer neles.

---

## O que continua por resolver

Estas coisas **não** estão corrigidas e não são apresentadas como se estivessem:

1. **O modelo não bate o trigrama.** 2,6770 vs 2,4788. É a limitação central e nenhuma
   correcção acima a resolve.
2. **O texto gerado é mau.** Palavras inventadas, concordância quebrada, fragmentos.
   A penalização de repetição remove o colapso, não a incompetência.
3. **Contexto 256 é pior que 128** (2,7018 vs 2,6796) e o mecanismo está identificado
   (tabela de posições aprendida: 5,45 amostras por parâmetro contra 10,9) mas a
   correcção exige RoPE, que está congelada.
4. **3 de 9 finalidades** do cruzamento têm dados. Cobertura 16,61%.
5. **Um só corpus de autor.** 938 807 caracteres de Machado. "5 obras" ≠ "5 autores".

## Números desta correcção (todos reproduzíveis)

```
python -m nucleo.experimentos.comparacao_final      # ~30 s
python -m nucleo.experimentos.treinar_checkpoint_bom # ~460 s
pytest nucleo/testes/ -o addopts="" -q              # ~12 s, 211 passed
pytest testes/ nucleo/testes/ -o addopts="" -q      # ~660 s
```
