# Contratos da Fase 2 — interfaces antes da implementação

**Natureza:** especificação. Nenhum código de modelo foi escrito ainda.
**Regra que governa este documento:** um contrato serve para poder **testar uma
peça sem as outras**. Se uma interface não permite isso, está mal definida.

---

## Porque escrever contratos antes do código

A alternativa já foi tentada neste projecto: `mente.py` cresceu com 1 393 linhas
onde orquestração, composição e personalidade se chamam mutuamente. O resultado é
que nada pode ser substituído nem medido em isolamento — e por isso 972 linhas de
núcleo neuronal ficaram dois meses fora do caminho sem que ninguém notasse.

A regra para a Fase 2: **cada componente tem uma fronteira testável sem GPU, sem
treino e sem corpus.**

---

## 1. Tokenizador ↔ Modelo

```python
class Tokenizador(Protocol):
    tamanho_vocab: int
    def codificar(self, texto: str) -> list[int]: ...
    def descodificar(self, ids: list[int]) -> str: ...

assert descodificar(codificar(t)) == t     # para todo o t
```

**Contrato:**
- `tamanho_vocab` é fixo depois do treino e define a forma da matriz de embeddings.
- Todas as saídas de `codificar` estão em `[0, tamanho_vocab)`.
- `descodificar` é exacta — não «aproximada». Já há 54 testes a provar isto.

**Testável sem o modelo:** sim. É o que já existe.

---

## 2. Modelo — a interface mínima

```python
class Modelo(Protocol):
    def __call__(self, ids: list[int]) -> list[list[float]]:
        """ids (T,) -> logits (T, vocab). logits[t] é a distribuição para t+1."""

    def parametros(self) -> list[Parametro]: ...
    def num_parametros(self) -> int: ...
```

**Contrato crítico:** `saida[t]` depende **apenas** de `ids[:t+1]`.
É a causalidade, e testa-se sem treinar:

```python
base = modelo([1, 2, 3, 4, 5])
alt  = modelo([1, 2, 3, 4, 9])   # muda só o último
assert base[:4] == pytest.approx(alt[:4])   # os primeiros 4 logits iguais
```

Se esta asserção falhar, há fuga de informação do futuro e **tudo o que se
aprender depois é inválido** — o modelo teria visto a resposta.

---

## 3. Bloco Transformer

```python
class Bloco(Protocol):
    def __call__(self, x: Tensor) -> Tensor:
        """x (lote, seq, d_model) -> (lote, seq, d_model). Mesma forma."""
```

**Contrato:** entrada e saída têm a **mesma forma**. É o que permite empilhar N
blocos sem recalcular dimensões, e o que faz um erro de forma aparecer na
construção e não no treino três horas depois.

---

## 4. Atenção causal

```python
class Atencao(Protocol):
    def __call__(self, x: Tensor, mascara: Tensor | None = None) -> Tensor: ...
    n_cabecas: int
    d_cabeca: int
```

**Contrato:** `n_cabecas * d_cabeca == d_model`. Verificado na construção.

**Teste sem treino:** com a máscara causal, `saida[:, t]` não muda quando se
alteram posições `> t` na entrada. É a mesma propriedade do §2, mas ao nível da
peça — e falha mais perto da causa.

---

## 5. Função de perda

```python
def perda(logits: Tensor, alvos: np.ndarray) -> Tensor
```

**Contrato:** `logits` de forma `(lote*seq, vocab)`, `alvos` de forma `(lote*seq,)`
com valores em `[0, vocab)`. Devolve escalar.

**Teste sem treino:** com logits uniformes e `vocab=V`, a perda é `log(V)`.
É o valor calculável à mão; uma implementação que erre por um factor de 1/T
passa despercebida em treino e envenena todas as comparações.

---

## 6. Optimizador

```python
class Optimizador(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
```

**Contrato:** depois de `step()`, pelo menos um parâmetro mudou de valor.
Depois de `zero_grad()`, todos os gradientes são `None` ou zero.

**Teste sem treino:** um parâmetro conhecido, um passo, verificar que mudou na
direcção oposta ao gradiente. Já existe em `test_autograd.py`.

---

## 7. Geração autoregressiva

```python
def gerar(modelo, ids: list[int], n: int, temperatura: float = 1.0,
          top_k: int | None = None) -> list[int]
```

**Contrato:**
- Devolve **apenas** tokens novos; o prompt não é repetido.
- `temperatura=0` é greedy: determinístico, duas chamadas dão o mesmo resultado.
- `temp > 0` é amostragem: duas chamadas **podem** diferir, e isso testa-se.
- `top_k` restringe os candidatos aos `k` mais prováveis.

**Teste sem treino:** um modelo falso que devolve sempre logits fixos. A geração
tem de produzir a sequência esperada. Isto prova o motor de geração **sem**
depender de um modelo treinado — e é a diferença entre «o motor funciona» e «o
modelo aprendeu».

---

## 8. Separação que a Fase 6 vai precisar

Estes contratos definem o **modelo**. Não definem:

| Camada | Contrato futuro | Depende do modelo treinado? |
|---|---|---|
| Contexto | montar o prompt a partir do histórico | não |
| Memória | decidir o que guardar | não |
| Avaliação | pontuar respostas | **sim** |
| Cognição | interpretar o turno | não |

A coluna da direita é a razão pela qual a Fase 1.5 foi dividida em a/b/c: os
componentes que não dependem do modelo podem ser construídos hoje; os que
dependem esperam por um gerador real.

---

## O que este documento não decide

- Número de camadas, cabeças, dimensão. **Decide-se por medição** no hardware
  onde vai treinar, não por preferência.
- Se o autograd próprio chega. O benchmark mediu 153 733 tokens/s contra 246 919
  do PyTorch — 1,6×. Para 1–3 M parâmetros, o autograd próprio é viável; para
  muito mais, não. **A decisão é sua e os números estão na mesa.**
- O tamanho do vocabulário. Depende do benchmark que está a correr.
