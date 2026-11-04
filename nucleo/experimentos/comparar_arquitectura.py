"""Comparação da arquitectura própria contra nanoGPT, LitGPT e OLMo.

NÃO É UM ESTUDO DE OPINIÃO. Cada linha foi lida no código-fonte, clonado
em /tmp/estudo nesta sessão, e o caminho do ficheiro está citado. O que não
foi lido está marcado UNKNOWN.

Serve para responder a uma pergunta concreta da directiva: «a arquitectura
actual é adequada à escala?» — e a resposta tem de ser sobre decisões, não
sobre gosto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Decisao:
    aspecto: str
    nosso: str
    referencia: str
    fonte: str
    veredicto: str          # "igual" | "difere" | "falta" | "unknown"
    porque_importa: str


DECISOES = [
    Decisao(
        "Normalização", "LayerNorm pós? NÃO — pré-norm (ln antes de cada sub-camada)",
        "Pré-norm em todos: nanoGPT (ln_1 antes da atenção), LitGPT e OLMo2 usam RMSNorm",
        "nanoGPT/model.py; litgpt/litgpt/config.py:2190; OLMo2-1B.yaml layer_norm_type: rms",
        "difere",
        "RMSNorm não tem média nem beta (menos 2*d por camada) e é o padrão desde LLaMA. "
        "Não muda a matemática aprendível; muda estabilidade e número de parâmetros."),
    Decisao(
        "Activação da MLP", "ReLU",
        "GELU no nanoGPT (nn.GELU), SwiGLU no LitGPT/OLMo (LLaMAMLP)",
        "nanoGPT/model.py:83; litgpt config mlp_class_name=LLaMAMLP",
        "difere",
        "ReLU é a escolha mais fraca das três. SwiGLU custa ~1,5x params na MLP por "
        "ganho consistente. Não é razão do resultado fraco, mas é dívida técnica."),
    Decisao(
        "Posição", "Embedding posicional aprendido, absoluto",
        "RoPE (rotary) em LitGPT e OLMo; embedding aprendido no nanoGPT",
        "OLMo2-1B.yaml rope: true, rope_theta: 500000; nanoGPT wpe",
        "difere",
        "RoPE generaliza para contexto maior que o treinado; o embedding absoluto não. "
        "Relevante se a Fase 3 quiser contexto 512: com absoluto, é preciso re-treinar."),
    Decisao(
        "Peso da cabeça de saída", "Matriz separada (não partilhada)",
        "Weight tying no nanoGPT: wte.weight = lm_head.weight",
        "nanoGPT/model.py:138",
        "difere",
        "Partilhar poupa vocab*d_model parâmetros — aqui 1 048 576, ou 38% do modelo "
        "— e costuma melhorar a perda em modelos pequenos. É a única diferença desta "
        "tabela que reduz parâmetros em vez de os aumentar."),
    Decisao(
        "Bias nas lineares", "Sem bias (viés=False em todas)",
        "Sem bias no LitGPT (bias=False) e OLMo; com bias no nanoGPT (config.bias)",
        "litgpt config.py:2189",
        "igual",
        "Está alinhado com as referências modernas. Sem acção."),
    Decisao(
        "Optimizador", "Adam simples com grad_clip",
        "AdamW com weight decay separado para pesos 2D e não-2D (nanoGPT), fused",
        "nanoGPT/model.py:263-285",
        "difere",
        "Sem weight decay, os embeddings crescem sem travão. Num corpus pequeno isso "
        "agrava o sobreajuste. É uma correcção de 10 linhas, não uma reescrita."),
    Decisao(
        "Precisão", "float32 em tudo",
        "bf16 misto no OLMo (amp_bf16); nanoGPT usa autocast bf16/fp16 em GPU",
        "OLMo2-1B.yaml precision: amp_bf16",
        "difere",
        "Sem GPU, bf16 não traz ganho. Mas o fp32 é a razão de o gradcheck bater no "
        "limite de resolução — está documentado na auditoria."),
    Decisao(
        "Dados por parâmetro", "≈24 tokens/parâmetro (66k / 2,77M)",
        "OLMo2-1B: 4e12 tokens / ~1e9 params = ~4 000 tokens/parâmetro",
        "OLMo2-1B.yaml max_duration: 4e12T",
        "difere (2 ordens de grandeza)",
        "AQUI está a diferença que explica o resultado, e não é de arquitectura. "
        "Chinchilla (20 tok/param) é o óptimo de CÁLCULO; o OLMo treina a 4000, "
        "muito acima, porque quer capacidade e não eficiência de treino."),
    Decisao(
        "Curadoria de dados", "Limpeza textual mínima; filtros implementados agora",
        "Dolma: taggers Gopher/C4 + dedup MinHash; Tucano: classificador BERT+XGBoost",
        "dolma/python/dolma/taggers/gopher.py; Tucano/gigaverbo/text-filter/",
        "falta (parcialmente)",
        "O Tucano filtra 780 GB com um classificador treinado. Os filtros "
        "determinísticos do Gopher estão agora em nucleo/dados/curadoria.py; o "
        "classificador não é reproduzível sem dados rotulados."),
    Decisao(
        "Pipeline pós-treino", "Inexistente",
        "nanochat: tok_train -> tok_eval -> base_train -> base_eval (CORE, BPB) "
        "-> chat_sft -> chat_eval -> chat_cli",
        "nanochat/runs/speedrun.sh",
        "falta",
        "Sem SFT não há comportamento conversacional, por muito bom que o "
        "pré-treino seja. É a distinção que a directiva impõe: língua != diálogo."),
    Decisao(
        "Avaliação", "Perda de validação + baselines n-grama",
        "CORE metric + bits-per-byte (nanochat base_eval); benchmarks por tarefa",
        "nanochat/scripts/base_eval.py",
        "falta",
        "Perda não mede capacidade. Bits/byte já está implementado aqui; falta um "
        "conjunto de tarefas com resposta verificável."),
]


def main() -> None:
    print(f"{'aspecto':<26} {'estado':<14} {'nosso'}")
    print("-" * 100)
    for d in DECISOES:
        print(f"{d.aspecto:<26} {d.veredicto:<14} {d.nosso[:45]}")
    print()
    conta = {}
    for d in DECISOES:
        conta[d.veredicto.split(" ")[0]] = conta.get(d.veredicto.split(" ")[0], 0) + 1
    print("resumo:", conta)

    destino = Path(__file__).resolve().parent / "resultado_comparacao_arquitectura.json"
    destino.write_text(json.dumps([asdict(d) for d in DECISOES],
                                  indent=2, ensure_ascii=False))
    print(f"\n{destino}")


if __name__ == "__main__":
    main()
