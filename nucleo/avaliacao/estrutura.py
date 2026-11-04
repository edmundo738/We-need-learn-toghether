"""Estrutura de avaliação: 9 dimensões, cada uma com o seu baseline.

O princípio que governa isto: **nenhuma dimensão é avaliada sem um baseline
que também se pode obter sem modelo.** Se um trigrama, um dicionário ou uma
resposta constante igualam o modelo, a dimensão não mede capacidade — mede
a existência do modelo.

Cada dimensão declara: o que mede, como se mede, qual é o baseline, e o que
conta como «melhor que o baseline».

O que se pode medir HOJE, com o corpus e o modelo que existem, está marcado
IMPLEMENTADO. O resto está DESENHADO — e um teste que se desenha e não se
corre não é um resultado.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Dimensao:
    nome: str
    mede: str
    metrica: str
    baseline: str
    criterio: str
    estado: str          # IMPLEMENTADO | DESENHADO


DIMENSOES = [
    Dimensao(
        "Língua (perplexidade)",
        "capacidade de prever o próximo token em texto novo",
        "bits por byte (comparável entre tokenizadores)",
        "trigrama com backoff (1,8947 bpb) e 2-grama (1,9272)",
        "bpb do modelo < bpb do trigrama, no mesmo conjunto",
        "IMPLEMENTADO (baselines.py + bits/byte convertido)",
    ),
    Dimensao(
        "Língua (gramática)",
        "concordância, regência, colocação de pronomes",
        "taxa de acerto em frases com uma violação plantada vs versão correcta",
        "moeda ao ar (50%)",
        "acima de 50% com intervalo de confiança que exclua 0,5",
        "IMPLEMENTADO e MEDIDO (era 0) — 26 pares em 14 fenómenos, "
        "resultado_gramatical.json: Transformer 0,462 (p=1,00 vs moeda), "
        "trigrama 0,712 (p=0,0755). O Transformer NÃO julga gramática.",
    ),
    Dimensao(
        "Conhecimento",
        "factos verificáveis sobre o domínio",
        "acerto em perguntas com resposta única e fonte declarada",
        "responder sempre o token mais frequente",
        "acima do baseline, e a resposta citável na fonte",
        "DESENHADO — exige corpus factual, que não existe",
    ),
    Dimensao(
        "Instruções",
        "seguir uma ordem de 2-3 passos",
        "execução correcta de todos os passos, verificável",
        "ignorar a instrução (0%)",
        "> 0% e cada passo verificável",
        "DESENHADO — exige conjunto instrução/resposta",
    ),
    Dimensao(
        "Conversação multi-turno",
        "manter referência a turnos anteriores",
        "acerto em pergunta cuja resposta depende de um turno prévio",
        "responder sem ver o histórico",
        "modelo com histórico > modelo sem histórico",
        "DESENHADO — exige modelo treinado em diálogo",
    ),
    Dimensao(
        "Memória",
        "recordar um facto introduzido na sessão",
        "recuperação exacta do facto N turnos depois",
        "não guardar nada (0%)",
        "> 0% até N turnos declarados",
        "DESENHADO — é do orquestrador, não do LM (ver ESTRATEGIA.md)",
    ),
    Dimensao(
        "Raciocínio",
        "aritmética e lógica com resposta única",
        "acerto em problemas cuja resposta se confere por cálculo",
        "aritmética do próprio avaliador (100%)",
        "o modelo tem de acertar onde a conta é verificável",
        "DESENHADO — e nota: um LM de 2,77 M treinado em prosa não fará isto",
    ),
    Dimensao(
        "Factualidade",
        "não inventar quando não sabe",
        "taxa de «não sei» correcto em perguntas fora do domínio",
        "inventar sempre (0% de «não sei»)",
        "declara ignorância em vez de inventar",
        "DESENHADO — o corpus actual não tem nenhum exemplo de «não sei»",
    ),
    Dimensao(
        "Generalização",
        "desempenho em autor/fonte nunca vistos",
        "diferença de perda entre validação do mesmo autor e de autor novo",
        "a própria validação actual (mesmo autor)",
        "perda em autor novo não muito acima da do mesmo autor",
        "PARCIAL — e MEDIDO em diálogo: avaliacao_limpa.py mostra que 50->80 "
        "conversas melhora a validação de 9,11 para 8,27 (acaso 9,011) mas a "
        "folga treino-val fica em 8,17 nats: decora, não generaliza",
    ),
]


def resumo() -> dict:
    """Conta por estado. O prefixo do estado tem de ser lido do texto real.

    A primeira versão fazia `d.estado.split(" ")[0]`, o que dava
    «IMPLEMENTADO» e «DESENHADO» — mas os estados com prefixo composto
    («IMPLEMENTADO e MEDIDO...») eram contados como implementados por
    acidente do prefixo, e «PARCIAL — e MEDIDO...» como parcial. O resultado
    impresso («1 implementada») estava certo por sorte e errado na leitura:
    eram 2 implementadas. Fica explícito.
    """
    conta = {"IMPLEMENTADO": 0, "PARCIAL": 0, "DESENHADO": 0}
    for d in DIMENSOES:
        e = d.estado.upper()
        if e.startswith("IMPLEMENTADO"):
            conta["IMPLEMENTADO"] += 1
        elif e.startswith("PARCIAL"):
            conta["PARCIAL"] += 1
        else:
            conta["DESENHADO"] += 1
    return conta


def main() -> None:
    print(f"{len(DIMENSOES)} dimensões de avaliação\n")
    for d in DIMENSOES:
        print(f"{d.nome:<26} [{d.estado.split(' ')[0]}]")
        print(f"   mede:     {d.mede}")
        print(f"   métrica:  {d.metrica}")
        print(f"   baseline: {d.baseline}")
        print(f"   critério: {d.criterio}")
        print()
    print("resumo:", resumo())
    print()
    r = resumo()
    print(f"LEITURA HONESTA: {r['IMPLEMENTADO']} implementadas, {r['PARCIAL']} parciais, "
          f"{r['DESENHADO']} desenhadas.")
    print("Uma dimensão desenhada e não executada NÃO é um resultado — é um plano.")


if __name__ == "__main__":
    main()
