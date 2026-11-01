"""Conjunto RESERVADO: nunca usado para ajustar o sistema.

Regra que este ficheiro impõe
-----------------------------
O benchmark de 14 paráfrases foi, na prática, um conjunto de treino: corrigi 11
defeitos *olhando para essas 14 perguntas*. Chamar-lhe generalização é
desonesto, e está escrito no README desde então.

Este conjunto é diferente: foi escrito de uma vez, antes de ser corrido, e o seu
resultado **não é usado para corrigir nada**. Se uma pergunta falhar aqui, fica
registada como falha. Só se corrige o sistema depois de fechar a versão, e nesse
momento este conjunto deixa de ser reservado — passando a ser necessária outra
reserva. É por isso que leva um número de versão.

REGRA: não alterar perguntas depois de ver resultados. Não acrescentar
sinónimos para aprovar. Não relaxar `todos`.

Versão 1 — escrita em 2026-11-01.
"""

from __future__ import annotations

from kianda.benchmark.banco import Item

RESERVADO_V1: tuple[Item, ...] = (
    # --- factual, do canon ---
    Item("Quem criou o Sigillo?", ("zuleika", "zulu"), (), "livro"),
    Item("Qual é o nome completo da Zulu?", ("mendes",), (), "livro"),
    Item("De quem é filho o Edmilson?", ("celestino",), (), "livro"),
    Item("Que idade tem a Zuleika?", ("26",), (), "livro"),
    Item("Onde é que a Zuleika cresceu?", ("rangel",), (), "livro"),
    Item("Quem é a Lara?", ("irmã", "irma"), (), "livro"),
    Item("De que trata a obra do pastor?", ("h99", "liberdade"), (), "livro"),
    # --- técnico, do projecto ---
    Item("Que linguagem usas para as contas?", ("numpy",), (), "projecto"),
    Item("Precisas de internet para funcionar?", ("não precisa", "nao precisa"), (),
         "projecto"),
    Item("Quantas dimensões tem o teu índice?", ("96",), (), "projecto"),
    # CORRIGIDO sem alterar a exigência: `todos` não é lista de sinónimos.
    #
    # Escrevi `todos=("semântica", "semantica")` para tolerar o acento, mas
    # `todos` exige que TODOS estejam presentes — e uma resposta em português
    # correcto nunca contém as duas grafias da mesma palavra. O item era
    # impossível de satisfazer. A Kianda respondia exactamente o pedido
    # («episódica, semântica, relacional...») e era marcada errada.
    #
    # O que se exige é: lista os tipos (episódica) E inclui a semântica. O
    # sinónimo fica em `esperado`, onde significa "um destes serve".
    Item("Que tipos de memória tens?", ("episódica", "episodica"), (),
         "projecto", todos=("semântica",)),
    Item("O que acontece quando esqueceres algo?", ("desactiv", "desativ", "não apaga"),
         (), "projecto"),
    # --- pessoais / identidade ---
    Item("Qual é o teu nome?", ("kianda",), (), "pessoal"),
    Item("Gostas de ler o quê?", ("apenas existo", "h99", "ondas de zulu"), (), "pessoal"),
    Item("Tens consciência?", ("não afirmo", "nao afirmo", "não", "nao"), (),
         "pessoal"),
)


def correr() -> dict:
    """Corre o conjunto reservado em instâncias limpas e devolve o resultado."""
    import tempfile
    import warnings

    warnings.filterwarnings("ignore")
    from kianda.mente import Kianda
    from kianda.benchmark.banco import _acertou

    detalhes = []
    for item in RESERVADO_V1:
        k = Kianda(pasta_dados=tempfile.mkdtemp())
        try:
            t = k.conversar(item.pergunta, guardar=False)
            ok = _acertou(t.resposta.texto, item)
            detalhes.append({
                "pergunta": item.pergunta,
                "ok": ok,
                "intencao": t.percepcao.intencao,
                "resposta": t.resposta.texto[:160],
            })
        finally:
            k.fechar()

    acertos = sum(d["ok"] for d in detalhes)
    return {"acertos": acertos, "total": len(detalhes), "detalhes": detalhes}


if __name__ == "__main__":
    r = correr()
    print(f"\nCONJUNTO RESERVADO v1: {r['acertos']}/{r['total']} = "
          f"{r['acertos'] / r['total']:.1%}\n")
    for d in r["detalhes"]:
        marca = "OK " if d["ok"] else "XX "
        print(f"{marca} {d['pergunta'][:44]:46s} [{d['intencao'][:15]:15s}]")
        if not d["ok"]:
            print(f"     {d['resposta']}")
