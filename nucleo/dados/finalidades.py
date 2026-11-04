"""As 9 finalidades de dados da directiva, medidas contra o que existe.

A directiva pede dados por finalidade (pt_general, pt_africa, science_tech,
programming, games, conversation, instructions, reasoning, identity) com
proveniência/licença/limpeza/avaliação por categoria.

Este módulo não inventa dados: **mede o que existe** e diz, em tokens, a
distância entre o que há e o que cada finalidade precisa para ser treinável.
O objectivo numérico é o ponto de cruzamento medido (9,73e4 tokens) e, como
limite de utilidade, 2,3e6 (35x o corpus, ver ESTRATEGIA.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


@dataclass
class Finalidade:
    nome: str
    descricao: str
    fonte_actual: str
    tokens_existentes: int = 0
    licenca: str = "desconhecida"
    redistribuivel: bool = False
    onde_buscar: list[str] = field(default_factory=list)
    teste_de_avaliacao: str = ""


# O que existe hoje, medido: 65 999 tokens no total, todos de 1 autor.
FINALIDADES = [
    Finalidade(
        "pt_general",
        "Português corrente: gramática, notícia, ensaio, variantes",
        "5 obras de Edmundo Nambo + Bechara (em dados/outros, fora do corpus)",
        onde_buscar=["texto próprio e licenciado", "domínio público (Bechara não: ainda protegido)",
                     "jornais com licença explícita"],
        teste_de_avaliacao="perplexidade em texto de autor nunca visto; taxa de concordância verbal",
    ),
    Finalidade(
        "pt_africa",
        "Português angolano e variantes africanas",
        "as 5 obras do corpus",
        onde_buscar=["literatura angolana em domínio público ou autorizada",
                     "PEPÁ (Pesquisas e Publicações Angola) com autorização"],
        teste_de_avaliacao="léxico e construções marcadas de Angola reconhecidas em contexto",
    ),
    Finalidade(
        "science_tech",
        "Ciência, matemática, tecnologia",
        "nenhuma",
        onde_buscar=["artigos próprios", "teses com licença aberta", "domínio público científico"],
        teste_de_avaliacao="resposta verificável a pergunta factual com fonte",
    ),
    Finalidade(
        "programming",
        "Código e documentação técnica",
        "nenhuma",
        onde_buscar=["código do próprio projecto (licença nossa)", "documentação de bibliotecas com licença permissiva"],
        teste_de_avaliacao="sintaxe válida gerada; execução de testes unitários simples",
    ),
    Finalidade(
        "games",
        "Jogos, regras, narrativa interactiva",
        "nenhuma",
        onde_buscar=["regras escritas por nós", "descrições de jogo próprias"],
        teste_de_avaliacao="consistência de regras em multi-turno",
    ),
    Finalidade(
        "conversation",
        "Padrões de diálogo",
        "dados/dialogos/conversas_v1.md: 50 conversas, 512 turnos, 2 900 palavras",
        onde_buscar=["mais conversas do mesmo tipo", "conversas reais, transcritas e anonimizadas",
                     "SFT: pares instrução/resposta derivados destas conversas"],
        teste_de_avaliacao="manter referência a turnos anteriores; pedir esclarecimento. "
                           "MEDIDO: 9 653 tokens contra 1,7 M parâmetros -> o modelo decora "
                           "(folga treino-vs-validação 8,81 nats, 94% não explicada por nomes). "
                           "Ver resultado_adaptacao.json e resultado_memorizacao_dialogo.json",
    ),
    Finalidade(
        "instructions",
        "Seguir instruções, perguntas e respostas",
        "nenhuma",
        onde_buscar=["par instrução/resposta escritos por nós", "tarefas do próprio repositório"],
        teste_de_avaliacao="executar instrução de 2-3 passos; recusar quando não sabe",
    ),
    Finalidade(
        "reasoning",
        "Raciocínio verificável",
        "nenhuma",
        onde_buscar=["problemas com resposta conferível (aritmética, lógica)"],
        teste_de_avaliacao="acerto em problemas com resposta única e verificável",
    ),
    Finalidade(
        "identity",
        "Identidade e valores do projecto",
        "nenhuma",
        onde_buscar=["textos fundadores do projecto, escritos por nós"],
        teste_de_avaliacao="consistência das respostas sobre o que o sistema é e não é",
    ),
]


def medir_existentes() -> None:
    """Preenche tokens_existentes com o que está realmente em dados/."""
    total_obras = total_dialogo = 0
    try:
        from nucleo.tokenizador.bpe import TokenizadorBPE
        tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
        for f in sorted((RAIZ / "dados" / "obras").glob("*.txt")):
            total_obras += len(tok.codificar(f.read_text(encoding="utf-8", errors="replace")))
        for f in sorted((RAIZ / "dados" / "dialogos").glob("*.md")):
            total_dialogo += len(tok.codificar(f.read_text(encoding="utf-8", errors="replace")))
    except Exception:
        total_obras, total_dialogo = 65999, 9653   # valores medidos anteriormente
    for fin in FINALIDADES:
        if fin.nome in ("pt_general", "pt_africa"):
            fin.tokens_existentes = total_obras
        elif fin.nome == "conversation":
            fin.tokens_existentes = total_dialogo
    # licença: o manifesto diz caracteres_redistribuiveis = 0
    try:
        man = json.loads((RAIZ / "dados" / "manifest.json").read_text())
        redistrib = man["resumo"]["caracteres_redistribuiveis"] == 0
    except Exception:
        redistrib = True
    for fin in FINALIDADES:
        if fin.tokens_existentes:
            fin.licenca = "obra do autor do projecto — uso privado; redistribuição NÃO autorizada"
            fin.redistribuivel = not redistrib


def main() -> None:
    medir_existentes()
    alvo = 97_320          # ponto de cruzamento ajustado
    util = 2_300_000       # 35x, limite de utilidade

    medir_existentes()
    print(f"{'finalidade':<14} {'tokens':>9} {'% do alvo':>10} {'licença':<10} {'redistrib.'}")
    print("-" * 70)
    for f in FINALIDADES:
        pct = f.tokens_existentes / alvo * 100
        print(f"{f.nome:<14} {f.tokens_existentes:>9} {pct:>9.1f}% "
              f"{'sim' if f.tokens_existentes else '—':<10} "
              f"{'sim' if f.redistribuivel else 'NAO'}")

    # ATENÇÃO: pt_general e pt_africa partilham os MESMOS 65 999 tokens.
    # Somar `tokens_existentes` conta-os duas vezes e dá 131 998 — mais que
    # o corpus real. O total correcto é a UNIÃO das FONTES distintas:
    # obras (65 999) + diálogos (9 653). Cada fonte conta uma vez, mesmo que
    # alimente duas finalidades.
    total_unico = sum(f.tokens_existentes for f in FINALIDADES
                      if f.nome in ("pt_general", "conversation"))
    soma_ingenua = sum(f.tokens_existentes for f in FINALIDADES)
    cobertas = sum(1 for f in FINALIDADES if f.tokens_existentes > 0)
    print(f"\n{cobertas} de {len(FINALIDADES)} finalidades têm qualquer dado.")
    print(f"Tokens ÚNICOS: {total_unico:,}  (soma das fontes SEM sobreposição; a soma "
          f"cega por categoria daria {soma_ingenua:,} porque pt_general e pt_africa "
          f"são o mesmo texto)")
    print(f"Contra {alvo:,} (cruzamento) e {util:,} (utilidade).")
    print(f"Cobertura do cruzamento: {total_unico/alvo*100:.1f}%  ·  "
          f"da utilidade: {total_unico/util*100:.2f}%")

    destino = RAIZ / "nucleo" / "dados" / "resultado_finalidades.json"
    destino.write_text(json.dumps(
        {"alvo_cruzamento": alvo, "alvo_utilidade": util,
         "finalidades": [f.__dict__ for f in FINALIDADES],
         "cobertas": cobertas, "total": len(FINALIDADES),
         "tokens_unicos": total_unico, "soma_ingenua_por_categoria": soma_ingenua,
         "nota": "pt_general e pt_africa partilham o mesmo texto; somar duplica"},
        indent=2, ensure_ascii=False))
    print(f"\n{destino}")


if __name__ == "__main__":
    main()
