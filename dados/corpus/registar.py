"""Congelamento e manifesto do corpus — a base de tudo o resto.

Porque este ficheiro existe
---------------------------
A Fase 1 produziu um tokenizador cujos números **não são reproduzíveis**. O
corpus estava em `dados/`, que estava no `.gitignore`; um clone limpo não tinha
texto nenhum, e os relatórios referenciavam um corpus de 254 KB que ninguém
conseguia reconstruir. Pior: o próprio `dados/` desapareceu entre sessões e eu
estive a medir um corpus de 13 KB a pensar que era de 254 KB.

A regra que sai daí, e que este módulo implementa:

    Corpus v0.1 + tokenizer v0.2 + arquitetura X + seed Y = pesos Z

Tudo à esquerda do sinal é um artefacto com hash. Se o hash não bate, o
resultado não é comparável.

O problema jurídico, e como é resolvido
---------------------------------------
Tamanho não é o problema: **direito de redistribuição é**. Dos 3,1 MB
disponíveis, só uma parte pode ir para o repositório. O manifesto separa
deliberadamente duas coisas que se confundem:

- `redistribuivel` — posso publicar o texto? (licença)
- `versionado`    — ponho os bytes no git deste repositório?

Uma obra pode ser `versionado: true` e `redistribuivel: false`: é o caso das
obras do Edmundo neste repositório **privado** — são dele, mas não são para
publicar. Essa distinção é a única forma honesta de não transformar «cabe no
git» em «é legal».

Para as fontes que não podemos versionar, o manifesto guarda nome, SHA-256,
tamanho, origem e instruções de obtenção. Assim uma pessoa consegue verificar
que tem os ficheiros certos sem que o repositório os contenha.

Divisões train/validation/test
------------------------------
Feitas por **bloco contíguo**, com hash determinística de (ficheiro, índice do
bloco). Porquê blocos e não ficheiros: a gramática do Bechara tem 1,4 MB e
ficaria toda num lado; porquê contíguos: blocos aleatórios de linhas deixariam
frases cortadas entre treino e validação. O corte é feito em fronteira de
parágrafo e o hash garante que a mesma versão do corpus produz sempre a mesma
divisão — sem isso, «o resultado melhorou» pode ser só uma divisão diferente.

Uso:
    python -m dados.corpus.registar
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
VERSAO_CORPUS = "0.1"
SEMENTE_DIVISAO = 20261101

# --------------------------------------------------------------------------
# Política de cada fonte. Este dicionário é a decisão jurídica, escrita.
# --------------------------------------------------------------------------
#
# `redistribuivel` = licença permite publicar o texto.
# `versionado`     = os bytes vão para o git DESTE repositório.
#
# As obras do projecto são do autor deste repositório. O repositório é privado.
# Por isso são versionadas, mas marcadas como não-redistribuíveis: se um dia o
# repositório se tornar público, estas entradas têm de ser revistas ANTES.
FONTES = {
    "obras/apenas_existo_parte1.txt": dict(
        titulo="APENAS EXISTO — Ondas de Zulu (parte 1)",
        autor="Edmundo Nambo", categoria="kianda", idioma="pt-AO",
        origem="Original do autor, convertido de PDF em /tmp/arena-coding-uploads",
        licenca="© Edmundo Nambo — todos os direitos reservados",
        redistribuivel=False, versionado=True,
        nota="Obra do próprio autor do projecto. Versionada por ser repositório privado.",
    ),
    "obras/apenas_existo_parte2.txt": dict(
        titulo="APENAS EXISTO — Ondas de Zulu (parte 2)",
        autor="Edmundo Nambo", categoria="kianda", idioma="pt-AO",
        origem="Original do autor, convertido de PDF",
        licenca="© Edmundo Nambo — todos os direitos reservados",
        redistribuivel=False, versionado=True,
        nota="Obra do próprio autor do projecto.",
    ),
    "obras/apenas_existo_parte3.txt": dict(
        titulo="APENAS EXISTO — Ondas de Zulu (parte 3)",
        autor="Edmundo Nambo", categoria="kianda", idioma="pt-AO",
        origem="Original do autor, convertido de PDF",
        licenca="© Edmundo Nambo — todos os direitos reservados",
        redistribuivel=False, versionado=True,
        nota="Obra do próprio autor do projecto.",
    ),
    "obras/H99_Parte_I.txt": dict(
        titulo="H99 (parte I)", autor="Edmundo Nambo", categoria="kianda",
        idioma="pt-AO", origem="Original do autor, convertido de PDF",
        licenca="© Edmundo Nambo — todos os direitos reservados",
        redistribuivel=False, versionado=True,
        nota="Obra do próprio autor do projecto.",
    ),
    "obras/H99_Parte_II.txt": dict(
        titulo="H99 (parte II)", autor="Edmundo Nambo", categoria="kianda",
        idioma="pt-AO", origem="Original do autor, convertido de PDF",
        licenca="© Edmundo Nambo — todos os direitos reservados",
        redistribuivel=False, versionado=True,
        nota="Obra do próprio autor do projecto.",
    ),
    # --- a partir daqui: NÃO versionado, só manifesto -------------------
    "outros/bechara_gramatica.txt": dict(
        titulo="Moderna Gramática Portuguesa", autor="Evanildo Bechara",
        categoria="geral", idioma="pt-BR",
        origem="PDF fornecido pelo autor do projecto (574 pág.)",
        licenca="© Nova Fronteira / autor — uso restrito, não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Registo: gramática normativa. Registo culto, PT-BR — enviesa o modelo.",
    ),
    "outros/rogers_tornar_se_pessoa.txt": dict(
        titulo="Tornar-se Pessoa", autor="Carl R. Rogers",
        categoria="geral", idioma="pt-BR",
        origem="PDF fornecido pelo autor do projecto (243 pág.)",
        licenca="© editora / tradutor — uso restrito, não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Tradução brasileira de obra original em inglês.",
    ),
    "outros/adoum_eu_sou.txt": dict(
        titulo="Eu Sou", autor="Jorge Adoum", categoria="geral", idioma="pt-BR",
        origem="PDF fornecido pelo autor do projecto (104 pág.)",
        licenca="© editora / tradutor — uso restrito, não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Tradução; registo esotérico.",
    ),
    "outros/conversas_amor.txt": dict(
        titulo="Conversas Sobre o Amor", autor="(a confirmar)",
        categoria="conversa", idioma="pt",
        origem="PDF fornecido pelo autor do projecto",
        licenca="© — a confirmar, presumido não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Registo dialógico — o mais valioso para treino conversacional.",
    ),
    "outros/falar_qualquer_pessoa.txt": dict(
        titulo="Como Falar com Qualquer Pessoa", autor="(a confirmar)",
        categoria="conversa", idioma="pt",
        origem="PDF fornecido pelo autor do projecto",
        licenca="© — a confirmar, presumido não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Registo dialógico.",
    ),
    "outros/conversando_ia.txt": dict(
        titulo="Conversando com IA", autor="(a confirmar)",
        categoria="geral", idioma="pt",
        origem="PDF fornecido pelo autor do projecto",
        licenca="© — a confirmar, presumido não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Ensino; registo expositivo.",
    ),
    "outros/situacoes_variadas.txt": dict(
        titulo="Situações Variadas — Básico", autor="(a confirmar)",
        categoria="conversa", idioma="pt",
        origem="PDF fornecido pelo autor do projecto (8 pág.)",
        licenca="© — a confirmar, presumido não redistribuível",
        redistribuivel=False, versionado=False,
        nota="Material didáctico pequeno. Registar, não incluir sem verificar.",
    ),
}

# Ficheiros deliberadamente excluídos, com a razão. Excluir em silêncio é a
# forma mais fácil de mentir num relatório de corpus.
EXCLUIDOS = {
    "outros/who_am_i.txt": (
        "2 caracteres extraídos de 17 páginas — PDF sem camada de texto "
        "utilizável (provavelmente digitalizado). Não é um problema de código: "
        "exigiria OCR. Excluído em vez de incluído como ficheiro vazio."
    ),
}

TAMANHO_BLOCO = 2000          # caracteres por bloco, cortado em parágrafo
FRACOES = {"train": 0.90, "validacao": 0.05, "teste": 0.05}


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for pedaco in iter(lambda: f.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def _limpar(texto: str) -> str:
    """Normalização mínima — o que é claramente lixo de extracção, nada mais.

    Deliberadamente conservadora: qualquer normalização agressiva (minúsculas,
    remoção de acentos) destrói informação que o modelo precisa de aprender.
    Só se remove o que sabidamente não é texto: hífen de quebra de linha,
    hífenes suaves, e excesso de linhas vazias.
    """
    texto = texto.replace("\u00ad", "")
    texto = re.sub(r"-\n(?=[a-záàâãéêíóôõúüç])", "", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _blocos(texto: str, tamanho: int = TAMANHO_BLOCO) -> list[str]:
    """Divide em blocos contíguos, cortando em fronteira de parágrafo."""
    blocos, atual = [], ""
    for paragrafo in texto.split("\n\n"):
        if len(atual) + len(paragrafo) + 2 > tamanho and atual:
            blocos.append(atual.strip())
            atual = paragrafo
        else:
            atual = f"{atual}\n\n{paragrafo}" if atual else paragrafo
    if atual.strip():
        blocos.append(atual.strip())
    return [b for b in blocos if b]


def _dividir(ficheiro: str, n_blocos: int) -> str:
    """Divisão determinística de um bloco, por hash de (ficheiro, índice).

    Hash e não ordem sequencial para que os três conjuntos fiquem misturados ao
    longo do livro — se o teste fosse o último 5% de cada ficheiro, mediria o
    fim dos livros e não o que o modelo generaliza.
    """
    chave = f"{SEMENTE_DIVISAO}:{ficheiro}:{n_blocos}".encode()
    h = int(hashlib.sha256(chave).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < FRACOES["train"]:
        return "train"
    if h < FRACOES["train"] + FRACOES["validacao"]:
        return "validacao"
    return "teste"


def construir(raiz: Path = RAIZ) -> dict:
    pasta_dados = raiz / "dados"
    entradas, avisos = [], []

    for relativo, politica in sorted(FONTES.items()):
        caminho = pasta_dados / relativo
        if not caminho.exists():
            avisos.append(f"AUSENTE: {relativo} — declarado no manifesto mas não está no disco")
            continue
        bruto = caminho.read_text(encoding="utf-8", errors="replace")
        texto = _limpar(bruto)
        blocos = _blocos(texto)
        divisao = {}
        for i, bloco in enumerate(blocos):
            d = _dividir(relativo, i)
            divisao[d] = divisao.get(d, 0) + len(bloco)
        entradas.append({
            "caminho": f"dados/{relativo}",
            **politica,
            "sha256": _sha256(caminho),
            "bytes": caminho.stat().st_size,
            "caracteres": len(texto),
            "blocos": len(blocos),
            "char_por_divisao": divisao,
        })

    for relativo, razao in EXCLUIDOS.items():
        caminho = pasta_dados / relativo
        if caminho.exists():
            avisos.append(f"EXCLUÍDO: {relativo} — {razao}")

    total = sum(e["caracteres"] for e in entradas)
    versionado = sum(e["caracteres"] for e in entradas if e["versionado"])
    redistribuivel = sum(e["caracteres"] for e in entradas if e["redistribuivel"])

    manifesto = {
        "corpus_versao": VERSAO_CORPUS,
        "data": date.today().isoformat(),
        "semente_divisao": SEMENTE_DIVISAO,
        "fracoes": FRACOES,
        "tamanho_bloco_caracteres": TAMANHO_BLOCO,
        "resumo": {
            "ficheiros": len(entradas),
            "caracteres_totais": total,
            "caracteres_versionados_no_git": versionado,
            "caracteres_redistribuiveis": redistribuivel,
            "caracteres_apenas_no_manifesto": total - versionado,
        },
        "regra": (
            "Corpus v{} + tokenizer + arquitectura + seed = pesos. "
            "Se o sha256 de um ficheiro não bater, o resultado não é comparável."
        ).format(VERSAO_CORPUS),
        "avisos": avisos,
        "ficheiros": entradas,
        "excluidos": EXCLUIDOS,
    }
    return manifesto


def main() -> int:
    manifesto = construir()
    saida = RAIZ / "dados" / "manifest.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False))

    r = manifesto["resumo"]
    print("=" * 68)
    print(f"CORPUS v{manifesto['corpus_versao']} — congelado em {manifesto['data']}")
    print("=" * 68)
    print(f"ficheiros                          : {r['ficheiros']}")
    print(f"caracteres totais                  : {r['caracteres_totais']:>12,}")
    print(f"  versionados no git               : {r['caracteres_versionados_no_git']:>12,}")
    print(f"  só no manifesto (não versionados): {r['caracteres_apenas_no_manifesto']:>12,}")
    print(f"  redistribuíveis                  : {r['caracteres_redistribuiveis']:>12,}")
    print()
    print(f"{'ficheiro':38s} {'chars':>9s} {'git':>4s} {'redis':>6s}")
    print("-" * 68)
    for e in manifesto["ficheiros"]:
        nome = e["caminho"].replace("dados/", "")
        print(f"{nome:38s} {e['caracteres']:>9,} "
              f"{'sim' if e['versionado'] else '—':>4s} "
              f"{'sim' if e['redistribuivel'] else 'não':>6s}")
    print("-" * 68)
    if manifesto["avisos"]:
        print("\nAVISOS:")
        for a in manifesto["avisos"]:
            print(f"  · {a[:100]}")
    print(f"\nmanifesto: {saida.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
