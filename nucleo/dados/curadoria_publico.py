"""Curadoria do corpus de domínio público (Machado de Assis, 1839-1908).

Porque é que este ficheiro existe
---------------------------------
O corpus do projecto tinha 65 999 tokens de um só autor. Machado de Assis,
falecido em 1908, está em domínio público (mais de 70 anos após a morte, na
lei brasileira e na angolana), e os quatro PDFs somam 944 276 caracteres.
Isso é 3,9x o corpus anterior — a maior mudança de regime de dados do projecto.

Mas texto extraído de PDF não é texto: tem cabeçalhos de página, créditos da
biblioteca digital, marcas de paginação e artefactos de OCR. Treinar nisso
ensina o modelo a gerar «www.nead.unama.br». Este curador remove isso e,
mais importante, **mede e reporta** o que removeu — porque uma remoção
silenciosa é indistinguível de um corpus que nunca teve lixo.

REGRA: nenhuma linha é removida por parecer estranha. Cada filtro tem nome,
motivo e contagem. Se um filtro remover demasiado, vê-se.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


@dataclass
class Filtro:
    nome: str
    motivo: str
    padrao: re.Pattern


FILTROS = [
    Filtro("creditos_da_biblioteca",
           "créditos de digitalização da biblioteca/editora, não do autor",
           re.compile(r"(biblioteca virtual|bibvirt|futuro\.usp|elogica|virtualstore|"
                      r"aportis|onetap|unama|nead|universidade da amaz|"
                      r"curso objetivo|pdf-tools|www\.|http)", re.I)),
    Filtro("marca_de_paginacao",
           "números de página isolados injectados pela digitalização",
           re.compile(r"^\s*\d{1,4}\s*$")),
    Filtro("credito_de_ocr",
           "metadados do processo de digitalização",
           re.compile(r"(scannea|scanner|\bocr\b|reconhecimento óptico|digitaliz|"
                      r"todos os direitos|proibida a reprodu)", re.I)),
    Filtro("linha_sem_conteudo",
           "linhas só com pontuação ou símbolos, sem conteúdo lexical",
           re.compile(r"^\s*[-–—_=*~.·•\s]{0,6}\s*$")),
]


@dataclass
class Resultado:
    ficheiro: str
    caracteres_antes: int = 0
    caracteres_depois: int = 0
    linhas_antes: int = 0
    linhas_depois: int = 0
    palavras_antes: int = 0
    palavras_depois: int = 0
    filtros: list[dict] = field(default_factory=list)


def curar(texto: str, nome: str) -> tuple[str, Resultado]:
    r = Resultado(ficheiro=nome)
    r.caracteres_antes, r.palavras_antes = len(texto), len(texto.split())
    linhas = texto.split("\n")
    r.linhas_antes = len(linhas)

    contagem = {f.nome: [0, 0] for f in FILTROS}
    saida = []
    for linha in linhas:
        normalizada = re.sub(r"[ \t]+", " ", linha)
        removida = False
        for f in FILTROS:
            if f.padrao.search(normalizada):
                contagem[f.nome][0] += 1
                contagem[f.nome][1] += len(linha)
                removida = True
                break
        if not removida:
            saida.append(normalizada)

    limpo = re.sub(r"\n{3,}", "\n\n", "\n".join(saida)).strip()
    r.caracteres_depois, r.palavras_depois = len(limpo), len(limpo.split())
    r.linhas_depois = len(saida)
    r.filtros = [{"nome": f.nome, "motivo": f.motivo,
                  "linhas": contagem[f.nome][0],
                  "caracteres": contagem[f.nome][1]} for f in FILTROS]
    return limpo, r


def main() -> None:
    pasta = RAIZ / "dados" / "publico"
    limpa = RAIZ / "dados" / "publico_limpo"
    limpa.mkdir(parents=True, exist_ok=True)

    resultados = []
    for f in sorted(pasta.glob("*.txt")):
        texto = f.read_text(encoding="utf-8", errors="replace")
        limpo, r = curar(texto, f.name)
        (limpa / f.name).write_text(limpo, encoding="utf-8")
        resultados.append(r)
        print(f"{f.name:<20} {r.caracteres_antes:>8,} -> {r.caracteres_depois:>8,} car "
              f"({(1-r.caracteres_depois/max(r.caracteres_antes,1))*100:>5.1f}% removido) "
              f"| {r.palavras_antes:>7,} -> {r.palavras_depois:>7,} palavras")

    tot_a = sum(r.caracteres_antes for r in resultados)
    tot_d = sum(r.caracteres_depois for r in resultados)
    tot_pa = sum(r.palavras_antes for r in resultados)
    tot_pd = sum(r.palavras_depois for r in resultados)
    print(f"\nTOTAL: {tot_a:,} -> {tot_d:,} caracteres "
          f"({(1-tot_d/max(tot_a,1))*100:.1f}% removido) | "
          f"{tot_pa:,} -> {tot_pd:,} palavras")

    print("\npor filtro:")
    agg: dict[str, list[int]] = {}
    for r in resultados:
        for f in r.filtros:
            a = agg.setdefault(f["nome"], [0, 0])
            a[0] += f["linhas"]
            a[1] += f["caracteres"]
    for nome, (lin, car) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {nome:<26} {lin:>5} linhas  {car:>8,} caracteres")

    res = {"classe": "MEASURED",
           "fonte": "Machado de Assis (1839-1908), domínio público",
           "total_antes": tot_a, "total_depois": tot_d,
           "palavras_antes": tot_pa, "palavras_depois": tot_pd,
           "removido_pct": round((1 - tot_d / max(tot_a, 1)) * 100, 2),
           "ficheiros": [{"ficheiro": r.ficheiro, "antes": r.caracteres_antes,
                          "depois": r.caracteres_depois,
                          "palavras_antes": r.palavras_antes,
                          "palavras_depois": r.palavras_depois,
                          "filtros": r.filtros} for r in resultados]}
    (RAIZ / "nucleo" / "dados" / "resultado_curadoria_publico.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
