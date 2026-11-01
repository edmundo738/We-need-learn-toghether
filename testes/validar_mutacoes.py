"""Mutation testing: provar que os testes detectam os defeitos que os motivaram.

Sem isto, um teste verde pode estar verde porque não testa nada. Este script
reintroduz, um a um, defeitos reais que estiveram neste código, corre a suite, e
verifica se ela FALHA. Se um mutante sobreviver (suite continua verde), o teste
que o devia apanhar é inútil.

Uso:
    python testes/validar_mutacoes.py

Saída: uma linha por mutante, com MATADO (bom) ou SOBREVIVEU (teste fraco).
Código de saída 0 apenas se todos os mutantes forem mortos.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PY = sys.executable

# (nome, ficheiro, procurar, substituir, testes que devem falhar)
MUTANTES = [
    (
        "recuperacao-sempre-devolve",
        "kianda/memoria/recuperador.py",
        "            if rb is None and rs is None:\n                continue  # não é um resultado, é um documento",
        "            if rb is None and rs is None:\n                pass  # MUTANTE: sem abstinência",
        ["test_recuperacao_abstem_se_nao_ha_sinal"],
    ),
    (
        "acertou-usa-any",
        "kianda/benchmark/banco.py",
        "    if item.todos and not all(x.lower() in t for x in item.todos):\n        return False",
        "    if False and item.todos:  # MUTANTE: ignora requisitos\n        return False",
        ["test_acertou_exige_todos_os_requisitos"],
    ),
    (
        "guardar-false-volta-a-gravar",
        "kianda/mente.py",
        "        mid = \"\"\n        if guardar:",
        "        mid = \"\"\n        if True:  # MUTANTE: guardar ignorado",
        ["test_guardar_false_nao_escreve_nada"],
    ),
    (
        "stemmer-sem-vogal-tematica",
        "kianda/memoria/recuperador.py",
        '               "a", "e", "o")',
        '               )  # MUTANTE: sem vogal temática',
        ["test_stemmer_converge_formas_da_mesma_palavra"],
    ),
    (
        "parentesco-ignora-relacao",
        "kianda/cognicao/raciocinio.py",
        "            if _parentesco:",
        "            if False:  # MUTANTE: parentesco ignorado",
        # Teste focado, não o reservado. A primeira versão usava o conjunto
        # reservado e o mutante SOBREVIVEU duas vezes: o limiar de 12/15
        # continuava atingido (13/15), e mais tarde 13/15 já era o valor
        # normal. Uma barreira agregada não distingue "perdi um caso" de
        # "este caso sempre falhou" — tem de haver um teste por comportamento.
        ["test_parentesco_resolve_a_relacao_pedida"],
    ),
    (
        "regra-mecanismo-desligada",
        "kianda/cognicao/percepcao.py",
        '        if (intencao in ("pergunta_pessoal", "conversa")\n                and _SINAIS_MECANISMO.search(texto)',
        '        if (False\n                and _SINAIS_MECANISMO.search(texto)',
        ["test_pergunta_sobre_mecanismo_nao_vira_pessoal"],
    ),
    (
        "indexavel-ignora-texto-falado",
        "kianda/linguagem/corpus.py",
        '        idx = (prefixo + v) if prefixo else ""',
        '        idx = (prefixo + v[:-40]) if prefixo else ""  # MUTANTE: trunca',
        ["test_indexavel_contem_sempre_o_texto_falado"],
    ),
    (
        "corte-de-evidencia-ignora-a-pergunta",
        "kianda/mente.py",
        "            if len(frases) > 1 and perguntou:",
        "            if False:  # MUTANTE: corte cego por posição",
        ["test_pergunta_sobre_mecanismo_nao_vira_pessoal"],
    ),
    (
        "aviso-de-obras-removido",
        "kianda/mente.py",
        '            self.corpus_info["_por_carregar"] = [aviso]',
        '            pass  # MUTANTE: aviso removido',
        ["test_ausencia_de_obras_e_declarada"],
    ),
]


def correr_suite(seleccao: list[str]) -> bool:
    """True se a suite PASSA (o que significa mutante sobreviveu)."""
    r = subprocess.run(
        [PY, "-m", "pytest", "testes/", "-q", "-x", "--no-header",
         "-p", "no:cacheprovider", *seleccao],
        cwd=RAIZ, capture_output=True, text=True,
    )
    return r.returncode == 0


def main() -> int:
    print("=" * 74)
    print("MUTATION TESTING — os testes detectam os defeitos que os motivaram?")
    print("=" * 74)

    sobreviventes = []
    for nome, ficheiro, alvo, troca, testes in MUTANTES:
        caminho = RAIZ / ficheiro
        original = caminho.read_text()

        if alvo not in original:
            print(f"  IGNORADO {nome}: alvo não encontrado (código mudou)")
            sobreviventes.append((nome, "alvo ausente"))
            continue

        backup = Path(tempfile.mkdtemp()) / caminho.name
        shutil.copy(caminho, backup)
        try:
            caminho.write_text(original.replace(alvo, troca, 1))
            if testes == ["_reservado"]:
                # O conjunto reservado é a barreira que detecta estes dois
                # mutantes: não há teste unitário que os apanhe, porque o
                # defeito é "o sistema não percebe a pergunta".
                r = subprocess.run([PY, "testes/conjunto_reservado.py"],
                                   cwd=RAIZ, capture_output=True, text=True)
                linha = [l for l in r.stdout.splitlines() if l.startswith("CONJUNTO")]
                passou = False  # reprovar a versão base é o normal
                if linha:
                    # formato: "CONJUNTO RESERVADO v1: 12/15 = 80.0%"
                    val = linha[0].split(":")[1].strip()
                    ok = int(val.split("/")[0])
                    passou = ok >= 12  # linha de base medida: 12/15
                print(f"    reservado: {linha[0] if linha else 'sem saida'}")
            else:
                passou = correr_suite([f"testes/test_adversarial.py::{'::'.join(t.split('::'))}"
                                       for t in testes])
            if passou:
                print(f"  SOBREVIVEU  {nome}  <- teste não detecta este defeito")
                sobreviventes.append((nome, "sobreviveu"))
            else:
                print(f"  MATADO      {nome}")
        finally:
            shutil.copy(backup, caminho)

    print("-" * 74)
    total = len(MUTANTES)
    mortos = total - len(sobreviventes)
    print(f"{mortos}/{total} mutantes mortos")
    if sobreviventes:
        print("\nTestes a reforçar:")
        for n, m in sobreviventes:
            print(f"  - {n} ({m})")
        return 1
    print("Todos os defeitos conhecidos são detectados pela suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
