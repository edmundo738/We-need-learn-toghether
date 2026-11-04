"""Testes do tokenizador BPE — critérios de aceitação da Fase 1.

A directiva exige seis condições para a Fase 1. Este ficheiro testa cada uma:

| Condição (directiva §5 de resposta)        | Teste neste ficheiro                     |
|--------------------------------------------|------------------------------------------|
| 2. testes de codificação e descodificação  | `test_roundtrip_*`                       |
| 3. preservar acentos/pontuação/especiais   | `test_preserva_*`                        |
| 4. guardar vocabulário e metadados         | `test_guarda_e_carrega`                  |
| 6. testar texto fora do treino             | `test_roundtrip_validacao_holdout`       |
| (1. não alterar sistema principal)         | implícito: o pacote não importa `kianda` |

O teste de fuzz gera combinações aleatórias de caracteres Unicode, incluindo
os que o corpus nunca viu. É o único teste que prova a propriedade de forma
geral, e não por exemplos escolhidos por mim — que é a diferença entre testar
o que programei e testar o que afirmo.

Verificação:
    /home/user/.venv/bin/pytest nucleo/testes/test_tokenizador.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from nucleo.tokenizador.bpe import VOCAB_BASE, TokenizadorBPE  # noqa: E402

VALIDACAO = RAIZ / "nucleo" / "dados" / "validacao_tokenizador.txt"


def _texto_validacao() -> str:
    linhas = VALIDACAO.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in linhas if l.strip() and not l.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def corpus_pequeno() -> str:
    """Corpus de treino das fusões — separado do texto de validação.

    Só as obras entram aqui. O ficheiro de validação é lido no teste e nunca
    é dado ao `treinar()`, que é o que torna a condição 6 verificável em vez
    de declarada.
    """
    from kianda.linguagem import corpus

    return corpus.construir_tudo()["texto_treino"]


@pytest.fixture(scope="module")
def tok(corpus_pequeno: str) -> TokenizadorBPE:
    t = TokenizadorBPE()
    t.treinar(corpus_pequeno, vocab_alvo=1200)
    return t


# --------------------------------------------------------------------------
# Condição 2 e 6 — codificação e descodificação, incluindo texto não visto
# --------------------------------------------------------------------------


def test_treinou_alguma_coisa(tok: TokenizadorBPE, corpus_pequeno: str) -> None:
    assert tok.treinado
    assert tok.tamanho_vocab > VOCAB_BASE + 100, (
        f"só {tok.tamanho_vocab - VOCAB_BASE} fusões aprendidas: o corpus pode "
        f"estar vazio ou o limite demasiado baixo"
    )
    assert len(corpus_pequeno) > 100_000


def test_roundtrip_corpus_de_treino(tok: TokenizadorBPE, corpus_pequeno: str) -> None:
    """O corpus inteiro sobrevive ao ciclo texto → ids → texto, sem excepção."""
    ids = tok.codificar(corpus_pequeno)
    assert tok.descodificar(ids) == corpus_pequeno


def test_roundtrip_validacao_holdout(tok: TokenizadorBPE) -> None:
    """Texto que nunca viu o treino. Condição 6 da directiva."""
    texto = _texto_validacao()
    assert len(texto) > 500, "o ficheiro de validação encolheu"
    ids = tok.codificar(texto)
    assert tok.descodificar(ids) == texto


def test_roundtrip_linha_a_linha_validacao(tok: TokenizadorBPE) -> None:
    """Falha por linha, para localizar a corrupção quando acontecer."""
    for linha in _texto_validacao().splitlines():
        assert tok.descodificar(tok.codificar(linha)) == linha, f"corrompeu: {linha!r}"


# --------------------------------------------------------------------------
# Condição 3 — acentos, pontuação, caracteres especiais e não-ASCII
# --------------------------------------------------------------------------

CASOS_ESPECIAIS = [
    "coração", "ação", "órgão", "açúcar", "êxito", "ímpar", "úlcera",
    "ãõẽ", "çÇ", "«aspas portuguesas»", "— travessão", "… reticências",
    "1.º", "2.ª", "3.º", "R$", "42%", "3,14159", "10:30", "2026-11-01",
    "H99", "Z.M.", "KIANDA", "kixi", "musseque", "capulana", "Sambizanga",
    "ñ", "ü", "ß", "日本語", "Ελληνικά", "🙂", "→", "\t", "\n",
    " ", "a", "", "  espaços  duplos  ",
]


@pytest.mark.parametrize("texto", CASOS_ESPECIAIS)
def test_preserva_caracteres_especiais(tok: TokenizadorBPE, texto: str) -> None:
    assert tok.descodificar(tok.codificar(texto)) == texto


def test_nao_ha_perda_de_bytes(tok: TokenizadorBPE) -> None:
    """O número de bytes descodificados tem de bater com o codificado."""
    for texto in CASOS_ESPECIAIS:
        n_bytes = len(texto.encode("utf-8"))
        assert len(b"".join(tok.vocab[i] for i in tok.codificar(texto))) == n_bytes, (
            f"perda de bytes em {texto!r}"
        )


# --------------------------------------------------------------------------
# Propriedade geral — fuzz. O único teste que não fui eu a escolher.
# --------------------------------------------------------------------------


def test_fuzz_roundtrip(tok: TokenizadorBPE) -> None:
    """10 000 strings aleatórias de Unicode arbitrário.

    Cobre o que nenhum caso à mão cobre: combinações de caracteres que o
    corpus não tem, incluindo pares surrogados já resolvidos pelo Python e
    símbolos de planos altos.
    """
    rng = random.Random(20261101)
    alfabeto = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ"
        "0123456789 \n\t.,;:!?«»—…€$%&()[]{}"
        "ñßαβγ日🙂→"
    )
    for _ in range(10_000):
        n = rng.randint(0, 60)
        s = "".join(rng.choice(alfabeto) for _ in range(n))
        assert tok.descodificar(tok.codificar(s)) == s, f"fuzz falhou em {s!r}"


def test_fuzz_code_points_arbitrarios(tok: TokenizadorBPE) -> None:
    """Code points fora de qualquer alfabeto, incluindo o plano astral."""
    rng = random.Random(7)
    for _ in range(500):
        pontos = [rng.randint(0x20, 0x10FFFF) for _ in range(rng.randint(1, 20))]
        # Evita a faixa de surrogados, que não é UTF-8 válido.
        pontos = [p for p in pontos if not (0xD800 <= p <= 0xDFFF)]
        s = "".join(chr(p) for p in pontos)
        assert tok.descodificar(tok.codificar(s)) == s


# --------------------------------------------------------------------------
# Condição 4 — guardar e recarregar o vocabulário treinado
# --------------------------------------------------------------------------


def test_guarda_e_carrega(tok: TokenizadorBPE, tmp_path: Path) -> None:
    alvo = tok.guardar(tmp_path, nota="teste de persistência")
    assert alvo.exists()
    assert (tmp_path / "vocabulario.txt").exists(), "falta o vocabulário legível"

    outro = TokenizadorBPE.carregar(tmp_path)
    assert outro.tamanho_vocab == tok.tamanho_vocab
    assert outro.fusões == tok.fusões

    texto = _texto_validacao()
    assert outro.codificar(texto) == tok.codificar(texto), (
        "o vocabulário recarregado segmenta de forma diferente do original"
    )
    assert outro.descodificar(outro.codificar(texto)) == texto


# --------------------------------------------------------------------------
# Propriedades estruturais
# --------------------------------------------------------------------------


def test_ids_dentro_do_vocabulario(tok: TokenizadorBPE, corpus_pequeno: str) -> None:
    amostra = corpus_pequeno[:5000] + _texto_validacao()
    ids = tok.codificar(amostra)
    assert all(0 <= i < tok.tamanho_vocab for i in ids)


def test_compressao_melhor_que_bytes(tok: TokenizadorBPE, corpus_pequeno: str) -> None:
    """Se o BPE não comprime, não está a aprender nada de útil."""
    n_bytes = len(corpus_pequeno.encode("utf-8"))
    n_tokens = len(tok.codificar(corpus_pequeno))
    razao = n_tokens / n_bytes
    assert razao < 0.75, f"compressão fraca: {razao:.3f} tokens por byte"


def test_vocab_consistente(tok: TokenizadorBPE) -> None:
    """Cada id do vocabulário mapeia para bytes, sem buracos nem colisões."""
    assert len(tok.vocab) == tok.tamanho_vocab
    assert sorted(tok.vocab) == list(range(tok.tamanho_vocab))
    assert len(set(tok.vocab.values())) == tok.tamanho_vocab, "bytes duplicados"
    for novo_id, (a, b) in enumerate(tok.fusões, start=VOCAB_BASE):
        assert tok.vocab[novo_id] == tok.vocab[a] + tok.vocab[b]


def test_pre_tokenizacao_e_injectiva() -> None:
    """A propriedade que um bug real quebrou.

    A regex de pré-tokenização tem de cumprir `"".join(findall(t)) == t` para
    todo o `t`. Quando não cumpria, 23 underscores desapareciam em silêncio do
    corpus do Bechara — o round-trip falhava, mas com uma diferença de 23
    caracteres em 2,99 milhões, o que nenhum teste de amostra apanharia.

    Este teste compara o comprimento e o conteúdo, e falha com a primeira
    divergência localizada para diagnóstico.
    """
    from nucleo.tokenizador.bpe import _BLOCO

    casos = [
        "Estive três   \n ___________   \nem Fortaleza",
        "a_b __init__ x_1", "日本語 🙂", "", "   ", "a\tb\nc",
        "©®™ § ¶", "«aspas» — travessão…", "1.º 2.ª 3,14159",
    ]
    for c in casos:
        reconstruido = "".join(_BLOCO.findall(c))
        assert reconstruido == c, f"perda na pré-tokenização: {c!r} -> {reconstruido!r}"


def test_pre_tokenizacao_no_corpus_completo() -> None:
    """A mesma propriedade sobre o corpus inteiro, se os ficheiros existirem.

    Salta se o corpus não estiver no disco (é gitignored): um teste que falha
    por ausência de dados que não são versionados seria ruído, não sinal.
    """
    from pathlib import Path

    from nucleo.tokenizador.bpe import _BLOCO

    pasta = RAIZ / "dados"
    if not pasta.exists():
        pytest.skip("corpus não está no disco (dados/ é gitignored)")
    for f in sorted(pasta.glob("*/*.txt")):
        texto = f.read_text(encoding="utf-8", errors="replace")
        assert "".join(_BLOCO.findall(texto)) == texto, (
            f"perda de caracteres na pré-tokenização de {f.name}"
        )


def test_nao_importa_o_sistema_principal() -> None:
    """A área experimental tem de ser independente para a prova ser estrutural.

    O tokenizador não pode importar nada de `kianda/`. Isto é o que permite
    afirmar, mais tarde, que o texto do núcleo novo não vem do sistema antigo.
    """
    fonte = (RAIZ / "nucleo" / "tokenizador" / "bpe.py").read_text()
    assert "from kianda" not in fonte
    assert "import kianda" not in fonte
