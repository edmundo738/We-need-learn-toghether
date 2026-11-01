"""Tokenizer BPE byte-level, treinado do zero.

Porque byte-level e não caracteres
----------------------------------
O português tem acentos que, com caracteres, exigiriam um vocabulário base
grande e a palavra «coração» seria unidade atómica — o modelo nunca aprenderia
que «coração» e «corações» partilham o radical. Com BPE sobre **bytes UTF-8**,
«ç» e «ã» são duas sequências de 2 bytes cada, o vocabulário base são os 256
valores possíveis, e nenhum texto é irrepresentável.

A propriedade que se ganha, e que é o critério de aceitação da Fase 1:
`descodificar(codificar(t)) == t` para **qualquer** `t`, incluindo texto que o
treino nunca viu. Não é «quase»: é exacto, e há um teste que o prova por
fuzz.

O que o treino aprende
----------------------
As fusões (merges). Começa com 256 bytes e aprende, iterativamente, qual o par
de símbolos adjacentes mais frequente e o funde num símbolo novo. O resultado
são subpalavras: «coração» acaba partido em algo como «cor» + «aç» + «ão»,
que é partilhado com «corações».

Dependência: nenhuma além da biblioteca padrão. Um tokenizador é a peça que
mais convém compreender linha a linha, porque todas as medidas de perplexidade
e de custo dependem dele.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# Limite prático: com 256 símbolos base, o vocabulário final é 256 + fusões.
# Um vocabulário de 4096 tokens dá 3840 fusões aprendidas.
VOCAB_BASE = 256


@dataclass
class TokenizadorBPE:
    """BPE byte-level com treino incremental.

    Atributos públicos: `fusões` (lista ordenada de pares), `vocab` (bytes por
    id), `indice` (bytes -> id). Tudo serializável em JSON, porque o
    vocabulário treinado tem de ser um artefacto guardado e auditável — não
    algo que se reconstrói ao arrancar.
    """

    fusões: list[tuple[int, int]] = field(default_factory=list)
    vocab: dict[int, bytes] = field(default_factory=dict)
    indice: dict[bytes, int] = field(default_factory=dict)
    treinado: bool = False
    metadados: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            self._reiniciar_base()

    # -- base -------------------------------------------------------------
    def _reiniciar_base(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(VOCAB_BASE)}
        self.indice = {bytes([i]): i for i in range(VOCAB_BASE)}
        self.fusões = []
        self.treinado = False

    @property
    def tamanho_vocab(self) -> int:
        return len(self.vocab)

    # -- treino -----------------------------------------------------------
    def treinar(self, texto: str, vocab_alvo: int = 4096, verbose: bool = False) -> dict:
        """Aprende fusões até `vocab_alvo` símbolos.

        O algoritmo é o clássico: representar o corpus como lista de ids,
        contar pares adjacentes, fundir o mais frequente, repetir. É O(n) por
        fusão com o corpus em memória; para 254 KB e 3840 fusões o custo é
        aceitável e não vale a pena a maquinaria de índices incrementais.

        `verbose` imprime a cada 500 fusões — para se ver que está vivo.
        """
        if vocab_alvo <= VOCAB_BASE:
            raise ValueError(f"vocab_alvo tem de ser > {VOCAB_BASE}")

        self._reiniciar_base()
        ids = list(texto.encode("utf-8"))
        if not ids:
            raise ValueError("corpus vazio: não há nada para treinar")

        n_fusões = vocab_alvo - VOCAB_BASE
        for passo in range(n_fusões):
            pares = Counter(zip(ids, ids[1:]))
            if not pares:
                if verbose:
                    print(f"  paragem em {passo}: corpus reduzido a um símbolo")
                break
            (a, b), freq = pares.most_common(1)[0]
            if freq < 2:
                if verbose:
                    print(f"  paragem em {passo}: nenhum par repetido")
                break

            novo_id = VOCAB_BASE + len(self.fusões)
            self.fusões.append((a, b))
            self.vocab[novo_id] = self.vocab[a] + self.vocab[b]
            self.indice[self.vocab[novo_id]] = novo_id
            ids = self._fundir(ids, (a, b), novo_id)

            if verbose and (passo + 1) % 500 == 0:
                print(f"  fusão {passo + 1}/{n_fusões} — vocab {self.tamanho_vocab}")

        self.treinado = True
        self.metadados = {
            "vocab_alvo": vocab_alvo,
            "vocab_final": self.tamanho_vocab,
            "fusoes": len(self.fusões),
            "caracteres_corpus": len(texto),
            "bytes_corpus": len(texto.encode("utf-8")),
        }
        return self.metadados

    @staticmethod
    def _fundir(ids: list[int], par: tuple[int, int], novo: int) -> list[int]:
        a, b = par
        saida: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                saida.append(novo)
                i += 2
            else:
                saida.append(ids[i])
                i += 1
        return saida

    # -- codificação ------------------------------------------------------
    def codificar(self, texto: str) -> list[int]:
        """Texto -> ids, aplicando as fusões pela ordem em que foram aprendidas.

        A ordem não é arbitrária: BPE aprendido define uma precedência. Aplicar
        as fusões fora de ordem produz uma segmentação diferente e a
        descodificação deixa de ser exacta.
        """
        if not self.treinado:
            raise RuntimeError("tokenizador não treinado: chame treinar() primeiro")
        ids = list(texto.encode("utf-8"))
        # Cache por par melhora muito a codificação de texto repetido.
        for novo_id, par in enumerate(self.fusões, start=VOCAB_BASE):
            if par[0] in ids or par[1] in ids:
                ids = self._fundir(ids, par, novo_id)
        return ids

    def descodificar(self, ids: list[int]) -> str:
        """Ids -> texto. Reconstrói bytes e decodifica UTF-8.

        `errors="replace"` numa primeira passagem esconderia corrupção; por
        isso deixa-se levantar. Se a descodificação falhar, é um erro e deve
        aparecer, não virar «�» silencioso.
        """
        bruto = b"".join(self.vocab[i] for i in ids)
        return bruto.decode("utf-8")

    def codificar_lote(self, textos: list[str]) -> list[list[int]]:
        return [self.codificar(t) for t in textos]

    # -- persistência -----------------------------------------------------
    def guardar(self, pasta: Path, nota: str = "") -> Path:
        """Guarda vocabulário e metadados como artefacto auditável."""
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        dados = {
            "versao": 1,
            "tipo": "bpe_byte_level",
            "vocab_base": VOCAB_BASE,
            "fusoes": [[a, b] for a, b in self.fusões],
            "tamanho_vocab": self.tamanho_vocab,
            "meta": self.metadados,
            "nota": nota,
        }
        alvo = pasta / "vocabulario.json"
        alvo.write_text(json.dumps(dados, indent=1, ensure_ascii=False))

        # Vocabulário legível, para inspecção humana. Não é usado no código.
        legivel = pasta / "vocabulario.txt"
        linhas = []
        for i in sorted(self.vocab):
            b = self.vocab[i]
            try:
                s = b.decode("utf-8")
                s = repr(s)[1:-1]
            except UnicodeDecodeError:
                s = " ".join(f"\\x{x:02x}" for x in b)
            linhas.append(f"{i}\t{s}")
        legivel.write_text("\n".join(linhas))
        return alvo

    @classmethod
    def carregar(cls, pasta: Path) -> "TokenizadorBPE":
        dados = json.loads((Path(pasta) / "vocabulario.json").read_text())
        t = cls()
        t.fusões = [(a, b) for a, b in dados["fusoes"]]
        t.vocab = {i: bytes([i]) for i in range(VOCAB_BASE)}
        t.indice = {bytes([i]): i for i in range(VOCAB_BASE)}
        for novo_id, (a, b) in enumerate(t.fusões, start=VOCAB_BASE):
            t.vocab[novo_id] = t.vocab[a] + t.vocab[b]
            t.indice[t.vocab[novo_id]] = novo_id
        t.metadados = dados.get("meta", {})
        t.treinado = True
        return t
