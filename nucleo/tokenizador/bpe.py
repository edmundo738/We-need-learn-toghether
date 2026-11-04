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
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# Limite prático: com 256 símbolos base, o vocabulário final é 256 + fusões.
# Um vocabulário de 4096 tokens dá 3840 fusões aprendidas.
VOCAB_BASE = 256

# --------------------------------------------------------------------------
# Pré-tokenização. A decisão mais importante do ficheiro.
# --------------------------------------------------------------------------
#
# Sem pré-tokenização, o BPE aprendeu fusões que atravessam fronteiras de
# palavra («de uma », « uma »). Duas consequências medidas:
#
#   1. a compressão em texto de treino era boa (1,223 tokens/palavra) mas a
#      cache por palavra era impossível — segmentar por palavra impede as
#      fusões de fronteira, e a compressão piorou 92% (24 -> 46 tokens);
#   2. o espaço gasto em fragmentos de frase nunca se reaproveita: no hold-out
#      a cobertura de palavra inteira caía para 32,4%.
#
# Com pré-tokenização, cada bloco é a unidade atómica. As fusões só acontecem
# dentro de um bloco, por isso a segmentação de um bloco é independente do que
# está à volta — e a cache por bloco passa a ser exacta, não uma aproximação.
#
# O espaço fica colado à esquerda, como no GPT-2: «de» e « de» são tokens
# diferentes e essa diferença é informação (início de palavra).
# A armadilha, e ela custou um round-trip falhado: `\w` inclui o underscore,
# mas `[^\W\d_]` exclui-o da classe de palavra (é o idioma habitual para não
# apanhar identificadores). Resultado: `_` não casava em nenhuma das quatro
# alternativas e **23 underscores desapareciam em silêncio** no corpus do
# Bechara, cheio de lacunas para preencher («Estive três ___ em Fortaleza»).
#
# A correcção é separar explicitamente o underscore como símbolo. A lição é
# maior do que o bug: uma regex de pré-tokenização tem de ter a propriedade
# `"".join(findall(t)) == t` para todo o `t`, e isso testa-se, não se assume.
_BLOCO = re.compile(
    r" ?[^\W\d_]+"      # palavra com espaço opcional à esquerda
    r"| ?\d+"           # número
    r"| ?_+"             # underscores (lacunas, identificadores)
    r"| ?[^\s\w]+"     # pontuação / símbolos
    r"|\s+",            # espaços restantes (incluindo quebras de linha)
    re.UNICODE,
)


def _tem_par(ids: list[int], a: int, b: int) -> bool:
    """True se o par (a, b) aparece adjacente. Evita fundir blocos inúteis."""
    for i in range(len(ids) - 1):
        if ids[i] == a and ids[i + 1] == b:
            return True
    return False


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
    _cache_palavras: dict[str, list[int]] = field(default_factory=dict, repr=False,
                                                  compare=False)
    _cache_versao: int = field(default=-1, repr=False, compare=False)

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

        **Porque este método parece complicado.** A primeira versão percorria a
        lista de blocos a cada fusão: O(fusões × blocos). Com 254 KB e 1 200
        fusões eram 35 s; com os 3 MB do corpus v0.1 e 3 840 fusões, **mais de
        28 minutos sem terminar**. Um tokenizador que não acompanha o corpus é
        inútil exactamente quando o corpus cresce.

        A correcção é a das implementações de produção: contar **blocos únicos
        com a sua frequência** e fundir sobre os tipos, não sobre as ocorrências.
        A língua repete-se muito — 490 810 palavras no corpus, mas poucos
        milhares de tipos distintos — e uma fusão muda o tipo uma vez, não
        centenas de vezes. O custo passa de O(fusões × ocorrências) para
        O(fusões × tipos), que é duas ordens de grandeza menor.

        Isto é possível precisamente por causa da pré-tokenização: como nenhuma
        fusão cruza fronteiras de bloco, cada tipo pode ser fundido em
        isolamento e a contagem multiplicada pela frequência. Sem
        pré-tokenização esta optimização seria incorrecta.
        """
        if vocab_alvo <= VOCAB_BASE:
            raise ValueError(f"vocab_alvo tem de ser > {VOCAB_BASE}")

        self._reiniciar_base()

        # Tipos: bloco -> lista de ids. Frequência: bloco -> ocorrências.
        contagem_blocos: Counter = Counter(_BLOCO.findall(texto))
        if not contagem_blocos:
            raise ValueError("corpus vazio: não há nada para treinar")
        tipos: dict[str, list[int]] = {
            b: list(b.encode("utf-8")) for b in contagem_blocos
        }
        if verbose:
            print(f"  {sum(contagem_blocos.values()):,} blocos, "
                  f"{len(tipos):,} tipos distintos")

        n_fusões = vocab_alvo - VOCAB_BASE
        for passo in range(n_fusões):
            # Pares ponderados pela frequência do tipo que os contém. Só se
            # conta dentro de cada tipo: a pré-tokenização está aqui.
            pares: Counter = Counter()
            for bloco, ids in tipos.items():
                f = contagem_blocos[bloco]
                for par in zip(ids, ids[1:]):
                    pares[par] += f
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

            # Funde nos tipos que contêm o par — nunca nas ocorrências.
            for bloco, ids in tipos.items():
                if a in ids and _tem_par(ids, a, b):
                    tipos[bloco] = self._fundir(ids, (a, b), novo_id)

            if verbose and (passo + 1) % 500 == 0:
                print(f"  fusão {passo + 1}/{n_fusões} — vocab {self.tamanho_vocab}")

        self.treinado = True
        self.metadados = {
            "vocab_alvo": vocab_alvo,
            "vocab_final": self.tamanho_vocab,
            "fusoes": len(self.fusões),
            "caracteres_corpus": len(texto),
            "bytes_corpus": len(texto.encode("utf-8")),
            "blocos": sum(contagem_blocos.values()),
            "tipos_de_bloco": len(contagem_blocos),
            "pre_tokenizacao": True,
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
        if self._cache_versao != len(self.fusões):
            self._cache_versao = len(self.fusões)
            self._cache_palavras.clear()

        saida: list[int] = []
        for bloco in _BLOCO.findall(texto):
            ids = self._cache_palavras.get(bloco)
            if ids is None:
                ids = self._segmentar_bloco(bloco)
                self._cache_palavras[bloco] = ids
            saida.extend(ids)
        return saida

    def _segmentar_bloco(self, bloco: str) -> list[int]:
        """Segmenta um bloco isolado. Igual ao laço antigo, mas por bloco.

        É a mesma operação de antes; a diferença é que agora é aplicada a uma
        unidade pequena e o resultado é memorizável. A compressão mantém-se
        porque as fusões aprendidas também só cruzam símbolos dentro de blocos.
        """
        ids = list(bloco.encode("utf-8"))
        for novo_id, par in enumerate(self.fusões, start=VOCAB_BASE):
            if par[0] in ids or par[1] in ids:
                ids = self._fundir(ids, par, novo_id)
        return ids

    def codificar_ficheiro(self, caminho, cache: dict | None = None) -> list[int]:
        """Codifica um ficheiro, atravessando-o uma vez com cache de linha.

        LIMITAÇÃO CONHECIDA, e é uma medida, não uma desculpa: `codificar` é
        O(n × fusões). Com 254 KB termina em segundos; com os 3 MB do corpus
        completo não terminou em 15 minutos.

        Tentei a correcção clássica — cache por palavra, como faz o GPT-2 — e
        **piorou a compressão em 92%** (24 → 46 tokens na mesma frase). A razão
        é uma descoberta sobre este tokenizador, não sobre o código: foi treinado
        **sem pré-tokenização**, por isso aprendeu fusões que atravessam
        fronteiras de palavra (« uma », «de uma »). Segmentar por palavra
        impede-o de as usar. A cache por palavra só é correcta num BPE treinado
        com pré-tokenização.

        Consequência para a Fase 2: o tokenizador deve ser **retreinado com
        pré-tokenização**. Isso torna a cache correcta (rápida), e deve também
        melhorar a cobertura de hold-out de 32%, porque o vocabulário passará a
        gastar-se em morfologia («coraç», «ão», «ões») em vez de fragmentos de
        frase que nunca se reaproveitam.
        """
        from pathlib import Path

        cache = cache if cache is not None else {}
        saida: list[int] = []
        for linha in Path(caminho).read_text(encoding="utf-8", errors="replace").splitlines(True):
            ids = cache.get(linha)
            if ids is None:
                ids = self.codificar(linha)
                cache[linha] = ids
            saida.extend(ids)
        return saida

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
