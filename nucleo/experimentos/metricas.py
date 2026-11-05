"""Uma única fonte para a conversão nats<->bits/byte.

Havia `TOKENS_POR_BYTE = 0.313` copiado à mão em cinco ficheiros. Se o
tokenizador mudar, os bits/byte de todos os experimentos ficam errados em
silêncio e ninguém dá por isso — cada ficheiro continua a produzir "números".

Aqui o valor é MEDIDO a partir do corpus real, uma vez, e guardado. Cada
experimento importa daqui. Se o corpus ou o tokenizador mudar, muda-se num
sítio e os testes apanham a diferença.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
_CACHE = RAIZ / "nucleo" / "experimentos" / "medida_tokens_por_byte.json"


@lru_cache(maxsize=1)
def tokens_por_byte() -> float:
    """Rácio MEDIDO entre tokens e caracteres no corpus de domínio público."""
    if _CACHE.exists():
        return json.loads(_CACHE.read_text())["tokens_por_byte"]

    from nucleo.tokenizador.bpe import TokenizadorBPE
    tok = TokenizadorBPE.carregar(RAIZ / "nucleo" / "tokenizador" / "vocabulario_v1")
    tot_t = tot_c = 0
    for f in sorted((RAIZ / "dados" / "publico_limpo").glob("*.txt")):
        t = f.read_text(encoding="utf-8")
        tot_c += len(t)
        tot_t += len(tok.codificar(t))
    valor = tot_t / max(tot_c, 1)
    _CACHE.write_text(json.dumps({
        "tokens_por_byte": round(valor, 6),
        "tokens": tot_t, "caracteres": tot_c,
        "corpus": "dados/publico_limpo",
        "nota": "medido, não estimado. Cada experimento importa daqui.",
    }, indent=2))
    return valor


def nats_para_bits_por_byte(nats_por_token: float,
                            tpb: float | None = None) -> float:
    """Converte nats/token em bits/byte — a métrica comparável entre tokenizadores."""
    tpb = tokens_por_byte() if tpb is None else tpb
    return (nats_por_token / math.log(2)) / (1.0 / tpb)


def main() -> None:
    tpb = tokens_por_byte()
    print(f"tokens/byte medido: {tpb:.6f}")
    print(f"0,313 escrito à mão em cinco ficheiros vs {tpb:.4f} medido")
    print(f"erro de nats=6,0: {nats_para_bits_por_byte(6.0, 0.313):.4f} (com 0,313) "
          f"vs {nats_para_bits_por_byte(6.0, tpb):.4f} (medido)")


if __name__ == "__main__":
    main()
