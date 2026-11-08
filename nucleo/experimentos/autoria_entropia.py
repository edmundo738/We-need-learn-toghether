"""Atribuição de autoria por entropia cruzada de caracteres, com calibração.

Porquê este método e não o Delta:
  O Delta de Burrows mede distância entre *perfis* e não separou nada neste
  corpus (o controlo de Machado ficou mais perto das obras do que as obras
  estão entre si).  A entropia cruzada mede o que interessa: quanto um modelo
  treinado numa obra se surpreende com este texto.

Calibração — a parte que torna o número interpretável:
  Cortamos cada obra em blocos do tamanho dos textos em questão.  Para cada
  bloco, treinamos o modelo na obra *excluindo* esse bloco (leave-out real,
  não fingido) e medimos a que obra o bloco fica mais próximo.  Isso dá a
  distribuição de referência para "texto do mesmo autor".

  Sem esta calibração, "está 0,2 bits mais perto de X" não significa nada.

Nota metodológica: bits/char menor = modelo surpreende-se menos.

Uso:  python -m nucleo.experimentos.autoria_entropia
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
OBRAS = RAIZ / "dados" / "obras"
AUTOR = OBRAS / "autor"
PUBLICO = RAIZ / "dados" / "publico_limpo"

ORDEM = 5  # n-grama de caracteres
DESCONTO = 0.75
SEMENTE = 20261101


def normalizar(texto: str) -> str:
    t = texto.lower()
    t = re.sub(r"[ \t]+", " ", t)
    return t


def contar(texto: str, ordem: int) -> tuple[dict, dict]:
    ctx: dict[str, dict[str, int]] = {}
    tot: dict[str, int] = {}
    pad = "\x02" * (ordem - 1)
    t = pad + texto
    for i in range(len(t) - ordem + 1):
        g = t[i : i + ordem]
        c, ch = g[:-1], g[-1]
        d = ctx.setdefault(c, {})
        d[ch] = d.get(ch, 0) + 1
        tot[c] = tot.get(c, 0) + 1
    return ctx, tot


def logp(ctx: dict, tot: dict, contexto: str, ch: str, vocab: int) -> float:
    """Absolute discounting com backoff uniforme (simples mas honesto)."""
    p = 1.0 / vocab
    # do contexto mais longo para o mais curto
    for k in range(min(len(contexto), ORDEM - 1), -1, -1):
        c = contexto[-k:] if k else ""
        d = ctx.get(c)
        if not d:
            continue
        n = tot[c]
        cu = d.get(ch, 0)
        if cu:
            return math.log((cu - DESCONTO) / n + DESCONTO * p)
        p = DESCONTO * p if n else p
    return math.log(p)


def entropia(modelo: tuple[dict, dict, int], texto: str) -> float:
    ctx, tot, vocab = modelo
    n = len(texto)
    if n == 0:
        return float("nan")
    pad = "\x02" * (ORDEM - 1)
    t = pad + texto
    # para ser rápido, amostramos no máximo 60k caracteres de forma uniforme
    posicoes = range(ORDEM - 1, len(t))
    if len(t) > 60000:
        passo = max(1, (len(t) - ORDEM) // 60000)
        posicoes = range(ORDEM - 1, len(t), passo)
    total, k = 0.0, 0
    for i in posicoes:
        total += logp(ctx, tot, t[max(0, i - ORDEM + 1) : i], t[i], vocab)
        k += 1
    return -total / (k * math.log(2))


def modelo_de(texto: str, vocab: int):
    ctx, tot = contar(normalizar(texto), ORDEM)
    return (ctx, tot, vocab)


def main() -> None:
    ae = "".join((OBRAS / f"apenas_existo_parte{i}.txt").read_text("utf-8") for i in (1, 2, 3))
    h99 = "".join((OBRAS / f"H99_Parte_{p}.txt").read_text("utf-8") for p in ("I", "II", "III", "IV"))
    luna = (OBRAS / "depths_luna_EXCERTO.txt").read_text("utf-8")

    livros = {
        "AE": ae,
        "H99": h99,
        "LUNA": luna,
    }
    for nome in ("dom_casmurro", "bras_cubas", "cartomante", "alienista"):
        p = PUBLICO / f"{nome}.txt"
        if p.exists():
            livros[f"MACHADO_{nome[:6]}"] = p.read_text("utf-8")

    vocab = 256  # bytes de facto; usamos a contagem de caracteres distintos globais
    todos = "".join(normalizar(v) for v in livros.values())
    vocab = len(set(todos))

    # --- textos em questão ---------------------------------------------------
    querys = {}
    for chave, ficheiro in [
        ("Q_bencao_de_dor", "uma_bencao_de_dor.txt"),
        ("Q_edmundo_sem_ia", "edmundo_sem_ia.txt"),
        ("Q_atraves_de_alguem", "atraves_de_alguem_que_erra.txt"),
        ("Q_aventuras_zulu", "aventuras_aleatorias_da_zulu.txt"),
    ]:
        p = AUTOR / ficheiro
        if p.exists():
            querys[chave] = p.read_text("utf-8")

    modelo = {nome: modelo_de(t, vocab) for nome, t in livros.items()}

    resultado_q = {}
    for q, texto in querys.items():
        bpc = {nome: round(entropia(m, texto), 4) for nome, m in modelo.items()}
        ordenado = sorted(bpc.items(), key=lambda kv: kv[1])
        obra = [k for k in bpc if not k.startswith("MACHADO")]
        melhor_obra = min(obra, key=lambda k: bpc[k])
        resultado_q[q] = {
            "bits_por_char": bpc,
            "obra_mais_proxima": melhor_obra,
            "margem_sobre_a_2a": round(
                sorted(bpc[k] for k in obra)[1] - bpc[melhor_obra], 4
            ),
            "machado_mais_proximo": round(min(bpc[k] for k in bpc if k.startswith("MACHADO")), 4),
        }

    # --- calibração: blocos do próprio livro --------------------------------
    rng = __import__("random").Random(SEMENTE)
    tamanhos = [len(t) for t in querys.values()] or [4000]
    tam = int(statistics.fmean(tamanhos))

    calib = {}
    for alvo in ("AE", "H99"):
        texto = livros[alvo]
        blocos = []
        sobra = texto
        while len(sobra) > tam * 2 and len(blocos) < 8:
            ini = rng.randrange(0, len(sobra) - tam)
            blocos.append(sobra[ini : ini + tam])
            sobra = sobra[:ini] + sobra[ini + tam :]
        acertos, registos = 0, []
        for b in blocos:
            # modelo treinado no resto do próprio livro (leave-out verdadeiro)
            m_self = modelo_de(sobra if len(sobra) > len(b) else texto.replace(b, ""), vocab)
            bpc = {}
            bpc[f"{alvo}_resto"] = entropia(m_self, b)
            for outro in livros:
                if outro == alvo:
                    continue
                bpc[outro] = entropia(modelo[outro], b)
            vencedor = min(bpc, key=bpc.get)
            acertos += vencedor == f"{alvo}_resto"
            registos.append({"vencedor": vencedor, "bits": {k: round(v, 4) for k, v in bpc.items()}})
        calib[alvo] = {"n_blocos": len(blocos), "acertos_self": acertos, "registos": registos}

    saida = {
        "metodo": {
            "n_grama": ORDEM,
            "desconto": DESCONTO,
            "vocabulario": vocab,
            "tamanho_bloco_calibracao": tam,
        },
        "querys": resultado_q,
        "calibracao": calib,
        "interpretacao": (
            "bits/char menor = modelo menos surpreendido. A calibração diz quantas "
            "vezes um bloco de um livro é reconhecido como sendo desse livro quando "
            "o modelo é treinado no resto desse mesmo livro."
        ),
    }

    destino = Path(__file__).with_name("resultado_autoria_entropia.json")
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ordem={ORDEM}  vocab={vocab}  bloco={tam} chars\n")
    for alvo, c in calib.items():
        print(f"CALIBRAÇÃO {alvo}: {c['acertos_self']}/{c['n_blocos']} blocos reconhecidos como {alvo}")
        for r in c["registos"]:
            b = r["bits"]
            print(f"   vencedor={r['vencedor']:14s} " + "  ".join(f"{k}={v}" for k, v in b.items()))
        print()
    print("=== TEXTOS EM QUESTÃO (bits/char) ===")
    for q, r in resultado_q.items():
        print(f"\n{q}  -> mais próximo: {r['obra_mais_proxima']}  (margem {r['margem_sobre_a_2a']} bits)")
        for k, v in sorted(r["bits_por_char"].items(), key=lambda kv: kv[1]):
            print(f"    {k:20s} {v}")
    print(f"\nJSON: {destino}")


if __name__ == "__main__":
    main()
