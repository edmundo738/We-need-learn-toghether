"""Auditoria técnica independente — Fase 2 e cadeia completa.

Este script NÃO treina nada e NÃO altera o modelo. Só mede e verifica.
Cada verificação devolve um resultado classificado. A classificação é o ponto:
o relatório anterior misturou hipóteses com factos, e esta auditoria existe
para corrigir isso.

Classificações usadas:
    FACT       demonstrado por construção/símbolo (ex.: contagem de parâmetros)
    MEASURED   número obtido por execução neste ambiente, com comando reproduzível
    OBSERVED   visto numa execução concreta, mas depende dessa execução
    INFERRED   dedução a partir de medições, não medida directamente
    HYPOTHESIS proposta não demonstrada
    UNKNOWN    não sabemos
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]   # nucleo/
RAIZ_REPO = RAIZ.parent                      # raiz do repositório
res: list[dict] = []


def regista(id_: str, pergunta: str, classe: str, valor, nota: str = "") -> None:
    res.append({"id": id_, "pergunta": pergunta, "classe": classe,
                "valor": valor, "nota": nota})
    print(f"[{classe:<10}] {id_:<28} {valor}")


# ==========================================================================
# A. GIT E REPRODUTIBILIDADE
# ==========================================================================

def auditar_git() -> None:
    def _run(*a):
        try:
            return subprocess.run(a, cwd=RAIZ, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    commits = _run("git", "log", "--oneline", "-1")
    regista("A1.head", "qual é o commit actual", "MEASURED", commits or "vazio")

    reportados = ["edf5b54", "592977c", "8e70ed6", "63cddb1", "1b7cbd7", "2323f76"]
    faltam = []
    for c in reportados:
        if _run("git", "cat-file", "-t", c) != "commit":
            faltam.append(c)
    regista("A2.commits_em_falta",
            "dos commits reportados, quantos existem",
            "MEASURED", f"{len(faltam)}/{len(reportados)} NÃO existem",
            "o repositório foi re-clonado; o trabalho só existe em ficheiros por versionar")

    nao_versionados = _run("git", "status", "--porcelain").splitlines()
    regista("A3.ficheiros_sem_commit", "quantos ficheiros não estão em git",
            "MEASURED", len(nao_versionados))


# ==========================================================================
# B. CONTAGEM DE PARÂMETROS
# ==========================================================================

def auditar_parametros() -> None:
    from nucleo.modelo.transformer import Config, Transformer

    # B1: a fórmula e a construção coincidem?
    for nome, cfg in (("pequena", Config(vocab=64, d_model=32, n_camadas=2,
                                        n_cabecas=2, ctx=12, d_ff=64)),
                      ("directiva", Config())):
        m = Transformer(cfg, semente=0)
        igual = cfg.n_parametros() == m.n_parametros()
        regista(f"B1.{nome}", f"fórmula == construção ({nome})", "FACT",
                f"{cfg.n_parametros():,} == {m.n_parametros():,} -> {igual}")

    # B2: decomposição — de onde vêm os parâmetros
    c = Config()
    emb = c.vocab * c.d_model
    pos = c.ctx * c.d_model
    por_camada = c.d_model * c.d_model * 4 + 2 * c.d_model * c.d_ff + 4 * c.d_model
    cabeca = c.d_model * c.vocab
    total = emb + pos + c.n_camadas * por_camada + 2 * c.d_model + cabeca
    regista("B2.decomposicao", "fracção de parâmetros em matrizes de vocabulário",
            "FACT", f"{(emb+cabeca)/total*100:.1f}% "
                    f"({emb+cabeca:,} de {total:,})",
            "as duas matrizes de vocabulário dominam o modelo")

    # B3: porque 2787584 e 2771200 no treino real
    regista("B3.divergencia", "porque o treino real tinha 2 771 200",
            "FACT",
            "ctx=128 em vez de 256: tabela de posição 128*128=16 384 "
            "em vez de 256*128=32 768. 32 768-16 384 = 16 384 de diferença "
            "e 2 787 584-16 384 = 2 771 200. Mesma configuração, contexto menor.")

    # B4: reprodutibilidade da contagem com seeds diferentes
    ns = {Transformer(Config(vocab=64, d_model=32, n_camadas=2, n_cabecas=2,
                             ctx=12, d_ff=64), semente=s).n_parametros()
          for s in range(5)}
    regista("B4.seed_nao_afecta", "a contagem depende da seed", "FACT",
            f"{len(ns)} valor(es) distintos em 5 seeds -> {ns}")


# ==========================================================================
# C. AUDITORIA DA ATENÇÃO — as quatro verificações exigidas
# ==========================================================================

def auditar_atencao() -> None:
    from nucleo.modelo.transformer import Config, Transformer

    cfg = Config(vocab=32, d_model=16, n_camadas=1, n_cabecas=2, ctx=10, d_ff=32)
    m = Transformer(cfg, semente=7)
    rng = np.random.default_rng(0)

    # C1 — alterar a sequência A não muda nenhuma saída de B
    a = rng.integers(0, cfg.vocab, (1, 6))
    b = rng.integers(0, cfg.vocab, (1, 6))
    juntas_ref = m.forward(np.concatenate([a, b], axis=0)).data.copy()
    a2 = rng.integers(0, cfg.vocab, (1, 6))
    juntas_novo = m.forward(np.concatenate([a2, b], axis=0)).data
    delta_b = float(np.abs(juntas_novo[6:] - juntas_ref[6:]).max())
    regista("C1.isolamento_A_nao_afecta_B",
            "mudar A muda alguma saída de B", "MEASURED",
            f"max|Δ| em B = {delta_b:.3e}",
            "0.0 exacto: a máscara impede, não é aproximação")

    # C2 — resultado no lote == resultado isolado
    so_b = m.forward(b).data
    delta_iso = float(np.abs(juntas_ref[6:] - so_b).max())
    regista("C2.lote_igual_isolado", "lote == processado isoladamente",
            "MEASURED", f"max|Δ| = {delta_iso:.3e}",
            "diferença de ordem de somatório em fp32, não vazamento")

    # C3 — nenhuma posição futura influencia anterior (causalidade)
    ids = rng.integers(0, cfg.vocab, (1, 8))
    ref = m.forward(ids).data.copy()
    viol = []
    for k in range(1, 8):
        alt = ids.copy()
        alt[0, k] = (alt[0, k] + 13) % cfg.vocab
        novo = m.forward(alt).data
        if not np.allclose(ref[:k], novo[:k], atol=1e-6):
            viol.append(k)
    regista("C3.causalidade", "posições futuras influenciam anteriores",
            "MEASURED", f"{len(viol)} violações em 7 posições",
            "0 violações = causalidade estrita")

    # C4 — os gradientes também respeitam o isolamento.
    #
    # A primeira versão deste teste estava ERRADA: calculava a perda sobre as
    # 12 posições (A e B) e comparava gradientes entre dois A diferentes. A
    # perda incluía A, por isso o gradiente DEPENDIA de A — correctamente. O
    # teste media o que não devia e deu 5,71.
    # A versão certa isola a perda nas 6 posições de B (Tensor.fatiar) e só
    # então compara.
    from nucleo.modelo.transformer import perda_entropia_cruzada
    alvos_b = rng.integers(0, cfg.vocab, 6)

    def _grad_com_a(a_seq):
        m_ = Transformer(cfg, semente=7)
        for t in m_.parametros().values():
            t.zero_grad()
        lg = m_.forward(np.concatenate([a_seq, b], axis=0))
        perda_entropia_cruzada(lg.fatiar(6, 12), alvos_b).backward()
        return {k: (None if t.grad is None else t.grad.copy())
                for k, t in m_.parametros().items()}

    g1 = _grad_com_a(rng.integers(0, cfg.vocab, (1, 6)))
    g2 = _grad_com_a(rng.integers(0, cfg.vocab, (1, 6)))
    maior = 0.0
    for k in g1:
        if g1[k] is not None and g2[k] is not None:
            maior = max(maior, float(np.abs(g1[k] - g2[k]).max()))
    regista("C4.gradiente_isolado",
            "gradiente de B depende da sequência A", "MEASURED",
            f"max|Δ| entre A e A' = {maior:.3e}",
            "0.0 exacto = o gradiente também respeita o isolamento. "
            "A 1ª versão do teste dava 5.71 porque media a perda de A também")

    # C5 — a máscara é triangular por construção
    m.forward(np.zeros((1, 5), dtype=np.int64))
    ms = m.blocos[0].atencao._ultima_mascara
    tri = np.allclose(ms, np.tril(np.ones((5, 5), dtype=bool)))
    regista("C5.mascara_triangular", "a máscara é triangular inferior",
            "FACT", tri, "verificado contra np.tril, não por inspecção")


# ==========================================================================
# D. AUDITORIA DO AUTOGRAD
# ==========================================================================

def auditar_autograd() -> None:
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada

    cfg = Config(vocab=48, d_model=24, n_camadas=2, n_cabecas=2, ctx=10, d_ff=48)
    m = Transformer(cfg, semente=5)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg.vocab, (1, 6))
    alvos = rng.integers(0, cfg.vocab, 6)

    def perda() -> float:
        return float(perda_entropia_cruzada(m.forward(ids), alvos).data.reshape(-1)[0])

    for t in m.parametros().values():
        t.zero_grad()
    perda_entropia_cruzada(m.forward(ids), alvos).backward()

    # D1 — todos os tensores recebem gradiente
    sem = [k for k, t in m.parametros().items() if t.grad is None]
    regista("D1.todos_com_gradiente", "tensores sem gradiente", "MEASURED",
            f"{len(sem)} de {len(m.parametros())}", "0 é o esperado")

    # D2 — convergência do gradiente numérico com epsilon
    p = m.parametros()
    resultados = {}
    for nome in ("cabeca.W", "emb.peso", "bloco0.atencao.q.W", "bloco0.ln1.gama"):
        t = p[nome]
        idx = np.unravel_index(int(np.argmax(np.abs(t.grad))), t.data.shape)
        analitico = float(t.grad[idx])
        orig = t.data[idx]
        linha = {}
        for eps in (1e-2, 1e-3, 1e-4, 1e-5):
            t.data[idx] = orig + eps
            c1 = perda()
            t.data[idx] = orig - eps
            c2 = perda()
            t.data[idx] = orig
            num = (c1 - c2) / (2 * eps)
            linha[eps] = abs(num - analitico) / max(abs(analitico), 1e-12)
        resultados[nome] = {k: round(v * 100, 4) for k, v in linha.items()}
    regista("D2.erro_relativo_por_eps",
            "erro relativo do gradiente numérico por epsilon (%)",
            "MEASURED", json.dumps(resultados),
            "convergir com eps é a prova; um gradiente errado não converge")

    melhor = {k: min(v.values()) for k, v in resultados.items()}
    regista("D3.melhor_erro", "menor erro relativo alcançado por tensor",
            "MEASURED", json.dumps({k: f"{v:.3f}%" for k, v in melhor.items()}),
            "todos abaixo de 0.1% = gradiente correcto")


# ==========================================================================
# E. PERFORMANCE MEDIDA
# ==========================================================================

def auditar_performance() -> None:
    import os
    import resource

    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada

    cfg = Config(vocab=8192, d_model=128, n_camadas=4, n_cabecas=4,
                 ctx=128, d_ff=384)
    m = Transformer(cfg, semente=0)
    from nucleo import nn
    opt = nn.Adam(m.parametros(), eta=1e-3)

    # E1 — throughput real de um passo
    for lote in (1, 2, 4, 8):
        d = np.zeros((lote, cfg.ctx + 1), dtype=np.int64)
        # aquecimento
        opt.zero_grad()
        perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
        opt.step()
        rep = 3
        t0 = time.perf_counter()
        for _ in range(rep):
            opt.zero_grad()
            perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
            opt.step()
        dt = (time.perf_counter() - t0) / rep
        tokens = lote * cfg.ctx
        regista(f"E1.throughput_lote{lote}",
                f"passo com lote {lote} a ctx {cfg.ctx}", "MEASURED",
                f"{dt*1000:.0f} ms/passo | {tokens/dt:.0f} tokens/s "
                f"| {tokens/dt/1000:.3f} k tokens/s")

    # E2 — CPU e memória do ambiente
    ncpu = os.cpu_count()
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    regista("E2.ambiente", "CPU e RSS pico deste processo", "MEASURED",
            f"{ncpu} CPUs | RSS pico {ram:.0f} MB")


# ==========================================================================
# F. BASELINES — o que o Transformer tem de vencer
# ==========================================================================

def auditar_baselines() -> None:
    """Unigrama, bigrama e trigrama com suavização, no mesmo corpus e divisão."""
    from nucleo.tokenizador.bpe import TokenizadorBPE

    
    tok = TokenizadorBPE.carregar(RAIZ_REPO / "nucleo" / "tokenizador" / "vocabulario_v1")
    texto = "\n\n".join(f.read_text(encoding="utf-8", errors="replace")
                        for f in sorted((RAIZ_REPO / "dados" / "obras").glob("*.txt")))
    cache = Path("/tmp/kianda_ids_corpus.json")
    ids = json.loads(cache.read_text()) if cache.exists() else tok.codificar(texto)
    ids = np.array(ids, dtype=np.int64)
    V = tok.tamanho_vocab

    ctx = 128
    total = len(ids) // (ctx + 1)
    dados = np.empty((total, ctx + 1), dtype=np.int64)
    for i in range(total):
        dados[i] = ids[i * (ctx + 1):(i + 1) * (ctx + 1)]
    corte = int(total * 0.9)
    treino, val = dados[:corte], dados[corte:]

    y_tr = treino[:, 1:].reshape(-1)
    y_va = val[:, 1:].reshape(-1)

    def unigrama():
        c = np.bincount(y_tr, minlength=V).astype(np.float64) + 1.0
        p = c / c.sum()
        return float(-np.log(p[y_va]).mean())

    def ngrama(n: int, alpha: float = 0.1):
        """N-grama com contexto de n-1 tokens, suavizado.

        A versão anterior contava os alvos por posição e colhia um contexto
        de n tokens (devia ser n-1), pelo que bigrama e trigrama quase nunca
        encontravam contexto e caíam no backoff uniforme — resultado: 9.0106
        e 9.0109, praticamente log(V). Estava errado e dava a falsa ideia de
        que os n-gramas não funcionam.
        """
        flat = treino[:, :].reshape(-1)     # fluxo contínuo de treino
        cont: dict[tuple, dict[int, int]] = {}
        for i in range(n - 1, len(flat)):
            ctx_ = tuple(flat[i - n + 1:i]) if n > 1 else ()
            d = cont.setdefault(ctx_, {})
            d[int(flat[i])] = d.get(int(flat[i]), 0) + 1

        vflat = val[:, :].reshape(-1)
        tot, k = 0.0, 0
        for i in range(n - 1, len(vflat)):
            ctx_ = tuple(vflat[i - n + 1:i]) if n > 1 else ()
            d = cont.get(ctx_)
            if d is None:
                p = 1.0 / V
            else:
                total_ctx = sum(d.values())
                p = (d.get(int(vflat[i]), 0) + alpha) / (total_ctx + alpha * V)
            tot += -math.log(p)
            k += 1
        return tot / k

    ppl = lambda l: math.exp(l)
    for nome, val_l in (("unigrama", unigrama()),
                        ("bigrama", ngrama(2)),
                        ("trigrama", ngrama(3))):
        regista(f"F.{nome}", f"perda do modelo de {nome} na validação",
                "MEASURED", f"{val_l:.4f} | PPL {ppl(val_l):.1f}")

    regista("F.uniforme", "perda do baseline uniforme", "FACT",
            f"{math.log(V):.4f} | PPL {V}")
    regista("F.transformer", "perda do Transformer na validação (execução anterior)",
            "OBSERVED", "5,8962 | PPL 363,4",
            "valor do relatório anterior; reproduzir exige retreino")


# ==========================================================================
# G. MEMÓRIA — de onde vem
# ==========================================================================

def auditar_memoria() -> None:
    """Separa o custo da atenção quadrática do custo do grafo do autograd."""
    from nucleo.modelo.transformer import Config, Transformer, perda_entropia_cruzada

    cfg = Config(vocab=8192, d_model=128, n_camadas=4, n_cabecas=4,
                 ctx=64, d_ff=384)

    # G1 — custo teórico da matriz de atenção
    for n_tok in (512, 1024, 2048):
        # (B*T, B*T) fp32 por camada, e save no grafo
        bytes_1 = n_tok * n_tok * 4
        total = bytes_1 * cfg.n_camadas * 2   # forward + gradiente guardado
        regista(f"G1.atencao_{n_tok}tok",
                f"memória teórica só dos scores ({n_tok} tokens)", "FACT",
                f"{total/1048576:.0f} MB",
                "(B*T)^2 * 4 bytes * camadas * 2 (forward+grad)")

    # G2 — memória medida do passo completo
    for lote in (1, 2, 4):
        m = Transformer(cfg, semente=0)
        d = np.zeros((lote, cfg.ctx + 1), dtype=np.int64)
        import resource
        antes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        perda_entropia_cruzada(m.forward(d[:, :-1]), d[:, 1:].reshape(-1)).backward()
        depois = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        regista(f"G2.passo_lote{lote}", f"RSS depois de um passo lote {lote}",
                "MEASURED", f"{depois:.0f} MB (antes {antes:.0f} MB)",
                "inclui parâmetros 10.5 MB + grafo + activações")

    # G3 — parâmetros vs grafo
    m = Transformer(cfg, semente=0)
    params = m.n_parametros() * 4 / 1048576
    regista("G3.parametros", "memória só dos parâmetros", "FACT",
            f"{params:.1f} MB", "o resto é grafo, activações e temporários")


# ==========================================================================
# H. CONTAMINAÇÃO ENTRE TREINO E VALIDAÇÃO
# ==========================================================================

def auditar_contaminacao() -> None:
    """As janelas de 128 *não* se sobrepõem? E há texto repetido no corpus?"""
    from nucleo.tokenizador.bpe import TokenizadorBPE

    
    tok = TokenizadorBPE.carregar(RAIZ_REPO / "nucleo" / "tokenizador" / "vocabulario_v1")
    cache = Path("/tmp/kianda_ids_corpus.json")
    ids = json.loads(cache.read_text()) if cache.exists() else None
    if ids is None:
        texto = "\n\n".join(f.read_text(encoding="utf-8", errors="replace")
                            for f in sorted((RAIZ_REPO / "dados" / "obras").glob("*.txt")))
        ids = tok.codificar(texto)
    ids = np.array(ids, dtype=np.int64)

    ctx = 128
    total = len(ids) // (ctx + 1)
    regista("H1.janelas_nao_sobrepostas",
            f"{total} janelas de {ctx+1} tokens contíguos", "FACT",
            "passo fixo de 129 tokens: sem sobreposição por construção",
            "cada janela usa ids[i*129:(i+1)*129]; a seguinte começa em (i+1)*129")

    # H2 — há 20-gramas repetidos DENTRO do corpus? (contaminação interna
    #      entre uma janela de treino e uma de validação)
    n = 20
    vistos: dict[tuple, list[int]] = {}
    dup = 0
    for i in range(len(ids) - n):
        g = tuple(ids[i:i + n])
        if g in vistos:
            dup += 1
            vistos[g].append(i)
        else:
            vistos[g] = [i]
    regista("H2.ngramas_repetidos_corpus",
            f"{n}-gramas que aparecem mais de uma vez no corpus", "MEASURED",
            f"{dup} de {len(ids)-n} ({dup/(len(ids)-n)*100:.2f}%)",
            "repetição interna pode colocar o mesmo texto em treino e validação")

    # H3 — quantos desses repetidos atravessam o corte treino/validação
    corte = total * 0.9
    fronteira_token = int(corte) * (ctx + 1)
    atravessa = sum(1 for pos in vistos.values() if len(pos) > 1
                    and min(pos) < fronteira_token <= max(pos))
    regista("H3.contaminacao_treino_val",
            "20-gramas que aparecem em treino E em validação", "MEASURED",
            atravessa,
            ">0 significa que o mesmo texto foi visto no treino e medido na validação")


# ==========================================================================
# I. O QUE O RELATÓRIO AFIRMOU E NÃO ESTÁ DEMONSTRADO
# ==========================================================================

def auditar_afirmacoes() -> None:
    # I1 — a aritmética dos "5 tokens/ms"
    passo_s = 0.391
    tokens = 2 * 128
    regista("I1.tokens_por_ms", "throughput real do treino citado", "FACT",
            f"{tokens/passo_s:.0f} tokens/s = {tokens/passo_s/1000:.3f} k tokens/s",
            "o relatório disse '5 tokens/ms'; o correcto é ~0.66 tokens/ms "
            "— errei por um factor de ~7.8")

    # I2 — o factor "50x" do PyTorch nunca foi medido
    regista("I2.pytorch_50x", "evidência para 'PyTorch faria 50x mais depressa'",
            "UNKNOWN",
            "nenhuma: o número foi escrito sem benchmark",
            "afirmação retirada")

    # I3 — a inferência morfológica
    regista("I3.morfologia",
            "'o modelo inventa palavras por regras de formação'", "HYPOTHESIS",
            "não demonstrado",
            "colagens de subpalavras BPE não provam morfologia; exigiria "
            "testes de flexão, concordância e palavras novas em contexto")

    # I4 — a inferência de memorização
    regista("I4.memorizacao", "'0% de 20-gramas = não memoriza'", "HYPOTHESIS",
            "só 20-gramas exactos foram testados",
            "memorização pode existir a 5, 10 ou 15 tokens, ou em forma "
            "aproximada; o que se mediu foi 'não há 20-gramas exactos'")

    # I5 — os "58 752 tokens vistos"
    regista("I5.tokens_vistos", "'58 752 tokens vistos'", "FACT",
            "459 janelas x 128 = 58 752 posições ÚNICAS no conjunto de treino",
            "não é exposição: com 1500 passos x lote 2 x 128 o modelo "
            "processou 384 000 posições, ou seja 6.5 passagens pelo corpus")

    # I6 — os '0.02% do necessário'
    regista("I6.percentagem", "'o corpus é 0.02% do necessário'", "UNKNOWN",
            "o cálculo nunca foi apresentado; não há regra universal",
            "Chinchilla dá ~20 tokens/parâmetro como heurística de CÁLCULO "
            "óptimo, não como lei. Aplicada aqui: 2.77M*20 = 55.5M tokens, "
            "e o corpus tem 66k, ou seja 0.12% — mas é heurística, "
            "não necessidade demonstrada")

    # I7 — a divergência do corpus
    regista("I7.corpus_253589_vs_244372", "corpus de 253 589 ou 244 372 chars",
            "FACT",
            "253 589 = tamanho dos .txt em disco; 244 372 = tamanho depois "
            "da limpeza (_limpar) que remove hífenes de quebra e linhas "
            "vazias repetidas",
            "as duas contagens são correctas para coisas diferentes; o "
            "relatório devia ter dito qual estava a usar")

    # I8 — a divisão 90/10 por janelas contíguas
    regista("I8.divisao_contigua", "divisão 90/10 contígua por janelas",
            "OBSERVED",
            "mede continuação temporal do MESMO autor, não generalização "
            "linguística",
            "não há conjunto de teste independente nem autor diferente")


def main() -> int:
    print("=" * 78)
    print("AUDITORIA — FASE 2 E CADEIA COMPLETA")
    print("=" * 78)

    for nome, fn in (
        ("A. Git e reprodutibilidade", auditar_git),
        ("B. Parâmetros", auditar_parametros),
        ("C. Atenção (4 verificações)", auditar_atencao),
        ("D. Autograd", auditar_autograd),
        ("E. Performance", auditar_performance),
        ("G. Memória", auditar_memoria),
        ("H. Contaminação", auditar_contaminacao),
        ("I. Afirmações do relatório", auditar_afirmacoes),
    ):
        print(f"\n--- {nome} " + "-" * (60 - len(nome)))
        try:
            fn()
        except Exception as e:
            regista(nome, "auditoria falhou", "UNKNOWN", f"{type(e).__name__}: {e}")

    print(f"\n--- F. Baselines " + "-" * 50)
    try:
        auditar_baselines()
    except Exception as e:
        regista("F.baselines", "baselines falharam", "UNKNOWN", f"{type(e).__name__}: {e}")

    saida = RAIZ / "experimentos" / "resultado_auditoria.json"
    saida.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n{len(res)} verificações gravadas em {saida}")

    from collections import Counter
    c = Counter(r["classe"] for r in res)
    print("classificação:", dict(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
