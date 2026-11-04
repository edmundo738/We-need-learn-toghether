"""Mede o Piper de verdade: carga do modelo, RAM, latência por segundo de áudio."""
import time, os, resource, numpy as np, onnxruntime as ort

caminho = '/tmp/piper_src/etc/test_voice.onnx'
print(f'modelo: {os.path.getsize(caminho)/1048576:.1f} MB')

t0=time.perf_counter()
so = ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
sess = ort.InferenceSession(caminho, sess_options=so, providers=['CPUExecutionProvider'])
t_carga = time.perf_counter()-t0

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(f'carga: {t_carga:.2f} s | RSS pico apos carga: {rss:.0f} MB')
print('entradas:', [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print('saidas:  ', [(o.name, o.shape) for o in sess.get_outputs()])

# ids de entrada: assume fonemas; usar comprimento variável para medir escalabilidade
import json
cfg = json.load(open(caminho+'.json'))
n_sym = len(cfg['phoneme_id_map'])
print(f'simbolos foneticos: {n_sym} | sample_rate: {cfg["audio"]["sample_rate"]}')

for n_fonemas in (20, 50, 100):
    ids = np.array([[1]+[list(range(10, n_sym-2))[i % (n_sym-12)] for i in range(n_fonemas)]+[2]], dtype=np.int64)
    lens = np.array([ids.shape[1]], dtype=np.int64)
    sc = np.ones(3, dtype=np.float32)
    for _ in range(2):  # aquecimento
        sess.run(None, {'input': ids, 'input_lengths': lens, 'scales': sc})
    ts=[]
    for _ in range(5):
        t=time.perf_counter(); out = sess.run(None, {'input': ids, 'input_lengths': lens, 'scales': sc}); ts.append(time.perf_counter()-t)
    audio = out[0]
    ns = audio.shape[-1]/cfg['audio']['sample_rate']
    med = sorted(ts)[len(ts)//2]
    print(f'  {n_fonemas:>3} fonemas -> {audio.shape[-1]:>6} amostras = {ns:>5.2f}s audio | {med*1000:>7.1f} ms | RTF {med/ns:>5.3f}')
