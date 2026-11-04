"""A pergunta que decide: 8,6% de erro nos pesos degrada a saída em quanto?"""
import sys, numpy as np
sys.path.insert(0,'.')
from nucleo.autograd import Tensor
from nucleo import nn

rng = np.random.default_rng(7)
D, V = 128, 8192

print("Simulando a camada de saída do KIANDA (d_model=128, vocab=8192)")
print(f"{'cenario':<26} {'top1 igual':>11} {'KL media':>10} {'logit max dif':>14}")
print("-"*66)

def q4(x):
    QK=32; plano=x.reshape(-1,QK).astype(np.float32); nb=plano.shape[0]
    e=np.zeros(nb,np.float16); q=np.zeros((nb,QK//2),np.uint8)
    for i in range(nb):
        b=plano[i]; d=b[np.argmax(np.abs(b))]/-8.0; e[i]=np.float16(d)
        id_=1.0/d if d else 0.0
        v=np.clip(b*id_+8.5,0,15).astype(np.uint8); q[i]=v[:16]|(v[16:]<<4)
    out=np.zeros((nb,QK),np.float32); dd=e.astype(np.float32)
    out[:,:16]=((q&0x0F).astype(np.int8)-8).astype(np.float32)*dd[:,None]
    out[:,16:]=(((q>>4)&0x0F).astype(np.int8)-8).astype(np.float32)*dd[:,None]
    return out.reshape(x.shape)

W = rng.normal(0, 0.02, (D, V)).astype(np.float32)
x = rng.normal(0, 1.0, (64, D)).astype(np.float32)   # 64 posicoes
logits_ref = x @ W
mq = q4(W)
for nome, Wq in (("Q4_0 (32 valores)", mq),):
    logits_q = x @ Wq
    top1 = (logits_ref.argmax(-1)==logits_q.argmax(-1)).mean()
    def softmax(z):
        z=z-z.max(-1,keepdims=True); p=np.exp(z); return p/p.sum(-1,keepdims=True)
    p,q_=softmax(logits_ref),softmax(logits_q)
    kl=np.sum(p*(np.log(p+1e-9)-np.log(q_+1e-9)),axis=-1).mean()
    print(f"{nome:<26} {top1*100:>10.2f}% {kl:>10.5f} {np.abs(logits_ref-logits_q).max():>14.5f}")

# distribuicao dos pesos com dispersao realista (com outliers, como pesos treinados)
W2 = (rng.standard_t(df=3, size=(D,V))*0.02).astype(np.float32)
lr = x @ W2; lq = x @ q4(W2)
s=lambda z:(lambda p:p/p.sum(-1,keepdims=True))(np.exp(z-z.max(-1,keepdims=True)))
p,qq=s(lr),s(lq)
kl=np.sum(p*(np.log(p+1e-9)-np.log(qq+1e-9)),axis=-1).mean()
print(f"{'Q4_0 (pesos c/ outliers)':<26} {(lr.argmax(-1)==lq.argmax(-1)).mean()*100:>10.2f}% {kl:>10.5f} {np.abs(lr-lq).max():>14.5f}")
