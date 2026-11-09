import numpy as np, torch, scipy.optimize as so, scipy.sparse as sp
def acv(w,K):
    n=len(w); L=1<<(2*n).bit_length()
    F=np.fft.rfft(w,L); ac=np.fft.irfft(F*np.conj(F),L)[:K+1]
    ac=np.concatenate([ac,np.zeros(max(0,K+1-len(ac)))])
    return K*ac/w.sum()**2
def adam(w0,K,steps=3000,lr=0.02,b0=1e2,b1=1e5,dev='cuda'):
    th=torch.tensor(np.log(np.maximum(w0,1e-12)),dtype=torch.float64,device=dev).requires_grad_()
    opt=torch.optim.Adam([th],lr=lr); n=len(w0); L=1<<(2*n).bit_length()
    best=(-1,None)
    for s in range(steps):
        beta=b0*(b1/b0)**(s/steps); w=th.exp()
        F=torch.fft.rfft(w,L); ac=torch.fft.irfft(F*F.conj(),L)[1:K+1]
        r=K*ac/w.sum()**2; soft=-torch.logsumexp(-beta*r,0)/beta
        if r.min().item()>best[0]: best=(r.min().item(),w.detach().cpu().numpy().copy())
        opt.zero_grad(); (-soft).backward(); opt.step()
    return best
def slp(w,K,iters=200,delta=0.3,margin=0.02,verbose=False,tol=1e-12):
    w=np.asarray(w,float); w=w/w.sum(); n=len(w)
    cur=acv(w,K)[1:].min()
    for it in range(iters):
        r=acv(w,K)[1:]   # k=1..K
        act=np.where(r<cur+margin)[0]+1
        # gradient of K*ac_k wrt w_j (S fixed=1 via equality): K*(w_{j+k}+w_{j-k})
        rows=[]
        for k in act:
            g=np.zeros(n); g[:n-k]+=w[k:]; g[k:]+=w[:n-k]; rows.append(K*g)
        A=np.array(rows)
        # vars d (n), t ; maximize t: -A d + t <= r_act ; sum d=0; -w<=d (w+d>=0), |d|<=delta*(w+eps)
        lo=np.maximum(-w,-delta*(w+1e-3/n)); hi=delta*(w+1e-3/n)
        c=np.zeros(n+1); c[-1]=-1
        Aub=np.hstack([-A,np.ones((len(act),1))]); bub=r[act-1]
        Aeq=np.hstack([np.ones((1,n)),np.zeros((1,1))])
        res=so.linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=[0],bounds=list(zip(lo,hi))+[(None,None)],method='highs')
        if res.status!=0: delta/=2; continue
        d=res.x[:n]; wn=np.maximum(w+d,0); wn/=wn.sum(); new=acv(wn,K)[1:].min()
        if new>cur+tol: w,cur=wn,new; delta=min(delta*1.5,1.0)
        else: delta/=2
        if verbose and it%10==0: print(it,cur,delta,len(act),flush=True)
        if delta<1e-7: break
    return w,cur
def write_cert(w,K,path,scale=2**24):
    n=len(w); s=scale
    while n*s*s>=2**61: s//=2
    wi=np.round(w/w.max()*s).astype(np.int64)
    open(path,'w').write(f"{K}\n"+" ".join(map(str,wi.tolist())))
