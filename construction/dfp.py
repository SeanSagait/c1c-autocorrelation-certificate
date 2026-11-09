import numpy as np
from lib import acv
def selfcorr(u):
    n=len(u); L=1<<(2*n).bit_length(); F=np.fft.rfft(u,L); return np.fft.irfft(F*np.conj(F),L)[:n]
def dsolve(K,Vs,extra=1,it=20000,eta=0.3,u0=None,tol=1e-15):
    # w0=1 (point-mass cell), u = w[1:] length K+extra-1 ; target: ac_k = Vs/K for k=1..K  (Vs ~ V*S^2 normalised by K)
    n=K+extra; x=np.arange(1,n)/K
    u=np.clip((x-0.17)*2.0,0,Vs)/K if u0 is None else u0.copy()
    tgt=Vs/K
    for i in range(it):
        C=selfcorr(u)     # C[k] = sum_{i>=1} u_i u_{i+k} (indices in u)
        # for cell k (=u[k-1]) target: w0*w_k + sum_{i>=1} w_i w_{i+k} = tgt
        un=np.empty_like(u); kk=np.arange(1,n)
        Ck=np.where(kk<len(u),C[np.minimum(kk,len(u)-1)],0.0)
        un=np.maximum(0,tgt-Ck)
        un[kk>K]=u[kk>K]  # cells beyond K: keep (not constrained)
        err=np.abs(un-u).max()*K; u=(1-eta)*u+eta*un
        if err<tol: break
    w=np.r_[1.0,u]; return w,err,i
if __name__=='__main__':
    import sys
    K=int(sys.argv[1])
    for Vs in np.linspace(float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])):
        w,err,i=dsolve(K,Vs)
        r=acv(w,K)[1:]; print(f'{Vs:.4f}',f'{r.min():.10f}',f'err {err:.1e} it {i}',np.argmin(r),flush=True)
