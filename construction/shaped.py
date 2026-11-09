# "noise-shaped spike" construction.
# Spike = m cells with weights s_i ∝ C(m-1,i) rho^i. Then ac_k = sum_i s_i g_{k+i} + (g*g)_k, and the
# back-substitution g_k = (R_k - sum_{i>=1} s_i g_{k+i})/s_0 is stable (roots of s at -1/rho), with quantum s_0 << max(s).
import numpy as np, math, sys, time
def selfc(u, K):
    n = len(u); L = 1 << (2*n).bit_length(); F = np.fft.rfft(u, L); return np.fft.irfft(F*np.conj(F), L)[:K+1]
def spike(m, rho):
    return np.array([math.comb(m-1, i)*rho**i for i in range(m)], float)
def backsub(R, s, K, n, integer=False):
    m = len(s); g = np.zeros(n); s0 = s[0]; s1 = s[1:]
    # g index j>=m ; window k uses g_{k..k+m-1}; equation for k=K..1 descending determines g_k (k>=m)
    for k in range(K, m-1, -1):
        rest = float(np.dot(s1, g[k+1:k+m])) if m > 1 else 0.0
        v = (R[k] - rest)/s0
        if integer: v = max(0, math.ceil(v - 1e-9))
        else: v = max(0.0, v)
        g[k] = v
    return g
def backsub_fast(R, s, K, n, endlen=None):
    # descending recursion s0 g_k + sum_{i>=1} s_i g_{k+i} = R_k (k=K..m), cells beyond K are 0.
    # Right end (last endlen cells): exact clamped python recursion; rest: IIR filter (lfilter) seeded from it;
    # then zero below the last negative (gap edge).
    from scipy.signal import lfilter, lfiltic
    m = len(s); g = np.zeros(n); s1 = s[1:]
    E = endlen if endlen is not None else min(K-m, 6*m+50)
    k = K
    for k in range(K, K-E, -1):
        g[k] = max(0.0, (R[k] - (float(np.dot(s1, g[k+1:k+m])) if m > 1 else 0.0))/s[0])
    k0 = K-E   # next index to compute
    if m > 1:
        zi = lfiltic([1.0], s, y=list(g[k0+1:k0+m]))
        h, _ = lfilter([1.0], s, R[m:k0+1][::-1], zi=zi)
    else:
        h = lfilter([1.0], s, R[m:k0+1][::-1])
    gg = h[::-1].copy()
    neg = np.where(gg < 0)[0]
    if len(neg): gg[:neg.max()+1] = 0
    g[m:k0+1] = gg; return g
FAST = True
def value(w, K):
    ac = selfc(w, K); return K*ac[1:].min()/w.sum()**2, ac
def fixed_point(K, s, Vs, g0=None, it=300, eta=0.5, tol=1e-14):
    m = len(s); n = K+m
    sp = np.zeros(n); sp[:m] = s/s.sum()            # spike mass 1
    T = Vs/K
    x = np.arange(n)/K
    g = np.clip((x-0.17)*2.0, 0, Vs)/K if g0 is None else g0.copy()
    g[:m] = 0
    for i in range(it):
        G = selfc(g, K)
        R = T - G
        gn = (backsub_fast if FAST else backsub)(R, sp[:m], K, n); gn[:m] = 0
        err = np.abs(gn-g).max()*K; g = (1-eta)*g + eta*gn
        if err < tol: break
    w = g.copy(); w[:m] = sp[:m]
    return w, err, i
if __name__ == "__main__":
    K = int(sys.argv[1]); m = int(sys.argv[2]); rho = float(sys.argv[3])
    Vss = [float(v) for v in sys.argv[4].split(",")]
    s = spike(m, rho); t0 = time.time(); g0 = None; best = None
    for Vs in Vss:
        w, err, i = fixed_point(K, s, Vs, g0); g0 = w.copy(); g0[:m] = 0
        v, ac = value(w, K); print(f"m={m} rho={rho} Vs={Vs} value={v:.10f} err={err:.1e} it={i} {time.time()-t0:.0f}s", flush=True)
        if best is None or v > best[0]: best = (v, w, Vs)
    np.save(f"A/sh{K}_{m}_{rho}.npy", best[1]); print("best", best[0], best[2])

def fixed_point_aa(K, s, Vs, g0=None, it=300, eta=0.5, tol=1e-13, hist=6, verbose=False):
    """Anderson-accelerated version of fixed_point (type-II AA on the damped map)."""
    m = len(s); n = K+m
    sp = np.zeros(n); sp[:m] = s/s.sum(); T = Vs/K
    x = np.arange(n)/K
    g = np.clip((x-0.17)*2.0, 0, Vs)/K if g0 is None else g0.copy(); g[:m] = 0
    Xs, Fs = [], []
    err = np.inf
    for i in range(it):
        G = selfc(g, K); gn = backsub_fast(T - G, sp[:m], K, n); gn[:m] = 0
        f = gn - g; err = np.abs(f).max()*K
        if verbose and i % 10 == 0: print(i, err)
        if err < tol: break
        Xs.append(g.copy()); Fs.append(f.copy())
        if len(Xs) > hist+1: Xs.pop(0); Fs.pop(0)
        if len(Xs) >= 2:
            dF = np.array([Fs[j+1]-Fs[j] for j in range(len(Fs)-1)]).T
            dX = np.array([Xs[j+1]-Xs[j] for j in range(len(Xs)-1)]).T
            gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
            g = g + eta*f - (dX + eta*dF) @ gam
        else:
            g = g + eta*f
        g = np.maximum(g, 0); g[:m] = 0
    w = g.copy(); w[:m] = sp[:m]
    return w, err, i
