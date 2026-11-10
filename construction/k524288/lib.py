import numpy as np, math, sys
from scipy.signal import lfilter, lfiltic
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shaped import selfc, value

def backsub2(R, s, K, n, gapfrac=0.30):
    """Descending recursion s0 g_k + sum_{i>=1} s_i g_{k+i} = R_k, k=K..m. Right end exact clamped; middle linear IIR;
    gap edge = last negative index below gapfrac*K (everything below zeroed); stray negatives above clamped to 0."""
    m = len(s); g = np.zeros(n); s1 = s[1:]
    E = min(K - m, 6 * m + 400)
    for k in range(K, K - E, -1):
        g[k] = max(0.0, (R[k] - float(np.dot(s1, g[k + 1:k + m]))) / s[0])
    k0 = K - E
    zi = lfiltic([1.0], s, y=list(g[k0 + 1:k0 + m]))
    h, _ = lfilter([1.0], s, R[m:k0 + 1][::-1], zi=zi)
    gg = h[::-1].copy()
    lim = int(gapfrac * K) - m
    neg = np.where(gg[:max(lim, 1)] < 0)[0]
    if len(neg): gg[:neg.max() + 1] = 0
    gg = np.maximum(gg, 0)
    g[m:k0 + 1] = gg
    return g

def fp(K, s, Vs, g0=None, it=300, eta=0.5, tol=1e-12, hist=0, verbose=False):
    m = len(s); n = K + m
    sp = np.zeros(n); sp[:m] = s / s.sum(); T = Vs / K
    x = np.arange(n) / K
    g = np.clip((x - 0.17) * 2.0, 0, Vs) / K if g0 is None else g0.copy(); g[:m] = 0
    Xs, Fs = [], []; err = np.inf
    for i in range(it):
        G = selfc(g, K); gn = backsub3(T - G, sp[:m], K, n); gn[:m] = 0
        f = gn - g; err = np.abs(f).max() * K
        if verbose and i % 10 == 0: print(i, err, flush=True)
        if err < tol: break
        if hist:
            Xs.append(g.copy()); Fs.append(f.copy())
            if len(Xs) > hist + 1: Xs.pop(0); Fs.pop(0)
            if len(Xs) >= 2:
                dF = np.array([Fs[j + 1] - Fs[j] for j in range(len(Fs) - 1)]).T
                dX = np.array([Xs[j + 1] - Xs[j] for j in range(len(Xs) - 1)]).T
                gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
                g = g + eta * f - (dX + eta * dF) @ gam
            else: g = g + eta * f
            g = np.maximum(g, 0); g[:m] = 0
        else:
            g = (1 - eta) * g + eta * gn
    w = g.copy(); w[:m] = sp[:m]
    return w, err, i

def backsub3(R, s, K, n, pad=None, P=None):
    """True clamped descending recursion, fast: linear IIR above, exact clamped python recursion from (highest negative+pad) down,
    stopping after P consecutive nonpositive raw values (gap) and zeroing the rest."""
    m = len(s); g = np.zeros(n); s1 = s[1:]; s0 = s[0]
    pad = pad or (m + 60); P = P or (4 * m + 100)
    E = min(K - m, 6 * m + 400)
    for k in range(K, K - E, -1):
        g[k] = max(0.0, (R[k] - float(np.dot(s1, g[k + 1:k + m]))) / s0)
    k0 = K - E
    zi = lfiltic([1.0], s, y=list(g[k0 + 1:k0 + m]))
    h, _ = lfilter([1.0], s, R[m:k0 + 1][::-1], zi=zi)
    gg = h[::-1]                      # gg[j] = linear solution at index m+j
    neg = np.where(gg < 0)[0]
    if len(neg) == 0:
        g[m:k0 + 1] = gg; return g
    jt = min(k0, m + int(neg.max()) + pad)     # exact recursion starts here (index in g)
    g[jt + 1:k0 + 1] = gg[jt + 1 - m:]
    g[jt + 1:k0 + 1] = np.maximum(g[jt + 1:k0 + 1], 0)
    run = 0
    for k in range(jt, m - 1, -1):
        v = (R[k] - float(np.dot(s1, g[k + 1:k + m]))) / s0
        if v > 0: g[k] = v; run = 0
        else:
            run += 1
            if run >= P: break
    return g

def backsub_fixed(R, s, K, n, je):
    """Linear (unclamped) descending recursion for k=K..je, zeros below je. Right end handled by same recursion."""
    m = len(s)
    h, _ = lfilter([1.0], s, R[je:K + 1][::-1], zi=np.zeros(m - 1))
    g = np.zeros(n); g[je:K + 1] = h[::-1]
    return g

def fp2(K, s, Vs, g0=None, it=300, eta=0.5, tol=1e-12, hist=0, freeze_at=int(__import__("os").environ.get("FZ","25")), verbose=False, je=None):
    m = len(s); n = K + m
    sp = s / s.sum(); T = Vs / K
    x = np.arange(n) / K
    g = np.clip((x - 0.17) * 2.0, 0, Vs) / K if g0 is None else g0.copy(); g[:m] = 0
    Xs, Fs = [], []; err = np.inf
    for i in range(it):
        G = selfc(g, K); R = T - G
        if je is None or (i < freeze_at and g0 is None):
            gn = backsub3(R, sp, K, n); gn[:m] = 0
            nz = np.nonzero(gn)[0]
            if i == freeze_at - 1 or g0 is not None and i == 0:
                je = int(nz[0]) if g0 is None else int(np.nonzero(g0)[0][0]); 
        else:
            gn = backsub_fixed(R, sp, K, n, je)
        f = gn - g; err = np.abs(f).max() * K
        if verbose and i % 10 == 0: print(i, err, je, gn.min() * K, flush=True)
        if err < tol and je is not None: break
        if hist and je is not None:
            Xs.append(g.copy()); Fs.append(f.copy())
            if len(Xs) > hist + 1: Xs.pop(0); Fs.pop(0)
            if len(Xs) >= 2:
                dF = np.array([Fs[j + 1] - Fs[j] for j in range(len(Fs) - 1)]).T
                dX = np.array([Xs[j + 1] - Xs[j] for j in range(len(Xs) - 1)]).T
                gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
                g = g + eta * f - (dX + eta * dF) @ gam
            else: g = g + eta * f
            g[:m] = 0
        else:
            g = (1 - eta) * g + eta * gn
    w = g.copy(); w[:m] = sp
    return w, err, i, je
