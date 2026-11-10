# small-pivot spike = geometric(m,r) * (1+rho z)^q ; float fixed point (Anderson) + integer descending recursion.
# usage: python mk.py K m r q rho out.txt [--Vs 2.5469,...] [--warm file.npy] [--it N] [--taus ..] [--outer 4]
import numpy as np, sys, math, time, argparse
import os; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]; from shaped import fixed_point_aa, fixed_point, value, selfc

def spike_poly(m, r, q, rho):
    s = r ** np.arange(m)
    b = np.array([math.comb(q, j) * rho ** j for j in range(q + 1)], float)
    return np.convolve(s, b)

def upsample(w, Kold, mold, K, m):
    f = K / Kold
    g = w.copy(); g[:mold] = 0
    x = np.arange(K + m) / f
    gn = np.interp(x, np.arange(len(g)), g, right=0.0) / f
    gn[:m] = 0
    return gn

def int_pass(x, Sint, K, T, G, tau):
    n = len(x); m = len(Sint)
    u = np.zeros(n); u[:m] = Sint; S0 = float(Sint[0]); S1 = Sint[1:].astype(float)
    R = T - G
    for k in range(K, m - 1, -1):
        rest = float(np.dot(S1, u[k + 1:k + m]))
        u[k] = max(0.0, math.floor((R[k] - rest) / S0 + tau))
    return u

def exact_val(u, K):
    ui = u.astype(np.int64)
    ac = selfc(ui.astype(float), K)
    return K * ac[1:].min() / float(ui.sum()) ** 2, ac

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("K", type=int); p.add_argument("m", type=int); p.add_argument("r", type=float)
    p.add_argument("q", type=int); p.add_argument("rho", type=float); p.add_argument("out")
    p.add_argument("--Vs", default="2.5469"); p.add_argument("--warm", default=None)
    p.add_argument("--warmK", type=int, default=0); p.add_argument("--warmm", type=int, default=0)
    p.add_argument("--it", type=int, default=300); p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--hist", type=int, default=6)
    p.add_argument("--taus", default="0.5"); p.add_argument("--outer", type=int, default=3)
    p.add_argument("--Msc", type=float, default=1.0)
    p.add_argument("--noint", action="store_true"); p.add_argument("--plain", action="store_true")
    a = p.parse_args(); K, m_, r = a.K, a.m, a.r; t0 = time.time()
    s = spike_poly(m_, r, a.q, a.rho); m = len(s)
    print(f"spike len {m} s0/max={s[0]/s.max():.4g} mass/max={s.sum()/s.max():.2f} mass/s0={s.sum()/s[0]:.2f}", flush=True)
    g0 = None
    if a.warm:
        w0 = np.load(a.warm); g0 = upsample(w0, a.warmK, a.warmm, K, m) if a.warmK else w0.copy()
        if a.warmK == 0: g0[:m] = 0
    best = (-1, None)
    for Vs in map(float, a.Vs.split(",")):
        import lib
        fn = lib.fp
        kw = dict(it=a.it, eta=a.eta, tol=1e-12, hist=0 if a.plain else a.hist)
        w, err, i = fn(K, s, Vs, g0=g0, **kw)
        vf, _ = value(w, K); g0 = w.copy(); g0[:m] = 0
        n = len(w); print(f"Vs={Vs} float {vf:.12f} err={err:.1e} it={i} ({time.time()-t0:.0f}s)", flush=True)
        np.save(a.out + f".Vs{Vs}.npy", w)
        if a.noint: continue
        M = int(math.floor(math.sqrt((2 ** 62 - 1) / n))) - 2
        Sc = M * a.Msc / s.max()
        Sint = np.round(s * Sc).astype(np.int64)
        assert Sint.max() <= M
        lam = float(Sint @ w[:m]) / float(w[:m] @ w[:m]); x = w * lam; x[:m] = Sint
        T = selfc(x, K)[m:K + 1].min()
        for tau in map(float, a.taus.split(",")):
            G = selfc(np.where(np.arange(n) >= m, x, 0.0), K)
            for o in range(a.outer):
                u = int_pass(x, Sint, K, T, G, tau)
                g = u.copy(); g[:m] = 0; G = selfc(g, K)
                v, ac = exact_val(u, K)
                if v > best[0]: best = (v, u.astype(np.int64).copy())
                print(f"   tau={tau} pass{o} int {v:.12f} loss {vf-v:.3e} S0={Sint[0]} M={M} ({time.time()-t0:.0f}s)", flush=True)
    if best[1] is not None:
        u = best[1]; assert u.min() >= 0 and len(u) * int(u.max()) ** 2 < 2 ** 62
        open(a.out, "w").write(f"{a.K}\n" + " ".join(map(str, u.tolist())))
        print("wrote", a.out, best[0])
