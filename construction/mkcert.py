# integer certificate from a geometric-spike fixed point.
# usage: python A/mkcert.py K m r out.txt [--taus 0.5,0.7,0.9] [--outer 3] [--Vs lo,hi]
import numpy as np, sys, math, time, argparse
sys.path.insert(0, "A"); from shaped import fixed_point, fixed_point_aa, value, selfc
from scipy.optimize import minimize_scalar
p = argparse.ArgumentParser(); p.add_argument("K", type=int); p.add_argument("m", type=int); p.add_argument("r", type=float); p.add_argument("out")
p.add_argument("--taus", default="0.5,0.6,0.7,0.8,0.9,1.0"); p.add_argument("--outer", type=int, default=4)
p.add_argument("--Vs", default="2.540,2.552"); p.add_argument("--Mshrink", type=int, default=0); p.add_argument("--it", type=int, default=400)
p.add_argument("--aa", action="store_true"); p.add_argument("--eta", type=float, default=0.5); p.add_argument("--tol", type=float, default=1e-13); p.add_argument("--xatol", type=float, default=3e-5)
a = p.parse_args(); K, m, r = a.K, a.m, a.r; t0 = time.time()
s = r**np.arange(m); cache = {}; warm = [None]
def f(Vs):
    w, err, i = (fixed_point_aa if a.aa else fixed_point)(K, s, Vs, g0=warm[0], it=a.it, eta=a.eta, tol=a.tol); v, _ = value(w, K); cache[Vs] = (v, w, err)
    g = w.copy(); g[:m] = 0; warm[0] = g; print(f"  Vs={Vs:.6f} v={v:.12f} err={err:.1e} it={i} ({time.time()-t0:.0f}s)", flush=True); return -v
lo, hi = map(float, a.Vs.split(","))
res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": a.xatol})
vf, w, err = cache[res.x]; n = len(w)
print(f"float: K={K} m={m} r={r} Vs={res.x:.5f} value={vf:.12f} err={err:.1e} ({time.time()-t0:.0f}s)", flush=True)
np.save(a.out + ".npy", w)
M = int(math.floor(math.sqrt((2**62-1)/n))) - 2 - a.Mshrink
Sint = np.array([round(M*r**i) for i in range(m)], dtype=np.int64)
lam = Sint[0]/w[0]; x = w*lam; x[:m] = Sint
acx = selfc(x, K); T = acx[m:K+1].min()       # common float target (scaled)
def greedy(G, tau):
    u = np.zeros(n); u[:m] = Sint; S0 = float(Sint[0]); S1 = Sint[1:].astype(float)
    R = T - G
    for k in range(K, m-1, -1):
        rest = float(np.dot(S1, u[k+1:k+m]))
        v = (R[k] - rest)/S0
        u[k] = max(0.0, math.floor(v + tau))
    # zero below the gap: cells where float x is 0 and greedy produced junk are allowed (they are >=0); keep
    return u
def exact_val(u):
    ui = u.astype(np.int64); ac = selfc(ui.astype(float), K); return K*ac[1:].min()/float(ui.sum())**2, ac
best = (-1, None, None)
for tau in map(float, a.taus.split(",")):
    G = selfc(np.where(np.arange(n) >= m, x, 0.0), K)   # g*g term from float
    for o in range(a.outer):
        u = greedy(G, tau)
        g = u.copy(); g[:m] = 0; G = selfc(g, K)
        v, ac = exact_val(u)
    print(f"tau={tau}: value {v:.12f} loss {vf-v:.3e} dS={u.sum()-x.sum():.0f} ({time.time()-t0:.0f}s)", flush=True)
    if v > best[0]: best = (v, u.astype(np.int64), tau)
u = best[1]
assert u.min() >= 0 and len(u)*int(u.max())**2 < 2**62
open(a.out, "w").write(f"{K}\n" + " ".join(map(str, u.tolist())))
print("wrote", a.out, "tau", best[2], "approx value", best[0])
