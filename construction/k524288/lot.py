# integer-rounding lottery over (M offset, tau, T shift) for a converged float solution
import numpy as np, sys, math, time, itertools
import os; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]
from shaped import selfc
from mk import int_pass, exact_val
from multiprocessing import Pool
wf, K, m, r, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
nw = int(sys.argv[6]) if len(sys.argv) > 6 else 3
w = np.load(wf); n = len(w); s = r**np.arange(m)
Mmax = int(math.floor(math.sqrt((2**62-1)/n))) - 2
def job(args):
    dM, tau, dT = args
    M = Mmax - dM; Sint = np.round(s*M/s.max()).astype(np.int64)
    lam = float(Sint @ w[:m]) / float(w[:m] @ w[:m]); x = w*lam; x[:m] = Sint
    T = selfc(x, K)[m:K+1].min()*(1+dT)
    G = selfc(np.where(np.arange(n) >= m, x, 0.0), K)
    u = int_pass(x, Sint, K, T, G, tau); v, _ = exact_val(u, K)
    return v, args, u.astype(np.int64)
if __name__ == "__main__":
    combos = [(dM, tau, dT) for dM in (0, 300, 700, 1100) for tau in (0.75, 0.8, 0.85, 1.0) for dT in (-1e-8, 1e-8, 2e-8, 3e-8)]
    best = (-1, None, None); t0 = time.time()
    with Pool(nw) as p:
        for v, a, u in p.imap_unordered(job, combos):
            print(f"{v:.12f} {a} ({time.time()-t0:.0f}s)", flush=True)
            if v > best[0]: best = (v, a, u)
    u = best[2]; assert len(u)*int(u.max())**2 < 2**62
    open(out, "w").write(f"{K}\n" + " ".join(map(str, u.tolist()))); print("BEST", best[0], best[1])
