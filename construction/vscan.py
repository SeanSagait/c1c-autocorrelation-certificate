# scan Vs near the optimum (warm-started fixed point) and keep the best integer certificate.
import numpy as np, sys, math, time
sys.path.insert(0, "A"); from shaped import fixed_point, selfc, value
K, m, r = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); out = sys.argv[4]
Vss = [float(v) for v in sys.argv[5].split(",")]; w0 = np.load(sys.argv[6]); t0 = time.time()
s = r**np.arange(m); g0 = w0.copy(); g0[:m] = 0; best = (-1, None)
for Vs in Vss:
    w, err, i = fixed_point(K, s, Vs, g0=g0, it=400, eta=0.3, tol=1e-12); vf, _ = value(w, K)
    n = len(w); M = int(math.floor(math.sqrt((2**62-1)/n))) - 2
    Sint = np.array([round(M*r**j) for j in range(m)], dtype=np.int64)
    x = w*(Sint[0]/w[0]); x[:m] = Sint; T = selfc(x, K)[m:K+1].min()
    S0 = float(Sint[0]); S1 = Sint[1:].astype(float)
    G = selfc(np.where(np.arange(n) >= m, x, 0.0), K)
    for o in range(3):
        u = np.zeros(n); u[:m] = Sint; R = T - G
        for k in range(K, m-1, -1):
            u[k] = max(0.0, math.floor((R[k] - float(np.dot(S1, u[k+1:k+m])))/S0 + 0.5))
        g = u.copy(); g[:m] = 0; G = selfc(g, K)
        ac = selfc(u, K); v = K*ac[1:].min()/u.sum()**2
        if v > best[0]: best = (v, u.astype(np.int64).copy())
    print(f"Vs={Vs} float {vf:.12f} err {err:.1e} int {v:.12f} best {best[0]:.12f} ({time.time()-t0:.0f}s)", flush=True)
    if time.time()-t0 > 520: break
u = best[1]; assert u.min() >= 0 and len(u)*int(u.max())**2 < 2**62
open(out, "w").write(f"{K}\n" + " ".join(map(str, u.tolist()))); print("wrote", out, best[0])
