# one-cell spike (w_0 = 1) + profile solving w_k + sum_{i>=1} w_i w_{i+k} = Vs/K for k=1..K; then round at scale 2^bits.
# usage: python one.py K Vs1,Vs2,... bits out.txt [--warm file.npy]
import numpy as np, scipy.fft as sf, sys, time, argparse
def corr(u):
    n = len(u); L = sf.next_fast_len(2 * n, real=True)
    F = sf.rfft(u, L, workers=-1); return sf.irfft(F * np.conj(F), L, workers=-1)[:n]
def solve(K, Vs, u0=None, it=5000, eta=0.3, tol=1e-13, hist=5):
    x = np.arange(1, K + 1) / K
    u = np.clip((x - 0.17) * 2.0, 0, Vs) / K if u0 is None else u0.copy()
    tgt = Vs / K; X, Fs = [], []
    for i in range(it):
        C = corr(u); C = np.r_[C[1:], 0.0]          # C[k-1] = sum_{i>=1} u_i u_{i+k}
        f = np.maximum(0, tgt - C) - u; err = np.abs(f).max() * K
        if err < tol: break
        X.append(u.copy()); Fs.append(f.copy())
        if len(X) > hist + 1: X.pop(0); Fs.pop(0)
        if len(X) >= 2:   # Anderson mixing
            dF = np.array([Fs[j+1]-Fs[j] for j in range(len(Fs)-1)]).T; dX = np.array([X[j+1]-X[j] for j in range(len(X)-1)]).T
            g, *_ = np.linalg.lstsq(dF, f, rcond=None); u = np.maximum(u + eta*f - (dX + eta*dF) @ g, 0)
        else: u = u + eta * f
    return np.r_[1.0, u], err, i
def fval(w, K):
    return K * corr(w)[1:K+1].min() / w.sum() ** 2
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("K", type=int); p.add_argument("Vs"); p.add_argument("bits", type=int)
    p.add_argument("out"); p.add_argument("--warm"); p.add_argument("--it", type=int, default=5000); p.add_argument("--hist", type=int, default=5)
    a = p.parse_args(); K = a.K; t0 = time.time()
    u0 = None
    if a.warm:
        w0 = np.load(a.warm); K0 = len(w0) - 1; f = K // K0
        u0 = np.interp(np.arange(1, K + 1) / K, np.arange(1, K0 + 1) / K0, w0[1:]) / f
    best = (-1, None)
    for Vs in map(float, a.Vs.split(",")):
        w, err, i = solve(K, Vs, u0, it=a.it, hist=a.hist); v = fval(w, K)
        print(f"Vs={Vs} float {v:.13f} err {err:.1e} it {i} ({time.time()-t0:.0f}s)", flush=True)
        if v > best[0]: best = (v, w, Vs)
        u0 = w[1:]
    v, w, Vs = best; np.save(a.out + ".npy", w)
    W = np.array([int(round(x)) for x in (w * 2.0 ** a.bits).tolist()], dtype=object)
    open(a.out, "w").write(f"{K}\n" + " ".join(map(str, W.tolist())) + "\n")
    print("wrote", a.out, "Vs", Vs, "float", f"{v:.13f}", flush=True)
