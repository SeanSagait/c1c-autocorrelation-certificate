# Memory-lean version of one.py for K = 2^25 and beyond: one-cell atom (w_0 = 1) + profile u on cells 1..K
# solving u_k + sum_{i>=1} u_i u_{i+k} = Vs/K (k = 1..K), Anderson-accelerated fixed point without the big
# dX/dF matrices (Gram matrix from dot products, updates accumulated in place).  Then rounding at scale 2^bits.
# usage: python lean.py K Vs bits out --warm file.npy [--it N] [--hist m] [--tol t]
import numpy as np, scipy.fft as sf, sys, time, argparse, gc


def corr_profile(u, L):
    """C[k] = sum_{i} u_i u_{i+k}, k = 0..K-1 (linear, no wrap since L = 2K >= 2K-1)."""
    F = sf.rfft(u, L, workers=-1)
    A = np.abs(F); del F
    np.multiply(A, A, out=A)
    C = sf.irfft(A, L, workers=-1); del A
    return C[:len(u)].copy()


def solve(K, Vs, u, it=400, eta=0.3, tol=1e-13, hist=4, log=print):
    L = 2 * K
    T = Vs / K
    X, Fs = [], []
    best = (np.inf, None)
    since = 0
    t0 = time.time()
    for i in range(it):
        C = corr_profile(u, L)                     # C[k] = sum_i u_i u_{i+k}, k = 0..K-1
        # profile self term at lag k (1..K): sum_{i>=1} u_i u_{i+k} = C[k] for k <= K-1, 0 for k = K
        C = np.r_[C[1:], 0.0]
        f = np.maximum(0.0, T - C) - u; del C
        err = float(np.abs(f).max()) * K
        log(f"it {i}: err {err:.3e} ({time.time() - t0:.0f}s)")
        if not np.isfinite(err) or err > 1e6:
            break
        if err < best[0]:
            best = (err, u.copy()); since = 0
        else:
            since += 1
            if since > 25 and best[0] < 1e-10:
                log("stalled at the noise floor; keeping the best iterate"); break
        if err < tol:
            break
        X.append(u); Fs.append(f)
        if len(X) > hist + 1:
            X.pop(0); Fs.pop(0)
        m = len(X) - 1
        if m >= 1:
            G = np.zeros((m, m)); b = np.zeros(m)
            for j in range(m):
                dFj = Fs[j + 1] - Fs[j]
                b[j] = dFj @ f
                for l in range(j + 1):
                    dFl = dFj if l == j else Fs[l + 1] - Fs[l]
                    G[j, l] = G[l, j] = dFj @ dFl
                    del dFl
                del dFj
            g = np.linalg.lstsq(G, b, rcond=None)[0]
            unew = u + eta * f
            for j in range(m):
                d = X[j + 1] - X[j]; d *= g[j]; unew -= d; del d
                d = Fs[j + 1] - Fs[j]; d *= eta * g[j]; unew -= d; del d
            np.maximum(unew, 0.0, out=unew)
            u = unew
        else:
            u = u + eta * f
        gc.collect()
    return best[1] if best[1] is not None else u, best[0], i


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("K", type=int); p.add_argument("Vs", type=float); p.add_argument("bits", type=int)
    p.add_argument("out"); p.add_argument("--warm", required=True); p.add_argument("--it", type=int, default=400)
    p.add_argument("--hist", type=int, default=4); p.add_argument("--tol", type=float, default=1e-13)
    a = p.parse_args(); K = a.K; t0 = time.time()
    logf = open(a.out + ".log", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    w0 = np.load(a.warm); K0 = len(w0) - 1
    xo = np.arange(1, K0 + 1) / K0
    u = np.interp(np.arange(1, K + 1) / K, xo, w0[1:]) * (K0 / K)
    del w0, xo; gc.collect()
    log(f"K={K} Vs={a.Vs} warm K0={K0} loaded ({time.time() - t0:.0f}s)")
    u, err, i = solve(K, a.Vs, u, it=a.it, tol=a.tol, hist=a.hist, log=log)
    log(f"done: err {err:.3e} after {i} iterations ({time.time() - t0:.0f}s)")
    # exact float value of the step function [1, u]: K * min_k a_k / S^2 with a_k = u_k + sum_{i>=1} u_i u_{i+k}
    C = corr_profile(u, 2 * K); C = np.r_[C[1:], 0.0]; A = u + C; del C
    S = 1.0 + u.sum(); kmin = int(np.argmin(A)) + 1
    log(f"float value {K * A.min() / S ** 2:.13f} at lag {kmin} (t={kmin / K:.6f}); S={S:.12f}; first active cell {int(np.nonzero(u > 1e-12)[0][0]) + 1} ({(int(np.nonzero(u > 1e-12)[0][0]) + 1) / K:.6f})")
    del A
    w = np.r_[1.0, u]; np.save(a.out + ".npy", w)
    W = np.rint(w * 2.0 ** a.bits).astype(np.int64)
    np.save(a.out + ".int.npy", W)
    with open(a.out, "w") as fh:
        fh.write(f"{K}\n")
        for s in range(0, len(W), 4096):
            fh.write(" ".join(map(str, W[s:s + 4096].tolist())) + "\n")
    log(f"wrote {a.out} ({len(W)} weights, max {W.max()}) ({time.time() - t0:.0f}s)")
