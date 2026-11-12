"""Memory-lean exact verifier for big step-function certificates (same method as verify_limbs.py,
rewritten for K = 2^25 and beyond).

Certificate: first token K, then n non-negative integer weights w_i (< 2^63), cell i = [i/K, (i+1)/K).
Value = K * min_{0<=k<=K} a_k / S^2,  a_k = sum_i w_i w_{i+k},  S = sum_i w_i  (exact rational).

Method: w_i = sum_p 2^(Bp) x_p[i] with B-bit limbs x_p[i] < 2^B.  For d = 0..2P-2 the sums
    s_d[k] = sum_{p+q=d} c_pq[k],   c_pq[k] = sum_i x_p[i] x_q[i+k]
are integers below (#terms) * n * (2^B-1)^2 < 2^52 and are computed by one float64 FFT correlation each
(one irfft of  sum_{p+q=d} conj(X_p) X_q ).  Rounding error bound of the FFT correlation (Higham-type, with
accurately computed twiddles): |error| <= c * u * log2(L) * ||x||_2 ||y||_2 with u = 2^-53, c a small constant;
here ||x||_2 ||y||_2 <= n (2^B-1)^2 so the bound is ~ 1e-2 for n = 2^25+1, B = 8: every s_d[k] is recovered
exactly by rounding.  The empirical maximum distance from an integer is printed as a check (must be << 0.5).
a_k = sum_d 2^(Bd) s_d[k] is then assembled in Python integers chunk by chunk; S, a_0 and the deciding lag are
re-computed by direct summation in Python integers.
usage: python verify_lean.py cert.txt [B]
"""
import sys, time
from fractions import Fraction
import numpy as np, scipy.fft as sf

t0 = time.time()
fn = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
chunks = []
with open(fn) as fh:                      # whitespace-separated integers, any line structure
    for line in fh:
        if line.strip():
            chunks.append(np.array(line.split(), dtype=np.int64))
arr = np.concatenate(chunks); del chunks
K = int(arr[0]); w = arr[1:].copy(); del arr
n = len(w)
if not (K > 0 and n > 0 and int(w.min()) >= 0): sys.exit("bad certificate: need K > 0 and non-negative weights")
S = sum(int(np.sum(w[i:i + (1 << 20)].astype(object))) for i in range(0, n, 1 << 20))      # exact Python integer
if S <= 0: sys.exit("bad certificate: zero mass")
M = (1 << B) - 1
P = (int(w.max()).bit_length() + B - 1) // B
if (2 * P - 1) * n * M ** 2 >= 2 ** 52: sys.exit("too many weights / too large limbs for exact float64 limb correlations")
L = sf.next_fast_len(2 * n, real=True)
print(f"K={K} n={n} S={S} limbs={P} B={B} L={L} ({time.time()-t0:.0f}s)", flush=True)

def limb(p):
    return ((w >> (B * p)) & M).astype(np.float64)

def rfft_limb(p):
    return sf.rfft(limb(p), L, workers=-1)

CH = 1 << 20
nch = (n + CH - 1) // CH
# s_d for all d first (int64, 2P-1 arrays of n), then a_k assembled chunk by chunk in Python integers
S_d = []
worst = 0.0
for d in range(2 * P - 1):
    acc = None
    for p in range(P):
        q = d - p
        if q < 0 or q >= P: continue
        Xp = rfft_limb(p); Xq = rfft_limb(q) if q != p else Xp
        term = np.conj(Xp); term *= Xq; del Xp
        if q != p: del Xq
        if acc is None: acc = term
        else: acc += term; del term
    r = sf.irfft(acc, L, workers=-1)[:n]; del acc
    ri = np.rint(r); worst = max(worst, float(np.abs(r - ri).max())); del r
    S_d.append(ri.astype(np.int64)); del ri
    print(f"  d={d} done, worst rounding distance so far {worst:.3g} ({time.time()-t0:.0f}s)", flush=True)
if worst >= 0.25: sys.exit(f"FFT rounding distance {worst} too large")

# assemble a_k exactly in chunks, track the minimum over 0 <= k <= K
best = None; kbest = -1; a0 = None
for c in range(nch):
    lo, hi = c * CH, min(n, (c + 1) * CH)
    if lo > K: break
    a = np.zeros(hi - lo, dtype=object)
    for d in range(2 * P - 1):
        a += S_d[d][lo:hi].astype(object) * (1 << (B * d))
    if c == 0: a0 = int(a[0])
    if hi > K + 1: a = a[:K + 1 - lo]
    m = int(a.min()); k = lo + int(np.argmin(a))
    if best is None or m < best: best, kbest = m, k
del S_d
if n < K + 1 and best > 0: best, kbest = 0, n      # lags k >= n carry no mass: a_k = 0
# direct exact checks
def exact_lag(k):
    tot = 0
    for lo in range(0, n - k, CH):
        hi = min(n - k, lo + CH)
        tot += int(np.sum(w[lo:hi].astype(object) * w[lo + k:hi + k].astype(object)))
    return tot
if a0 != exact_lag(0): sys.exit("a_0 mismatch")
if best != exact_lag(kbest): sys.exit("argmin lag mismatch")
val = Fraction(K * best, S * S)
print(f"K={K} n={n} argmin k={kbest} (t={kbest/K:.6f}) a_k={best} S={S} max FFT rounding distance={worst:.3g}")
print("ratio =", val)
print(f"value = {float(val):.13f}   ({time.time()-t0:.0f}s)")
