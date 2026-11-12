"""Independent exact check of an equal-step certificate (int64 weights).
Pass 1: limb FFTs (B bits, each limb correlation an exact integer, rounding checked < 0.25) folded into a
float a_k (non-negative terms, relative error < 1e-14). Candidates: lags within 1e-12 of the smallest.
Pass 2: the same exact limb integers are summed per limb degree at the candidate lags only and combined in
Python integers, so the exact minimum over all lags is the minimum over the candidates. S, a_0 exact."""
import sys, time
from fractions import Fraction
import numpy as np, scipy.fft as sf

t0 = time.time()
raw = np.fromfile(sys.argv[1], dtype=np.int64, sep=' ')
K, w = int(raw[0]), raw[1:]; del raw
n = len(w)
assert K > 0 and n == K + 1 and w.min() >= 0 and w.max() < 2**62, "bad certificate"
ch = 1 << 20
S = sum(sum(w[c:c + ch].tolist()) for c in range(0, n, ch))
a0 = sum(sum(x * x for x in w[c:c + ch].tolist()) for c in range(0, n, ch))
B = 9; M = (1 << B) - 1
P = (int(w.max()).bit_length() + B - 1) // B
assert n * M * M * P < 2**62 and n * M * M < 2**52
L = sf.next_fast_len(2 * n, real=True)
X = [sf.rfft((w >> (B * p) & M).astype(np.float64), L, workers=-1) for p in range(P)]
print(f"n={n} limbs={P} ({time.time()-t0:.0f}s)", flush=True)

def limb_pairs():
    for p in range(P):
        for q in range(P):
            r = sf.irfft(np.conj(X[p]) * X[q], L, workers=-1)[:K + 1]
            ri = np.rint(r)
            yield p + q, ri, float(np.abs(r - ri).max())

approx = np.zeros(K + 1); worst = 0.0
for d, ri, e in limb_pairs():
    worst = max(worst, e); approx += ri * 2.0 ** (B * d)
assert worst < 0.25, f"FFT rounding {worst}"
assert abs(approx[0] / a0 - 1) < 1e-13, "a_0 mismatch"
cand = np.nonzero(approx <= approx.min() * (1 + 1e-12))[0]; del approx
print(f"pass 1 done, worst rounding {worst:.4g}, {len(cand)} candidates ({time.time()-t0:.0f}s)", flush=True)
s = np.zeros((2 * P - 1, len(cand)), dtype=np.int64)
for d, ri, e in limb_pairs():
    s[d] += ri[cand].astype(np.int64)
a = [sum(int(s[d, j]) << (B * d) for d in range(2 * P - 1)) for j in range(len(cand))]
j = min(range(len(cand)), key=a.__getitem__); k, m = int(cand[j]), a[j]
assert m == sum(sum(x * y for x, y in zip(w[c:min(c + ch, n - k)].tolist(), w[c + k:min(c + ch, n - k) + k].tolist()))
                for c in range(0, n - k, ch)), "argmin lag mismatch"
val = Fraction(K * m, S * S)
print(f"K={K} argmin k={k} a_k={m} S={S}")
print("ratio =", val)
print(f"value = {float(val):.13f}  ({time.time()-t0:.0f}s)")
