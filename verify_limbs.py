"""Second exact verifier, for certificates with weights too large for verify.py's int64 check.
Splits each weight into B-bit limbs, w = sum_p 2^(Bp) x_p. Every limb correlation
c_pq[k] = sum_i x_p[i] x_q[i+k] is at most n * (2^B-1)^2 < 2^52, so a float64 FFT gives it
exactly after rounding (checked: every value is within 0.25 of an integer). The lag sums
a_k = sum_{p,q} 2^(B(p+q)) c_pq[k] are then recombined in Python integers. B is an optional second argument (default 9)."""
import sys
from fractions import Fraction
import numpy as np, scipy.fft as sf

toks = open(sys.argv[1]).read().split()
B = int(sys.argv[2]) if len(sys.argv) > 2 else 9
K = int(toks[0]); w = [int(x) for x in toks[1:]]
if not (K > 0 and w and min(w) >= 0 and sum(w) > 0): sys.exit("bad certificate")
n = len(w); S = sum(w); M = (1 << B) - 1
P = (max(w).bit_length() + B - 1) // B
if n * M ** 2 >= 2 ** 52: sys.exit("too many weights for exact float64 limb correlations")
L = sf.next_fast_len(2 * n, real=True)
X = [sf.rfft(np.array([(x >> (B * p)) & M for x in w], dtype=np.float64), L, workers=-1) for p in range(P)]
s = [np.zeros(n, dtype=np.int64) for _ in range(2 * P - 1)]
worst = 0.0
for p in range(P):
    for q in range(P):
        r = sf.irfft(np.conj(X[p]) * X[q], L, workers=-1)[:n]
        ri = np.rint(r); worst = max(worst, float(np.abs(r - ri).max()))
        s[p + q] += ri.astype(np.int64)
if worst >= 0.25: sys.exit(f"FFT rounding error {worst} too large")
a = np.zeros(n, dtype=object)
for d in range(2 * P - 1):
    a += s[d].astype(object) * (1 << (B * d))
a = list(a) + [0] * max(0, K + 1 - n)
if a[0] != sum(x * x for x in w): sys.exit("a_0 mismatch")
m = min(a[:K + 1]); k = a.index(m)
if m != sum(w[i] * w[i + k] for i in range(n - k)): sys.exit("argmin lag mismatch")
val = Fraction(K * m, S * S)
print(f"K={K} n={n} limbs={P} limb bits={B} max FFT rounding error={worst:.3g} argmin k={k} a_k={m} S={S}")
print("ratio =", val)
print(f"value = {float(val):.12f}")
