"""Exact verifier using integer arithmetic only (no floating point).
Computes a_k = sum_i w_i w_{i+k} modulo several NTT primes, then rebuilds every a_k exactly by the
Chinese remainder theorem.  This is valid because 0 <= a_k <= a_0 = sum_i w_i^2 (Cauchy-Schwarz),
a_0 is computed exactly in Python integers, and the primes are chosen so that their product exceeds a_0.
Scores ratio = K * min_{0<=k<=K} a_k / S^2 as in README.md.
All primes are < 2^31.1, so every product of two residues is < 2^63 and int64 arithmetic cannot
overflow."""
import sys
from fractions import Fraction
from math import prod
import numpy as np

# (p, primitive root, e) with 2^e | p-1; all primes are below 2.3e9 so (p-1)^2 < 2^63.
TABLE = [(167772161, 3, 25), (469762049, 3, 26), (754974721, 11, 24), (1224736769, 3, 24),
         (2013265921, 31, 27), (1811939329, 13, 26), (2113929217, 5, 25), (2281701377, 3, 27)]

def ntt(x, p, g, invert=False):
    N = len(x); bits = N.bit_length() - 1
    idx = np.arange(N); rev = np.zeros(N, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    x = x[rev]; del rev, idx
    h = 1
    while h < N:
        root = pow(g, (p - 1) // (2 * h), p)
        if invert: root = pow(root, p - 2, p)
        tw = np.ones(1, dtype=np.int64)
        while len(tw) < h:
            tw = np.concatenate([tw, tw * pow(root, len(tw), p) % p])
        x = x.reshape(-1, 2, h)
        x0 = x[:, 0, :]; x1 = x[:, 1, :] * tw % p
        x = np.stack([(x0 + x1) % p, (x0 - x1) % p], axis=1).reshape(-1)
        h *= 2
    return x * pow(N, p - 2, p) % p if invert else x

toks = open(sys.argv[1]).read().split()
K = int(toks[0]); w = [int(x) for x in toks[1:]]
if not (K > 0 and w and min(w) >= 0 and sum(w) > 0): sys.exit("bad certificate")
n = len(w); S = sum(w); a0 = sum(x * x for x in w)
N = 1 << (2 * n - 1).bit_length()
PRIMES = []
for p, g, e in TABLE:
    if (1 << e) % N == 0: PRIMES.append((p, g))
    if prod(p for p, _ in PRIMES) > a0: break
if prod(p for p, _ in PRIMES) <= a0: sys.exit("not enough NTT primes for this certificate")
print(f"N={N} primes={[p for p, _ in PRIMES]}", flush=True)
res = []
for p, g in PRIMES:
    f = np.zeros(N, dtype=np.int64); f[:n] = [x % p for x in w]
    r = np.zeros(N, dtype=np.int64); r[:n] = f[:n][::-1]
    c = ntt(ntt(f, p, g) * ntt(r, p, g) % p, p, g, invert=True)
    res.append(c[n - 1:2 * n - 1].copy())          # coefficient n-1+k of w * reversed(w) is a_k
    del f, r, c
    print(f"prime {p} done", flush=True)
# Garner: a = v0 + p0 v1 + p0 p1 v2 + ..., each v_i computed mod p_i in int64
v = []
for i, (p, _) in enumerate(PRIMES):
    acc = np.zeros(n, dtype=np.int64); coef = 1
    for j in range(i):
        acc = (acc + coef * v[j]) % p; coef = coef * PRIMES[j][0] % p
    v.append((res[i] - acc) % p * pow(coef, p - 2, p) % p)
a = v[-1].astype(object)
for j in range(len(PRIMES) - 2, -1, -1):
    a = a * PRIMES[j][0] + v[j].astype(object)
a = list(a) + [0] * max(0, K + 1 - n)
if a[0] != a0: sys.exit("a_0 mismatch")
m = min(a[:K + 1]); k = a.index(m)
if m != sum(w[i] * w[i + k] for i in range(n - k)): sys.exit("argmin lag mismatch")
val = Fraction(K * m, S * S)
print(f"K={K} n={n} argmin k={k} a_k={m} S={S}")
print("ratio =", val)
print(f"value = {float(val):.12f}")
