# Exact verifier for constant 1c lower bounds.
# Cert file: first line K, then non-negative integer weights w_i (step i covers [i/K,(i+1)/K)).
# value = K * min_{0<=k<=K} sum_i w_i w_{i+k} / (sum w)^2  (exact: autocorrelation is piecewise linear between grid points)
import sys, numpy as np
from fractions import Fraction
RECORD = Fraction("0.41033335274")
K, *w = map(int, open(sys.argv[1]).read().split())
if not (K > 0 and w and min(w) >= 0 and sum(w) > 0): sys.exit("bad certificate: need K > 0 and non-negative, not all zero weights")
w = np.array(w, dtype=np.int64)
if len(w) * int(w.max())**2 >= 2**62: sys.exit("weights too large for exact int64; rescale")
ac = np.correlate(w, w, "full")[len(w)-1:]          # ac[k] = sum_i w_i w_{i+k}, exact int64
ac = np.concatenate([ac, np.zeros(max(0, K+1-len(ac)), np.int64)])[:K+1]
k = int(ac.argmin()); S = int(w.sum())
val = Fraction(K*int(ac[k]), S*S)
print(f"K={K} n={len(w)} argmin k={k} value={float(val):.12f}  record={float(RECORD):.12f}  NEW RECORD: {val > RECORD}")
