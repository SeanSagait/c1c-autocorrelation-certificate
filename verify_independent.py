"""Independent exact verifier for C_1c lower bound from a step-function certificate.
ratio(f) = min_{t in [0,1]} int f(x)f(x+t)dx / (int f)^2 ; any f>=0 gives C_1c >= ratio(f).
f = w_i on [i/K,(i+1)/K). g(t)=int f f(.+t) is piecewise linear in t with nodes at j/K,
g(j/K) = a_j / K, a_j = sum_i w_i w_{i+j}. A continuous piecewise-linear function on [0,1]
attains its min at a node, so min over t in [0,1] = min_{0<=j<=K} a_j / K.
int f = S/K, so ratio = K * min_j a_j / S^2 (exact Fraction)."""
import sys
from fractions import Fraction

def main(path):
    toks = open(path).read().split()
    K = int(toks[0]); w = [int(x) for x in toks[1:]]
    if not (K > 0 and w and all(x >= 0 for x in w) and any(w)): sys.exit("bad certificate")
    n = len(w); S = sum(w)
    nz = [i for i, x in enumerate(w) if x]
    print("K", K, "n", n, "S", S, "max w", max(w), "first/last nonzero", nz[0], nz[-1],
          "support length (units of 1)", Fraction(nz[-1] + 1 - nz[0], K))
    # exact autocorrelation by Kronecker substitution with Python big ints
    B = (2 * max(w).bit_length() + n.bit_length() + 2)
    # build integers: A = sum w_i 2^{B i}, R = sum w_i 2^{B (n-1-i)}
    hexd = (B + 3) // 4; B = 4 * hexd
    A = int("".join(format(x, "0%dx" % hexd) for x in reversed(w)), 16)
    R = int("".join(format(x, "0%dx" % hexd) for x in w), 16)
    prod = A * R  # coefficient of 2^{B(n-1+j)} is a_j for j>=0
    h = format(prod, "x").rjust(hexd * (2 * n - 1), "0")
    coeffs = [int(h[k * hexd:(k + 1) * hexd], 16) for k in range(2 * n - 1)]
    coeffs.reverse()  # coeffs[m] = coefficient of 2^{Bm}
    a = coeffs[n - 1:]  # a[j], j=0..n-1
    assert a[0] == sum(x * x for x in w)
    # spot-check a few lags directly
    for j in (1, 7, K // 2, min(K, n - 1)):
        if j < n:
            assert a[j] == sum(w[i] * w[i + j] for i in range(n - j)), j
    a = a + [0] * max(0, K + 1 - len(a))
    m = min(a[: K + 1]); jm = a.index(m)
    ratio = Fraction(K * m, S * S)
    print("argmin lag j", jm, "a_j", m, "a_K", a[K])
    print("ratio =", ratio)
    print("decimal ~", f"{ratio.numerator * 10**15 // ratio.denominator}e-15")

if __name__ == "__main__":
    main(sys.argv[1])
