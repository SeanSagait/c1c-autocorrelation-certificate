"""Independent verification of a piecewise-polynomial certificate written by pwcert.py.

Reads the JSON (t1, s, T, margin, M, bound, pieces = [lo, hi, coeffs in u = x - lo]) and checks, independently of
the bivariate-polynomial machinery of pwcert.py:
  (1) f >= 0: every piece polynomial is >= 0 on its interval (Bernstein subdivision), box height 1/s > 0;
  (2) M = int h exactly (sum of exact piece integrals) and bound = (T - margin) / (1 + M)^2 exactly;
  (3) g(t) = int f(x) f(x+t) dx >= T - margin for all t in [0, 1]:
      g is a polynomial of degree <= 2p+1 on each piece of the lag mesh; its coefficients are recovered here by
      EXACT direct integration at 2p+2 distinct rational sample points per lag piece (a different computation
      from pwcert.py: piece-by-piece exact integration of the product at fixed t) followed by exact polynomial
      interpolation; then the Bernstein-subdivision test is applied with a separate implementation.
  (4) a floating-point sanity check: f sampled on 2^22 cells, FFT autocorrelation, min over [0,1] vs T.
usage: python verify_pwcert.py certificate.json [--samples 3]
"""
import sys, json, math, time
import numpy as np
from fractions import Fraction as Fr
from mpmath import mp, mpf

def peval(a, x):
    r = Fr(0)
    for c in reversed(a): r = r * x + c
    return r
def pint(a): return [Fr(0)] + [c / (k + 1) for k, c in enumerate(a)]
def pmul(a, b):
    r = [Fr(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b): r[i + j] += ca * cb
    return r
def pshift(a, c):      # P(u + c)
    r = [Fr(0)] * len(a)
    for k, ck in enumerate(a):
        for m in range(k + 1): r[m] += ck * math.comb(k, m) * c ** (k - m)
    return r

def bern(a):           # Bernstein coefficients on [0,1] from monomial coefficients
    n = len(a) - 1
    return [sum(Fr(math.comb(k, j), math.comb(n, j)) * a[j] for j in range(k + 1)) for k in range(n + 1)]
def nonneg_on_01(bc, depth=0, maxdepth=80):
    """Independent subdivision test (bisection via de Casteljau with explicit convex combinations)."""
    if all(c >= 0 for c in bc): return True
    if bc[0] < 0 or bc[-1] < 0 or depth >= maxdepth: return False
    n = len(bc) - 1
    tri = [list(bc)]
    for r in range(n):
        prev = tri[-1]; tri.append([(prev[i] + prev[i + 1]) / 2 for i in range(len(prev) - 1)])
    left = [tri[r][0] for r in range(n + 1)]; right = [tri[n - r][r] for r in range(n + 1)]
    return nonneg_on_01(left, depth + 1, maxdepth) and nonneg_on_01(right, depth + 1, maxdepth)
def to_01(a, lo, hi):  # P(lo + (hi-lo) sigma)
    r = [Fr(0)]
    for c in reversed(a):
        r = [x * lo for x in r] + [Fr(0)]
        # r = r*(lo + (hi-lo) sigma) + c  computed via explicit multiplication
    # simpler: Horner with the linear polynomial
    r = [Fr(0)]
    for c in reversed(a):
        r = pmul(r, [lo, hi - lo]); r[0] += c
    return r

def h_value(pieces, x):
    for lo, hi, c in pieces:
        if lo <= x < hi: return peval(c, x - lo)
    if x == pieces[-1][1]: return peval(pieces[-1][2], x - pieces[-1][0])
    return Fr(0)

def g_direct(pieces, s, t):
    """Exact g(t) by direct integration: box*box + cross + h*h (piece by piece in x)."""
    val = Fr(0)
    if t < s: val += (1 / s) * (1 - t / s)
    # cross: (1/s) int_{t-s/2}^{t+s/2} h
    a_, b_ = t - s / 2, t + s / 2
    for lo, hi, c in pieces:
        l, r = max(lo, a_), min(hi, b_)
        if l < r:
            P = pint(c); val += (peval(P, r - lo) - peval(P, l - lo)) / s
    # h*h: sum over piece pairs of int P_i(x) P_j(x + t) dx
    for loi, hii, ci in pieces:
        for loj, hij, cj in pieces:
            l, r = max(loi, loj - t), min(hii, hij - t)
            if l < r:
                # in u = x - loi:  P_i(u) * P_j(u + loi + t - loj)
                Q = pmul(ci, pshift(cj, loi + t - loj)); P = pint(Q)
                val += peval(P, r - loi) - peval(P, l - loi)
    return val

def interp(xs, ys):
    """Exact polynomial interpolation (Newton divided differences) -> monomial coefficients."""
    n = len(xs); coef = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    # Newton form -> monomial
    poly = [Fr(0)]
    for i in range(n - 1, -1, -1):
        poly = pmul(poly, [-xs[i], Fr(1)]); poly[0] += coef[i]
    return poly

if __name__ == "__main__":
    t0 = time.time()
    d = json.load(open(sys.argv[1]))
    pieces = [(Fr(lo), Fr(hi), [Fr(c) for c in cs]) for lo, hi, cs in d["pieces"]]
    s, T, margin, M, bound = Fr(d["s"]), Fr(d["T"]), Fr(d["margin"]), Fr(d["M"]), Fr(d["bound"])
    p = max(len(c) for _, _, c in pieces) - 1
    print(f"certificate: {len(pieces)} pieces, degree {p}, s = {float(s):.3e}, T - margin = {float(T - margin):.17f}")
    # (1) f >= 0
    for lo, hi, c in pieces:
        assert lo < hi and nonneg_on_01(bern(to_01(c, lo, hi))), ("piece not nonneg", float(lo), float(hi))
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1)), "pieces must be contiguous"
    print("(1) f >= 0 on every piece: OK; support of h =", float(pieces[0][0]), "..", float(pieces[-1][1]))
    # (2) mass and bound
    Mchk = sum(peval(pint(c), hi - lo) for lo, hi, c in pieces)
    assert Mchk == M, "M mismatch"
    assert bound == (T - margin) / (1 + M) ** 2, "bound formula mismatch"
    print(f"(2) M = int h exact: OK; bound = (T - margin)/(1+M)^2 = {float(bound):.17f}")
    # (3) exact g by direct integration + interpolation on the lag mesh, then Bernstein
    ends = sorted({lo for lo, _, _ in pieces} | {hi for _, hi, _ in pieces})
    mesh = {Fr(0), s, Fr(1)}
    for b in ends:
        for v in (b - s / 2, b + s / 2):
            if 0 < v < 1: mesh.add(v)
    for bi in ends:
        for bj in ends:
            v = bi - bj
            if 0 < v < 1: mesh.add(v)
    mesh = sorted(mesh)
    deg = 2 * p + 1
    mprime = T - margin
    worst = None
    nsamp = deg + 2
    for a_, b_ in zip(mesh[:-1], mesh[1:]):
        xs = [a_ + (b_ - a_) * Fr(i + 1, nsamp + 1) for i in range(nsamp)]        # interior rational points
        ys = [g_direct(pieces, s, x) for x in xs]
        poly = interp(xs[:deg + 1], ys[:deg + 1])
        # consistency: the extra sample must be reproduced exactly (degree bound check)
        assert peval(poly, xs[-1]) == ys[-1], ("degree bound violated on", float(a_), float(b_))
        bc = bern(to_01([c for c in poly], a_, b_))
        bc = [c - mprime for c in bc]
        ok = nonneg_on_01(bc)
        assert ok, ("g < m' possible on piece", float(a_), float(b_), float(min(bc)))
        mn = min(ys)
        if worst is None or mn < worst[0]: worst = (mn, a_, b_)
    print(f"(3) exact g >= T - margin on all {len(mesh)-1} lag pieces: OK  (smallest sampled g - T = {float(worst[0]-T):+.3e} at t in [{float(worst[1]):.4f},{float(worst[2]):.4f}])   ({time.time()-t0:.0f}s)")
    # (4) float sanity check by FFT
    K = 1 << 22
    xg = (np.arange(K) + 0.5) / K
    fv = np.zeros(K)
    for lo, hi, c in pieces:
        sel = (xg >= float(lo)) & (xg < float(hi))
        cf = np.array([float(x) for x in c])
        u = xg[sel] - float(lo)
        fv[sel] = np.polyval(cf[::-1], u)
    # the box is narrower than a cell: treat it as an atom of mass 1 at x = 0 for the float check
    gf = np.fft.irfft(np.abs(np.fft.rfft(fv, 4 * K)) ** 2, 4 * K)[:K] / K        # h*h on lags k/K
    gf += fv                                                                       # atom x profile (cell average)
    kk = np.arange(K); tl = kk / K
    act = (tl > float(pieces[0][0]) + 0.01) & (tl <= 1.0)
    print(f"(4) float FFT check (K=2^22, atom as point mass): min over [{float(pieces[0][0])+0.01:.2f},1] of g/T - 1 = {gf[act].min()/float(T)-1:+.2e}, max = {gf[act].max()/float(T)-1:+.2e} (O(1/K) discretisation)")
    mp.dps = 50
    print("VERIFIED: C_1c >=", mp.nstr(mpf(bound.numerator) / bound.denominator, 45))
    print(f"total {time.time()-t0:.0f}s")
