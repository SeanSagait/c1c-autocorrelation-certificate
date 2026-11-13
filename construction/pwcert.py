"""Exact piecewise-polynomial lower-bound certificate for C_1c.

Certificate:  f = (1/s) 1_{[-s/2, s/2]} + h,  s = 2^-S a dyadic rational,
              h = 0 on (-inf, t1), h = polynomial with rational coefficients on each piece of the breakpoint mesh
              B = {k t1} U {1 - k t1} U {1} (t1 dyadic rational), h = T on [1 - t1, 1 + s/2], h = 0 beyond.
Everything is a rational number, so  g(t) = int f(x) f(x+t) dx  is, on each piece of the lag mesh
   {0, s} U {b +- s/2} U {b_i - b_j} (b in B U {1 + s/2}),
a polynomial in t with rational coefficients, computed EXACTLY here (bivariate polynomial algebra over Fractions).
Then  g(t) >= m'  on [0, 1] is certified by exact Bernstein-coefficient subdivision on every lag piece, and
   C_1c >= m' / (1 + M)^2,  M = int h  (exact rational).
Only step 1 (finding h) uses floating point / mpmath; steps 2-3 are exact.
usage: python pwcert.py dps p S [--rb RB] [--margin 1e-27]
   dps: mpmath digits for the solve, p: polynomial degree per piece, S: box width s = 2^-S,
   RB: rounding bits for the polynomial values (default dps*3.4), margin: m' = T - margin (then verified).
"""
import sys, time, math, json
import numpy as np
from fractions import Fraction as Fr
from mpmath import mp, mpf
import cont, contmp

# ---------------------------------------------------------------- univariate / bivariate polynomial algebra
def padd(a, b):
    n = max(len(a), len(b)); r = [Fr(0)] * n
    for i, c in enumerate(a): r[i] += c
    for i, c in enumerate(b): r[i] += c
    return r
def psub(a, b): return padd(a, [-c for c in b])
def pmul(a, b):
    r = [Fr(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca == 0: continue
        for j, cb in enumerate(b):
            r[i + j] += ca * cb
    return r
def pscale(a, c): return [c * x for x in a]
def peval(a, x):
    r = Fr(0)
    for c in reversed(a): r = r * x + c
    return r
def pint(a):           # antiderivative, zero constant
    return [Fr(0)] + [c / (k + 1) for k, c in enumerate(a)]
def pshift(a, c):      # P(u + c) as polynomial in u
    n = len(a); r = [Fr(0)] * n
    for k, ck in enumerate(a):
        if ck == 0: continue
        for m in range(k + 1):
            r[m] += ck * math.comb(k, m) * c ** (k - m)
    return r
def pcompose_affine(a, e0, e1):   # P(e0 + e1 t) as polynomial in t
    r = [Fr(0)]
    for c in reversed(a):
        r = padd(pmul(r, [e0, e1]), [c])
    return r
def ptrim(a):
    while len(a) > 1 and a[-1] == 0: a = a[:-1]
    return a

def bivar_shift(a, d):
    """P(u + t + d) for univariate P (coeffs a): dict (i, j) -> coeff of u^i t^j."""
    out = {}
    for k, ck in enumerate(a):
        if ck == 0: continue
        # (u + t + d)^k = sum_{i+j+l=k} k!/(i! j! l!) u^i t^j d^l
        for i in range(k + 1):
            for j in range(k - i + 1):
                l = k - i - j
                coef = ck * (math.factorial(k) // (math.factorial(i) * math.factorial(j) * math.factorial(l))) * d ** l
                out[(i, j)] = out.get((i, j), Fr(0)) + coef
    return out
def bivar_mul_uni(P, a):
    """(univariate in u with coeffs a) * (bivariate P)."""
    out = {}
    for (i, j), c in P.items():
        for k, ak in enumerate(a):
            if ak == 0: continue
            out[(i + k, j)] = out.get((i + k, j), Fr(0)) + c * ak
    return out
def bivar_int_u(P):
    return {(i + 1, j): c / (i + 1) for (i, j), c in P.items()}
def bivar_eval_u_affine(P, e0, e1):
    """R(e0 + e1 t, t) as univariate polynomial in t."""
    # group by power of u
    byu = {}
    for (i, j), c in P.items():
        byu.setdefault(i, {})[j] = c
    r = [Fr(0)]
    for i, tj in byu.items():
        base = [Fr(0)] * (max(tj) + 1)
        for j, c in tj.items(): base[j] = c
        r = padd(r, pmul(base, pcompose_affine([Fr(0)] * i + [Fr(1)], e0, e1)))
    return ptrim(r)

# ---------------------------------------------------------------- Bernstein certification
def bernstein_coeffs(a):
    """Bernstein coefficients on [0,1] of the polynomial with monomial coeffs a (degree n = len(a)-1)."""
    n = len(a) - 1
    b = []
    for k in range(n + 1):
        s = Fr(0)
        for j in range(k + 1):
            if a[j] == 0: continue
            s += Fr(math.comb(k, j), math.comb(n, j)) * a[j]
        b.append(s)
    return b
def decasteljau_split(b):
    n = len(b) - 1
    left = [b[0]]; right = [b[-1]]
    cur = list(b)
    for r in range(1, n + 1):
        cur = [(cur[i] + cur[i + 1]) / 2 for i in range(len(cur) - 1)]
        left.append(cur[0]); right.append(cur[-1])
    return left, right[::-1]
def certify_nonneg(b, depth=0, maxdepth=60):
    """True iff the polynomial with Bernstein coefficients b is >= 0 on [0,1] (proved by subdivision)."""
    if min(b) >= 0:
        return True, depth
    if b[0] < 0 or b[-1] < 0:
        return False, depth          # endpoint value negative: genuinely negative
    if depth >= maxdepth:
        return False, depth
    l, r = decasteljau_split(b)
    okl, dl = certify_nonneg(l, depth + 1, maxdepth)
    if not okl: return False, dl
    okr, dr = certify_nonneg(r, depth + 1, maxdepth)
    return okr, max(dl, dr)

# ---------------------------------------------------------------- step 1: the profile as rational polynomials
def solve_profile(t1r, p, dps, verbose=True):
    mp.dps = dps
    w = np.load("K4194304.npy")  # warm start: float weights of a K = 2^22 solution (lean.py)
    t1 = mpf(t1r.numerator) / t1r.denominator
    fam, z, phi, M, err = contmp.solve_mp(t1, p, w, verbose=verbose, maxit=12)
    return fam, z, phi, M, err

def rational_pieces(fam, z, rb):
    """For each polynomial piece j: (lo, hi, coeffs) with lo, hi exact (dyadic) and coeffs rational, obtained by
    interpolating the mp piece at p+1 equispaced RATIONAL points (exact Vandermonde solve) after rounding the
    values to multiples of 2^-rb.  Returns also T rounded."""
    h, T = fam.full(z); p = fam.p
    bexact = []  # exact breakpoints as Fractions: rebuild from t1r
    pieces = []
    return h, T

def mp_to_fr(x, rb):
    return Fr(int(mp.nint(x * (mpf(2) ** rb))), 2 ** rb)

def build_h(t1r, fam, z, rb, S):
    """Exact mesh and rational polynomials.  Pieces: 0..npc-1 polynomial, then plateau [1-t1, 1], extension [1, 1+s/2]."""
    h, T = fam.full(z); p = fam.p
    s = Fr(1, 2 ** S)
    # exact breakpoints (same construction as contmp.breakpoints_mp, in Fractions)
    pts = {t1r, 1 - t1r, Fr(1)}
    k = 1
    while k * t1r < 1 - t1r: pts.add(k * t1r); k += 1
    k = 1
    while 1 - k * t1r > t1r: pts.add(1 - k * t1r); k += 1
    b = sorted(pts)
    assert len(b) - 2 == fam.npc, (len(b), fam.npc)
    Tr = mp_to_fr(T, rb)
    pieces = []
    for j in range(fam.npc):
        lo, hi = b[j], b[j + 1]
        wj = hi - lo
        # equispaced rational nodes in u = x - lo
        us = [wj * Fr(i, p) for i in range(p + 1)]
        vals = []
        for u in us:
            x = mpf(lo.numerator) / lo.denominator + mpf(u.numerator) / u.denominator
            if j == fam.npc - 1 and u == wj:
                v = T
            else:
                row = fam.basis_row(x)
                v = fam.ev(row, h, T)
            vals.append(mp_to_fr(v, rb))
        # exact Vandermonde solve: coefficients c with sum c_k u^k = vals
        n = p + 1
        A = [[u ** k for k in range(n)] + [vals[i]] for i, u in enumerate(us)]
        # Gaussian elimination in Fractions
        for col in range(n):
            piv = next(r for r in range(col, n) if A[r][col] != 0)
            A[col], A[piv] = A[piv], A[col]
            pv = A[col][col]
            A[col] = [x / pv for x in A[col]]
            for r in range(n):
                if r != col and A[r][col] != 0:
                    fac = A[r][col]
                    A[r] = [x - fac * y for x, y in zip(A[r], A[col])]
        coeffs = [A[i][n] for i in range(n)]
        pieces.append((lo, hi, coeffs))
    pieces.append((1 - t1r, Fr(1), [Tr]))               # plateau
    pieces.append((Fr(1), 1 + s / 2, [Tr]))             # extension so that the box reads T at lag 1
    return b, pieces, Tr, s

def h_integral(pieces):
    M = Fr(0)
    for lo, hi, c in pieces:
        M += peval(pint(c), hi - lo)
    return M

# ---------------------------------------------------------------- step 2: exact g on the lag mesh
def lag_mesh(pieces, s):
    pts = {Fr(0), s, Fr(1)}
    ends = sorted({lo for lo, hi, c in pieces} | {hi for lo, hi, c in pieces})
    for bb in ends:
        for v in (bb - s / 2, bb + s / 2):
            if 0 < v < 1: pts.add(v)
    for bi in ends:
        for bj in ends:
            v = bi - bj
            if 0 < v < 1: pts.add(v)
    return sorted(pts)

def exact_g(pieces, s, mesh, verbose=True):
    """Returns list of (ta, tb, poly in t) covering [0, 1]."""
    t0 = time.time()
    npieces = len(pieces)
    # precompute, for each ordered pair (i, j), the antiderivative in u of P_i(u) P_j(u + t + d), d = lo_i - lo_j,
    # evaluated at the four possible affine u-bounds  u = e0 + e1 t  (e1 in {0, -1})
    pre = {}
    for i, (loi, hii, ci) in enumerate(pieces):
        for j, (loj, hij, cj) in enumerate(pieces):
            d = loi - loj
            Q = bivar_mul_uni(bivar_shift(cj, d), ci)
            R = bivar_int_u(Q)
            cand = {
                "lo_i": bivar_eval_u_affine(R, Fr(0), Fr(0)),
                "hi_i": bivar_eval_u_affine(R, hii - loi, Fr(0)),
                "lo_j": bivar_eval_u_affine(R, loj - loi, Fr(-1)),
                "hi_j": bivar_eval_u_affine(R, hij - loi, Fr(-1)),
            }
            pre[(i, j)] = cand
    if verbose: print(f"  pair antiderivatives: {len(pre)} ({time.time()-t0:.1f}s)", flush=True)
    # antiderivative H of h (for the cross term), as piecewise polynomial in the local variable
    Hcum = []; acc = Fr(0)
    for lo, hi, c in pieces:
        P = pint(c)
        Hcum.append((lo, hi, padd(P, [acc])))
        acc += peval(P, hi - lo)
    def H_poly_at(shift, tm):
        """H(t + shift) as a polynomial in t, valid on a lag piece whose midpoint is tm."""
        x = tm + shift
        if x < pieces[0][0]: return [Fr(0)]
        if x >= pieces[-1][1]: return [acc]
        for lo, hi, P in Hcum:
            if lo <= x < hi:
                return pcompose_affine(P, shift - lo, Fr(1))      # u = (t + shift) - lo
        raise RuntimeError
    out = []
    t1 = pieces[0][0]
    for a_, b_ in zip(mesh[:-1], mesh[1:]):
        tm = (a_ + b_) / 2
        g = [Fr(0)]
        # box * box
        if tm < s:
            g = padd(g, [Fr(1) / s, -Fr(1) / (s * s)])
        # cross term (1/s) [H(t + s/2) - H(max(t - s/2, t1))]
        if tm + s / 2 > t1:
            cross = H_poly_at(s / 2, tm)
            if tm - s / 2 > t1:
                cross = psub(cross, H_poly_at(-s / 2, tm))
            g = padd(g, pscale(cross, Fr(1) / s))
        # h * h
        for i, (loi, hii, ci) in enumerate(pieces):
            for j, (loj, hij, cj) in enumerate(pieces):
                alpha = max(loi, loj - tm); beta = min(hii, hij - tm)
                if alpha >= beta: continue
                cand = pre[(i, j)]
                pa = cand["lo_i"] if loi >= loj - tm else cand["lo_j"]
                pb = cand["hi_i"] if hii <= hij - tm else cand["hi_j"]
                g = padd(g, psub(pb, pa))
        out.append((a_, b_, ptrim(g)))
    if verbose: print(f"  lag pieces: {len(out)} ({time.time()-t0:.1f}s)", flush=True)
    return out

# ---------------------------------------------------------------- step 3: certification
def certify(gpieces, mprime, verbose=True):
    t0 = time.time(); worst = None; maxd = 0
    for (a_, b_, g) in gpieces:
        loc = pcompose_affine(psub(g, [mprime]), a_, b_ - a_)        # sigma in [0,1]
        bc = bernstein_coeffs(loc)
        ok, d = certify_nonneg(bc)
        maxd = max(maxd, d)
        if not ok:
            return False, (a_, b_, float(min(bc)), d)
        if worst is None or min(bc) < worst[0]:
            worst = (min(bc), a_, b_)
    if verbose: print(f"  certified g >= m' on all {len(gpieces)} pieces, max subdivision depth {maxd} ({time.time()-t0:.1f}s)")
    return True, (maxd,)

def lower_bound_g(gpieces, maxdepth=40):
    """Rigorous lower bound of min g over [0,1] via Bernstein subdivision (used to pick m')."""
    lb = None
    for (a_, b_, g) in gpieces:
        loc = pcompose_affine(g, a_, b_ - a_)
        stack = [(bernstein_coeffs(loc), 0)]
        piece_lb = None
        # breadth-limited: refine the intervals whose min coeff is below the current best candidate
        best_here = None
        while stack:
            bc, d = stack.pop()
            m = min(bc)
            v = min(bc[0], bc[-1])         # true values at the endpoints are upper bounds for the min
            if best_here is None or v < best_here: best_here = v
            if d < maxdepth and m < best_here - Fr(1, 10 ** 40):
                l, r = decasteljau_split(bc); stack.append((l, d + 1)); stack.append((r, d + 1))
            else:
                piece_lb = m if piece_lb is None else min(piece_lb, m)
        if lb is None or piece_lb < lb: lb = piece_lb
    return lb

if __name__ == "__main__":
    dps = int(sys.argv[1]); p = int(sys.argv[2]); S = int(sys.argv[3])
    rb = int(dps * 3.4) if "--rb" not in sys.argv else int(sys.argv[sys.argv.index("--rb") + 1])
    margin = Fr("1e-27") if "--margin" not in sys.argv else Fr(sys.argv[sys.argv.index("--margin") + 1])
    t1bits = 48
    t1star = "0.16019716051723549160488473885984496639208845032573"
    t1r = Fr(int(round(float(Fr(t1star)) * 2 ** t1bits)), 2 ** t1bits)
    t1r = Fr(round(Fr(t1star) * 2 ** t1bits), 2 ** t1bits)
    print(f"t1 = {t1r} = {float(t1r):.17f}  (t1* - t1 = {float(Fr(t1star) - t1r):.2e})")
    T0 = time.time()
    fam, z, phi, M, err = solve_profile(t1r, p, dps)
    print(f"mp solve: p={p} dps={dps} N={fam.N} |F|={mp.nstr(err, 3)} Phi={mp.nstr(phi, dps)} M={mp.nstr(M, 25)} ({time.time()-T0:.0f}s)")
    b, pieces, Tr, s = build_h(t1r, fam, z, rb, S)
    Mr = h_integral(pieces)
    print(f"rational h: {len(pieces)} pieces, T = {float(Tr):.17f}, M = {float(Mr):.17f}, s = 2^-{S}")
    # float sanity check of the equioscillation residual of the rational polynomials
    hp = fam.full(z)[0]
    xs = np.linspace(float(t1r), 1.0, 2000)
    def hr(x):
        for lo, hi, c in pieces:
            if float(lo) <= x <= float(hi): return float(peval(c, Fr(x) - lo))
        return 0.0
    hv = np.array([hr(x) for x in xs])
    hm = np.array([float(fam.ev(fam.basis_row(mpf(x)), hp, fam.full(z)[1])) for x in xs])
    print(f"  max |h_rational - h_mp| on a grid: {np.abs(hv - hm).max():.2e}")
    mesh = lag_mesh(pieces, s)
    gp = exact_g(pieces, s, mesh)
    # float check of g against a direct numerical convolution
    tt = np.linspace(0.2, 0.8, 7)
    for t in tt:
        for a_, b_, g in gp:
            if a_ <= Fr(t) <= b_:
                print(f"   g({t:.2f}) - T = {float(peval(g, Fr(t)) - Tr):+.3e}"); break
    lb = lower_bound_g(gp)
    print(f"rigorous lower bound of min g on [0,1] (Bernstein): T - {float(Tr - lb):.3e}")
    if "--margin" not in sys.argv:          # automatic: twice the observed deficit, rounded up to 2 digits
        d = Tr - lb
        if d <= 0:
            margin = Fr(1, 10 ** 60)
        else:
            e = math.floor(math.log10(float(d))) - 1
            margin = Fr(math.ceil(float(2 * d) / 10 ** e)) * Fr(10) ** e
        print(f"  margin set to {float(margin):.2e}")
    mprime = Tr - margin
    ok, info = certify(gp, mprime)
    print("certify g >= T - margin:", ok, info)
    if ok:
        val = mprime / (1 + Mr) ** 2
        print("RESULT: C_1c >=", val.numerator, "/", val.denominator)
        mp.dps = 50
        print("       =", mp.nstr(mpf(val.numerator) / val.denominator, 45))
        with open(f"pwcert_p{p}_dps{dps}_S{S}.json", "w") as fh:
            json.dump({"t1": str(t1r), "s": str(s), "T": str(Tr), "margin": str(margin), "M": str(Mr),
                       "bound": str(val), "pieces": [[str(lo), str(hi), [str(c) for c in cs]] for lo, hi, cs in pieces]}, fh)
    print(f"total {time.time()-T0:.0f}s")
