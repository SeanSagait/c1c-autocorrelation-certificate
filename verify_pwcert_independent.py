"""Second exact checker for pwcert_p20_dps40_S120.json, written independently of verify_pwcert.py.

Written from the certificate format description only. All quantities that matter use fractions.Fraction.

f = (1/s) 1_[-s/2, s/2) + h,  h(x) = sum_k c_k (x - lo)^k on [lo, hi) for each JSON piece, 0 elsewhere.
g(t) = int f(x) f(x+t) dx computed exactly as a piecewise polynomial in t on [0, 1+s] (full support of g on t>=0).
"""
import json, sys, time, hashlib
from fractions import Fraction as Fr
from math import comb

t0 = time.time()
path = sys.argv[1]
raw = open(path, 'rb').read()
print('sha256', hashlib.sha256(raw).hexdigest())
D = json.loads(raw)
F = lambda x: Fr(x)
s = F(D['s']); T = F(D['T']); margin = F(D['margin']); Mclaim = F(D['M']); bound_claim = F(D['bound'])
t1 = F(D['t1'])
print('s == 2^-120:', s == Fr(1, 2**120))

# pieces: (lo, hi, coeffs in local variable u = x - lo)
pieces = [(s * Fr(-1, 2), s / 2, [1 / s])]
for lo, hi, c in D['pieces']:
    pieces.append((F(lo), F(hi), [F(x) for x in c]))
pieces.sort(key=lambda p: p[0])
for a, b, c in pieces:
    assert a < b
for k in range(len(pieces) - 1):
    assert pieces[k][1] <= pieces[k + 1][0], 'overlap'
gaps = [(pieces[k][1], pieces[k + 1][0]) for k in range(len(pieces) - 1) if pieces[k][1] < pieces[k + 1][0]]
print('pieces', len(pieces), 'gaps (f=0 there):', [(float(a), float(b)) for a, b in gaps])
print('first h piece starts at t1:', D['pieces'][0][0] == D['t1'])
print('plateau value equals T:', all(F(p[2][0]) == T and len(p[2]) == 1 for p in D['pieces'][-2:]))

# ---------- polynomial helpers (coeff lists, low degree first) ----------
def padd(p, q):
    n = max(len(p), len(q)); r = [Fr(0)] * n
    for i, x in enumerate(p): r[i] += x
    for i, x in enumerate(q): r[i] += x
    return r

def pscale(p, c):
    return [x * c for x in p]

def taylor_shift(p, d):
    """return q with q(v) = p(v + d)"""
    q = list(p); n = len(q)
    for i in range(n - 1):
        for j in range(n - 2, i - 1, -1):
            q[j] += d * q[j + 1]
    return q

def peval(p, x):
    r = Fr(0)
    for c in reversed(p): r = r * x + c
    return r

def bernstein(p, w):
    """Bernstein coefficients on [0, w] of p(v), degree n = len(p)-1"""
    n = len(p) - 1
    a = [p[j] * w ** j for j in range(n + 1)]
    return [sum(Fr(comb(i, j), comb(n, j)) * a[j] for j in range(i + 1)) for i in range(n + 1)]

def casteljau(b):
    n = len(b) - 1; L = [b[0]]; R = [b[-1]]; cur = list(b)
    for k in range(n):
        cur = [(cur[i] + cur[i + 1]) / 2 for i in range(len(cur) - 1)]
        L.append(cur[0]); R.append(cur[-1])
    return L, R[::-1]

def bern_lower(b, target, depth=0, maxdepth=16, stats=None):
    """rigorous lower bound of the polynomial with Bernstein coeffs b on its interval,
    subdividing until the bound reaches `target` (or maxdepth)."""
    lb = min(b)
    # stop if target reached, depth exhausted, or an endpoint (an attained value) is already below target
    if lb >= target or depth >= maxdepth or min(b[0], b[-1]) < target:
        if stats is not None: stats[0] += 1; stats[1] = max(stats[1], depth)
        return lb
    L, R = casteljau(b)
    return min(bern_lower(L, target, depth + 1, maxdepth, stats), bern_lower(R, target, depth + 1, maxdepth, stats))

# ---------- 1. f >= 0 ----------
fmin_ok = True
for a, b, c in pieces:
    if len(c) == 1:
        ok = c[0] >= 0
    else:
        bb = bernstein(c, b - a)
        lb = bern_lower(bb, Fr(0))
        ok = lb >= 0
        print('  piece [%.6f,%.6f) deg %d: value at lo %.3e, at hi- %.3e, Bernstein lower bound %.3e'
              % (a, b, len(c) - 1, float(c[0]), float(peval(c, b - a)), float(lb)))
    fmin_ok &= ok
print('f >= 0 on every piece:', fmin_ok)

# ---------- 2. integral ----------
I = Fr(0)
for a, b, c in pieces:
    L = b - a
    I += sum(ck * L ** (k + 1) / (k + 1) for k, ck in enumerate(c))
M = I - 1
print('int f = 1 + M; M matches JSON:', M == Mclaim, ' M =', float(M))
print('t=%.1fs' % (time.time() - t0))

# ---------- 3. g as piecewise polynomial ----------
# pair (i,j): int_{x in piece i, x+t in piece j} p_i(x-a_i) p_j(x+t-a_j) dx.
# u = x - a_i, tau = t - d with d = a_j - a_i, then x + t - a_j = u + tau.
# u ranges over [max(0, -tau), min(L_i, L_j - tau)].
# G(u,tau) = p_i(u) p_j(u+tau); Fb(U,tau) = int_0^U G du.  Fb[n][r] = coeff of U^n tau^r.
npc = len(pieces)
pairdata = {}

def pair_branches(i, j):
    if (i, j) in pairdata: return pairdata[(i, j)]
    ai, bi, ci = pieces[i]; aj, bj, cj = pieces[j]
    Li = bi - ai; Lj = bj - aj; d = aj - ai
    di = len(ci) - 1; dj = len(cj) - 1
    G = [[Fr(0)] * (dj + 1) for _ in range(di + dj + 1)]
    for k, ck in enumerate(cj):
        if ck == 0: continue
        for l in range(k + 1):
            v = ck * comb(k, l)
            for m, am in enumerate(ci):
                G[m + l][k - l] += am * v
    Fb = [[Fr(0)] * (dj + 1)] + [[x / (n + 1) for x in row] for n, row in enumerate(G)]
    # branch polys in tau
    # U = L_i
    hi_Li = [sum(Fb[n][r] * Li ** n for n in range(len(Fb))) for r in range(dj + 1)]
    # U = -tau
    deg = len(Fb) - 1 + dj
    lo_mt = [Fr(0)] * (deg + 1)
    for n in range(len(Fb)):
        sg = -1 if n % 2 else 1
        for r in range(dj + 1):
            if Fb[n][r]: lo_mt[n + r] += sg * Fb[n][r]
    # U = L_j - tau : (L_j - tau)^n = sum_q C(n,q) L_j^(n-q) (-tau)^q
    hi_Ljmt = [Fr(0)] * (deg + 1)
    Ljp = [Lj ** e for e in range(len(Fb))]
    for n in range(len(Fb)):
        expn = [comb(n, q) * Ljp[n - q] * (-1 if q % 2 else 1) for q in range(n + 1)]
        for r in range(dj + 1):
            fr = Fb[n][r]
            if not fr: continue
            for q in range(n + 1):
                hi_Ljmt[q + r] += fr * expn[q]
    # convert tau-polys to t-polys: tau = t - d -> p(t - d)
    res = {k: taylor_shift(v, -d) for k, v in (('Li', hi_Li), ('-tau', lo_mt), ('Lj-tau', hi_Ljmt))}
    res['geom'] = (Li, Lj, d)
    pairdata[(i, j)] = res
    return res

# lag breakpoints: all differences of piece endpoints
ends = sorted(set([p[0] for p in pieces] + [p[1] for p in pieces]))
tmax = ends[-1] - ends[0]
bps = sorted(set([Fr(0), Fr(1), tmax] + [y - x for x in ends for y in ends if 0 < y - x < tmax]))
lagints = list(zip(bps[:-1], bps[1:]))
print('lag mesh: %d intervals on [0, %s]; %d inside [0,1]' % (len(lagints), float(tmax), sum(1 for a, b in lagints if b <= 1)))

gpolys = []   # (ta, tb, poly in v = t - ta)
for ta, tb in lagints:
    tm = (ta + tb) / 2
    acc = [Fr(0)]
    for i in range(npc):
        ai, bi, _ = pieces[i]
        for j in range(npc):
            aj, bj, _ = pieces[j]
            d = aj - ai; Li = bi - ai; Lj = bj - aj
            lo = max(Fr(0), d - tm); hi = min(Li, Lj + d - tm)
            if lo >= hi: continue
            br = pair_branches(i, j)
            hp = br['Li'] if Li <= Lj + d - tm else br['Lj-tau']
            # sanity: branch choice must be constant over the whole lag interval
            assert (Li <= Lj + d - ta) == (Li <= Lj + d - tb) or Li == Lj + d - ta or Li == Lj + d - tb
            assert (d - ta >= 0) == (d - tb >= 0) or d == ta or d == tb
            contrib = list(hp)
            if d - tm > 0:
                contrib = padd(contrib, pscale(br['-tau'], -1))
            acc = padd(acc, contrib)
    while len(acc) > 1 and acc[-1] == 0: acc.pop()
    gpolys.append((ta, tb, taylor_shift(acc, ta)))
print('g built, t=%.1fs; max degree %d' % (time.time() - t0, max(len(p) - 1 for _, _, p in gpolys)))

# consistency checks
cont = all(peval(gpolys[k][2], gpolys[k][1] - gpolys[k][0]) == gpolys[k + 1][2][0] for k in range(len(gpolys) - 1))
print('g continuous across all lag breakpoints (exact):', cont)
print('g(tmax) == 0:', peval(gpolys[-1][2], gpolys[-1][1] - gpolys[-1][0]) == 0)
g0 = gpolys[0][2][0]
f2 = Fr(0)
for a, b, c in pieces:
    L = b - a; sq = [Fr(0)] * (2 * len(c) - 1)
    for p_, x in enumerate(c):
        for q_, y in enumerate(c): sq[p_ + q_] += x * y
    f2 += sum(ck * L ** (k + 1) / (k + 1) for k, ck in enumerate(sq))
print('g(0) == int f^2 (exact):', g0 == f2)
intg = sum(sum(ck * (tb - ta) ** (k + 1) / (k + 1) for k, ck in enumerate(p)) for ta, tb, p in gpolys)
print('2 int_0^tmax g == (int f)^2 (exact):', 2 * intg == I * I)
print('t=%.1fs' % (time.time() - t0))

# ---------- 4. rigorous lower bound of min_[0,1] g ----------
target = T
mins = []
stats = [0, 0]
for ta, tb, p in gpolys:
    if ta >= 1: break
    b = bernstein(p, tb - ta)
    lb = bern_lower(b, target, stats=stats)
    exact_end = min(p[0], peval(p, tb - ta))
    const = all(x == 0 for x in p[1:])
    mins.append((lb, ta, tb, const, exact_end))
L = min(m[0] for m in mins)
print('Bernstein leaves %d, max depth %d' % tuple(stats))
for lb, ta, tb, const, ee in mins:
    print('  [%.17f, %.17f] const=%s  (lb - T) = %.3e  (min endpoint - T) = %.3e' % (ta, tb, const, float(lb - T), float(ee - T)))
print('rigorous lower bound L of min_[0,1] g: L - T =', float(L - T), ' L >= T - margin:', L >= T - margin)
mprime = T - margin
ratio = L / I ** 2
print('bound from JSON m\' = T - margin:  m\'/(1+M)^2 == JSON bound:', mprime / I ** 2 == bound_claim)
print('our bound L/(1+M)^2 =', ratio)
from decimal import Decimal, getcontext
getcontext().prec = 60
dec = lambda q: Decimal(q.numerator) / Decimal(q.denominator)
print('our bound decimal  :', dec(ratio))
print('JSON bound decimal :', dec(bound_claim))
print('claimed            : 0.410378195722515456620959705443384843306')
print('our bound >= JSON bound:', ratio >= bound_claim, '; JSON bound >= claimed decimal:', bound_claim >= Fr('0.410378195722515456620959705443384843306'))
print('total time %.1fs' % (time.time() - t0))
