"""High-precision (mpmath) version of cont.py: the residual of the piecewise-Chebyshev collocation is evaluated
in mp arithmetic, Newton uses the double-precision Jacobian of cont.py (linear convergence, 16 digits per step).
Phi(t1) = T/(1+M)^2 is maximised over t1 by successive parabolic interpolation in mp.
usage: python contmp.py dps p [rounds]
"""
import sys, time, numpy as np
from mpmath import mp, mpf, fdot, fsum
from cont import Family as FamilyF, solve_t1

def gauss_legendre(n):
    """n-point Gauss-Legendre nodes/weights at the current mp precision."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = mp.cos(mp.pi * (i - mpf(1) / 4) / (n + mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp; x -= dx
            if abs(dx) < mpf(10) ** (-mp.dps + 2):
                break
        p0, p1 = mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def breakpoints_mp(t1):
    pts = {t1, 1 - t1, mpf(1)}
    k = 1
    while k * t1 < 1 - t1:
        pts.add(k * t1); k += 1
    k = 1
    while 1 - k * t1 > t1:
        pts.add(1 - k * t1); k += 1
    return sorted(pts)


class FamilyMP:
    def __init__(self, t1, p, verbose=False):
        t0 = time.time()
        self.t1, self.p = t1, p
        b = breakpoints_mp(t1); self.b = b; self.npc = len(b) - 2
        s = [mp.cos(mp.pi * j / p) for j in range(p, -1, -1)]; self.s = s
        wb = [mpf((-1) ** j) for j in range(p + 1)]; wb[0] /= 2; wb[-1] /= 2; self.wb = wb
        nodes, gidx = [], []
        for j in range(self.npc):
            lo, hi = b[j], b[j + 1]
            x = [(lo + hi) / 2 + (hi - lo) / 2 * sj for sj in s]
            if j == 0:
                idx = list(range(0, p + 1)); nodes.extend(x)
            else:
                idx = [len(nodes) - 1] + list(range(len(nodes), len(nodes) + p)); nodes.extend(x[1:])
            gidx.append(idx)
        self.x = nodes; self.gidx = gidx; self.N = len(nodes)
        self.gl_x, self.gl_w = gauss_legendre(p + 1)
        # quadrature data per collocation node
        self.Q = []
        for i in range(self.N - 1):
            tau = nodes[i]
            xq, wq = self.quad(tau)
            rows_x = [self.basis_row(xx) for xx in xq]
            rows_y = [self.basis_row(xx + tau) for xx in xq]
            self.Q.append((wq, rows_x, rows_y))
        # integral row
        self.int_rows = []
        for j in range(self.npc):
            lo, hi = b[j], b[j + 1]
            for gx, gw in zip(self.gl_x, self.gl_w):
                self.int_rows.append(((hi - lo) / 2 * gw, self.basis_row((lo + hi) / 2 + (hi - lo) / 2 * gx)))
        if verbose:
            print(f"    mp mesh p={p} N={self.N} built ({time.time()-t0:.0f}s)", flush=True)

    def piece(self, x):
        b = self.b
        j = 0
        while j < self.npc and x >= b[j + 1]:
            j += 1
        return j

    def basis_row(self, x):
        """(indices, values, plateau flag): h(x) = sum values*h[idx] + T*flag."""
        if x >= 1 - self.t1:
            return ([], [], 1)
        if x < self.t1:
            return ([], [], 0)
        j = self.piece(x)
        lo, hi = self.b[j], self.b[j + 1]
        s = (2 * x - (lo + hi)) / (hi - lo)
        r = []
        for k in range(self.p + 1):
            d = s - self.s[k]
            if abs(d) < mpf(10) ** (-mp.dps + 5):
                return ([self.gidx[j][k]], [mpf(1)], 0)
            r.append(self.wb[k] / d)
        tot = fsum(r)
        return (self.gidx[j], [v / tot for v in r], 0)

    def quad(self, tau):
        lo, hi = self.t1, 1 - tau
        if hi <= lo:
            return [], []
        c = [lo] + [v for v in (self.b + [bb - tau for bb in self.b]) if lo < v < hi] + [hi]
        c = sorted(set(c))
        xq, wq = [], []
        for a, bq in zip(c[:-1], c[1:]):
            if bq - a < mpf(10) ** (-mp.dps + 5):
                continue
            mid, half = (a + bq) / 2, (bq - a) / 2
            for gx, gw in zip(self.gl_x, self.gl_w):
                xq.append(mid + half * gx); wq.append(half * gw)
        return xq, wq

    def full(self, z):
        return [mpf(0)] + list(z[:-1]) + [z[-1]], z[-1]

    def ev(self, row, h, T):
        idx, vals, fl = row
        v = fdot(vals, [h[i] for i in idx]) if idx else mpf(0)
        return v + T if fl else v

    def residual(self, z):
        h, T = self.full(z)
        F = []
        for i in range(self.N - 1):
            wq, rows_x, rows_y = self.Q[i]
            acc = fdot(wq, [self.ev(rx, h, T) * self.ev(ry, h, T) for rx, ry in zip(rows_x, rows_y)]) if wq else mpf(0)
            F.append(h[i] + acc - T)
        return F

    def value(self, z):
        h, T = self.full(z)
        M = fsum(wq * self.ev(row, h, T) for wq, row in self.int_rows) + T * self.t1
        return T / (1 + M) ** 2, M


def solve_mp(t1, p, w, verbose=False, maxit=8):
    """Newton in mp with the float Jacobian.  Returns (fam, z, Phi, M, residual)."""
    t0 = time.time()
    famf, zf, err, phif, Mf = solve_t1(float(t1), p, w=w)
    J = famf.jacobian(zf)
    Jinv = np.linalg.inv(J)
    fam = FamilyMP(t1, p, verbose)
    z = [mpf(v) for v in zf]
    for it in range(maxit):
        F = fam.residual(z)
        err = max(abs(v) for v in F)
        if verbose:
            print(f"    mp newton {it}: |F| = {mp.nstr(err, 3)} ({time.time()-t0:.0f}s)", flush=True)
        if err < mpf(10) ** (-mp.dps + 3):
            break
        Ff = np.array([float(v) for v in F])
        dz = Jinv @ Ff
        # scale: F may be far below float range for tiny residuals -> split exponent
        z = [zi - mpf(float(d)) for zi, d in zip(z, dz)]
    phi, M = fam.value(z)
    return fam, z, phi, M, err


if __name__ == "__main__":
    dps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mp.dps = dps
    w = np.load("K4194304.npy")  # warm start: float weights of a K = 2^22 solution (lean.py)
    t1c = mpf(float(np.load("t1star.npy")[0]))
    t0 = time.time()
    cache = {}

    def Phi(t1):
        if t1 not in cache:
            fam, z, phi, M, err = solve_mp(t1, p, w, verbose=True)
            cache[t1] = (phi, M, z[-1], err)
            print(f"  t1={mp.nstr(t1, 25)}  Phi={mp.nstr(phi, dps)}  T={mp.nstr(z[-1], 25)}  M={mp.nstr(M, 25)} |F|={mp.nstr(err, 3)} ({time.time()-t0:.0f}s)", flush=True)
        return cache[t1][0]

    delta = mpf("1e-6")
    c = t1c
    for r in range(rounds):
        fm, f0, fp = Phi(c - delta), Phi(c), Phi(c + delta)
        denom = fp - 2 * f0 + fm
        shift = -delta * (fp - fm) / (2 * denom)
        print(f" round {r}: delta={mp.nstr(delta, 3)} curvature Phi''={mp.nstr(denom / delta ** 2, 10)} shift={mp.nstr(shift, 10)}", flush=True)
        c = c + shift
        delta = delta * delta * 100 if delta > mpf("1e-9") else delta * delta
    phi, M, T, err = cache[c] if c in cache else (None, None, None, None)
    if phi is None:
        Phi(c); phi, M, T, err = cache[c]
    a = 1 / (1 + M)
    print("RESULT")
    print(" t1* =", mp.nstr(c, dps))
    print(" T*  =", mp.nstr(T, dps))
    print(" M*  =", mp.nstr(M, dps))
    print(" a*  =", mp.nstr(a, dps))
    print(" V*  = Phi* =", mp.nstr(phi, dps))
    print(" 1-6t1* =", mp.nstr(1 - 6 * c, dps))
    with open(f"mp_result_dps{dps}_p{p}.txt", "w") as fh:
        fh.write(f"dps={dps} p={p}\nt1={mp.nstr(c, dps)}\nT={mp.nstr(T, dps)}\nM={mp.nstr(M, dps)}\na={mp.nstr(a, dps)}\nV={mp.nstr(phi, dps)}\n")
    # convergence check in p at the final t1
    for p2 in [p + 4, p + 8]:
        fam2, z2, phi2, M2, err2 = solve_mp(c, p2, w)
        print(f" check p={p2}: Phi={mp.nstr(phi2, dps)}  diff vs p={p}: {mp.nstr(phi2 - phi, 3)}  T diff {mp.nstr(z2[-1] - T, 3)} |F|={mp.nstr(err2, 3)} ({time.time()-t0:.0f}s)", flush=True)
