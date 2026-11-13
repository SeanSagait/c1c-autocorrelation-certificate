"""Continuum limit of the one-atom family by piecewise Chebyshev collocation (the "Sundman" step:
regularise the singularities first, then expand).

Normalisation: atom of mass 1 at 0, profile h >= 0 on [t1, 1], h = 0 on (0, t1) (the gap).
Autocorrelation on lags 0 < t <= 1:  g(t) = h(t) + (h*h)(t),  (h*h)(t) = int_{t1}^{1-t} h(x) h(x+t) dx.
Equioscillation:  h(t) + (h*h)(t) = T on [t1, 1],  h(t1) = 0,  so h = T on the plateau [1-t1, 1]
(where h*h = 0).  Value of the family member:  Phi = T / (1 + M)^2,  M = int h.  C_fam = max_t1 Phi(t1).

The kink of h at t1 propagates through the correlation integral: h is analytic on each piece between
consecutive points of  {k t1} U {1 - k t1}  (closed under the maps p -> 1 - p and (p, q) -> p - q that the
integral generates), with singularities of increasing order (C^0 at t1, C^1 at 1 - t1, C^2 at 1 - 2 t1, ...).
We therefore represent h by a polynomial of degree p on each piece (Chebyshev-Lobatto nodes, barycentric
interpolation, shared end nodes => C^0) and collocate the equation at the nodes; the correlation integral of
the piecewise polynomial is computed EXACTLY by Gauss-Legendre quadrature on the sub-intervals cut by both
mesh copies.  Unknowns: h at the interior nodes and T, for a given t1; Newton with an analytic Jacobian
(the map is bilinear).  Phi(t1) is then maximised by Brent.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def breakpoints(t1):
    """Sorted breakpoints of h in [t1, 1]: k t1 and 1 - k t1 inside [t1, 1 - t1], plus 1."""
    pts = {t1, 1.0 - t1, 1.0}
    k = 1
    while k * t1 < 1.0 - t1 - 1e-13:
        pts.add(k * t1); k += 1
    k = 1
    while 1.0 - k * t1 > t1 + 1e-13:
        pts.add(1.0 - k * t1); k += 1
    b = np.array(sorted(pts))
    return b


class Mesh:
    def __init__(self, t1, p, split=True):
        self.t1, self.p = t1, p
        b = breakpoints(t1) if split else np.array([t1, 1.0 - t1, 1.0])
        self.b = b                       # piece j = [b[j], b[j+1]], last piece = plateau
        self.npc = len(b) - 2            # number of polynomial pieces (plateau excluded)
        s = np.cos(np.pi * np.arange(p, -1, -1) / p)       # Lobatto nodes on [-1,1], increasing
        self.s = s
        wb = np.ones(p + 1); wb[1::2] = -1; wb[0] *= 0.5; wb[-1] *= 0.5
        self.wb = wb                                       # barycentric weights
        # global nodes: shared ends
        nodes, gidx = [], []
        for j in range(self.npc):
            lo, hi = b[j], b[j + 1]
            x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * s
            if j == 0:
                idx = list(range(0, p + 1)); nodes.extend(x)
            else:
                idx = [len(nodes) - 1] + list(range(len(nodes), len(nodes) + p)); nodes.extend(x[1:])
            gidx.append(np.array(idx))
        self.x = np.array(nodes)          # global nodes, increasing; x[0] = t1, x[-1] = 1 - t1
        self.gidx = gidx
        self.N = len(self.x)
        gl_x, gl_w = leggauss(p + 1)
        self.gl_x, self.gl_w = gl_x, gl_w

    def basis(self, pts):
        """Dense (len(pts) x N) matrix of Lagrange basis values at pts in [t1, 1 - t1]; rows for pts on the
        plateau [1 - t1, 1] are zero (handled separately as the constant T)."""
        pts = np.asarray(pts, dtype=float)
        L = np.zeros((len(pts), self.N))
        b, p = self.b, self.p
        j = np.clip(np.searchsorted(b, pts, side="right") - 1, 0, self.npc)   # piece index
        inpoly = j < self.npc
        for jj in range(self.npc):
            sel = np.nonzero(inpoly & (j == jj))[0]
            if len(sel) == 0:
                continue
            lo, hi = b[jj], b[jj + 1]
            s = (2.0 * pts[sel] - (lo + hi)) / (hi - lo)
            d = s[:, None] - self.s[None, :]
            exact = np.abs(d) < 1e-15
            with np.errstate(divide="ignore", invalid="ignore"):
                r = self.wb[None, :] / d
                rowsum = r.sum(axis=1)
                lv = r / rowsum[:, None]
            hit = exact.any(axis=1)
            if hit.any():
                lv[hit] = exact[hit].astype(float)
            L[sel[:, None], self.gidx[jj][None, :]] = lv
        return L

    def plateau_mask(self, pts):
        return np.asarray(pts) >= 1.0 - self.t1 - 1e-15

    def subintervals(self, tau):
        """Breakpoints of x -> h(x) h(x + tau) on [t1, 1 - tau]."""
        lo, hi = self.t1, 1.0 - tau
        if hi <= lo + 1e-15:
            return np.array([lo, lo])
        c = np.concatenate([self.b, self.b - tau])
        c = c[(c > lo + 1e-14) & (c < hi - 1e-14)]
        c = np.unique(np.concatenate([[lo], c, [hi]]))
        return c

    def quad(self, tau):
        """Gauss-Legendre points and weights for int_{t1}^{1-tau} (exact for the piecewise polynomial)."""
        c = self.subintervals(tau)
        if len(c) < 2 or c[-1] - c[0] < 1e-15:
            return np.zeros(0), np.zeros(0)
        mid = 0.5 * (c[1:] + c[:-1]); half = 0.5 * (c[1:] - c[:-1])
        xq = (mid[:, None] + half[:, None] * self.gl_x[None, :]).ravel()
        wq = (half[:, None] * self.gl_w[None, :]).ravel()
        return xq, wq


class Family:
    """h on the mesh; unknowns z = (h at interior nodes 1..N-2, T); h[0] = 0, h[N-1] = T."""

    def __init__(self, t1, p, split=True):
        self.m = Mesh(t1, p, split)
        m = self.m
        # precompute quadrature data for every collocation node tau = x[i], i = 0..N-2
        self.Q = []
        for i in range(m.N - 1):
            tau = m.x[i]
            xq, wq = m.quad(tau)
            Lx = m.basis(xq)
            Ly = m.basis(xq + tau)
            py = m.plateau_mask(xq + tau).astype(float)
            self.Q.append((xq, wq, Lx, Ly, py))
        # integration of h over [t1, 1 - t1] (exact): GL on each piece
        xs, ws = [], []
        for j in range(m.npc):
            lo, hi = m.b[j], m.b[j + 1]
            xs.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * m.gl_x); ws.append(0.5 * (hi - lo) * m.gl_w)
        xs, ws = np.concatenate(xs), np.concatenate(ws)
        self.int_row = ws @ m.basis(xs)          # M = int_row . h + T * t1

    def full(self, z):
        h = np.empty(self.m.N); h[0] = 0.0; h[1:-1] = z[:-1]; h[-1] = z[-1]
        return h, z[-1]

    def residual(self, z):
        h, T = self.full(z)
        N = self.m.N
        F = np.empty(N - 1)
        for i in range(N - 1):
            xq, wq, Lx, Ly, py = self.Q[i]
            hx = Lx @ h; hy = Ly @ h + T * py
            F[i] = h[i] + wq @ (hx * hy) - T
        return F

    def jacobian(self, z):
        h, T = self.full(z)
        N = self.m.N
        D = np.zeros((N - 1, N)); P = np.zeros(N - 1)
        for i in range(N - 1):
            xq, wq, Lx, Ly, py = self.Q[i]
            hx = Lx @ h; hy = Ly @ h + T * py
            D[i] = (wq * hy) @ Lx + (wq * hx) @ Ly
            D[i, i] += 1.0
            P[i] = wq @ (hx * py)
        J = np.empty((N - 1, N - 1))
        J[:, :-1] = D[:, 1:-1]
        J[:, -1] = D[:, -1] + P - 1.0
        return J

    def newton(self, z, tol=1e-14, it=30, verbose=False):
        for k in range(it):
            F = self.residual(z)
            err = np.abs(F).max()
            if verbose:
                print(f"    newton {k}: |F|={err:.2e} T={z[-1]:.12f}")
            if err < tol:
                break
            J = self.jacobian(z)
            dz = np.linalg.solve(J, F)
            z = z - dz
        return z, err

    def corr(self, z, taus):
        """(h*h)(tau) for arbitrary lags in (0, 1]."""
        h, T = self.full(z); m = self.m
        out = []
        for tau in np.atleast_1d(taus):
            xq, wq = m.quad(tau)
            if len(xq) == 0:
                out.append(0.0); continue
            hx = m.basis(xq) @ h
            yy = xq + tau
            hy = m.basis(yy) @ h + T * m.plateau_mask(yy)
            out.append(wq @ (hx * hy))
        return np.array(out)

    def eval_h(self, z, pts):
        h, T = self.full(z); m = self.m
        pts = np.atleast_1d(pts)
        v = m.basis(pts) @ h + T * m.plateau_mask(pts)
        v[pts < m.t1] = 0.0
        return v

    def value(self, z):
        h, T = self.full(z)
        M = self.int_row @ h + T * self.m.t1
        return T / (1.0 + M) ** 2, M

    def init_from_discrete(self, w, T0):
        """Initial guess from a discrete family profile w (w[0] = 1 atom, cells 1..K)."""
        K = len(w) - 1
        xg = np.arange(1, K + 1) / K
        hd = w[1:] * K
        h = np.interp(self.m.x, xg, hd)
        z = np.empty(self.m.N - 1); z[:-1] = h[1:-1]; z[-1] = T0
        return z


def solve_t1(t1, p, z0=None, w=None, T0=2.5469, split=True, verbose=False, tol=1e-14):
    fam = Family(t1, p, split)
    if z0 is None:
        z0 = fam.init_from_discrete(w, T0)
    z, err = fam.newton(z0, tol=tol, verbose=verbose)
    phi, M = fam.value(z)
    return fam, z, err, phi, M


if __name__ == "__main__":
    import sys, time
    w = np.load("K4194304.npy")  # warm start: float weights of a K = 2^22 solution (lean.py)
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1602
    t0 = time.time()
    fam, z, err, phi, M = solve_t1(t1, p, w=w, verbose=True)
    h, T = fam.full(z)
    print(f"p={p} t1={t1} N={fam.m.N} pieces={fam.m.npc}: T={T:.13f} M={M:.13f} Phi={phi:.13f} |F|={err:.1e} ({time.time()-t0:.1f}s)")
    print("breakpoints:", np.array2string(fam.m.b, precision=5))
