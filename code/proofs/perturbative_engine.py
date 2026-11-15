"""Exact (sympy) engine for the perturbative route to Conjecture conj:char.

Setting (paper, Sections 3-4): node x_i = 0 with weight |x|^(-alpha) (rho_i = 1), other
nodes x_j with weights kappa_j |x - x_j|^(-alpha), alpha even, local polynomials Pi_m(R^d) in the
monomial basis.  Zero data at x_i (no loss of generality: SWIMLS reproduces Pi_m), arbitrary data
at the other nodes.  Two independent ways of computing the error q(D)u(0) - 0 of a datum q in Q_i:

  direct_error   the normal equations of |x|^alpha J_x, reparametrised as in the proof of
                 Theorem thm:structure (p_x = n(x) + t(x) r(x), n in N_i, r in R, t = |x|^alpha),
                 solved as a truncated power series in x; then u(x) = p_x(x) and its Taylor
                 coefficients.  Independent of the expansion below.

  formula_terms  the exact finite expansion of the paper (Theorem thm:expansion):
                 error = sum_{k>=1} (-1)^(k-1) T_{Delta^{k alpha/2} q}( Gamma^(k-1) gamma ),
                 T_s(phi) = sum_delta c(d^delta s) . [x^delta] phi,
                 with Gamma(x), gamma(x) the Hessian and gradient (in the coordinates of the tuple
                 at x_i) of the reduced far-field objective F_x(eta) = min{B_x(p): Lambda_i p = eta},
                 c(s) the coordinates of a symbol s in the tuple at x_i.  Returns the list of the
                 k-terms (so that the eps-expansion of regime (A) can be read off).

Polynomials/symbols are dicts {exponent tuple: coefficient}.  Series in x are dicts
{exponent tuple: value}, truncated at a total degree.
"""
import itertools
import sympy as sp


# ---------------------------------------------------------------- polynomials and functionals
def monomials(d, m):
    return [e for n in range(m + 1) for e in sorted(
        (e for e in itertools.product(range(n + 1), repeat=d) if sum(e) == n), reverse=True)]


def ff(n, k):
    return sp.ff(n, k) if n >= k else 0


def row(point, symbol, mons):
    """lambda(p) = (symbol(D) p)(point) on the monomials y^e."""
    out = []
    for e in mons:
        v = 0
        for beta, c in symbol.items():
            if all(e[i] >= beta[i] for i in range(len(e))):
                t = c
                for i in range(len(e)):
                    t *= ff(e[i], beta[i]) * sp.sympify(point[i]) ** (e[i] - beta[i])
                v += t
        out.append(v)
    return out


def d_symbol(s, delta):
    """partial^delta of a symbol (derivative in xi)."""
    out = {}
    for e, c in s.items():
        if all(e[i] >= delta[i] for i in range(len(e))):
            k = 1
            for i in range(len(e)):
                k *= ff(e[i], delta[i])
            ne = tuple(e[i] - delta[i] for i in range(len(e)))
            out[ne] = out.get(ne, 0) + c * k
    return {e: c for e, c in out.items() if c != 0}


def laplacian_power(s, ell):
    d = len(next(iter(s))) if s else 1
    for _ in range(ell):
        out = {}
        for j in range(d):
            dj = tuple(2 if i == j else 0 for i in range(d))
            for e, c in d_symbol(s, dj).items():
                out[e] = out.get(e, 0) + c
        s = {e: c for e, c in out.items() if c != 0}
    return s


def deg(s):
    return max((sum(e) for e in s), default=-1)


# ---------------------------------------------------------------- truncated series in x
def s_mul(a, b, order, zero):
    out = {}
    for ea, va in a.items():
        for eb, vb in b.items():
            e = tuple(i + j for i, j in zip(ea, eb))
            if sum(e) <= order:
                out[e] = out.get(e, zero) + va * vb
    return out


def s_add(a, b, zero):
    out = dict(a)
    for e, v in b.items():
        out[e] = out.get(e, zero) + v
    return out


def weight_series(a, alpha, order, d, with_t=True):
    """Taylor series at 0, total order <= order, of |x|^alpha / |x - a|^alpha (with_t) or of
    |x - a|^(-alpha)."""
    a = [sp.sympify(v) for v in a]
    A = sum(v ** 2 for v in a)
    unit = lambda i: tuple(1 if k == i else 0 for k in range(d))
    zero_e = (0,) * d
    u = {}
    for i in range(d):
        u[unit(i)] = u.get(unit(i), 0) - 2 * a[i] / A
        e2 = tuple(2 if k == i else 0 for k in range(d))
        u[e2] = u.get(e2, 0) + sp.Integer(1) / A
    tdeg = alpha if with_t else 0
    inner = order - tdeg
    ser, upow = {zero_e: sp.Integer(1)}, {zero_e: sp.Integer(1)}
    for k in range(1, inner + 1):
        upow = s_mul(upow, u, inner, 0)
        ser = s_add(ser, {e: sp.binomial(sp.Rational(-alpha, 2), k) * v for e, v in upow.items()}, 0)
    ser = {e: v * A ** sp.Rational(-alpha, 2) for e, v in ser.items()}
    if with_t:
        t = {}
        for kappa in monomials(d, alpha // 2):
            if sum(kappa) == alpha // 2:
                c = sp.factorial(alpha // 2)
                for k in kappa:
                    c /= sp.factorial(k)
                t[tuple(2 * k for k in kappa)] = c
        ser = s_mul(t, ser, order, 0)
    return {e: sp.simplify(v) if v.free_symbols else v for e, v in ser.items()}


# ---------------------------------------------------------------- the configuration
class Config:
    """nodes: list of (point, list of symbols, multiplier); node 0 must be at the origin and
    have multiplier 1 (it is x_i).  m, alpha as in the paper."""

    def __init__(self, nodes, m, alpha):
        self.d = len(nodes[0][0])
        assert all(sp.sympify(v) == 0 for v in nodes[0][0])
        self.nodes, self.m, self.alpha = nodes, m, alpha
        self.mons = monomials(self.d, m)
        self.n = len(self.mons)
        self.Li = sp.Matrix([row(nodes[0][0], s, self.mons) for s in nodes[0][1]])
        self.Lj = [sp.Matrix([row(p, s, self.mons) for s in syms]) for p, syms, _ in nodes[1:]]
        self.kap = [sp.sympify(k) for _, _, k in nodes[1:]]
        ns = self.Li.nullspace()
        self.BN = sp.Matrix.hstack(*ns) if ns else sp.zeros(self.n, 0)
        rs = self.Li.T.columnspace()              # R = orthogonal complement of N in coefficients
        self.BR = sp.Matrix.hstack(*rs)
        self.P = self.BR * (self.Li * self.BR).inv()   # right inverse of Lambda_i with range R
        self.tuple0 = nodes[0][1]

    def data_of(self, h):
        """Data of the polynomial h (dict) at the other nodes, as a list of column vectors."""
        hv = sp.Matrix([h.get(e, 0) for e in self.mons])
        return [L * hv for L in self.Lj], self.Li * hv

    def coords(self, s):
        """c(s): coordinates of the symbol s in the tuple at x_i (must lie in Q_i)."""
        T = sp.Matrix([[sy.get(e, 0) for e in self.mons] for sy in self.tuple0]).T
        sv = sp.Matrix([s.get(e, 0) for e in self.mons])
        sol = T.solve_least_squares(sv) if T.shape[0] != T.shape[1] else T.solve(sv)
        assert (T * sol - sv).is_zero_matrix, "symbol not in Q_i"
        return sol

    def _theta_b(self, ys, order, with_t):
        Z = sp.zeros(self.n, self.n)
        Th, b = {}, {}
        for (p, _, _), L, k, y in zip(self.nodes[1:], self.Lj, self.kap, ys):
            w = weight_series(p, self.alpha, order, self.d, with_t)
            M, v = L.T * L, L.T * y
            for e, c in w.items():
                Th[e] = Th.get(e, Z) + k * c * M
                b[e] = b.get(e, sp.zeros(self.n, 1)) + k * c * v
        return Th, b

    # ------------------------------------------------------------ direct route
    def direct_error(self, q, ys):
        """q(D)u(0) with zero data at 0 and data ys at the other nodes."""
        order = deg(q)
        d, n = self.d, self.n
        Th, b = self._theta_b(ys, order, with_t=False)          # Theta(x), b(x)
        tTh, _ = self._theta_b(ys, order, with_t=True)          # t(x) Theta(x)
        Hi = self.Li.T * self.Li
        nR, nN = self.BR.shape[1], self.BN.shape[1]
        # unknowns z = (r, n): Hi BR r + Theta BN n + t Theta BR r = b
        M = {}
        zero_e = (0,) * d
        for e in set(Th) | set(tTh):
            blkR = tTh.get(e, sp.zeros(n, n)) * self.BR
            if e == zero_e:
                blkR = blkR + Hi * self.BR
            blkN = Th.get(e, sp.zeros(n, n)) * self.BN
            M[e] = sp.Matrix.hstack(blkR, blkN)
        M0inv = M[zero_e].inv()
        z = {}
        for e in monomials(d, order):
            rhs = b.get(e, sp.zeros(n, 1))
            for f, Mf in M.items():
                if f != zero_e and all(f[i] <= e[i] for i in range(d)):
                    g = tuple(e[i] - f[i] for i in range(d))
                    rhs = rhs - Mf * z[g]
            z[e] = (M0inv * rhs).applyfunc(sp.cancel)
        # p_x = BN n(x) + t(x) BR r(x)
        tser = {}                                   # t(x) = |x|^alpha, a polynomial
        for kappa in monomials(d, self.alpha // 2):
            if sum(kappa) == self.alpha // 2:
                c = sp.factorial(self.alpha // 2)
                for k in kappa:
                    c /= sp.factorial(k)
                tser[tuple(2 * k for k in kappa)] = c
        nser = {e: self.BN * v[nR:, 0] for e, v in z.items()}
        rser = {e: self.BR * v[:nR, 0] for e, v in z.items()}
        pser = s_add(nser, s_mul(tser, rser, order, sp.zeros(n, 1)), sp.zeros(n, 1))
        # u(x) = sum_e p_e(x) x^e ; coefficient of x^beta
        val = 0
        for beta, c in q.items():
            coef = 0
            for k, e in enumerate(self.mons):
                if all(e[i] <= beta[i] for i in range(d)):
                    g = tuple(beta[i] - e[i] for i in range(d))
                    if g in pser:
                        coef += pser[g][k]
            bf = 1
            for v in beta:
                bf *= sp.factorial(v)
            val += c * bf * coef
        return sp.factor(sp.cancel(val))

    # ------------------------------------------------------------ the derived formula
    def gamma_series(self, ys, order):
        """Gamma(x) (n_i x n_i) and gamma(x) (n_i x 1) as series to the given order."""
        Th, b = self._theta_b(ys, order, with_t=False)
        d, P, BN = self.d, self.P, self.BN
        zero_e = (0,) * d
        TNN = {e: BN.T * M * BN for e, M in Th.items()}
        TNP = {e: BN.T * M * P for e, M in Th.items()}
        TPP = {e: P.T * M * P for e, M in Th.items()}
        bN = {e: BN.T * v for e, v in b.items()}
        bP = {e: P.T * v for e, v in b.items()}
        nN = BN.shape[1]
        if nN:
            T0inv = TNN[zero_e].inv()
            # series inverse of TNN
            inv = {zero_e: T0inv}
            for e in monomials(d, order)[1:]:
                acc = sp.zeros(nN, nN)
                for f, Mf in TNN.items():
                    if f != zero_e and all(f[i] <= e[i] for i in range(d)):
                        g = tuple(e[i] - f[i] for i in range(d))
                        acc += Mf * inv[g]
                inv[e] = -T0inv * acc
            X = s_mul(inv, TNP, order, sp.zeros(nN, P.shape[1]))     # TNN^{-1} TNP
            Xb = s_mul(inv, bN, order, sp.zeros(nN, 1))
            G = s_add(TPP, {e: -v for e, v in s_mul({e: M.T for e, M in TNP.items()}, X, order,
                                                    sp.zeros(P.shape[1], P.shape[1])).items()},
                      sp.zeros(P.shape[1], P.shape[1]))
            g = s_add(bP, {e: -v for e, v in s_mul({e: M.T for e, M in TNP.items()}, Xb, order,
                                                   sp.zeros(P.shape[1], 1)).items()},
                      sp.zeros(P.shape[1], 1))
        else:
            G, g = TPP, bP
        simp = lambda s: {e: v.applyfunc(sp.cancel) for e, v in s.items()}
        return simp(G), simp(g)

    def T(self, s, phi):
        """T_s(phi) = sum_delta c(d^delta s) . [x^delta] phi."""
        tot = 0
        for delta in monomials(self.d, deg(s)):
            ds = d_symbol(s, delta)
            if ds and delta in phi:
                tot += (self.coords(ds).T * phi[delta])[0, 0]
        return sp.cancel(tot)

    def formula_terms(self, q, ys):
        """[E_1, E_2, ...] with error = sum_k (-1)^(k-1) E_k."""
        order = deg(q)
        G, g = self.gamma_series(ys, order)
        terms, k, phi = [], 1, g
        while True:
            qk = laplacian_power(q, k * self.alpha // 2)
            if not qk:
                break
            terms.append(sp.factor(self.T(qk, phi)))
            phi = s_mul(G, phi, order, sp.zeros(len(self.tuple0), 1))
            k += 1
        return terms
