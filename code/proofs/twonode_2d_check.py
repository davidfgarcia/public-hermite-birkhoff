"""Exact check of the extended two-node proposition in two dimensions (alpha = 2, m = 4).

Nodes 0 and a, Hermite data of order 4 at both (each node unisolvent on Pi_4 by itself),
weights |x - x_j|^(-2).  Zero data at 0, data of a polynomial h at a.  We expand the normal
equations of |x|^2 J_x as a truncated bivariate power series (exact rationals), form
u(x) = p_x(x) to total order 4 and evaluate q(D)u(0) for
  q1 = xi1^4 - xi2^4   (deg 4 = 2 alpha, Delta q1 != 0, Delta^2 q1 = 0)  -> error expected != 0
  q2 = xi1^3 xi2 - xi1 xi2^3  (harmonic)                              -> reproduced (Cor. exact)
  q3 = xi1^4           (Delta^2 q3 != 0, outside the proposition)      -> reported only
Run from code/:  python3 proofs/twonode_2d_check.py
"""
import itertools
import sympy as sp

M = 4
MON = [(i, n - i) for n in range(M + 1) for i in range(n, -1, -1)]      # exponents of Pi_4
NB = len(MON)
ORD = [(i, n - i) for n in range(M + 1) for i in range(n + 1)]          # series indices


def jet_rows(c):
    """Rows of p -> D^beta p(c) for |beta| <= 4 (ordinary derivatives)."""
    rows = []
    for b in MON:
        r = []
        for e in MON:
            if e[0] >= b[0] and e[1] >= b[1]:
                r.append(sp.ff(e[0], b[0]) * sp.ff(e[1], b[1])
                         * sp.Integer(c[0]) ** (e[0] - b[0]) * sp.Integer(c[1]) ** (e[1] - b[1]))
            else:
                r.append(0)
        rows.append(r)
    return sp.Matrix(rows)


def weight_series(a):
    """Taylor coefficients at 0, total order <= 4, of |z|^2 / |z - a|^2."""
    z1, z2 = sp.symbols("z1 z2")
    f = (z1 ** 2 + z2 ** 2) / ((z1 - a[0]) ** 2 + (z2 - a[1]) ** 2)
    t = sp.symbols("t")
    ser = sp.expand(sp.series(f.subs({z1: t * z1, z2: t * z2}), t, 0, M + 1).removeO().subs(t, 1))
    P = sp.Poly(ser, z1, z2)
    return {k: P.coeff_monomial(z1 ** k[0] * z2 ** k[1]) for k in ORD}


def errors(a, h):
    L0, La = jet_rows((0, 0)), jet_rows(a)
    hv = sp.Matrix([sp.Poly(h, *sp.symbols("y1 y2")).coeff_monomial(
        sp.Symbol("y1") ** e[0] * sp.Symbol("y2") ** e[1]) for e in MON])
    ya = La * hv
    w = weight_series(a)
    H = {k: w[k] * La.T * La for k in ORD}
    H[(0, 0)] = H[(0, 0)] + L0.T * L0
    b = {k: w[k] * La.T * ya for k in ORD}
    Hinv = H[(0, 0)].inv()
    p = {}
    for k in ORD:                                   # increasing total degree
        rhs = b[k] - sum((H[j] * p[(k[0] - j[0], k[1] - j[1])] for j in ORD
                          if j != (0, 0) and j[0] <= k[0] and j[1] <= k[1]), sp.zeros(NB, 1))
        p[k] = Hinv * rhs
    # u(z) = sum_e p_e(z) z^e ; coefficient of z^beta
    ucoef = {beta: sum(p[(beta[0] - e[0], beta[1] - e[1])][MON.index(e)] for e in MON
                       if e[0] <= beta[0] and e[1] <= beta[1]) for beta in ORD}
    val = lambda q: sum(c * sp.factorial(bb[0]) * sp.factorial(bb[1]) * ucoef[bb] for bb, c in q.items())
    return {"q1 = xi1^4 - xi2^4": val({(4, 0): 1, (0, 4): -1}),
            "q2 = xi1^3 xi2 - xi1 xi2^3": val({(3, 1): 1, (1, 3): -1}),
            "q3 = xi1^4": val({(4, 0): 1})}


if __name__ == "__main__":
    y1, y2 = sp.symbols("y1 y2")
    for a in [(1, 0), (1, 2)]:
        for h in [sp.Integer(1), y1, y1 ** 2]:
            print(f"a = {a}, h = {h}: ", {k: v for k, v in errors(a, h).items()})
