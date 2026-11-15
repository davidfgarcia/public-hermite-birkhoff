"""Exact (rational-arithmetic) SWIMLS computations in one dimension for Section 4.6
(Example ex:cancellation and the three-node witness in the remark after Conjecture conj:char).

Weights |x - x_j|^(-alpha) (rho = 1), local polynomials of degree m in the monomial basis.
Each node carries a list of linear functionals on Pi_m, given as coefficient rows
(lambda(p) = sum_k row[k] * p_k for p = sum_k p_k y^k).  We multiply J_x by |x - x_i|^alpha
(x_i = 0), solve the normal equations as truncated power series in x, form u(x) = p_x(x) and
read off u^(4)(0) from its Taylor coefficient.  Everything is exact.  For the numerical data
the result is cross-checked independently: the normal equations are solved in closed form
(u as a rational function of x) and the fourth derivative of u is taken in the limit x -> 0.

  (a) two nodes 0, a, Hermite data of order 4 at 0 (normalised jets p^(j)(0)/j!), and at a
      the tuple (L0+L1, L0+L1+L2, L0, L3, L4), L_j(p) = p^(j)(a-1)/j!.  The error in
      u^(4)(0) as a function of a and of the data; it vanishes identically at a = 1.
      Same span at a with the plain tuple (L0,...,L4): the error at a = 1 is non-zero.
  (b) three nodes 0, 1, 2: jets of order <= 4 at 0, f, f', f'' at 1, f, f' at 2.
      Zero data at 0, data of the constant 1 elsewhere: u^(4)(0) = 623/24.

Run from code/:  python3 proofs/exact_1d_examples.py
"""
import sympy as sp

x, a = sp.symbols("x a")
M, ALPHA = 4, 2


def taylor_rows(c, normalised=True):
    """Rows of p -> p^(j)(c)/j! (or p^(j)(c)) for j = 0..M, in the monomial basis."""
    rows = []
    for j in range(M + 1):
        r = [sp.binomial(k, j) * c ** (k - j) if k >= j else 0 for k in range(M + 1)]
        if not normalised:
            r = [sp.factorial(j) * e for e in r]
        rows.append(r)
    return rows


def u4_at_0(nodes):
    """nodes: list of (position, rows, data); the first node must be at 0 and its data must be
    unisolvent on Pi_m by themselves, so that the normal matrix of |x|^alpha J_x is invertible
    at x = 0.  The normal equations are then solved as truncated power series in x (exact
    rational arithmetic); returns u^(4)(0) = 4! [x^4] p_x(x)."""
    assert nodes[0][0] == 0
    n = M + 1
    H = [sp.zeros(n, n) for _ in range(n)]          # H(x) = sum_k H[k] x^k + O(x^5)
    b = [sp.zeros(n, 1) for _ in range(n)]
    for i, (pos, rows, ys) in enumerate(nodes):
        if i == 0:
            wt = [1] + [0] * M
        else:                                       # x^alpha / (x - c)^alpha, alpha = 2
            wt = [0, 0] + [(k + 1) / sp.sympify(pos) ** (k + 2) for k in range(M - 1)]
        for r, yv in zip(rows, ys):
            R = sp.Matrix([r])
            for k in range(n):
                if wt[k] != 0:
                    H[k] += wt[k] * R.T * R
                    b[k] += wt[k] * yv * R.T
    H0inv = H[0].inv()
    p = []
    for k in range(n):
        rhs = b[k] - sum((H[j] * p[k - j] for j in range(1, k + 1)), sp.zeros(n, 1))
        p.append((H0inv * rhs).applyfunc(sp.cancel))
    c4 = sum(p[M - j][j] for j in range(M + 1))      # [x^4] sum_j p_j(x) x^j
    return sp.factor(sp.factorial(M) * c4)


def example_a(apos, ya, plain=False):
    L = taylor_rows(apos - 1)
    if plain:
        rows_a = L
    else:
        add = lambda *rs: [sum(t) for t in zip(*rs)]
        rows_a = [add(L[0], L[1]), add(L[0], L[1], L[2]), L[0], L[3], L[4]]
    return u4_at_0([(0, taylor_rows(0), [0] * (M + 1)), (apos, rows_a, ya)])


def crosscheck(nodes):
    """Independent check: closed-form rational u(x), then the limit of u^(4)(x) as x -> 0.
    nodes: list of (position, list of functionals on sympy polynomials in y, data)."""
    y = sp.Symbol("y")
    cs = sp.symbols("c0:5")
    p = sum(c * y ** k for k, c in enumerate(cs))
    J = sum((f(p) - v) ** 2 / (x - pos) ** ALPHA for pos, funs, ys in nodes for f, v in zip(funs, ys))
    sol = sp.solve([sp.diff(J, c) for c in cs], cs, dict=True)[0]
    u = sp.cancel(p.subs(sol).subs(y, x))
    return sp.limit(sp.diff(u, x, M), x, 0)


if __name__ == "__main__":
    print("(a) two nodes, zero data at 0; the target u^(4)(0) = 0 is the prescribed value")
    # Gram matrix of the tuple at a = 1 in the monomial basis
    L = taylor_rows(0)
    add = lambda *rs: [sum(t) for t in zip(*rs)]
    T = sp.Matrix([add(L[0], L[1]), add(L[0], L[1], L[2]), L[0], L[3], L[4]])
    print("    Gram matrix at a=1, first row:", list((T.T * T).row(0)))
    ys = sp.symbols("y0:5")
    e1 = example_a(1, list(ys))
    print("    a = 1, general data at a:  error =", e1)
    print("    a = 2, data of h = 1 at a: error =", example_a(2, [1, 1, 1, 0, 0]))
    print("    general a, data of h = 1:  error =", example_a(a, [1, 1, 1, 0, 0]))
    print("    a = 1, plain tuple (L0..L4), data (1,0,0,0,0): error =",
          example_a(1, [1, 0, 0, 0, 0], plain=True))
    # with the data at 0 general too (reproduction for all data at a = 1)
    y0s = sp.symbols("z0:5")
    rows_a = [add(L[0], L[1]), add(L[0], L[1], L[2]), L[0], L[3], L[4]]
    full = u4_at_0([(0, taylor_rows(0), list(y0s)), (1, rows_a, list(ys))])
    print("    a = 1, all data general: u^(4)(0) - 24*z4 =", sp.simplify(full - 24 * y0s[4]))

    print("(b) three nodes 0, 1, 2 (ordinary derivatives)")
    J = lambda c, k: taylor_rows(c, normalised=False)[: k + 1]
    val = u4_at_0([(0, J(0, 4), [0] * 5), (1, J(1, 2), [1, 0, 0]), (2, J(2, 1), [1, 0])])
    print("    u^(4)(0) =", val, " (prescribed 0)")

    print("cross-checks by closed-form solution")
    y = sp.Symbol("y")
    D = lambda k, c: (lambda p: sp.diff(p, y, k).subs(y, c))
    T = lambda j, c: (lambda p: sp.diff(p, y, j).subs(y, c) / sp.factorial(j))
    print("    (b) u^(4)(0) =", crosscheck([(0, [D(k, 0) for k in range(5)], [0] * 5),
                                          (1, [D(k, 1) for k in range(3)], [1, 0, 0]),
                                          (2, [D(k, 2) for k in range(2)], [1, 0])]))
    tup = [lambda p: T(0, 1)(p) + T(1, 1)(p), lambda p: T(0, 1)(p) + T(1, 1)(p) + T(2, 1)(p),
           T(0, 1), T(3, 1), T(4, 1)]                        # the tuple at a = 2 (Taylor at a - 1 = 1)
    print("    (a) a = 2, h = 1: u^(4)(0) =",
          crosscheck([(0, [T(k, 0) for k in range(5)], [0] * 5), (2, tup, [1, 1, 1, 0, 0])]))
