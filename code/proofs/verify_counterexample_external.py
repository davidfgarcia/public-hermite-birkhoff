"""Exact checks for the counterexample to the unrestricted conjecture (paper, Section 4:
d = 6, m = 6, alpha = 4, a non-homogeneous central tuple; the error vanishes identically).

Requires SymPy. Run from code/:  python3 proofs/verify_counterexample_external.py
The proof in the note establishes the identity for every off-node position.
These checks additionally verify the differential identities and the finite
Neumann remainder independently in six-variable polynomial arithmetic.
"""

from math import comb

import sympy as sp


def laplacian(polynomial, variables):
    return sp.expand(sum(sp.diff(polynomial, variable, 2) for variable in variables))


def at_origin(expression, variables):
    return expression.subs(dict.fromkeys(variables, 0))


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    print(f"PASS  {label}")


dimension = 6
x = sp.symbols("x0:6")
a = sp.symbols("a0:6", real=True)
radius_x = sum(variable**2 for variable in x)
radius_a = sum(variable**2 for variable in a)
dot = sum(left * right for left, right in zip(x, a))
c = sp.Rational(2, 3)
b = sp.Rational(11, 12)

# The homogeneous monomials of degrees >= 2 complete the seven modified
# low-degree features to a basis. The change-of-basis matrix has identity
# diagonal and only entries from degrees 0->2 and 1->3 below the diagonal.
feature_count = 1 + dimension + sum(comb(dimension + degree - 1, degree) for degree in range(2, 7))
check("feature count equals dim Pi_6(R^6) = 924", feature_count == comb(12, 6) == 924)

phi_0_a = 1 - c * radius_a
phi_i_a = [variable * (1 - b * radius_a) for variable in a]

# Only terms through degree two in x can affect Delta_x(w P)(0).
kernel_quadratic = (1 - c * radius_x) * phi_0_a
kernel_quadratic += sum(left * right for left, right in zip(x, phi_i_a))
kernel_quadratic += sum(x[i] ** 2 * a[i] ** 2 for i in range(dimension))
kernel_quadratic += sum(x[i] * x[j] * a[i] * a[j] for i in range(dimension) for j in range(i + 1, dimension))

gradient_p = [at_origin(sp.diff(kernel_quadratic, variable), x) for variable in x]
laplacian_p = at_origin(laplacian(kernel_quadratic, x), x)
check("kernel gradient", gradient_p == phi_i_a)
check("kernel Laplacian", sp.expand(laplacian_p + 8 - sp.Rational(22, 3) * radius_a) == 0)

weight = sum((left - right) ** 2 for left, right in zip(x, a)) ** -2
laplacian_weight = sp.factor(sum(sp.diff(weight, variable, 2) for variable in x))
check("|x-a|^-4 is harmonic in dimension six", laplacian_weight == 0)
gradient_w = [at_origin(sp.diff(weight, variable), x) for variable in x]
check("weight gradient", all(sp.cancel(value - 4 * ai / radius_a**3) == 0 for value, ai in zip(gradient_w, a)))

product_laplacian = laplacian_p / radius_a**2 + 2 * sum(left * right for left, right in zip(gradient_p, gradient_w))
check("Delta_x[weight * kernel](0) vanishes identically in a", sp.cancel(product_laplacian) == 0)

q = radius_x**3
delta_q = laplacian(laplacian(q, x), x)
check("Delta^2 |xi|^6 = 1920 |xi|^2", sp.expand(delta_q - 1920 * radius_x) == 0)
check("Delta^4 |xi|^6 = 0", laplacian(laplacian(delta_q, x), x) == 0)

# Independent direct order-six check. Multiplying the quadratic Taylor jet
# of w P by |x|^4 gives the entire part of u seen by q(D)=Delta^3.
# Both the denominator correction in a one-node Sherman-Morrison solve and
# all higher Neumann terms for multiple nodes begin at degree eight.
sample_points = [
    (1, 0, 0, 0, 0, 0),
    (1, 2, -1, sp.Rational(1, 2), 3, -2),
    (sp.Rational(1, 3), -2, sp.Rational(3, 5), 1, -1, 2),
]
for position in sample_points:
    substitutions = dict(zip(a, position))
    squared_norm = radius_a.subs(substitutions)
    linear = dot.subs(substitutions)
    p0 = phi_0_a.subs(substitutions)
    p1 = (1 - b * squared_norm) * linear
    kernel_degree_two = sp.expand(-c * radius_x * p0)
    kernel_degree_two += sum(x[i] ** 2 * position[i] ** 2 for i in range(dimension))
    kernel_degree_two += sum(x[i] * x[j] * position[i] * position[j] for i in range(dimension) for j in range(i + 1, dimension))
    w0 = squared_norm**-2
    w1 = 4 * linear / squared_norm**3
    w2 = -2 * radius_x / squared_norm**3 + 12 * linear**2 / squared_norm**4
    sixth_degree = sp.expand(radius_x**2 * (w0 * kernel_degree_two + w1 * p1 + w2 * p0))
    direct_error = laplacian(laplacian(laplacian(sixth_degree, x), x), x)
    check(f"direct degree-six error at a={position}", direct_error == 0)

# Recover the next nonzero coefficient in Remark 5.4.
tau, nu = sp.symbols("tau nu", real=True)
objective = (1 + nu) ** 2
objective += tau**2 * ((1 + 4 * nu) ** 2 + sp.Rational(1, 4) * (1 - 32 * nu) ** 2)
objective += tau**4 * ((12 * nu) ** 2 + sp.Rational(1, 4) * (48 * nu) ** 2)
minimizer = sp.solve(sp.diff(objective, nu), nu)[0]
reduced = sp.factor(objective.subs(nu, minimizer))
check("Remark 5.4 next coefficient is 3375/2", sp.limit(6 * reduced / tau**2, tau, 0) == sp.Rational(3375, 2))

print("All exact checks passed.")
