# SWIMLS: Hermite–Birkhoff interpolation by singular-weight moving least squares

Code, exact verification scripts, results and figures for the paper

> David Figueiras-Garcia, *Generalized Hermite–Birkhoff interpolation of arbitrary order in any
> dimension via singular-weight moving least squares*, preprint (2026), submitted for publication.

SWIMLS fits, at every evaluation point x, a polynomial of degree m by weighted least squares to
arbitrary linear differential data at the nodes (values, derivatives of any order, Laplacians,
directional derivatives), with weights |x − x_i|^(−α) times a smooth factor, and evaluates the fit
at x. The paper characterizes exactly which data are reproduced and proves convergence for
compactly supported weights. This repository is release **v0.2.1**, the one cited by the paper.

## Contents
- `code/hbmls/` – the library: `SingularMLS` (stable float evaluator with the exact singular limit
  at the nodes), `MPSingularMLS` (high-precision mpmath reference), Hermite–Birkhoff functionals,
  completion (`complete`), and the 1-D/2-D evaluators `SWIMLS1D`/`SWIMLS2D` (exact derivatives).
- `code/tests/` – correctness checks.
- `code/experiments/` – the numerical experiments of the paper.
- `code/proofs/` – exact (SymPy, rational arithmetic) checks of examples and identities in the paper.
- `results/` – logs and CSV files produced by the scripts; `results/proofs/` for the exact checks.
- `figures/` – the figures (PDF and PNG).

## Weights
`SingularMLS(..., weight=...)` takes
- `"bump"`: |x − x_i|^(−α) ψ(|x − x_i|²/δ²) with the compactly supported C∞ factor
  ψ(s) = exp(1 − 1/(1 − s)) for s < 1, 0 otherwise. **This is the weight to use in practice**
  (local, smooth, the setting of the convergence theorem); see the paper, "Compact support in practice".
- `"power"`: the pure power |x − x_i|^(−α), used where a result of the paper assumes it.
- `"gauss"`: |x − x_i|^(−α) exp(−|x − x_i|²/δ²), kept only for comparison.

Every script states its weight explicitly. (In this release the library default is still
`"gauss"` for backward compatibility; it will become `"bump"` in v0.3.0.)

## Installation and use
```
pip install -r requirements.txt
cd code
PYTHONPATH=. python3 tests/test_core.py          # 36 checks
PYTHONPATH=. python3 experiments/e1_exponent_rule.py
```
All scripts are run from `code/` with `PYTHONPATH=.`; they write to `../results/` and `../figures/`.

## Paper items and the scripts that produce them
| paper | script (`code/…`) | output |
|---|---|---|
| Figure 1 (one-dimensional example) | `experiments/e4_shapes.py` | `figures/fig1_one_dimensional_example.*`, `results/e4_shapes.log` |
| Two-node example in 2-D (error 150) | `proofs/twonode_2d_check.py` | `results/proofs/twonode_2d_check.log` |
| 1-D exact examples (cancellation example, three-node witness) | `proofs/exact_1d_examples.py` | `results/proofs/exact_1d_examples.log` |
| Counterexample to the unrestricted conjecture (d = 6) | `proofs/verify_counterexample_external.py` | `results/proofs/verify_counterexample_external.log` |
| Komargodski–Levin counterexamples (1-D, 2-D) | `proofs/kl_counterexample.py`, `proofs/kl_counterexample_2d.py` | `results/proofs/kl_counterexample*.log` |
| Exact finite expansion of the error (theorem, exact check in 4 cases) | `proofs/paper_thm_expansion_check.py` (engine `proofs/perturbative_engine.py`) | `results/paper_thm_expansion_check.log` |
| §7.1 Hermite data (92 data) | `experiments/e1_exponent_rule.py` | `results/e1_per_datum.*` |
| §7.1 error identities of the 2-D Hermite example; §7.2 special configurations (table) | `experiments/e6_degenerate.py` | `results/e6_identities.csv`, `results/e6_degenerate.*` |
| §7.1 patterns with gaps (83 data) | `experiments/e2_birkhoff.py` | `results/e2_birkhoff.*` |
| §7.3 Komargodski–Levin setting | `experiments/e2c_kl_faithful.py` | `results/e2c_kl_faithful.log` |
| §7.3 completion (repairs, Franke accuracy test) | `experiments/e2b_completion.py` | `results/e2b_*` |
| §7.4 convergence with compactly supported weights (figure, rates) | `experiments/e8_compact.py`, `experiments/plot_e8.py` | `results/e8_compact*`, `figures/e8_compact.*` |
| §7.4 periodically jittered grids | `experiments/e8b_periodic.py` | `results/e8b_periodic.*` |
| §7.4 comparison with Gaussian factors | `experiments/e7_parameters.py`, `experiments/plot_e7.py` | `results/e7_parameters.*`, `figures/e7_convergence.*` |
| Appendix A, accuracy near a node (figure) | `experiments/e5_stability.py` | `results/e5_stability.*`, `figures/e5_stability.*` |

Not in the paper (earlier studies, kept for completeness): `experiments/e3_convergence.py` and
`experiments/e7b_values_high_degree.py` (Gaussian weights), the 2-D shape-function figure of `e4_shapes.py`.

## License

The software in this repository is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE) (SPDX: `PolyForm-Noncommercial-1.0.0`).

- **Non-commercial use is free, with no permission needed:** personal study, research,
  experiment and teaching, and use by educational institutions, public research
  organizations, charities and government bodies. If you use it, a short note to
  the author (see the paper, or the owner of this repository on GitHub) is appreciated; this is a courtesy
  request, not a condition of the license.
- **Commercial use requires a separate commercial license**, which may involve fees. Contact
  the author (see the paper, or the owner of this repository on GitHub).

**Patent pending:** US provisional patent application 64/187,586 (filed October 29, 2026) covers
the methods implemented here. For non-commercial use, the license above includes the necessary
patent rights.


The paper itself is not part of this repository.
