# Framework review evidence — 7 October 2026

Baseline source: `19421a547f47c3106419c275b610a5ce5b8ca975`.
See [the framework review](../../FRAMEWORK_REVIEW.md) for findings and scope.

- `performance-paired.json`: five alternating before/after measurements in
  `results.paired_same_process`. The top-level `repeats: 3` applies to the other
  single-implementation timing workloads. Allocation peaks come from an additional
  separately traced invocation, not the paired timing loop.
  The explicit `paired_repeats: 5` metadata was added during review; measured
  samples were not changed. The harness now emits this field itself.
- `performance-before.json`: earlier baseline-only measurements, retained to
  substantiate the allocation tradeoff. Use the paired file for timing comparisons.
- `local-validation.json`: bounded local suite, static checks and example results.
  Counts overlap focused suites and should not be summed.

Run the no-download suite from a checkout with the corresponding optional test
dependencies installed:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
python -m pytest -q -m 'not slow' tests validation/test_scientific_contracts.py validation/test_finite_loss_nonfinite_gradients.py validation/test_core_independent.py
python -m ruff check src tests templates validation/benchmark_framework.py
python -m ruff format --check src tests templates validation/benchmark_framework.py
python -m mypy
python -m bandit -q -r src/clmkit
```

The environment inherits laptop system packages. Fresh-install and dependency
security conclusions come from hosted checks on their named commits, not from
this local result. No real model was downloaded locally for this review.
