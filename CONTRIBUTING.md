# Contributing

Contributions should preserve the project’s bounded-memory design and reproducibility conventions.

Before opening a change:

1. Run `python -m compileall -q mojopqc_sca scripts`.
2. Run `python -m unittest discover -s tests -v`.
3. Use a small synthetic dataset for local tests.
4. Do not commit generated data, checkpoints, or result files.
5. Document new experiment parameters in `config/` and record generated metrics as JSON/CSV.

Research claims should be based on saved artifacts and clearly identify whether results use synthetic or hardware traces.
