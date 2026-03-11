# Contributing to hazardpulse

Thank you for your interest in contributing to hazardpulse.

## Core Principle

**Pure Python + NumPy only.** This is a hard rule, not a suggestion. Every machine learning algorithm in this project is implemented from scratch. No sklearn, no TensorFlow, no PyTorch, no XGBoost. This exists for two reasons:

1. **Transparency**: Every line of the prediction pipeline is readable and auditable
2. **Reproducibility**: The only dependency is NumPy. The models will run identically on any machine

## How to Contribute

### Replication Reports

The most valuable contribution is running the models and reporting your results. Use the [Replication Report issue template](https://github.com/Jphilbrick10/hazardpulse/issues/new?template=replication_report.yml).

### Bug Reports

If you find a bug, please include:
- Which model script you ran
- Python version and NumPy version
- The error traceback or unexpected output
- Operating system

### New Features

Before submitting a PR for a new feature:
1. Open an issue describing what you want to add and why
2. Wait for discussion — the project has specific design constraints

### Adding a New Hazard Model

If you want to apply the Helmholtz coherence framework to a new hazard type:

1. Create `src/hazardpulse/newhazard/` with:
   - `features.py` — feature extraction from raw data
   - `model.py` — training and evaluation pipeline
2. Use the shared `core/` modules for ML algorithms (logistic regression, GBM, etc.)
3. Include a proper temporal train/test split
4. Report AUC, Brier Score, and comparison to existing operational models
5. Add tests in `tests/`

### Code Style

- Use `ruff` for linting: `ruff check src/`
- Line length: 100 characters
- Type hints are welcome but not required
- Docstrings for public functions

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Commit Messages

Use conventional commits:
- `feat: add volcanic eruption prediction model`
- `fix: correct b-value calculation for shallow events`
- `docs: update earthquake methodology section`
- `test: add unit tests for logistic regression`

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
