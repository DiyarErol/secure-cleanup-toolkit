# CI/CD Integration

The Secure Cleanup Toolkit integrates seamlessly with GitHub Actions to automatically verify and enforce code hygiene.

---

## 1. Default Workflow

Located at `.github/workflows/ci.yml`, the CI pipeline performs:
1. Linting via Ruff
2. Unit testing via Pytest
3. Secure cleanup validation via `secure_cleanup.py`

---

## 2. Example Workflow

```yaml
name: Secure Cleanup CI

on:
  push:
  pull_request:

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest ruff

      - name: Run cleanup preview
        run: python scripts/secure_cleanup.py --preview

      - name: Run lint and tests
        run: |
          ruff check .
          pytest -q
```

---

## 3. Failure Conditions

The CI job will fail if:

- Any file contains AI/Copilot/GPT traces.
- Lint or unit tests fail.
- YAML configuration is invalid.

---

## 4. Customizing CI

You can extend this workflow to include:

- Code coverage reports (via `coverage.py`)
- Security scanning (via `bandit`)
- Pre-deployment cleanup enforcement

---

## 5. CI Results

Successful builds display:

- ✅ Lint passed
- ✅ Unit tests passed
- ✅ Cleanup verified

Failed builds show error summaries and paths to affected files.

---

## 6. Integration Example

For enterprise environments:

```yaml
- name: Enforce cleanup policy
  run: python scripts/secure_cleanup.py --force
```

This ensures that every deployment is free of residual  metadata.

---

## Maintainer

Diyar — Independent Developer, Lucerne, Switzerland.
