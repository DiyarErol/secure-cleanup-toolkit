# Release Template — AxiomBridge-SeverityLab

## 🚀 Version X.Y.Z — [Release Title]

**Release Date**: YYYY-MM-DD  
**Tag**: `vX.Y.Z`

---

### 🎯 Highlights

- ✅ [Major feature or improvement 1]
- ✅ [Major feature or improvement 2]
- ✅ [Major feature or improvement 3]
- ✅ [Security/performance enhancement]

---

### 📦 What's New

#### Added
- [New feature description with details]
- [New module/script/tool added]

#### Changed
- [Updated behavior or API changes]
- [Configuration format updates]

#### Fixed
- [Bug fix description]
- [Edge case resolution]

#### Deprecated
- [Features marked for future removal]

---

### 🔒 Security

- [Security improvements or vulnerability fixes]
- [Updated dependencies for CVE resolution]

---

### 📊 Metrics

- **Test Coverage**: X% (Y tests passing)
- **Performance**: [Benchmark results if applicable]
- **Supported Python**: 3.10, 3.11
- **Supported OS**: Windows, macOS, Linux

---

### 🚀 Quick Start

#### Installation

```bash
# Clone repository
git clone https://github.com/USERNAME/MindForge-EventSeverity.git
cd MindForge-EventSeverity

# Setup environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

#### Training

```bash
# Train model with default config
python -m src.cli train --config configs/default.yaml

# Evaluate on test set
python -m src.cli evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
```

#### Secure Cleanup

```bash
# Preview cleanup (no changes)
python scripts/secure_cleanup.py --preview

# Apply cleanup (with backup)
python scripts/secure_cleanup.py --force
```

---

### ✅ Verification

```bash
# Run tests
pytest -q

# Lint code
ruff check .

# Type check
mypy src/
```

**Expected**: All tests pass, no lint errors.

---

### 📝 Breaking Changes

- [List any breaking API or config changes]
- [Migration guide if needed]

---

### 🤝 Contributors

- [@username1](https://github.com/username1) — [Contribution description]
- [@username2](https://github.com/username2) — [Contribution description]

---

### 📚 Documentation

- [Link to updated documentation sections]
- [New guides or tutorials added]

---

### 🐛 Known Issues

- [Issue #123] — [Brief description and workaround]
- [Issue #456] — [Brief description and workaround]

---

### 📦 Assets

**Downloadable binaries/packages** (if applicable):
- [Linux x86_64 package]
- [Windows installer]
- [macOS universal binary]

---

### 🔗 Full Changelog

See [CHANGELOG.md](../CHANGELOG.md) for detailed version history.

---

**For support**, open an issue or see [SECURITY.md](../SECURITY.md) for vulnerability reporting.
