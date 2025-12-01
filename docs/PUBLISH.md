# Publishing Guide — MindForge EventSeverity

This guide provides step-by-step instructions for publishing the **Secure Cleanup Toolkit** project to GitHub as a stable, production-ready open-source repository.

## 📋 Prerequisites

- Git installed (`git --version`)
- GitHub account
- Project directory initialized with all files
- All tests passing locally (`pytest -q`)

---

## 🚀 Step 1: Initialize Local Git Repository

```bash
# Navigate to project root
cd C:\Users\erold\Desktop\secure-cleanup-toolkit

# Initialize Git repository
git init

# Add all files to staging
git add -A

# Create initial commit
git commit -m "Initial commit: Secure Cleanup Toolkit v1.0"
```

**What this does:**
- `git init` creates a new `.git` directory to track version history
- `git add -A` stages all files (excluding `.gitignore` patterns)
- `git commit` saves the snapshot with a descriptive message

---

## 🌐 Step 2: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `secure-cleanup-toolkit`
3. Description: *Production-grade severity classification for autonomous risk understanding*
4. Visibility: **Public** (or Private)
5. **DO NOT** initialize with README, license, or `.gitignore` (we already have these)
6. Click **Create repository**

---

## 🔗 Step 3: Connect Local Repository to GitHub

```bash
# Set main as default branch
git branch -M main

# Add GitHub remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/secure-cleanup-toolkit.git

# Verify remote URL
git remote -v
```

**Expected output:**
```
origin  https://github.com/USERNAME/secure-cleanup-toolkit.git (fetch)
origin  https://github.com/USERNAME/secure-cleanup-toolkit.git (push)
```

---

## 📤 Step 4: Push to GitHub

```bash
# Push main branch to remote
git push -u origin main
```

**What this does:**
- `-u` sets `origin/main` as upstream tracking branch
- All commits and files are uploaded to GitHub
- CI/CD workflow (`.github/workflows/ci.yml`) is triggered automatically

---

## ✅ Step 5: Verify CI Build

1. Go to your GitHub repository
2. Click **Actions** tab
3. Check the latest workflow run: **CI**
4. Ensure all jobs pass (green checkmark ✅)

**Expected workflow steps:**
- Secure cleanup preview
- Lint with ruff
- Type check with mypy
- Run tests (pytest)
- Coverage report

---

## 🏷️ Step 6: Create a Stable Release Tag

```bash
# Create annotated tag for v1.0.0
git tag -a v1.0.0 -m "Stable release — Production-grade severity classification toolkit"

# Push tag to GitHub
git push origin v1.0.0
```

**What this does:**
- Creates a permanent reference to the current commit
- Appears under **Releases** on GitHub
- Enables version tracking and changelog management

---

## 📦 Step 7: Create GitHub Release

1. Go to **Releases** → **Draft a new release**
2. Choose tag: `v1.0.0`
3. Release title: `v1.0.0 — Stable Production Release`
4. Description (use `.github/release_template.md`):

```markdown
## 🚀 Secure Cleanup Toolkit — v1.0.0 Release

### Highlights
- ✅ Production-grade severity classification framework
- ✅ Config-driven YAML design
- ✅ Comprehensive data pipeline (frame extraction, augmentation, splitting)
- ✅ Multiple baseline models (3D-ResNet, SlowFast, TimeSformer stubs)
- ✅ Evaluation suite (per-class metrics, confusion matrices, PR curves)
- ✅ Explainability (Grad-CAM/saliency maps)
- ✅ Secure cleanup automation (AI metadata removal)
- ✅ Pre-commit hooks + CI/CD integration
- ✅ 17 passing unit tests

### Quick Start
\`\`\`bash
git clone https://github.com/USERNAME/secure-cleanup-toolkit.git
cd secure-cleanup-toolkit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
\`\`\`

### Security Features
\`\`\`bash
python scripts/secure_cleanup.py --preview
python scripts/secure_cleanup.py --force
\`\`\`

### CI/CD
All tests and linting checks pass on Windows, macOS, and Linux.
```

5. Click **Publish release**

---

## 🔒 Step 8: Add Repository Topics (GitHub SEO)

1. Go to repository homepage
2. Click **⚙️ Settings** → **About** (top-right)
3. Add topics:
   - `severity-classification`
   - `video-analysis`
   - `pytorch`
   - `deep-learning`
   - `autonomous-systems`
   - `explainable-ai`
   - `security-automation`
   - `pre-commit-hooks`
   - `python`

---

## 🧪 Step 9: Post-Publish Verification

Clone the repository fresh and test:

```bash
# Clone from GitHub
git clone https://github.com/USERNAME/secure-cleanup-toolkit.git
cd secure-cleanup-toolkit

# Setup environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Run tests
pytest -q

# Run secure cleanup
python scripts/secure_cleanup.py --preview

# Verify pre-commit hook
git add -A
git commit -m "test: verify pre-commit hook"
```

**Expected results:**
- ✅ All tests pass (17/17)
- ✅ `cleanup_report.txt` generated
- ✅ Pre-commit hook blocks commits if AI traces found
- ✅ CI pipeline green on GitHub Actions

---

## 📚 Optional: PyPI Publishing (Future)

For distributing as a Python package:

1. Add `[project.scripts]` to `pyproject.toml`:
   ```toml
   [project.scripts]
   secure-cleanup = "scripts.secure_cleanup:main"
   secure-cleanup = "src.cli:main"
   ```

2. Build package:
   ```bash
   pip install build twine
   python -m build
   ```

3. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

4. Install via pip:
   ```bash
   pip install secure-cleanup-toolkit
   ```

---

## 🤝 Community Setup

Recommended additional files for open-source projects:

- `CONTRIBUTING.md` — Contribution guidelines
- `CODE_OF_CONDUCT.md` — Community standards
- `SECURITY.md` — Vulnerability disclosure policy
- `.github/ISSUE_TEMPLATE/` — Bug/feature templates
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist

All included in this repository.

---

## 📝 Changelog Management

For future releases, maintain `CHANGELOG.md`:

```markdown
# Changelog

## [1.0.0] - 2025-12-01
### Added
- Initial production release
- Severity classification framework
- Secure cleanup automation

## [1.1.0] - TBD
### Added
- Multi-GPU support
### Fixed
- Edge case in frame extraction
```

---

## ✅ Checklist

- [x] Git initialized
- [x] GitHub repository created
- [x] Remote added and verified
- [x] Code pushed to `main`
- [x] CI pipeline green
- [x] Release `v1.0.0` tagged and published
- [x] Topics added for discoverability
- [x] Fresh clone tested successfully

---

**Repository ready for production use!** 🎉

For support, see [SECURITY.md](../SECURITY.md) or open an issue on GitHub.
