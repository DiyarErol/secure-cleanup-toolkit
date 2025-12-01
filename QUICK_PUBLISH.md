# 🚀 Quick Publish Commands — Copy & Paste

Replace `USERNAME` with your actual GitHub username before running.

## Initial Setup (One Time)

```bash
# 1. Verify current state
pwd
git status

# 2. Add all files and commit
git add -A
git commit -m "Initial commit: Secure Cleanup Toolkit v1.0 with secure cleanup automation"

# 3. Set default branch
git branch -M main

# 4. Add GitHub remote (REPLACE USERNAME!)
git remote add origin https://github.com/USERNAME/secure-cleanup-toolkit.git

# 5. Verify remote
git remote -v

# 6. Push to GitHub
git push -u origin main

# 7. Create and push release tag
git tag -a v1.0.0 -m "Stable release — Production-grade severity classification toolkit"
git push origin v1.0.0
```

## Verification

```bash
# Check CI status (should be green)
# Visit: https://github.com/USERNAME/secure-cleanup-toolkit/actions

# Test local clone
cd ..
git clone https://github.com/USERNAME/secure-cleanup-toolkit.git test-clone
cd test-clone
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
python scripts/secure_cleanup.py --preview
```

## GitHub UI Tasks

1. **Create Repository**:
   - Go to: https://github.com/new
   - Name: `secure-cleanup-toolkit`
   - Description: `Production-grade severity classification for autonomous risk understanding`
   - Public
   - DO NOT initialize with README
   - Click: Create repository

2. **Add Topics**:
   - Click: ⚙️ Settings icon (top-right of repo page)
   - Add: `severity-classification`, `video-analysis`, `pytorch`, `deep-learning`, `security-automation`, `pre-commit-hooks`, `python`

3. **Create Release**:
   - Go to: Releases → Draft a new release
   - Tag: `v1.0.0`
   - Title: `v1.0.0 — Stable Production Release`
   - Copy description from `.github/release_template.md`
   - Click: Publish release

## Done! ✅

Repository URL: `https://github.com/USERNAME/secure-cleanup-toolkit`
