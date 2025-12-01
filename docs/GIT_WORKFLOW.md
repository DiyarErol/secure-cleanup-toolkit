# GitHub Repository Setup — Git Commands

This file contains the complete git workflow for publishing **Secure Cleanup Toolkit** to GitHub.

---

## 📋 Prerequisites Checklist

- [x] All files committed locally
- [x] Tests passing (`pytest -q`)
- [x] Secure cleanup passed (`python scripts/secure_cleanup.py --preview`)
- [x] GitHub account ready
- [x] Repository name decided: `secure-cleanup-toolkit`

---

## 🚀 Step-by-Step Git Commands

### 1. Initialize Git Repository (if not already done)

```bash
# Navigate to project root
cd C:\Users\erold\Desktop\secure-cleanup-toolkit

# Initialize Git
git init

# Configure user (if not set globally)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files
git add -A

# Create initial commit
git commit -m "Initial commit: Secure Cleanup Toolkit v1.0 with secure cleanup automation"
```

**Output**: `[main (root-commit) abc1234] Initial commit: ...`

---

### 2. Set Default Branch to `main`

```bash
# Rename current branch to main
git branch -M main
```

**Why**: GitHub uses `main` as the default branch name.

---

### 3. Create GitHub Repository

**Go to**: [github.com/new](https://github.com/new)

**Settings**:
- **Name**: `secure-cleanup-toolkit`
- **Description**: `Production-grade severity classification for autonomous risk understanding`
- **Visibility**: Public
- **Initialize**: DO NOT check any boxes (no README, license, or .gitignore)

**Click**: Create repository

---

### 4. Add GitHub Remote

```bash
# Replace USERNAME with your GitHub username
git remote add origin https://github.com/USERNAME/secure-cleanup-toolkit.git

# Verify remote URL
git remote -v
```

**Expected output**:
```
origin  https://github.com/USERNAME/secure-cleanup-toolkit.git (fetch)
origin  https://github.com/USERNAME/secure-cleanup-toolkit.git (push)
```

---

### 5. Push to GitHub

```bash
# Push main branch and set upstream tracking
git push -u origin main
```

**Output**:
```
Enumerating objects: 150, done.
Counting objects: 100% (150/150), done.
Delta compression using up to 8 threads
Compressing objects: 100% (120/120), done.
Writing objects: 100% (150/150), 500 KiB | 2 MiB/s, done.
Total 150 (delta 30), reused 0 (delta 0), pack-reused 0
To https://github.com/USERNAME/secure-cleanup-toolkit.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

**Verify**: Visit `https://github.com/USERNAME/secure-cleanup-toolkit` — all files should appear.

---

### 6. Create Stable Release Tag

```bash
# Create annotated tag for v1.0.0
git tag -a v1.0.0 -m "Stable release — Production-grade severity classification toolkit with secure cleanup automation"

# Push tag to GitHub
git push origin v1.0.0
```

**Output**:
```
Total 0 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/USERNAME/secure-cleanup-toolkit.git
 * [new tag]         v1.0.0 -> v1.0.0
```

**Verify**: Go to **Releases** tab on GitHub — `v1.0.0` appears.

---

### 7. Publish GitHub Release

1. Go to **Releases** → **Draft a new release**
2. **Choose a tag**: Select `v1.0.0`
3. **Release title**: `v1.0.0 — Stable Production Release`
4. **Description**: Copy from `.github/release_template.md` and customize
5. **Click**: Publish release

---

## 🔄 Future Workflow (After Initial Publish)

### Making Changes

```bash
# Create feature branch
git checkout -b feature/new-awesome-feature

# Make changes, then stage and commit
git add .
git commit -m "feat: add new awesome feature"

# Push to remote
git push origin feature/new-awesome-feature
```

### Creating Pull Request

1. Go to GitHub repository
2. Click **Pull requests** → **New pull request**
3. Select `feature/new-awesome-feature` → `main`
4. Fill in PR template
5. Request review

### Merging to Main

```bash
# After PR approval, merge via GitHub UI or:
git checkout main
git pull origin main
git merge feature/new-awesome-feature
git push origin main
```

### Creating New Release

```bash
# Update version in pyproject.toml, CHANGELOG.md
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 1.1.0"
git push origin main

# Tag new version
git tag -a v1.1.0 -m "Release v1.1.0 — [Brief description]"
git push origin v1.1.0

# Create release on GitHub (via UI)
```

---

## 🛠️ Useful Git Commands

### Check Status

```bash
# Show working tree status
git status

# Show commit history
git log --oneline

# Show remote URLs
git remote -v
```

### Undo Changes

```bash
# Discard unstaged changes
git checkout -- <file>

# Unstage file
git reset HEAD <file>

# Amend last commit message
git commit --amend -m "New message"

# Delete local branch
git branch -d feature/branch-name

# Delete remote branch
git push origin --delete feature/branch-name
```

### Sync with Remote

```bash
# Fetch updates from remote
git fetch origin

# Pull and merge main
git pull origin main

# Push all tags
git push origin --tags
```

---

## 🔍 Troubleshooting

### Authentication Issues

**HTTPS**: Use Personal Access Token (PAT) instead of password:
1. Go to **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Generate new token with `repo` scope
3. Use token as password when prompted

**SSH**: Setup SSH key:
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Copy this to GitHub Settings → SSH keys
git remote set-url origin git@github.com:USERNAME/secure-cleanup-toolkit.git
```

### Push Rejected

```bash
# Pull remote changes first
git pull origin main --rebase

# Then push
git push origin main
```

### Large Files

If repository size exceeds 100MB, use Git LFS:
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
git commit -m "chore: add Git LFS for model checkpoints"
```

---

## ✅ Post-Publish Checklist

- [x] Repository visible on GitHub
- [x] CI/CD pipeline green (Actions tab)
- [x] Release `v1.0.0` published
- [x] README renders correctly
- [x] License file present
- [x] Topics added for discoverability
- [x] Security policy visible
- [x] Contributing guide accessible

---

## 🎉 Success!

Your repository is now live and ready for collaboration.

**Next steps**:
- Share the repository link
- Add collaborators (Settings → Manage access)
- Enable Dependabot alerts (Settings → Security)
- Setup branch protection rules (Settings → Branches)

---

**Repository URL**: `https://github.com/USERNAME/secure-cleanup-toolkit`

For detailed publishing guide, see [docs/PUBLISH.md](../docs/PUBLISH.md).
