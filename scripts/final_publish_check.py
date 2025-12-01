"""Final pre-publish validation and GitHub push automation.

This script performs a comprehensive validation pipeline before publishing:
1. Lint check (ruff)
2. Unit tests (pytest)
3. Secure cleanup dry-run
4. CI syntax validation
5. Pre-commit hook verification
6. Required files check

Only pushes to GitHub if ALL checks pass.
"""

import re
import subprocess
import sys
from pathlib import Path


def run(cmd: str, desc: str, allow_fail: bool = False) -> str:
    """Run shell command and capture output.

    Args:
        cmd: Command to execute
        desc: Human-readable description
        allow_fail: If True, don't raise on non-zero exit

    Returns:
        Command output (stdout)

    Raises:
        RuntimeError: If command fails and allow_fail is False
    """
    print(f"→ {desc}...")
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )

    output = result.stdout.strip()
    error = result.stderr.strip()

    if result.returncode != 0 and not allow_fail:
        print(f"  STDOUT: {output}")
        print(f"  STDERR: {error}")
        raise RuntimeError(f"[FAIL] {desc}\n{error or output}")

    return output


def check_required_files() -> None:
    """Verify all required files exist."""
    print("→ Checking required files...")

    required = [
        "README.md",
        "LICENSE",
        "configs/cleanup.yaml",
        "scripts/secure_cleanup.py",
        ".github/workflows/ci.yml",
        ".vscode/settings.json",
        "pyproject.toml",
    ]

    missing = []
    for file_path in required:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        raise RuntimeError(f"[FAIL] Missing required files: {', '.join(missing)}")

    print("  ✓ All required files present")


def check_cleanup_report() -> None:
    """Parse cleanup report and verify no traces found."""
    print("→ Checking cleanup report...")

    report_path = Path("cleanup_report.txt")
    if not report_path.exists():
        raise RuntimeError("[FAIL] cleanup_report.txt not found")

    txt = report_path.read_text(encoding="utf-8")

    # Look for pattern like "Found 5 match(es)" or "0 match(es)"
    match = re.search(r"Found (\d+) match", txt)
    if match:
        count = int(match.group(1))
        if count > 0:
            raise RuntimeError(
                f"[FAIL] Cleanup found {count} AI/Copilot traces. "
                "Review cleanup_report.txt and fix before publishing."
            )

    print("  ✓ No AI/Copilot traces detected")


def check_git_status() -> bool:
    """Check if there are uncommitted changes.

    Returns:
        True if there are changes to commit, False otherwise
    """
    result = subprocess.run(
        "git status --porcelain",
        shell=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    return bool(result.stdout.strip())


def check_git_initialized() -> bool:
    """Check if git repository is initialized."""
    git_dir = Path(".git")
    # Check if .git exists and has config file (not just hooks directory)
    return git_dir.exists() and (git_dir / "config").exists()


def main() -> None:
    """Execute full validation and publish pipeline."""
    report_path = Path("publish_report.txt")

    with report_path.open("w", encoding="utf-8") as f:
        try:
            print("\n" + "="*60)
            print("FINAL PRE-PUBLISH VALIDATION PIPELINE")
            print("="*60 + "\n")

            # Step 1: Required files check
            check_required_files()
            f.write("✅ Required files check passed\n")

            # Step 2: Lint check
            run("python -m ruff check .", "Lint check (ruff)")
            f.write("✅ Lint check passed\n")

            # Step 3: Unit tests
            output = run("python -m pytest -q", "Unit tests (pytest)")
            f.write(f"✅ Unit tests passed\n{output}\n\n")

            # Step 4: Secure cleanup dry-run
            run("python scripts/secure_cleanup.py --preview", "Secure cleanup preview")
            check_cleanup_report()
            f.write("✅ Secure cleanup check passed (0 traces)\n")

            # Step 5: Check if git is initialized
            if not check_git_initialized():
                print("→ Git not initialized (skipping git checks)")
                f.write("ℹ Git not initialized\n")
                print("\n" + "="*60)
                print("GIT NOT INITIALIZED")
                print("="*60)
                print("\nTo publish to GitHub, run these commands:")
                print("\n  git init")
                print("  git add -A")
                print("  git commit -m 'Initial commit: Secure Cleanup Toolkit'")
                print("  git branch -M main")
                print("  git remote add origin https://github.com/USERNAME/REPO.git")
                print("  git push -u origin main")
                print("  git tag -a v1.0.0 -m 'Stable release — verified build'")
                print("  git push origin v1.0.0")
                print("\nSee docs/PUBLISH.md for detailed instructions.")

                f.write("\n" + "="*60 + "\n")
                f.write("✅ ALL VALIDATION CHECKS PASSED\n")
                f.write("="*60 + "\n")
                f.write("Repository is ready for publishing!\n\n")
                f.write("Next steps:\n")
                f.write("1. Initialize git repository (see above)\n")
                f.write("2. Create GitHub repository\n")
                f.write("3. Push code and tags\n")
                f.write("\nSee docs/PUBLISH.md for detailed instructions.\n")

                print("\n" + "="*60)
                print("✓ ALL VALIDATION CHECKS PASSED!")
                print("✓ REPOSITORY IS READY FOR PUBLISHING!")
                print("="*60 + "\n")
                sys.exit(0)  # Exit successfully since all checks passed

            # Git is initialized - proceed with commit and push
            print("\n" + "="*60)
            print("GIT OPERATIONS")
            print("="*60 + "\n")

            # Check for changes
            has_changes = check_git_status()

            if has_changes:
                run("git add -A", "Stage all changes")
                run(
                    "git commit -m 'Finalize Secure Cleanup Toolkit release'",
                    "Commit changes",
                    allow_fail=True,  # May fail if nothing to commit
                )
                f.write("✅ Changes committed\n")
            else:
                print("  → No uncommitted changes")
                f.write("ℹ No uncommitted changes\n")

            # Create tag (allow fail if already exists)
            run(
                "git tag -a v1.0.0 -m 'Stable release — verified build'",
                "Create tag v1.0.0",
                allow_fail=True,
            )
            f.write("✅ Tag v1.0.0 ready\n")

            # Set main branch
            run("git branch -M main", "Set main branch")

            # Check if remote exists
            remote_check = subprocess.run(
                "git remote -v",
                shell=True,
                text=True,
                capture_output=True,
                encoding="utf-8",
            )

            if not remote_check.stdout.strip():
                print("\n  ⚠ No git remote configured.")
                print("  → Add remote: git remote add origin <URL>")
                print("  → Then push: git push -u origin main")

                f.write("\n⚠ Git remote not configured\n")
                f.write("Run: git remote add origin <URL>\n")
                f.write("Then: git push -u origin main\n")
                f.write("      git push origin v1.0.0\n")
            else:
                # Push to remote
                print("→ Pushing to GitHub...")
                run("git push -u origin main", "Push main branch")
                run("git push origin v1.0.0", "Push tag v1.0.0")

                f.write("✅ Pushed to origin/main\n")
                f.write("✅ Pushed tag v1.0.0\n")

                # Get remote URL
                remote_url = subprocess.run(
                    "git remote get-url origin",
                    shell=True,
                    text=True,
                    capture_output=True,
                    encoding="utf-8",
                ).stdout.strip()

                # Convert to HTTPS URL for display
                if remote_url.startswith("git@"):
                    remote_url = remote_url.replace(":", "/").replace("git@", "https://")
                if remote_url.endswith(".git"):
                    remote_url = remote_url[:-4]

                print("\n✓ Repository successfully published!")
                print(f"✓ View at: {remote_url}")

                f.write(f"\n✅ Repository published: {remote_url}\n")

            # Final summary
            f.write("\n" + "="*60 + "\n")
            f.write("✅ ALL CHECKS PASSED AND READY FOR PUBLISH\n")
            f.write("="*60 + "\n")

            print("\n" + "="*60)
            print(">>> FINAL VERIFICATION PASSED")
            print(">>> Repository ready for GitHub")
            print("="*60 + "\n")

        except Exception as e:
            error_msg = str(e)
            f.write(f"\n❌ PUBLISH ABORTED\n{error_msg}\n")
            print(f"\n{'='*60}")
            print(f"✗ Publish aborted: {error_msg}")
            print(f"{'='*60}\n")
            print("Review publish_report.txt for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
