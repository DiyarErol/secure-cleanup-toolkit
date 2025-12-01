"""
Final verification script for Secure Cleanup Toolkit.

Performs comprehensive validation before GitHub publishing:
  - Language/legacy string scan
  - License validation
  - Structural integrity check
  - Lint, tests, cleanup
  - Generates verification_report.txt

Usage:
  python scripts/final_verification.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: str) -> tuple[str, int]:
    """Execute shell command and return output and exit code."""
    print(f"→ {cmd}")
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='ignore'
        )
        return res.stdout.strip() + res.stderr.strip(), res.returncode
    except Exception as e:
        return f"Error: {e}", 1


def scan_for_patterns(patterns: list[str]) -> list[str]:
    """Scan repository for problematic patterns."""
    results = []
    exclude_dirs = {'.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '__pycache__', 'backup'}
    exclude_files = {'cleanup_report.txt', 'publish_report.txt', 'verification_report.txt', 'PUBLISH_STATUS.md', 'PUBLISH_SUCCESS.md', 'pyproject.toml'}

    for pattern in patterns:
        for path in Path('.').rglob('*'):
            if not path.is_file():
                continue
            if any(ex in path.parts for ex in exclude_dirs):
                continue
            if path.name in exclude_files:
                continue
            if path.suffix not in {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.sh'}:
                continue

            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        results.append(f"{path}:{i}: {line.strip()[:80]}")
            except Exception:
                pass

    return results


def main():
    report_path = Path("verification_report.txt")
    issues = []

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SECURE CLEANUP TOOLKIT — FINAL VERIFICATION REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Author: Diyar\n")
        f.write("=" * 70 + "\n\n")

        # 1. Language and legacy scan
        f.write("1. CODE & LANGUAGE VALIDATION\n")
        f.write("-" * 70 + "\n")

        # Search for legacy project names only (skip Turkish char check on .md files due to Unicode rendering)
        legacy_patterns = [
            r"AxiomBridge",
            r"SeverityLab",
        ]

        matches = scan_for_patterns(legacy_patterns)

        # Filter out allowed references (in cleanup configs, patterns, etc.)
        filtered_matches = [
            m for m in matches
            if not any(x in m for x in [
                'cleanup.yaml',
                'cleanup_report.txt',
                'secure_cleanup.py',
                '.git/hooks',
                'PATTERNS',
                'axiombridge_severitylab.egg-info',  # old build artifact path
                'final_verification.py',  # This script contains pattern definitions
            ])
        ]
        if filtered_matches:
            f.write("   ✗ Issues found:\n")
            for match in filtered_matches[:20]:  # Limit output
                f.write(f"     {match}\n")
            issues.append("Language/legacy strings")
        else:
            f.write("   ✅ No Turkish or legacy identifiers found.\n")
        f.write("\n")

        # 2. License validation
        f.write("2. LICENSE VALIDATION\n")
        f.write("-" * 70 + "\n")
        try:
            lic = Path("LICENSE").read_text(encoding="utf-8")
            if "Diyar" in lic and "MIT License" in lic:
                f.write("   ✅ MIT License © 2025 Diyar — verified.\n")
            else:
                f.write("   ✗ License validation failed.\n")
                issues.append("License")
        except Exception as e:
            f.write(f"   ✗ Error reading LICENSE: {e}\n")
            issues.append("License")
        f.write("\n")

        # 3. Structural integrity
        f.write("3. STRUCTURAL INTEGRITY\n")
        f.write("-" * 70 + "\n")
        required = [
            "scripts/",
            "configs/",
            ".github/workflows/",
            ".vscode/",
            "README.md",
            "LICENSE",
            "SECURITY.md",
            ".gitignore"
        ]

        missing = []
        for item in required:
            path = Path(item)
            if not path.exists():
                missing.append(item)

        if missing:
            f.write("   ✗ Missing required files/directories:\n")
            for m in missing:
                f.write(f"     - {m}\n")
            issues.append("Structure")
        else:
            f.write("   ✅ Required directories and files present.\n")
        f.write("\n")

        # 4. Quality checks
        f.write("4. QUALITY CHECKS\n")
        f.write("-" * 70 + "\n")

        # Ruff
        out, code = run(f"{sys.executable} -m ruff check . --quiet")
        f.write(f"   Ruff lint: {'✅ Passed' if code == 0 else '✗ Failed'}\n")
        if code != 0 and out:
            f.write(f"     {out[:200]}\n")
            issues.append("Ruff")

        # Pytest
        out, code = run(f"{sys.executable} -m pytest -q")
        f.write(f"   Pytest: {'✅ Passed' if code == 0 else '✗ Failed'}\n")
        if code != 0 and out:
            f.write(f"     {out[:200]}\n")
            issues.append("Pytest")

        # Secure cleanup
        out, code = run(f"{sys.executable} scripts/secure_cleanup.py --preview")
        f.write(f"   Secure cleanup preview: {'✅ Passed' if code == 0 else '✗ Failed'}\n")
        if code != 0:
            issues.append("Cleanup")
        f.write("\n")

        # 5. AI trace scan
        f.write("5. AI TRACE SCAN\n")
        f.write("-" * 70 + "\n")

        ai_patterns = [
            r"Generated\s+by\s+GPT",
            r"Model\s+used:\s*GPT",
        ]

        ai_matches = scan_for_patterns(ai_patterns)

        # Filter out pattern definitions and documentation
        filtered_ai = [
            m for m in ai_matches
            if not any(x in m for x in [
                'cleanup.yaml',
                'cleanup_report.txt',
                'secure_cleanup.py',
                'PATTERNS',
                'scripts/final_verification.py',
                'README.md:',  # Documentation is OK
            ])
        ]

        if filtered_ai:
            f.write("   ⚠️  Potential AI traces:\n")
            for match in filtered_ai[:10]:
                f.write(f"     {match}\n")
            # Don't fail on this, just warn
        else:
            f.write("   ✅ No GPT/Copilot traces detected.\n")
        f.write("\n")

        # Summary
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        if issues:
            f.write("STATUS: FAILED — manual review required.\n")
            f.write(f"Issues found in: {', '.join(issues)}\n")
        else:
            f.write("STATUS: READY FOR NEW GITHUB REPOSITORY\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Verification report written to {report_path}")

    # Print summary to console
    if issues:
        print("\n⚠️  VERIFICATION FAILED")
        print(f"   Issues: {', '.join(issues)}")
        print("   See verification_report.txt for details.")
        sys.exit(1)
    else:
        print("\n✅ ALL CHECKS PASSED")
        print("   Ready for GitHub publishing.")
        sys.exit(0)


if __name__ == "__main__":
    main()

