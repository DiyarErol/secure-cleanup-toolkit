"""
Secure cleanup utility for removing AI attribution traces.

Usage:
  python scripts/secure_cleanup.py --preview
  python scripts/secure_cleanup.py --force

This tool scans text-based files and removes entire lines containing AI/Copilot/GPT
attribution or product metadata, writes a detailed report, and backs up modified files.
It skips common binary/build folders and attempts to preserve legitimate academic citations
by only deleting lines that clearly look like attribution or meta banners.
"""

import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

PATTERNS = [
    r"Model\s+(used|olarak)\s*:?\s*GPT[-\\s]?[0-9]+",
    r"Generated\\s+by\\s+GPT[-\\s]?[0-9]+",
    r"Generated\\s+using\\s+Copilot",
    r"Copilot\\s+Chat",
    r"GitHub\\s+Copilot",
    r"AI[-\\s]?generated",
    r"AI[-\\s]?assisted",
]

EXCLUDES = {".git", "node_modules", "venv", ".venv", "data", "dist", "build", "assets", "media"}
TARGET_EXT = (".md", ".markdown", ".py", ".yaml", ".yml", ".toml", ".txt", ".ipynb", ".json")

# Load config if present
CONFIG_PATH = Path("configs/cleanup.yaml")
if HAS_YAML and CONFIG_PATH.exists():
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        PATTERNS = cfg.get("patterns", PATTERNS)
        EXCLUDES = set(cfg.get("excludes", list(EXCLUDES)))
    except Exception:
        pass  # fallback to defaults

BACKUP_DIR = Path("backup") / f"cleanup_{time.strftime('%Y%m%d_%H%M%S')}"
REPORT = Path("cleanup_report.txt")


def should_skip(path: Path) -> bool:
    # Check both directory parts and full relative path
    path_str = str(path).replace("\\", "/")
    return any(x in path.parts or path_str.startswith(x) or x in path_str for x in EXCLUDES)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TARGET_EXT


def purge_model_traces(path: Path) -> bool:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    original = txt
    for pat in PATTERNS:
        txt = re.sub(pat, "", txt, flags=re.IGNORECASE)
    if txt != original:
        Path(path).write_text(txt, encoding="utf-8")
        return True
    return False


def clean_file(path: Path, preview: bool = False):
    lines, removed = [], []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if any(re.search(p, line, re.IGNORECASE) for p in PATTERNS):
                    removed.append((i, line.rstrip()))
                else:
                    lines.append(line)
    except Exception as e:
        return removed, f"Error reading file: {e}"

    # In force mode, purge model traces
    if not preview and removed:
        try:
            purge_model_traces(path)
        except Exception:
            pass

    if preview:
        return removed, None

    if removed:
        try:
            backup_path = BACKUP_DIR / path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            with path.open("w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            return removed, f"Error writing backup or file: {e}"
    return removed, None


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        return out.returncode, (out.stdout + out.stderr)
    except Exception as e:
        return 1, f"Command failed: {e}"


def main():
    force = "--force" in sys.argv or "-f" in sys.argv

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    total_removed = 0
    processed_files = 0

    with REPORT.open("w", encoding="utf-8") as rep:
        rep.write(f"=== Secure Cleanup Report ({time.ctime()}) ===\n\n")
        rep.write("Scan patterns:\n")
        for p in PATTERNS:
            rep.write(f"  - {p}\n")
        rep.write("\nExcluded folders: " + ", ".join(sorted(EXCLUDES)) + "\n\n")

        for path in Path(".").rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path):
                continue
            if not is_text_file(path):
                continue

            processed_files += 1
            removed, err = clean_file(path, preview=not force)

            status_symbol = "✅"
            if err:
                status_symbol = "❌"
            elif removed:
                status_symbol = "⚠️"

            ts = time.strftime("%H:%M:%S")
            msg = f"[{ts}] {status_symbol} {path}"
            if removed:
                msg += f" — {len(removed)} match(es)"
            print(msg)
            rep.write(f"{status_symbol} {path}\n")

            if err:
                rep.write(f"  Error: {err}\n")
            if removed:
                rep.write(f"  Removed lines: {len(removed)}\n")
                for (ln, text) in removed:
                    # Log exact removed text for verification
                    rep.write(f"    L{ln}: {text}\n")
                total_removed += len(removed)

        rep.write(f"\nTotal processed files: {processed_files}\n")
        rep.write(f"Total removed lines: {total_removed}\n")

        # Post-run validation (ruff and pytest)
        rep.write("\n=== Post-run validation ===\n")
        null_dev = "NUL" if os.name == "nt" else "/dev/null"
        ruff_code = os.system(f"ruff check . >{null_dev} 2>&1")
        rep.write("\n[ruff check .]\n")
        rep.write(f"Status: {'OK' if ruff_code == 0 else 'Issues found'}\n")

        pytest_code = os.system(f"{sys.executable} -m pytest -q >{null_dev} 2>&1")
        rep.write("\n[pytest -q]\n")
        rep.write(f"Status: {'OK' if pytest_code == 0 else 'Failures'}\n")

        rep.write("\n=== Summary ===\n")
        rep.write(f"Ruff & Pytest checks: {'completed successfully' if ruff_code == 0 and pytest_code == 0 else 'issues detected'}\n")

    print("\nSummary written to cleanup_report.txt")
    if not force:
        print("\nPreview mode: Run again with --force to apply changes.")


if __name__ == "__main__":
    main()
