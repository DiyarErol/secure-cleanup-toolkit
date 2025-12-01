# Usage Guide

The Secure Cleanup Toolkit provides both command-line and automated workflows for cleaning repositories from metadata and AI traces.

---

## 1. Installation

```bash
git clone https://github.com/DiyarErol/secure-cleanup-toolkit.git
cd secure-cleanup-toolkit
pip install -e .
```

---

## 2. Quick Start

Preview cleanup without applying changes:

```bash
python scripts/secure_cleanup.py --preview
```

Run a full cleanup (creates backups automatically):

```bash
python scripts/secure_cleanup.py --force
```

---

## 3. Typical Workflow

| Step   | Command                                      | Description                        |
| ------ | -------------------------------------------- | ---------------------------------- |
| Scan   | `python scripts/secure_cleanup.py --preview` | Identify unwanted traces.          |
| Clean  | `python scripts/secure_cleanup.py --force`   | Remove traces and generate backup. |
| Verify | `ruff check . && pytest -q`                  | Confirm code integrity remains.    |
| Commit | `git commit -m "Apply cleanup"`              | Save verified changes.             |

---

## 4. Backup and Reports

- Backups are stored in `backup/cleanup_<timestamp>/`.
- A detailed report is generated in `cleanup_report.txt`, summarizing all actions performed.

---

## 5. Advanced Options

```bash
python scripts/secure_cleanup.py --config configs/cleanup.yaml
```

- `--preview`: Dry run without modifying files.
- `--force`: Apply cleanup and create backups.
- `--config`: Specify a custom configuration file.

---

## 6. Example Session Output

```
[INFO] Running cleanup in preview mode...
[OK]  No AI traces found.
[INFO] Backup directory created at /backup/cleanup_2025-12-01/
```

---

## Author

Developed independently by Diyar, Lucerne, Switzerland. Licensed under the MIT License.
