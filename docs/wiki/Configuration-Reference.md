# Configuration Reference

The Secure Cleanup Toolkit uses a flexible YAML configuration file (`configs/cleanup.yaml`) to define patterns and exclusions.

---

## 1. File Structure

```yaml
patterns:
  - 'AI[-\s]?assisted'
  - 'Model used: GPT[-\s]?[0-9]+'
excludes:
  - '.git'
  - 'node_modules'
  - 'venv'
  - '.venv'
```

---

## 2. Keys Overview

- **patterns**: List — Regex expressions used to detect and remove unwanted text.
- **excludes**: List — Directories or file paths ignored during cleanup.

---

## 3. Pattern Design

- Patterns are written using regular expressions (regex).
- They are matched case-insensitively across all text-based files.

Example:

```yaml
patterns:
```

---

## 4. Safe Exclusions

Use `excludes` to prevent cleaning non-source directories such as:

- `.git`, `.venv`, `dist`, `build`, or binary outputs.

These paths are skipped for performance and data integrity.

---

## 5. Extending Configurations

You can include additional cleanup definitions for language-specific comments:

```yaml
patterns:
  - '// Copilot'
```

---

## 6. Config Validation

Before running cleanup:

```bash
python scripts/secure_cleanup.py --check-config configs/cleanup.yaml
```

If invalid syntax or unrecognized keys are found, a warning will be displayed.

---

## Configuration Tip

Keep separate YAML files for different projects:

```
configs/
 ├── default.yaml
 └── enterprise_cleanup.yaml
```

Select configuration using:

```bash
python scripts/secure_cleanup.py --config configs/enterprise_cleanup.yaml
```
