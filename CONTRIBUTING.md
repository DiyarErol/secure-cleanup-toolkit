# Contributing to AxiomBridge-SeverityLab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

---

## 🚀 How to Contribute

### 1. Reporting Bugs

Before creating a bug report:
- Check [existing issues](https://github.com/USERNAME/MindForge-EventSeverity/issues) to avoid duplicates
- Use the bug report template
- Include detailed reproduction steps

**Bug Report Template**:
```markdown
**Description**: Brief summary of the bug

**Steps to Reproduce**:
1. Step 1
2. Step 2
3. Observe error

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- OS: [Windows 11 / macOS 14 / Ubuntu 22.04]
- Python: [3.10.x / 3.11.x]
- PyTorch: [version]

**Logs/Screenshots**: [Attach relevant output]
```

### 2. Suggesting Features

Feature requests are welcome! Please:
- Search existing feature requests first
- Explain the use case clearly
- Provide examples if possible

**Feature Request Template**:
```markdown
**Problem**: What problem does this solve?

**Proposed Solution**: How should it work?

**Alternatives**: Other approaches considered?

**Additional Context**: Mockups, examples, links
```

### 3. Contributing Code

#### Setup Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/MindForge-EventSeverity.git
cd MindForge-EventSeverity

# Add upstream remote
git remote add upstream https://github.com/USERNAME/MindForge-EventSeverity.git

# Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
chmod +x .git/hooks/pre-commit  # Unix/macOS only
```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following our style guide (see below)

3. **Write tests** for new functionality:
   ```bash
   pytest tests/test_your_feature.py -v
   ```

4. **Run full test suite**:
   ```bash
   pytest -v --cov=src --cov-report=term
   ```

5. **Lint and format**:
   ```bash
   ruff check src/ tests/
   black src/ tests/
   isort src/ tests/
   mypy src/
   ```

6. **Run secure cleanup**:
   ```bash
   python scripts/secure_cleanup.py --preview
   ```

7. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add awesome feature"
   ```
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation only
   - `style:` formatting, no code change
   - `refactor:` code restructuring
   - `test:` adding tests
   - `chore:` maintenance tasks

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Open a Pull Request** on GitHub

---

## 📝 Style Guide

### Python Code Style

- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type Checker**: Mypy (strict mode)
- **Docstrings**: Google style

**Example**:
```python
def train_model(config: dict[str, Any], epochs: int = 50) -> dict[str, float]:
    """
    Train a severity classification model.

    Args:
        config: Training configuration dictionary
        epochs: Number of training epochs

    Returns:
        Dictionary containing training metrics

    Raises:
        ValueError: If config is invalid
    """
    # Implementation here
    pass
```

### Configuration Files

- **YAML**: 2-space indentation
- **JSON**: 2-space indentation
- **TOML**: Follow pyproject.toml conventions

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

**Examples**:
```
feat(data): add video augmentation transforms
fix(train): resolve NaN loss issue with gradient clipping
docs(readme): update installation instructions
test(evaluate): add confusion matrix test case
```

---

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names: `test_video_dataset_loads_correctly`
- Use pytest fixtures for shared setup
- Aim for >80% code coverage

**Example Test**:
```python
import pytest
from src.data.dataset import VideoDataset

def test_video_dataset_length():
    """Test that dataset returns correct number of samples."""
    dataset = VideoDataset(data_dir="data/processed", split="train")
    assert len(dataset) > 0

def test_video_dataset_item_shape():
    """Test that dataset returns correct tensor shapes."""
    dataset = VideoDataset(data_dir="data/processed", split="train")
    video, label = dataset[0]
    assert video.ndim == 4  # (T, C, H, W)
    assert isinstance(label, int)
```

### Running Tests Locally

```bash
# All tests
pytest

# Specific test file
pytest tests/test_dataset.py

# With coverage
pytest --cov=src --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

---

## 📚 Documentation

### Updating Documentation

- **Code**: Use docstrings (Google style)
- **README**: Keep examples up to date
- **Guides**: Update `docs/` for major features
- **Changelog**: Add entry to `CHANGELOG.md`

### Building Documentation (if Sphinx is added)

```bash
cd docs
make html
```

---

## 🔄 Pull Request Process

1. **Ensure all tests pass** and coverage is maintained
2. **Update documentation** if adding new features
3. **Add changelog entry** to `CHANGELOG.md`
4. **Request review** from maintainers
5. **Address feedback** promptly
6. **Squash commits** if requested before merge

### PR Checklist

- [ ] Tests pass locally (`pytest -v`)
- [ ] Lint checks pass (`ruff check .`)
- [ ] Type checks pass (`mypy src/`)
- [ ] Code formatted (`black .`, `isort .`)
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] No merge conflicts with `main`
- [ ] Secure cleanup passed

---

## 🏅 Recognition

Contributors will be:
- Added to `CONTRIBUTORS.md`
- Mentioned in release notes
- Recognized in `README.md` (for significant contributions)

---

## 📞 Getting Help

- **Questions**: Open a [Discussion](https://github.com/USERNAME/MindForge-EventSeverity/discussions)
- **Bugs**: Open an [Issue](https://github.com/USERNAME/MindForge-EventSeverity/issues)
- **Chat**: Join our [Discord/Slack] (if available)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.

---

**Thank you for contributing!** 🎉

Every contribution, big or small, helps make this project better.
