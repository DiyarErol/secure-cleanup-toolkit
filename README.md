# Secure Cleanup Toolkit

> Production-grade severity classification for autonomous risk understanding

[![CI](https://github.com/DiyarErol/secure-cleanup-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/DiyarErol/secure-cleanup-toolkit/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security](https://img.shields.io/badge/security-automated_cleanup-green.svg)](docs/PUBLISH.md)

A comprehensive, research-grade framework for video-based severity classification with a focus on autonomous risk understanding. This project provides a complete pipeline from data preprocessing to model training, evaluation, and explainability.

## 🎯 Project Overview

**Domain:** Autonomous Risk Understanding
**Primary Modality:** Video (with hooks for Audio/Text)
**Classification Task:** 3-class severity prediction
**Labels:** Stable / Critical / Terminal
**Framework:** PyTorch (TensorFlow-ready architecture)
**Supported OS:** Windows, macOS, Linux

## 📋 Features

- **Config-driven design:** All hyperparameters, paths, and settings in YAML
- **Multiple baseline models:** 3D-ResNet, SlowFast, TimeSformer stubs
- **Comprehensive data pipeline:** Frame extraction, augmentation, stratified splitting
- **Evaluation suite:** Per-class metrics, confusion matrices, PR curves
- **Explainability:** Grad-CAM/saliency maps for video key frames
- **Production-ready:** Type hints, structured logging, deterministic training
- **Ethical AI:** Explicit guidance for sensitive media handling

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- 4GB+ GPU recommended (CPU training supported but slow)

### Installation

#### Windows (PowerShell)

````powershell
# Clone the repository
git clone https://github.com/DiyarErol/secure-cleanup-toolkit.git
cd secure-cleanup-toolkit

# Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install
python -m pip install --upgrade pip
pip install -e .
# GPU Setup (Optional)
To enable CUDA acceleration:

```powershell
# Windows (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
````

```bash
# Linux / macOS (MPS)
pip install torch torchvision torchaudio
```

Run a quick check:

```powershell
python -m src.cli check --config configs/default.yaml
```

## Install development dependencies (optional)

```bash
pip install -e ".[dev]"
```

## Verify installation

```bash
python -m src.cli --help
```



### Automated Checks

- **Pre-commit hook**: Commits are blocked if AI/Copilot/GPT traces are detected.
- **CI pipelines**: Jobs fail if cleanup findings remain unaddressed.
- **Configuration**: Patterns and excludes can be customized in `configs/cleanup.yaml`.
- **Final validation**: Run full pre-publish validation before releasing.

```bash
# Run comprehensive validation pipeline
python scripts/final_publish_check.py

# This checks:
# ✓ Required files present
# ✓ Lint checks pass (ruff)
# ✓ Unit tests pass (pytest)
# ✓ No AI traces (secure cleanup)
# ✓ Git status and readiness
```

### Automated Metadata Cleanup

Commit and CI pipelines enforce strict prevention, ensuring a fully clean, human-verified codebase.

```powershell
# Quick automated cleanup
python scripts/secure_cleanup.py --force
```

To bypass the pre-commit hook once (not recommended):

```bash
git commit -m "message" --no-verify
```

#### macOS / Linux (Bash/Zsh)

```bash
# Clone the repository
git clone https://github.com/DiyarErol/secure-cleanup-toolkit.git
cd secure-cleanup-toolkit

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install
python -m pip install --upgrade pip
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Verify installation
python -m src.cli --help
```

## 📁 Project Structure

```text
.
├── README.md                  # This file
├── LICENSE                    # MIT License
├── pyproject.toml            # Project metadata and dependencies
├── .gitignore                # Git ignore rules
├── .gitattributes            # Git attributes
├── .editorconfig             # Editor configuration
├── configs/                  # Configuration files
│   ├── default.yaml          # Main configuration
│   └── model_baselines.yaml  # Baseline model configs
├── data/                     # Data directory (add your data here)
│   ├── raw/                  # Original, immutable data
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final train/val/test splits
├── docs/                     # Documentation
│   ├── DATASET_CARD.md       # Dataset documentation
│   ├── ETHICS.md             # Ethical guidelines
│   ├── EXPERIMENTS.md        # Experiment tracking guide
│   └── MODEL_CARD.md         # Model documentation
├── notebooks/                # Jupyter notebooks
│   └── 00_exploration.ipynb  # Data exploration
├── scripts/                  # Utility scripts
│   ├── split_dataset.py      # Create train/val/test splits
│   ├── extract_frames.py     # Extract frames from videos
│   ├── augment_preview.py    # Preview augmentations
│   └── infer_folder.py       # Batch inference
├── src/                      # Source code
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation logic
│   ├── explain.py            # Explainability
│   ├── utils/                # Utilities
│   ├── data/                 # Data loading and transforms
│   └── models/               # Model architectures
├── tests/                    # Unit tests
└── .github/workflows/        # CI/CD
    └── ci.yml                # GitHub Actions
```

## ✅ Environment Check

```powershell
python -m src.cli check --config configs/default.yaml
```

- Detects CUDA and AMP availability
- Verifies config validity
- Confirms dataset folders exist

Expected output:

```text
✅ PyTorch 2.9.1
✅ CUDA available: True
✅ AMP enabled: True
✅ Config valid
✅ Data root found: data/processed
```

## 🔧 Usage

### 1. Prepare Your Data

Place your raw video files in `data/raw/` organized by class:

```text
data/raw/
├── stable/
│   ├── video001.mp4
│   └── video002.mp4
├── critical/
│   ├── video003.mp4
│   └── video004.mp4
└── terminal/
    ├── video005.mp4
    └── video006.mp4
```

### 2. Preprocess Data

Extract frames and create train/val/test splits:

```bash
# Extract frames from videos (configurable FPS, resolution)
python -m src.cli preprocess --config configs/default.yaml

# Or use the standalone script
python scripts/extract_frames.py --input data/raw --output data/interim/frames --fps 10

# Create stratified splits (80/10/10 train/val/test)
python scripts/split_dataset.py --input data/interim/frames --output data/processed --seed 42
```

### 3. Preview Augmentations (Optional)

```bash
python scripts/augment_preview.py --config configs/default.yaml --samples 5
```

### 4. Train a Model

```bash
# Train with default config
python -m src.cli train --config configs/default.yaml

# Train a specific baseline
python -m src.cli train --config configs/model_baselines.yaml --model resnet3d

# Resume from checkpoint
python -m src.cli train --config configs/default.yaml --resume checkpoints/last.pt
```

**Training Options:**

- Mixed precision training (FP16) enabled by default for speed
- Early stopping on validation loss (patience=10 epochs)
- Automatic checkpointing (best + last)
- Deterministic mode via `seed_everything()` for reproducibility

### 5. Evaluate

```bash
# Evaluate on test set
python -m src.cli evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt

# Generates:
# - reports/metrics.json (accuracy, precision, recall, F1 per class)
# - reports/confusion_matrix.png
# - reports/pr_curves.png
# - reports/evaluation_report.html
```

### 6. Explain Predictions

```bash
# Generate Grad-CAM visualizations
python -m src.cli explain --config configs/default.yaml --checkpoint checkpoints/best.pt

# Saves heatmaps to reports/explainability/
```

### 7. Batch Inference

```bash
# Infer on a folder of videos
python scripts/infer_folder.py --input path/to/videos --checkpoint checkpoints/best.pt --output predictions.csv
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

- **Data paths** and **label mapping**
- **Preprocessing** (FPS, resolution, normalization)
- **Augmentations** (flip, crop, color jitter, temporal jitter)
- **Model architecture** (backbone, hidden dims, dropout)
- **Training hyperparameters** (batch size, learning rate, epochs)
- **Logging and checkpointing**

Example snippet:

```yaml
data:
  labels: ["stable", "critical", "terminal"]
  fps: 10
  resolution: [224, 224]

model:
  backbone: "resnet3d_18"
  num_classes: 3
  dropout: 0.5

training:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 50
  optimizer: "adamw"
  scheduler: "cosine"
```

## 🧪 Testing

Run unit tests:

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/test_dataset.py -v
```

## 🛠️ Development

### Code Quality

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/

# Format
black src/ tests/
isort src/ tests/
```

### VS Code Tasks

Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS) and select:

- **Tasks: Run Task** → **Run Tests**
- **Tasks: Run Task** → **Lint**
- **Tasks: Run Task** → **Train Model**

## 📊 Experiment Tracking

By default, training logs are saved to CSV files in `logs/`. For advanced tracking:

```bash
# Enable Weights & Biases (optional)
pip install -e ".[wandb]"
export WANDB_API_KEY=your_key_here
python -m src.cli train --config configs/default.yaml --wandb
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for details.

## 📚 Documentation

- **[Dataset Card](docs/DATASET_CARD.md):** Dataset description, collection protocol, labeling guidelines
- **[Ethics Guide](docs/ETHICS.md):** Sensitive content policy, anonymization, usage boundaries
- **[Model Card](docs/MODEL_CARD.md):** Model intent, performance, limitations, known failure modes
- **[Experiments Guide](docs/EXPERIMENTS.md):** How to track and reproduce experiments

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Ensure tests pass and code is formatted before submitting.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you installed in editable mode:

```bash
pip install -e .
```

### Issue: CUDA out of memory

**Solutions:**

- Reduce `batch_size` in config (e.g., 8 → 4)
- Reduce video resolution (e.g., 224 → 112)
- Enable gradient accumulation (set `gradient_accumulation_steps: 2`)

### Issue: Empty data folders

**Solution:** Place your videos in `data/raw/` following the structure above, or update paths in `configs/default.yaml`.

### Issue: Slow training on CPU

**Solution:** Training on CPU is supported but slow. Consider:

- Using a GPU (CUDA)
- Reducing model size (smaller backbone)
- Decreasing number of frames per video

### Windows-specific: PowerShell execution policy error

**Solution:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS-specific: SSL certificate error

**Solution:**

```bash
/Applications/Python\ 3.10/Install\ Certificates.command
```

### Linux-specific: Permission denied on scripts

**Solution:**

```bash
chmod +x scripts/*.py
```

## 🙏 Acknowledgments

- **PyTorch** and **torchvision** for deep learning infrastructure
- **timm** for vision model backbones
- **grad-cam** for explainability
- Community contributors and researchers in video understanding

## 📧 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/DiyarErol/secure-cleanup-toolkit/issues).

---

Happy modeling! 🚀
