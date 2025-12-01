# Experiment Tracking Guide

## Overview

This guide explains how to track, reproduce, and analyze experiments in the Secure Cleanup Toolkit project. Proper experiment tracking is essential for reproducible research and systematic model development.

## Experiment Tracking Methods

### 1. Local Logging (Default)

By default, all training runs log to local CSV files in the `logs/` directory.

**Directory Structure:**

```
logs/
├── train_20251201_143022.csv      # Training metrics
├── train_20251201_143022.log      # Detailed logs
├── config_20251201_143022.yaml    # Config snapshot
└── summary.csv                     # Aggregate results
```

**What's Logged:**

- Training loss per epoch
- Validation loss per epoch
- Validation accuracy, precision, recall, F1
- Learning rate per epoch
- Epoch duration
- Best validation metric
- Final test metrics

**Accessing Logs:**

```python
import pandas as pd

# Read training log
df = pd.read_csv('logs/train_20251201_143022.csv')

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

### 2. Weights & Biases (Optional)

For advanced tracking with visualization, experiment comparison, and team collaboration, use Weights & Biases.

**Setup:**

```bash
# Install wandb
pip install -e ".[wandb]"

# Login (one-time)
wandb login
```

**Enable in Config:**

Edit `configs/default.yaml`:

```yaml
logging:
  wandb:
    enabled: true
    project: "secure-cleanup-toolkit"
    entity: "your-username"  # or team name
    tags: ["baseline", "resnet3d"]
    notes: "Initial baseline experiment"
```

**Or via CLI:**

```bash
python -m src.cli train --config configs/default.yaml --wandb
```

**What's Logged to W&B:**

- All metrics from local logging
- System metrics (GPU utilization, CPU, memory)
- Model gradients and parameters (optional)
- Training configuration
- Code version (git commit hash)
- Model checkpoints (optional)

**Features:**

- Interactive plots and dashboards
- Hyperparameter sweeps
- Model comparison
- Team collaboration
- Artifact versioning

**Access Dashboard:**

 Visit `https://wandb.ai/diyar/secure-cleanup-toolkit`

### 3. TensorBoard (Alternative)

For TensorBoard integration (not implemented by default, but easily added):

```python
# In src/train.py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f'runs/{experiment_name}')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/val', accuracy, epoch)
writer.close()
```

**Launch TensorBoard:**

```bash
tensorboard --logdir=runs
```

## Reproducibility

### Setting Random Seeds

For deterministic training, set `seed` in `configs/default.yaml`:

```yaml
seed: 42
```

This seeds:
- Python's `random` module
- NumPy's random generator
- PyTorch's CPU and CUDA random generators
- DataLoader worker initialization

**Note:** Full determinism on GPU requires additional settings and may reduce performance. See `src/utils/seed.py` for details.

### Config Versioning

Every training run saves a snapshot of the configuration:

```
logs/config_20251201_143022.yaml
```

To reproduce an experiment:

```bash
python -m src.cli train --config logs/config_20251201_143022.yaml
```

### Code Versioning

**Git Commit Tracking:**

Training logs include the git commit hash (if in a git repo):

```
Git Commit: a3f5b7c2
Git Branch: main
Git Status: clean
```

To reproduce from a specific commit:

```bash
git checkout a3f5b7c2
python -m src.cli train --config logs/config_20251201_143022.yaml
```

**Best Practice:** Always commit code before running experiments.

### Environment Tracking

**Record Environment:**

```bash
# Capture pip environment
pip list --format=freeze > logs/environment_20251201_143022.txt

# Or use pip-tools
pip-compile --output-file=logs/requirements_20251201_143022.txt
```

**Reproduce Environment:**

```bash
pip install -r logs/environment_20251201_143022.txt
```

## Experiment Organization

### Naming Conventions

Use descriptive names for experiments:

**Format:** `{model}_{dataset}_{config}_{date}`

Examples:
- `resnet3d18_full_baseline_20251201`
- `slowfast_augmented_focal_20251215`
- `timesformer_small_regularized_20251220`

**Implement via Config:**

```yaml
project:
  name: "resnet3d18_full_baseline_20251201"
```

### Directory Structure

Organize experiment artifacts:

```
experiments/
├── resnet3d18_baseline/
│   ├── config.yaml
│   ├── checkpoints/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── logs/
│   │   ├── train.csv
│   │   └── train.log
│   └── reports/
│       ├── metrics.json
│       ├── confusion_matrix.png
│       └── evaluation_report.html
└── slowfast_augmented/
    └── ...
```

### Experiment Metadata

Maintain an experiment registry (CSV or database):

**experiments_registry.csv:**

| exp_id | date       | model      | config    | val_acc | test_acc | notes             |
| ------ | ---------- | ---------- | --------- | ------- | -------- | ----------------- |
| 001    | 2025-12-01 | resnet3d18 | baseline  | 0.82    | 0.80     | Initial baseline  |
| 002    | 2025-12-05 | resnet3d34 | baseline  | 0.85    | 0.83     | Larger model      |
| 003    | 2025-12-10 | resnet3d18 | augmented | 0.84    | 0.82     | More augmentation |

## Hyperparameter Tuning

### Manual Grid Search

Create config variants:

```
configs/
├── default.yaml
├── hparam_lr_1e-3.yaml
├── hparam_lr_1e-4.yaml
├── hparam_lr_1e-5.yaml
└── ...
```

Run sequentially:

```bash
for config in configs/hparam_*.yaml; do
    python -m src.cli train --config $config
done
```

### Weights & Biases Sweeps

Create sweep config:

**sweep.yaml:**

```yaml
program: src/cli.py
method: bayes  # or 'grid', 'random'
metric:
  name: val_acc
  goal: maximize
parameters:
  learning_rate:
    min: 0.00001
    max: 0.001
  batch_size:
    values: [4, 8, 16]
  dropout:
    min: 0.3
    max: 0.7
```

**Run Sweep:**

```bash
# Initialize sweep
wandb sweep sweep.yaml

# Run agents (can run multiple in parallel)
wandb agent your-username/secure-cleanup-toolkit/sweep-id
```

### Optuna Integration

For advanced hyperparameter optimization, integrate Optuna (not implemented by default):

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    
    # Train model with these hyperparameters
    val_acc = train_and_evaluate(lr=lr, dropout=dropout)
    
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Analyzing Results

### Compare Experiments

**Using Local Logs:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple experiments
exp1 = pd.read_csv('logs/exp001_train.csv')
exp2 = pd.read_csv('logs/exp002_train.csv')

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(exp1['epoch'], exp1['val_acc'], label='Baseline')
plt.plot(exp2['epoch'], exp2['val_acc'], label='Augmented')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Experiment Comparison')
plt.savefig('reports/experiment_comparison.png')
```

**Using W&B:**

- Use the W&B dashboard to compare runs interactively
- Create custom plots and tables
- Export results for publication

### Statistical Significance

Run multiple seeds and test significance:

```bash
# Run same config with different seeds
for seed in 42 43 44 45 46; do
    python -m src.cli train --config configs/default.yaml --seed $seed
done
```

**Analyze:**

```python
import numpy as np
from scipy import stats

# Collect results from multiple seeds
baseline_accs = [0.80, 0.81, 0.79, 0.82, 0.80]
new_method_accs = [0.83, 0.84, 0.82, 0.85, 0.83]

# Paired t-test
t_stat, p_value = stats.ttest_rel(baseline_accs, new_method_accs)

print(f"Baseline: {np.mean(baseline_accs):.3f} ± {np.std(baseline_accs):.3f}")
print(f"New Method: {np.mean(new_method_accs):.3f} ± {np.std(new_method_accs):.3f}")
print(f"p-value: {p_value:.4f}")
```

## Best Practices

### Before Starting Experiments

1. **Define Goals:** What question are you trying to answer?
2. **Plan Metrics:** What metrics will you track?
3. **Set Baselines:** Establish baseline performance first
4. **Version Control:** Commit code before experimenting

### During Experiments

1. **Document Changes:** Keep notes on what you're testing
2. **Track Everything:** Better to log too much than too little
3. **Monitor Progress:** Check training curves regularly
4. **Save Checkpoints:** Don't lose progress to crashes

### After Experiments

1. **Analyze Results:** Don't just collect metrics—understand them
2. **Document Findings:** Update experiment registry and notes
3. **Share Results:** Communicate findings to team
4. **Clean Up:** Archive or delete unnecessary artifacts

## Troubleshooting

### Missing Logs

If logs aren't being created:
- Check `configs/default.yaml` → `logging.log_to_file: true`
- Verify `logs/` directory exists and is writable
- Check console output for error messages

### W&B Connection Issues

```bash
# Re-authenticate
wandb login --relogin

# Test connection
wandb online

# Check status
wandb status
```

### Incomplete Experiments

If training crashes:
- Resume from last checkpoint: `--resume checkpoints/last.pt`
- Check logs for error messages
- Verify GPU memory isn't exceeded

## Advanced Topics

### Distributed Training Tracking

For multi-GPU training:
- Only log from rank 0 to avoid duplicate entries
- Aggregate metrics across GPUs before logging

### Experiment Templates

Create reusable experiment templates:

```yaml
# template_augmentation_study.yaml
experiments:
  - name: "no_aug"
    data:
      augmentation:
        enabled: false
  - name: "light_aug"
    data:
      augmentation:
        horizontal_flip: 0.5
  - name: "heavy_aug"
    data:
      augmentation:
        horizontal_flip: 0.5
        color_jitter:
          brightness: 0.3
```

### Continuous Integration

Integrate experiment tracking with CI:
- Run smoke tests (1 epoch) on every commit
- Track test accuracy over time
- Alert on performance degradation

## Resources

### Tools

- **Weights & Biases:** https://wandb.ai
- **TensorBoard:** https://www.tensorflow.org/tensorboard
- **MLflow:** https://mlflow.org
- **Optuna:** https://optuna.org

### Reading

- *Experiment Tracking Best Practices:* https://neptune.ai/blog/ml-experiment-tracking
- *Reproducible ML:* https://www.google.com/url?sa=t&url=https://reproducible.cs.princeton.edu/

---

**Questions?** Open an issue on GitHub or contact the maintainers.

**Last Updated:** December 2025
