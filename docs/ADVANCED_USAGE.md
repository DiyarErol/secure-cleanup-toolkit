# Advanced Usage Guide

## Performance Monitoring and Benchmarking

### Model Benchmarking

Benchmark your trained model to measure FLOPs and inference speed:

```bash
python -m src.cli benchmark \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --output-dir benchmark_results
```

This generates:

- FLOPs calculation
- Inference time statistics (mean, std, min, max)
- Throughput (FPS)
- Model size information

### Data Validation

Validate your dataset quality before training:

```bash
python -m src.cli validate \
    --config configs/default.yaml \
    --output-dir validation_reports
```

Checks include:

- Directory structure validation
- Video file integrity
- Frame count statistics
- Resolution consistency
- Class balance analysis
- Empty or corrupted videos

## Model Export and Deployment

### Export for Production

Export your model in multiple formats:

```bash
python -m src.cli export \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --output-dir deployment \
    --format torchscript onnx
```

This creates a deployment package with:

- `pytorch/`: Original PyTorch checkpoint
- `torchscript/`: TorchScript format (CPU/GPU)
- `onnx/`: ONNX format (cross-platform)
- `metadata.json`: Model configuration
- `README.md`: Usage instructions

### TorchScript Usage

```python
import torch

# Load TorchScript model
model = torch.jit.load('deployment/torchscript/model.pt')
model.eval()

# Inference
input_tensor = torch.randn(1, 16, 3, 224, 224)
with torch.no_grad():
    output = model(input_tensor)
    predictions = torch.softmax(output, dim=1)
```

### ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('deployment/onnx/model.onnx')

# Prepare input
input_data = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)

# Inference
outputs = session.run(None, {'input': input_data})
predictions = outputs[0]
```

## Advanced Training Features

### Mixed Precision Training

Enable mixed precision for faster training on modern GPUs:

```yaml
training:
  mixed_precision: true
  gradient_clip_norm: 1.0
```

### Gradient Accumulation

Train with larger effective batch sizes:

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4 # Effective batch size: 16
```

### Custom Learning Rate Schedules

```yaml
training:
  scheduler:
    type: cosine_annealing
    T_max: 50
    eta_min: 1.0e-6
```

### Early Stopping

```yaml
training:
  early_stopping:
    patience: 10
    min_delta: 0.001
    monitor: val_loss
    mode: min
```

## Performance Optimization

### Model Quantization

Apply dynamic quantization for faster inference:

```python
from src.utils.export import quantize_model_dynamic

quantized_model = quantize_model_dynamic(
    model,
    output_path='models/quantized_model.pt',
    dtype=torch.qint8
)
```

Benefits:

- 50-75% model size reduction
- 2-3x faster inference on CPU
- Minimal accuracy loss

### Profiling

Profile your model to identify bottlenecks:

```python
from src.utils.metrics import compute_model_flops

input_shape = (1, 16, 3, 224, 224)
device = torch.device('cuda')

flops_info = compute_model_flops(model, input_shape, device)
print(f"Total FLOPs: {flops_info['total_flops_human']}")
print(f"Model size: {flops_info['model_size_mb']:.2f} MB")
```

## Monitoring and Visualization

### Training Progress Visualization

```python
from src.utils.visualization import plot_training_curves

plot_training_curves(
    history_csv='checkpoints/training_history.csv',
    output_dir='plots',
    metrics=['train_loss', 'val_loss', 'val_acc']
)
```

### Enhanced Confusion Matrix

```python
from src.utils.visualization import plot_confusion_matrix_advanced

plot_confusion_matrix_advanced(
    cm=confusion_matrix,
    labels=['stable', 'critical', 'terminal'],
    output_path='plots/confusion_matrix.png',
    normalize=True,
    show_percentages=True
)
```

### Training Summary Report

Generate comprehensive training summary:

```python
from src.utils.visualization import create_training_summary_report

create_training_summary_report(
    history_csv='checkpoints/training_history.csv',
    metrics_json='results/metrics.json',
    output_dir='reports'
)
```

## Weights & Biases Integration

Track experiments with W&B:

```bash
python -m src.cli train \
    --config configs/default.yaml \
    --wandb
```

Configure in `configs/default.yaml`:

```yaml
logging:
  wandb:
    enabled: true
    project: severity-classification
    entity: your-team
    name: experiment-1
    tags: [baseline, resnet3d]
```

## Best Practices

### 1. Data Validation First

Always validate your dataset before training:

```bash
python -m src.cli validate --config configs/default.yaml
```

### 2. Start with Smaller Models

Begin with ResNet3D-18 for faster iteration:

```yaml
model:
  backbone: resnet3d_18
  pretrained: true
```

### 3. Use Class Weights for Imbalanced Data

```python
from src.data.dataset import VideoDataset

dataset = VideoDataset(...)
class_weights = dataset.get_class_weights()
```

### 4. Monitor Multiple Metrics

Track accuracy, F1-score, and per-class metrics:

```yaml
training:
  metrics: [accuracy, f1_score, precision, recall]
```

### 5. Regular Checkpointing

```yaml
training:
  save_every_n_epochs: 5
  keep_last_n_checkpoints: 3
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or use gradient accumulation:

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### Slow Training

Enable mixed precision and optimize dataloader:

```yaml
training:
  mixed_precision: true

data:
  num_workers: 4
  pin_memory: true
```

### Poor Convergence

Adjust learning rate and use warmup:

```yaml
training:
  learning_rate: 1.0e-4
  warmup_epochs: 5
```

## Example Workflows

### Complete Training Pipeline

```bash
# 1. Validate data
python -m src.cli validate --config configs/default.yaml

# 2. Train model
python -m src.cli train --config configs/default.yaml --wandb

# 3. Evaluate
python -m src.cli evaluate \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt

# 4. Generate explanations
python -m src.cli explain \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt

# 5. Benchmark
python -m src.cli benchmark \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt

# 6. Export for deployment
python -m src.cli export \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --format torchscript onnx
```

### Production Deployment

```bash
# Export optimized model
python -m src.cli export \
    --config configs/production.yaml \
    --checkpoint checkpoints/best.pt \
    --output-dir production/models

# Run inference server
python scripts/inference_server.py \
    --model production/models/torchscript/model.pt \
    --host 0.0.0.0 \
    --port 8000
```
