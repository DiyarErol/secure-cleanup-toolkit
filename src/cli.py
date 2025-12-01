"""Command-line interface for Secure Cleanup Toolkit."""

import argparse
import sys

from src.utils.io import load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

# Constants for help strings
CONFIG_HELP = "Path to config file"
CHECKPOINT_HELP = "Path to model checkpoint"


def _load_config(config_path: str) -> dict:
    """Load and validate configuration file."""
    try:
        return load_yaml(config_path)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)


def _setup_logger(config: dict):
    """Setup logging from configuration."""
    log_cfg = config.get("logging", {})
    return setup_logging(
        log_dir=log_cfg.get("log_dir", "logs"),
        level=log_cfg.get("level", "INFO"),
        log_to_console=log_cfg.get("log_to_console", True),
        log_to_file=log_cfg.get("log_to_file", True),
    )


def _setup_seed(args, config: dict, logger):
    """Setup random seed from args or config."""
    seed = args.seed if hasattr(args, "seed") and args.seed is not None else config.get("seed")
    seed_everything(seed)
    if seed is not None:
        logger.info(f"Random seed set to: {seed}")


def _handle_check_command(config: dict, logger):
    """Handle check command - environment and config validation."""
    import os

    import torch

    cuda = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda}")
    if cuda:
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")

    data_root = config.get("data", {}).get("root") or config.get("data", {}).get("processed_dir")
    if not data_root:
        raise ValueError("Config missing data.root or data.processed_dir")
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    labels = config.get("data", {}).get("labels")
    if not labels or not isinstance(labels, list):
        raise ValueError("Config data.labels must be a non-empty list")

    logger.info(f"Labels: {labels}")
    logger.info("Environment & config checks passed")


def _handle_validate_command(args, config: dict, logger):
    """Handle validate command - data quality validation."""
    from src.utils.validation import DataValidator

    logger.info("Starting data validation...")
    data_dir = config["data"].get("root") or config["data"].get("processed_dir")
    labels = config["data"]["labels"]

    validator = DataValidator(data_dir, labels)
    is_valid = validator.run_full_validation(args.output_dir)

    if is_valid:
        logger.info("✓ Dataset validation passed!")
    else:
        logger.warning(f"⚠ Dataset validation found issues. Check {args.output_dir}")


def _handle_benchmark_command(args, config: dict, logger):
    """Handle benchmark command - model performance benchmarking."""
    import json
    import os

    import torch

    from src.models.builder import build_model
    from src.utils.metrics import compute_model_flops, measure_inference_time

    logger.info("Starting model benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Input shape
    input_shape = (
        1,
        config["data"]["num_frames"],
        3,
        config["data"]["input_size"],
        config["data"]["input_size"],
    )

    # Run benchmarks
    logger.info("Computing FLOPs...")
    flops_info = compute_model_flops(model, input_shape, device)

    logger.info("Measuring inference time...")
    timing_info = measure_inference_time(model, input_shape, device)

    # Save results
    results = {"flops": flops_info, "timing": timing_info, "device": str(device)}
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "benchmark_results.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark results saved to {output_file}")
    logger.info(f"FLOPs: {flops_info['total_flops_human']}")
    logger.info(f"Inference time: {timing_info['mean_ms']:.2f}ms ± {timing_info['std_ms']:.2f}ms")
    logger.info(f"Throughput: {timing_info['fps']:.2f} FPS")


def _handle_export_command(args, config: dict, logger):
    """Handle export command - model deployment export."""
    from src.utils.export import create_deployment_package

    logger.info("Exporting model for deployment...")
    create_deployment_package(
        model_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        include_onnx="onnx" in args.format,
        include_torchscript="torchscript" in args.format,
    )
    logger.info(f"Deployment package created at {args.output_dir}")


def _execute_command(args, config: dict, logger, parser):
    """Execute the requested command."""
    if args.command == "preprocess":
        logger.info("Starting preprocessing...")
        logger.info(
            "Please use scripts/extract_frames.py and scripts/split_dataset.py for preprocessing."
        )

    elif args.command == "train":
        from src.train import train

        if hasattr(args, "wandb") and args.wandb:
            config["logging"]["wandb"]["enabled"] = True
        logger.info("Starting training...")
        train(config, resume=args.resume)

    elif args.command == "evaluate":
        from src.evaluate import evaluate

        logger.info("Starting evaluation...")
        evaluate(config, checkpoint_path=args.checkpoint)

    elif args.command == "explain":
        from src.explain import explain

        logger.info("Generating explanations...")
        explain(config, checkpoint_path=args.checkpoint)

    elif args.command == "infer":
        logger.info("Running inference...")
        logger.info("Use scripts/infer_folder.py for batch inference")

    elif args.command == "check":
        _handle_check_command(config, logger)

    elif args.command == "validate":
        _handle_validate_command(args, config, logger)

    elif args.command == "benchmark":
        _handle_benchmark_command(args, config, logger)

    elif args.command == "export":
        _handle_export_command(args, config, logger)

    else:
        parser.print_help()
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Secure Cleanup Toolkit by Diyar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess videos (extract frames and create splits)"
    )
    preprocess_parser.add_argument("--config", type=str, required=True, help="Path to config file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)"
    )
    train_parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Generate explainability visualizations")
    explain_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    explain_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on videos")
    infer_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    infer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--input", type=str, required=True, help="Path to input video or directory"
    )
    infer_parser.add_argument(
        "--output", type=str, default="predictions.csv", help="Path to output CSV"
    )

    # Check command (environment & data validation)
    check_parser = subparsers.add_parser("check", help="Run environment and data checks")
    check_parser.add_argument("--config", type=str, required=True, help="Path to config file")

    # Validate command (data quality checks)
    validate_parser = subparsers.add_parser("validate", help="Validate dataset quality")
    validate_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    validate_parser.add_argument(
        "--output-dir", type=str, default="validation_reports", help="Output directory for reports"
    )

    # Benchmark command (model performance)
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    benchmark_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    benchmark_parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory for results"
    )

    # Export command (model deployment)
    export_parser = subparsers.add_parser("export", help="Export model for deployment")
    export_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    export_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    export_parser.add_argument(
        "--output-dir", type=str, default="deployment", help="Output directory"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["torchscript", "onnx"],
        help="Export formats (torchscript, onnx)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Load and setup
    config = _load_config(args.config)
    logger = _setup_logger(config)
    _setup_seed(args, config, logger)

    # Execute command
    try:
        _execute_command(args, config, logger, parser)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
