"""Command-line interface for Secure Cleanup Toolkit."""

import argparse
import sys

from src.utils.io import load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything


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
    preprocess_parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
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
    eval_parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Explain command
    explain_parser = subparsers.add_parser(
        "explain", help="Generate explainability visualizations"
    )
    explain_parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    explain_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on videos")
    infer_parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
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
    check_parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Load config
    # Load config with graceful error handling
    try:
        config = load_yaml(args.config)
    except FileNotFoundError as e:
        print(str(e))
        import sys as _sys
        _sys.exit(2)

    # Setup logging
    log_cfg = config.get("logging", {})
    logger = setup_logging(
        log_dir=log_cfg.get("log_dir", "logs"),
        level=log_cfg.get("level", "INFO"),
        log_to_console=log_cfg.get("log_to_console", True),
        log_to_file=log_cfg.get("log_to_file", True),
    )

    # Set seed
    seed = args.seed if hasattr(args, "seed") and args.seed is not None else config.get("seed")
    seed_everything(seed)
    if seed is not None:
        logger.info(f"Random seed set to: {seed}")

    # Execute command
    try:
        if args.command == "preprocess":
            logger.info("Starting preprocessing...")
            # Preprocessing would be implemented here or via scripts
            logger.info(
                "Please use scripts/extract_frames.py and scripts/split_dataset.py "
                "for preprocessing."
            )

        elif args.command == "train":
            from src.train import train

            # Enable W&B if specified
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
            import os

            import torch
            # Basic environment checks
            cuda = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda}")
            if cuda:
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"Current device: {torch.cuda.current_device()}")
            # Config validation
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

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
