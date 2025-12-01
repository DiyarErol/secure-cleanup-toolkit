"""Create stratified train/val/test splits."""

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Create stratified splits of dataset.

    Args:
        input_dir: Input directory with class subdirectories
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Find all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if len(class_dirs) == 0:
        print(f"No class directories found in {input_dir}")
        return

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        # Get all samples (each sample is a directory of frames)
        samples = [d for d in class_dir.iterdir() if d.is_dir()]

        if len(samples) == 0:
            print(f"  No samples found in {class_dir}, skipping")
            continue

        print(f"  Total samples: {len(samples)}")

        # First split: train vs. (val + test)
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            shuffle=True,
        )

        # Second split: val vs. test
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
            shuffle=True,
        )

        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        # Copy samples to splits
        for split_name, split_samples in [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ]:
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for sample in tqdm(split_samples, desc=f"  Copying to {split_name}"):
                dest = split_dir / sample.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(sample, dest)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    split_dataset(
        input_dir,
        output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    print(f"\nDone! Splits saved to: {output_dir}")


if __name__ == "__main__":
    main()
