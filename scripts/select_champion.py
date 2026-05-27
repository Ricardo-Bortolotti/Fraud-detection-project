"""Select the champion model from MLflow based on PR-AUC."""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def select_champion(
    experiment_name: str,
    output_dir: Path,
    tracking_uri: str,
) -> None:
    """Download the best run by PR-AUC and save champion artifacts locally.

    Args:
        experiment_name: MLflow experiment name to search.
        output_dir: Parent directory; champion files are written under
            ``output_dir / "champion"``.
        tracking_uri: MLflow tracking server URI.

    Raises:
        ValueError: If the experiment is missing or no run has ``pr_auc``.
    """
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    print(f"Searching runs for experiment: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if "metrics.pr_auc" not in runs.columns:
        raise ValueError("No run with pr_auc metric found")

    runs_with_pr_auc = runs[runs["metrics.pr_auc"].notna()]

    if runs_with_pr_auc.empty:
        raise ValueError("No run with pr_auc metric found")

    champion_run = runs_with_pr_auc.loc[runs_with_pr_auc["metrics.pr_auc"].idxmax()]
    run_id = champion_run["run_id"]
    pr_auc = champion_run["metrics.pr_auc"]

    print("\n=== Champion Model ===")
    print(f"Run ID: {run_id}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Model: {champion_run.get('tags.model_type', 'N/A')}")

    champion_dir = output_dir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading model from MLflow...")
    model_uri = f"runs:/{run_id}/model"
    model_path = champion_dir / "model"
    model_path.mkdir(exist_ok=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(model_path),
    )

    metadata = {
        "run_id": run_id,
        "pr_auc": float(pr_auc),
        "model_type": champion_run.get("tags.model_type", "unknown"),
        "metrics": {
            col: float(champion_run[col])
            for col in runs.columns
            if col.startswith("metrics.") and pd.notna(champion_run[col])
        },
        "params": {
            col: champion_run[col]
            for col in runs.columns
            if col.startswith("params.") and pd.notna(champion_run[col])
        },
    }

    metadata_path = champion_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {champion_dir}")
    print(f"Metadata saved to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for champion selection.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Select the champion model based on PR-AUC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the champion model",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load config and run champion selection."""
    args = parse_args()
    cfg = load_config(args.config)

    experiment_name = args.experiment_name or cfg["mlflow"]["experiment_name"]
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models"
    tracking_uri = cfg["mlflow"]["tracking_uri"]

    select_champion(experiment_name, output_dir, tracking_uri)


if __name__ == "__main__":
    main()
