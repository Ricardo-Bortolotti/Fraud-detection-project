"""Seleciona o modelo campeão do MLflow baseado em PR-AUC."""

import argparse
import json
import shutil
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
    """Seleciona o modelo com maior PR-AUC do experimento."""
    mlflow.set_tracking_uri(tracking_uri)

    # Buscar experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado")

    print(f"Buscando runs do experimento: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")

    # Buscar todas as runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Filtrar runs que têm métrica pr_auc
    runs_with_pr_auc = runs[runs["metrics.pr_auc"].notna()]

    if runs_with_pr_auc.empty:
        raise ValueError("Nenhuma run com métrica pr_auc encontrada")

    # Selecionar run com maior pr_auc
    champion_run = runs_with_pr_auc.loc[runs_with_pr_auc["metrics.pr_auc"].idxmax()]
    run_id = champion_run["run_id"]
    pr_auc = champion_run["metrics.pr_auc"]

    print(f"\n=== Modelo Campeão ===")
    print(f"Run ID: {run_id}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Modelo: {champion_run.get('tags.model_type', 'N/A')}")

    # Criar diretório de saída
    champion_dir = output_dir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)

    # Baixar modelo do MLflow
    print(f"\nBaixando modelo do MLflow...")
    model_uri = f"runs:/{run_id}/model"
    model_path = champion_dir / "model"
    model_path.mkdir(exist_ok=True)

    # Baixar artefatos do modelo
    artifact_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(model_path),
    )

    # Salvar metadados
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
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModelo salvo em: {champion_dir}")
    print(f"Metadados salvos em: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona o modelo campeão baseado em PR-AUC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho para config.yaml",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Nome do experimento no MLflow",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório de saída para o modelo campeão",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    experiment_name = args.experiment_name or cfg["mlflow"]["experiment_name"]
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models"
    tracking_uri = cfg["mlflow"]["tracking_uri"]

    select_champion(experiment_name, output_dir, tracking_uri)


if __name__ == "__main__":
    main()
