"""Treina Random Forest e registra no MLflow (parâmetros via config ou CLI)."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from src.data.load import load_creditcard_data
from src.data.preprocess import prepare_train_test
from src.models.estimators.random_forest import build_random_forest
from src.models.train import train_and_log
from src.utils.config import load_config, PROJECT_ROOT as ROOT

RF_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "n_jobs",
)
RF_CONFIG_KEYS = RF_PARAM_KEYS + ("train_sample_size",)


def _parse_class_weight(value: str | None) -> str | dict | None:
    if value is None or value.lower() in ("none", "null"):
        return None
    return value


def _extract_rf_params(cfg_block: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    params: dict[str, Any] = {}
    train_sample_size: int | None = None

    for key in RF_CONFIG_KEYS:
        if key not in cfg_block:
            continue
        if key == "train_sample_size":
            train_sample_size = cfg_block[key]
        elif key == "class_weight" and cfg_block[key] is None:
            params[key] = None
        else:
            params[key] = cfg_block[key]

    return params, train_sample_size


def _subsample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X_train) <= train_sample_size:
        return X_train, y_train

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_sample_size,
        random_state=random_state,
    )
    idx, _ = next(splitter.split(X_train, y_train))
    return X_train[idx], y_train[idx]


def _build_run_config(
    cfg: dict[str, Any],
    experiment_name: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    cli_train_sample_size: int | None = None,
) -> tuple[str, dict[str, Any], int | None]:
    rf_cfg = cfg["model"]["random_forest"]
    experiments = {e["name"]: e for e in rf_cfg.get("experiments", [])}

    if experiment_name:
        if experiment_name not in experiments:
            available = ", ".join(experiments) or "(nenhuma)"
            raise ValueError(
                f"Experimento '{experiment_name}' não encontrado. Disponíveis: {available}"
            )
        rf_params, train_sample_size = _extract_rf_params(experiments[experiment_name])
        run_name = experiment_name
    else:
        rf_params, train_sample_size = _extract_rf_params(rf_cfg)
        run_name = "random_forest"

    if cli_overrides:
        rf_params.update(cli_overrides)
    if cli_train_sample_size is not None:
        train_sample_size = cli_train_sample_size

    return run_name, rf_params, train_sample_size


def train_single_run(
    cfg: dict[str, Any],
    run_name: str,
    rf_params: dict[str, Any],
    train_sample_size: int | None,
) -> dict[str, float]:
    seed = cfg["project"]["random_seed"]
    raw_path = ROOT / cfg["paths"]["raw_data"]

    print(f"\n=== Run: {run_name} ===")
    print(f"Hiperparâmetros: {rf_params}")
    if train_sample_size:
        print(f"Amostra de treino (estratificada): {train_sample_size:,} linhas")

    df = load_creditcard_data(raw_path)
    X_train, X_test, y_train, y_test, _ = prepare_train_test(
        df,
        target_column=cfg["data"]["target_column"],
        test_size=cfg["data"]["test_size"],
        random_state=seed,
    )

    n_train_before = len(X_train)
    if train_sample_size:
        X_train, y_train = _subsample_train(X_train, y_train, train_sample_size, seed)

    model = build_random_forest(random_state=seed, **rf_params)

    log_params: dict[str, Any] = {
        **rf_params,
        "test_size": cfg["data"]["test_size"],
        "random_seed": seed,
        "n_train_before_subsample": n_train_before,
        "n_train_used": len(X_train),
    }
    if train_sample_size:
        log_params["train_sample_size"] = train_sample_size
    if log_params.get("class_weight") is None:
        log_params["class_weight"] = "none"
    if log_params.get("max_depth") is None:
        log_params["max_depth"] = "none"

    print("Treinando Random Forest...")
    metrics = train_and_log(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name=run_name,
        params=log_params,
    )

    print("Métricas no conjunto de teste:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina Random Forest com parâmetros customizados."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)
    parser.add_argument("--max-features", type=str, default=None)
    parser.add_argument("--class-weight", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--train-sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    cli_overrides: dict[str, Any] = {}
    if args.n_estimators is not None:
        cli_overrides["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        cli_overrides["max_depth"] = args.max_depth
    if args.min_samples_split is not None:
        cli_overrides["min_samples_split"] = args.min_samples_split
    if args.min_samples_leaf is not None:
        cli_overrides["min_samples_leaf"] = args.min_samples_leaf
    if args.max_features is not None:
        cli_overrides["max_features"] = args.max_features
    if args.class_weight is not None:
        cli_overrides["class_weight"] = _parse_class_weight(args.class_weight)
    if args.n_jobs is not None:
        cli_overrides["n_jobs"] = args.n_jobs

    if args.all_experiments:
        rf_cfg = cfg["model"]["random_forest"]
        for exp in rf_cfg.get("experiments", []):
            run_name, rf_params, train_sample_size = _build_run_config(
                cfg,
                experiment_name=exp["name"],
                cli_overrides=cli_overrides or None,
                cli_train_sample_size=args.train_sample_size,
            )
            if args.run_name:
                run_name = f"{args.run_name}_{exp['name']}"
            train_single_run(cfg, run_name, rf_params, train_sample_size)
    else:
        run_name, rf_params, train_sample_size = _build_run_config(
            cfg,
            experiment_name=args.experiment,
            cli_overrides=cli_overrides or None,
            cli_train_sample_size=args.train_sample_size,
        )
        if args.run_name:
            run_name = args.run_name
        train_single_run(cfg, run_name, rf_params, train_sample_size)

    print(f"\nExperimentos MLflow em: {ROOT / cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
