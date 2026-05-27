"""Train logistic regression and log runs to MLflow (config or CLI parameters)."""

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from src.data.load import load_creditcard_data
from src.data.preprocess import apply_balance_strategy, prepare_train_test
from src.models.estimators.logistic_regression import build_logistic_regression
from src.models.train import train_and_log
from src.utils.config import load_config, PROJECT_ROOT as ROOT

LR_PARAM_KEYS = ("max_iter", "class_weight", "C", "penalty", "solver", "tol")


def _parse_class_weight(value: str | None) -> str | dict | None:
    """Parse CLI class_weight string into a sklearn-compatible value.

    Args:
        value: Raw CLI value; ``None``, ``"none"``, or ``"null"`` map to ``None``.

    Returns:
        Parsed class weight or ``None``.
    """
    if value is None or value.lower() in ("none", "null"):
        return None
    return value


def _extract_lr_params(cfg_block: dict[str, Any]) -> dict[str, Any]:
    """Extract logistic regression hyperparameters from a config block.

    Args:
        cfg_block: YAML block for one experiment or the default model section.

    Returns:
        Dictionary of parameters accepted by :func:`build_logistic_regression`.
    """
    params: dict[str, Any] = {}
    for key in LR_PARAM_KEYS:
        if key in cfg_block:
            params[key] = cfg_block[key]
    if "class_weight" in params and params["class_weight"] is None:
        params["class_weight"] = None
    return params


def _build_run_config(
    cfg: dict[str, Any],
    experiment_name: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve MLflow run name and hyperparameters from config and CLI.

    Args:
        cfg: Full project configuration.
        experiment_name: Named experiment under ``model.logistic_regression.experiments``.
        cli_overrides: Optional CLI parameter overrides.

    Returns:
        Tuple of (run_name, hyperparameter dict).

    Raises:
        ValueError: If ``experiment_name`` is not defined in config.
    """
    lr_cfg = cfg["model"]["logistic_regression"]
    experiments = {e["name"]: e for e in lr_cfg.get("experiments", [])}

    if experiment_name:
        if experiment_name not in experiments:
            available = ", ".join(experiments) or "(none)"
            raise ValueError(
                f"Experiment '{experiment_name}' not found. Available: {available}"
            )
        base = _extract_lr_params(experiments[experiment_name])
        run_name = experiment_name
    else:
        base = _extract_lr_params(lr_cfg)
        run_name = "logistic_regression"

    if cli_overrides:
        base.update(cli_overrides)

    return run_name, base


def train_single_run(
    cfg: dict[str, Any],
    run_name: str,
    lr_params: dict[str, Any],
    balance_strategy: str = "none",
    sampling_ratio: float = 0.5,
) -> dict[str, float]:
    """Load data, train logistic regression, and log one MLflow run.

    Args:
        cfg: Full project configuration.
        run_name: MLflow run name.
        lr_params: Hyperparameters for the estimator.
        balance_strategy: Class balancing strategy for training data.
        sampling_ratio: Minority sampling ratio when balancing is enabled.

    Returns:
        Test-set metrics dictionary.
    """
    seed = cfg["project"]["random_seed"]
    raw_path = ROOT / cfg["paths"]["raw_data"]

    print(f"\n=== Run: {run_name} ===")
    print(f"Hyperparameters: {lr_params}")

    df = load_creditcard_data(raw_path)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(
        df,
        target_column=cfg["data"]["target_column"],
        test_size=cfg["data"]["test_size"],
        random_state=seed,
    )

    n_train_before_balance = len(X_train)
    if balance_strategy != "none":
        print(f"Applying balancing: {balance_strategy} (ratio={sampling_ratio})")
        X_train, y_train = apply_balance_strategy(
            X_train, y_train, strategy=balance_strategy, sampling_ratio=sampling_ratio, random_state=seed
        )

    model = build_logistic_regression(random_state=seed, **lr_params)

    log_params = {
        **lr_params,
        "test_size": cfg["data"]["test_size"],
        "random_seed": seed,
        "balance_strategy": balance_strategy,
    }
    if log_params.get("class_weight") is None:
        log_params["class_weight"] = "none"
    if balance_strategy != "none":
        log_params["sampling_ratio"] = sampling_ratio
        log_params["n_train_before_balance"] = n_train_before_balance

    metrics = train_and_log(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name=run_name,
        params=log_params,
        scaler=scaler,
    )

    print("Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for logistic regression training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train logistic regression with custom parameters."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Name of a variant in model.logistic_regression.experiments",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Train all variants listed in config",
    )
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--class-weight", type=str, default=None)
    parser.add_argument("--C", type=float, default=None, dest="c")
    parser.add_argument("--penalty", type=str, default=None, choices=("l1", "l2", "elasticnet"))
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Use SMOTE to oversample the minority class",
    )
    parser.add_argument(
        "--use-oversampling",
        action="store_true",
        help="Use RandomOverSampler to oversample the minority class",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.5,
        help="Sampling ratio for oversampling/SMOTE (0.0 to 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: configure MLflow and run training."""
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    cli_overrides: dict[str, Any] = {}
    if args.max_iter is not None:
        cli_overrides["max_iter"] = args.max_iter
    if args.class_weight is not None:
        cli_overrides["class_weight"] = _parse_class_weight(args.class_weight)
    if args.c is not None:
        cli_overrides["C"] = args.c
    if args.penalty is not None:
        cli_overrides["penalty"] = args.penalty
    if args.solver is not None:
        cli_overrides["solver"] = args.solver
    if args.tol is not None:
        cli_overrides["tol"] = args.tol

    if args.use_smote and args.use_oversampling:
        raise ValueError(
            "Only one balancing strategy may be used at a time. "
            "Use --use-smote or --use-oversampling, not both."
        )

    balance_strategy = "none"
    if args.use_smote:
        balance_strategy = "smote"
    elif args.use_oversampling:
        balance_strategy = "oversampling"

    if args.all_experiments:
        lr_cfg = cfg["model"]["logistic_regression"]
        for exp in lr_cfg.get("experiments", []):
            run_name, lr_params = _build_run_config(cfg, experiment_name=exp["name"])
            if cli_overrides:
                lr_params.update(cli_overrides)
            if args.run_name:
                run_name = f"{args.run_name}_{exp['name']}"
            train_single_run(cfg, run_name, lr_params, balance_strategy, args.sampling_ratio)
    else:
        run_name, lr_params = _build_run_config(
            cfg,
            experiment_name=args.experiment,
            cli_overrides=cli_overrides or None,
        )
        if args.run_name:
            run_name = args.run_name
        train_single_run(cfg, run_name, lr_params, balance_strategy, args.sampling_ratio)

    print(f"\nMLflow experiments at: {ROOT / cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
