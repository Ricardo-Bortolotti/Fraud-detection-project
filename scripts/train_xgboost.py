"""Train XGBoost and log runs to MLflow (config or CLI parameters)."""

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
from src.data.preprocess import apply_balance_strategy, prepare_train_test
from src.models.estimators.xgboost_model import build_xgboost
from src.models.train import train_and_log
from src.utils.config import load_config, PROJECT_ROOT as ROOT

XGB_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
    "n_jobs",
)
XGB_CONFIG_KEYS = XGB_PARAM_KEYS + ("train_sample_size", "auto_scale_pos_weight")


def _extract_xgb_params(cfg_block: dict[str, Any]) -> tuple[dict[str, Any], int | None, bool]:
    """Extract XGBoost hyperparameters from a config block.

    Args:
        cfg_block: YAML block for one experiment or the default model section.

    Returns:
        Tuple of (hyperparameters, optional ``train_sample_size``,
        ``auto_scale_pos_weight`` flag).
    """
    params: dict[str, Any] = {}
    train_sample_size: int | None = None
    auto_scale_pos_weight = cfg_block.get("auto_scale_pos_weight", False)

    for key in XGB_CONFIG_KEYS:
        if key not in cfg_block:
            continue
        if key == "train_sample_size":
            train_sample_size = cfg_block[key]
        elif key == "auto_scale_pos_weight":
            auto_scale_pos_weight = bool(cfg_block[key])
        elif key == "scale_pos_weight" and cfg_block[key] is None:
            params[key] = None
        else:
            params[key] = cfg_block[key]

    return params, train_sample_size, auto_scale_pos_weight


def _subsample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified training subsample.

    Args:
        X_train: Full training feature matrix.
        y_train: Full training labels.
        train_sample_size: Target number of training rows.
        random_state: Random seed for the split.

    Returns:
        Subsampled (X_train, y_train), unchanged if already small enough.
    """
    if len(X_train) <= train_sample_size:
        return X_train, y_train

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_sample_size,
        random_state=random_state,
    )
    idx, _ = next(splitter.split(X_train, y_train))
    return X_train[idx], y_train[idx]


def _resolve_scale_pos_weight(
    y_train: np.ndarray,
    xgb_params: dict[str, Any],
    auto_scale_pos_weight: bool,
) -> float | None:
    """Resolve ``scale_pos_weight`` from config, CLI, or class counts.

    Args:
        y_train: Training labels used when auto-scaling is enabled.
        xgb_params: Hyperparameters that may include an explicit weight.
        auto_scale_pos_weight: Whether to compute weight as negatives/positives.

    Returns:
        Positive-class weight for XGBoost, or ``None`` if not applicable.
    """
    if xgb_params.get("scale_pos_weight") is not None:
        return xgb_params["scale_pos_weight"]
    if auto_scale_pos_weight:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos == 0:
            return None
        return n_neg / n_pos
    return None


def _build_run_config(
    cfg: dict[str, Any],
    experiment_name: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    cli_train_sample_size: int | None = None,
    cli_auto_scale: bool | None = None,
) -> tuple[str, dict[str, Any], int | None, bool]:
    """Resolve MLflow run name, XGBoost params, subsample, and auto-scale flag.

    Args:
        cfg: Full project configuration.
        experiment_name: Named experiment under ``model.xgboost.experiments``.
        cli_overrides: Optional CLI parameter overrides.
        cli_train_sample_size: Optional CLI subsample size override.
        cli_auto_scale: Optional CLI override for auto ``scale_pos_weight``.

    Returns:
        Tuple of (run_name, hyperparameters, train_sample_size, auto_scale).

    Raises:
        ValueError: If ``experiment_name`` is not defined in config.
    """
    xgb_cfg = cfg["model"]["xgboost"]
    experiments = {e["name"]: e for e in xgb_cfg.get("experiments", [])}

    if experiment_name:
        if experiment_name not in experiments:
            available = ", ".join(experiments) or "(none)"
            raise ValueError(
                f"Experiment '{experiment_name}' not found. Available: {available}"
            )
        xgb_params, train_sample_size, auto_scale = _extract_xgb_params(
            experiments[experiment_name]
        )
        run_name = experiment_name
    else:
        xgb_params, train_sample_size, auto_scale = _extract_xgb_params(xgb_cfg)
        run_name = "xgboost"

    if cli_overrides:
        xgb_params.update(cli_overrides)
    if cli_train_sample_size is not None:
        train_sample_size = cli_train_sample_size
    if cli_auto_scale is not None:
        auto_scale = cli_auto_scale

    return run_name, xgb_params, train_sample_size, auto_scale


def train_single_run(
    cfg: dict[str, Any],
    run_name: str,
    xgb_params: dict[str, Any],
    train_sample_size: int | None,
    auto_scale_pos_weight: bool,
    balance_strategy: str = "none",
    sampling_ratio: float = 0.5,
) -> dict[str, float]:
    """Load data, train XGBoost, and log one MLflow run.

    Args:
        cfg: Full project configuration.
        run_name: MLflow run name.
        xgb_params: Hyperparameters for the estimator (except resolved weight).
        train_sample_size: Optional stratified subsample size before training.
        auto_scale_pos_weight: Whether to auto-compute ``scale_pos_weight``.
        balance_strategy: Class balancing strategy for training data.
        sampling_ratio: Minority sampling ratio when balancing is enabled.

    Returns:
        Test-set metrics dictionary.
    """
    seed = cfg["project"]["random_seed"]
    raw_path = ROOT / cfg["paths"]["raw_data"]

    print(f"\n=== Run: {run_name} ===")
    if train_sample_size:
        print(f"Stratified training sample: {train_sample_size:,} rows")

    df = load_creditcard_data(raw_path)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(
        df,
        target_column=cfg["data"]["target_column"],
        test_size=cfg["data"]["test_size"],
        random_state=seed,
    )

    n_train_before = len(X_train)
    if train_sample_size:
        X_train, y_train = _subsample_train(X_train, y_train, train_sample_size, seed)

    n_train_before_balance = len(X_train)
    if balance_strategy != "none":
        print(f"Applying balancing: {balance_strategy} (ratio={sampling_ratio})")
        X_train, y_train = apply_balance_strategy(
            X_train, y_train, strategy=balance_strategy, sampling_ratio=sampling_ratio, random_state=seed
        )

    scale_pos_weight = _resolve_scale_pos_weight(y_train, xgb_params, auto_scale_pos_weight)
    fit_params = {k: v for k, v in xgb_params.items() if k != "scale_pos_weight"}
    model = build_xgboost(random_state=seed, scale_pos_weight=scale_pos_weight, **fit_params)

    print(f"Hyperparameters: {fit_params}")
    print(f"scale_pos_weight: {scale_pos_weight}")

    log_params: dict[str, Any] = {
        **fit_params,
        "scale_pos_weight": scale_pos_weight if scale_pos_weight is not None else "none",
        "auto_scale_pos_weight": auto_scale_pos_weight,
        "test_size": cfg["data"]["test_size"],
        "random_seed": seed,
        "n_train_before_subsample": n_train_before,
        "n_train_used": len(X_train),
        "balance_strategy": balance_strategy,
    }
    if train_sample_size:
        log_params["train_sample_size"] = train_sample_size
    if balance_strategy != "none":
        log_params["sampling_ratio"] = sampling_ratio
        log_params["n_train_before_balance"] = n_train_before_balance

    print("Training XGBoost...")
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
    """Parse command-line arguments for XGBoost training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train XGBoost with custom parameters.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--min-child-weight", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--reg-alpha", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--scale-pos-weight", type=float, default=None)
    parser.add_argument(
        "--auto-scale-pos-weight",
        action="store_true",
        help="Set scale_pos_weight = negatives/positives on training data",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--train-sample-size", type=int, default=None)
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
    """Entry point: configure MLflow and run XGBoost training."""
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    cli_overrides: dict[str, Any] = {}
    if args.n_estimators is not None:
        cli_overrides["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        cli_overrides["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        cli_overrides["learning_rate"] = args.learning_rate
    if args.subsample is not None:
        cli_overrides["subsample"] = args.subsample
    if args.colsample_bytree is not None:
        cli_overrides["colsample_bytree"] = args.colsample_bytree
    if args.min_child_weight is not None:
        cli_overrides["min_child_weight"] = args.min_child_weight
    if args.gamma is not None:
        cli_overrides["gamma"] = args.gamma
    if args.reg_alpha is not None:
        cli_overrides["reg_alpha"] = args.reg_alpha
    if args.reg_lambda is not None:
        cli_overrides["reg_lambda"] = args.reg_lambda
    if args.scale_pos_weight is not None:
        cli_overrides["scale_pos_weight"] = args.scale_pos_weight
    if args.n_jobs is not None:
        cli_overrides["n_jobs"] = args.n_jobs

    active_strategies = []
    if args.train_sample_size is not None:
        active_strategies.append("subsampling")
    if args.use_smote:
        active_strategies.append("smote")
    if args.use_oversampling:
        active_strategies.append("oversampling")

    if len(active_strategies) > 1:
        raise ValueError(
            f"Only one balancing strategy may be used at a time. Active: {', '.join(active_strategies)}"
        )

    balance_strategy = "none"
    if args.use_smote:
        balance_strategy = "smote"
    elif args.use_oversampling:
        balance_strategy = "oversampling"

    cli_auto_scale: bool | None = True if args.auto_scale_pos_weight else None

    if args.all_experiments:
        xgb_cfg = cfg["model"]["xgboost"]
        for exp in xgb_cfg.get("experiments", []):
            run_name, xgb_params, train_sample_size, auto_scale = _build_run_config(
                cfg,
                experiment_name=exp["name"],
                cli_overrides=cli_overrides or None,
                cli_train_sample_size=args.train_sample_size,
                cli_auto_scale=cli_auto_scale,
            )
            if args.run_name:
                run_name = f"{args.run_name}_{exp['name']}"
            train_single_run(cfg, run_name, xgb_params, train_sample_size, auto_scale, balance_strategy, args.sampling_ratio)
    else:
        run_name, xgb_params, train_sample_size, auto_scale = _build_run_config(
            cfg,
            experiment_name=args.experiment,
            cli_overrides=cli_overrides or None,
            cli_train_sample_size=args.train_sample_size,
            cli_auto_scale=cli_auto_scale,
        )
        if args.run_name:
            run_name = args.run_name
        train_single_run(cfg, run_name, xgb_params, train_sample_size, auto_scale, balance_strategy, args.sampling_ratio)

    print(f"\nMLflow experiments at: {ROOT / cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
