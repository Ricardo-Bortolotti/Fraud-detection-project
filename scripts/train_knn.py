"""Treina KNN e registra no MLflow (parâmetros via config ou CLI)."""

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
from src.models.estimators.knn import build_knn
from src.models.train import train_and_log
from src.utils.config import load_config, PROJECT_ROOT as ROOT

KNN_PARAM_KEYS = ("n_neighbors", "weights", "algorithm", "metric", "p", "n_jobs")
KNN_CONFIG_KEYS = KNN_PARAM_KEYS + ("train_sample_size",)


def _extract_knn_params(cfg_block: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    """Extrai hiperparâmetros do KNN e tamanho opcional da amostra de treino."""
    params: dict[str, Any] = {}
    train_sample_size: int | None = None

    for key in KNN_CONFIG_KEYS:
        if key not in cfg_block:
            continue
        if key == "train_sample_size":
            train_sample_size = cfg_block[key]
        else:
            params[key] = cfg_block[key]

    return params, train_sample_size


def _subsample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra estratificada do treino (KNN é custoso em datasets grandes)."""
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
    knn_cfg = cfg["model"]["knn"]
    experiments = {e["name"]: e for e in knn_cfg.get("experiments", [])}

    if experiment_name:
        if experiment_name not in experiments:
            available = ", ".join(experiments) or "(nenhuma)"
            raise ValueError(
                f"Experimento '{experiment_name}' não encontrado. Disponíveis: {available}"
            )
        knn_params, train_sample_size = _extract_knn_params(experiments[experiment_name])
        run_name = experiment_name
    else:
        knn_params, train_sample_size = _extract_knn_params(knn_cfg)
        run_name = "knn"

    if cli_overrides:
        knn_params.update(cli_overrides)
    if cli_train_sample_size is not None:
        train_sample_size = cli_train_sample_size

    return run_name, knn_params, train_sample_size


def train_single_run(
    cfg: dict[str, Any],
    run_name: str,
    knn_params: dict[str, Any],
    train_sample_size: int | None,
    balance_strategy: str = "none",
    sampling_ratio: float = 0.5,
) -> dict[str, float]:
    seed = cfg["project"]["random_seed"]
    raw_path = ROOT / cfg["paths"]["raw_data"]

    print(f"\n=== Run: {run_name} ===")
    print(f"Hiperparâmetros: {knn_params}")
    if train_sample_size:
        print(f"Amostra de treino (estratificada): {train_sample_size:,} linhas")

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
        print(f"Aplicando balanceamento: {balance_strategy} (ratio={sampling_ratio})")
        X_train, y_train = apply_balance_strategy(
            X_train, y_train, strategy=balance_strategy, sampling_ratio=sampling_ratio, random_state=seed
        )

    model = build_knn(**knn_params)

    log_params: dict[str, Any] = {
        **knn_params,
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

    print("Treinando KNN...")
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

    print("Métricas no conjunto de teste:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina KNN com parâmetros customizados.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nome em model.knn.experiments",
    )
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        choices=("uniform", "distance"),
    )
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Subamostra estratificada do treino (recomendado para dataset grande)",
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Usa SMOTE para oversampling da classe minoritária",
    )
    parser.add_argument(
        "--use-oversampling",
        action="store_true",
        help="Usa RandomOverSampler para oversampling da classe minoritária",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.5,
        help="Ratio para oversampling/SMOTE (0.0 a 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    cli_overrides: dict[str, Any] = {}
    if args.n_neighbors is not None:
        cli_overrides["n_neighbors"] = args.n_neighbors
    if args.weights is not None:
        cli_overrides["weights"] = args.weights
    if args.metric is not None:
        cli_overrides["metric"] = args.metric
    if args.p is not None:
        cli_overrides["p"] = args.p
    if args.algorithm is not None:
        cli_overrides["algorithm"] = args.algorithm
    if args.n_jobs is not None:
        cli_overrides["n_jobs"] = args.n_jobs

    # Validação de estratégias de balanceamento mutuamente exclusivas
    active_strategies = []
    if args.train_sample_size is not None:
        active_strategies.append("subsampling")
    if args.use_smote:
        active_strategies.append("smote")
    if args.use_oversampling:
        active_strategies.append("oversampling")

    if len(active_strategies) > 1:
        raise ValueError(
            f"Apenas uma estratégia de balanceamento pode ser usada por vez. Ativas: {', '.join(active_strategies)}"
        )

    balance_strategy = "none"
    if args.use_smote:
        balance_strategy = "smote"
    elif args.use_oversampling:
        balance_strategy = "oversampling"

    if args.all_experiments:
        knn_cfg = cfg["model"]["knn"]
        for exp in knn_cfg.get("experiments", []):
            run_name, knn_params, train_sample_size = _build_run_config(
                cfg,
                experiment_name=exp["name"],
                cli_overrides=cli_overrides or None,
                cli_train_sample_size=args.train_sample_size,
            )
            if args.run_name:
                run_name = f"{args.run_name}_{exp['name']}"
            train_single_run(cfg, run_name, knn_params, train_sample_size, balance_strategy, args.sampling_ratio)
    else:
        run_name, knn_params, train_sample_size = _build_run_config(
            cfg,
            experiment_name=args.experiment,
            cli_overrides=cli_overrides or None,
            cli_train_sample_size=args.train_sample_size,
        )
        if args.run_name:
            run_name = args.run_name
        train_single_run(cfg, run_name, knn_params, train_sample_size, balance_strategy, args.sampling_ratio)

    print(f"\nExperimentos MLflow em: {ROOT / cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
