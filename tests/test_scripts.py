"""Tests for training and champion-selection scripts."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest


def test_select_champion_function(tmp_path):
    """Call select_champion with mocked MLflow responses."""
    from scripts.select_champion import select_champion

    mock_experiment = Mock()
    mock_experiment.experiment_id = "test_exp_id"

    mock_runs = pd.DataFrame({
        "run_id": ["run1", "run2", "run3"],
        "metrics.pr_auc": [0.75, 0.85, 0.80],
    })

    with patch("mlflow.get_experiment_by_name") as mock_get_exp, \
         patch("mlflow.search_runs") as mock_search_runs, \
         patch("mlflow.sklearn.load_model"), \
         patch("mlflow.artifacts.download_artifacts"):

        mock_get_exp.return_value = mock_experiment
        mock_search_runs.return_value = mock_runs

        output_dir = tmp_path / "champion"
        output_dir.mkdir()

        select_champion(
            experiment_name="fraud_detection",
            output_dir=output_dir,
            tracking_uri="http://localhost:5000",
        )

        mock_get_exp.assert_called_once_with("fraud_detection")
        mock_search_runs.assert_called_once_with(experiment_ids=["test_exp_id"])


def test_select_champion_no_experiment():
    """Raise ValueError when the experiment does not exist."""
    from scripts.select_champion import select_champion

    with patch("mlflow.get_experiment_by_name") as mock_get_exp:
        mock_get_exp.return_value = None

        with pytest.raises(ValueError, match="Experiment 'fraud_detection' not found"):
            select_champion(
                experiment_name="fraud_detection",
                output_dir=Path("/tmp"),
                tracking_uri="http://localhost:5000",
            )


def test_select_champion_no_pr_auc_runs():
    """Raise ValueError when no run has pr_auc."""
    from scripts.select_champion import select_champion

    mock_experiment = Mock()
    mock_experiment.experiment_id = "test_exp_id"

    mock_runs = pd.DataFrame({
        "run_id": ["run1", "run2"],
        "metrics.accuracy": [0.9, 0.85],
    })

    with patch("mlflow.get_experiment_by_name") as mock_get_exp, \
         patch("mlflow.search_runs") as mock_search_runs:

        mock_get_exp.return_value = mock_experiment
        mock_search_runs.return_value = mock_runs

        with pytest.raises(ValueError, match="No run with pr_auc metric found"):
            select_champion(
                experiment_name="fraud_detection",
                output_dir=Path("/tmp"),
                tracking_uri="http://localhost:5000",
            )
