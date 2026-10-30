"""Optional MLflow logging to a local SQLite store. View with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from src.config import ROOT

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def log_run(experiment: str, run_name: str, params: dict, metrics: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
