"""Train a small digits Pipeline and record a traceable, local MLflow run.

Concept sources: scikit-learn digits/classification and common-pitfalls guides;
MLflow Tracking documentation. This is an original small teaching implementation.
"""

import hashlib
import importlib.metadata
import platform
import subprocess

import joblib
import mlflow
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learning_service.common import ARTIFACTS, EVIDENCE, ROOT, sha256, write_json


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    data = load_digits()
    indices = np.arange(len(data.target))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=42, stratify=data.target
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        ]
    )
    model.fit(data.data[train_idx], data.target[train_idx])
    predicted = model.predict(data.data[test_idx])
    metrics = {
        "scope": "Local supplementary digits exercise; not ultrasound results",
        "dataset": "sklearn.datasets.load_digits",
        "samples_total": int(len(indices)),
        "samples_train": int(len(train_idx)),
        "samples_test": int(len(test_idx)),
        "split_seed": 42,
        "test_fraction": 0.25,
        "accuracy": float(accuracy_score(data.target[test_idx], predicted)),
        "macro_f1": float(f1_score(data.target[test_idx], predicted, average="macro")),
        "confusion_matrix": confusion_matrix(data.target[test_idx], predicted).tolist(),
        "preprocessing_fit_scope": "Training subset only, inside sklearn Pipeline",
    }
    model_path = ARTIFACTS / "digits_pipeline.joblib"
    joblib.dump(model, model_path)
    np.savez(ARTIFACTS / "split_indices.npz", train=train_idx, test=test_idx)
    write_json(EVIDENCE / "metrics.json", metrics)
    write_json(EVIDENCE / "example_request.json", {"pixels": data.data[test_idx[0]].tolist()})
    code_hashes = {
        str(p.relative_to(ROOT)): sha256(p)
        for folder in [ROOT / "learning_service", ROOT / "tests", ROOT / "scripts"]
        for p in sorted(folder.rglob("*.py"))
    }
    for relative in ["pyproject.toml", ".github/workflows/tests.yml"]:
        code_hashes[relative] = sha256(ROOT / relative)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    source_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "learning_service",
            "tests",
            "scripts",
            "pyproject.toml",
            "requirements.lock",
            ".github",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "fastapi",
            "starlette",
            "pydantic",
            "httpx",
            "pytest",
            "numpy",
            "scikit-learn",
            "scipy",
            "joblib",
            "mlflow-skinny",
            "uvicorn",
            "sqlalchemy",
            "alembic",
            "pip-audit",
            "pip",
            "ruff",
            "mypy",
        ]
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
    write_json(EVIDENCE / "environment.json", environment)
    mlflow.set_tracking_uri("sqlite:///" + str(ARTIFACTS / "mlflow.db"))
    mlflow.set_experiment("A3-supplementary-local-digits")
    with mlflow.start_run(run_name="standardised-logistic-regression") as run:
        manifest = {
            "model_version": "digits-logreg-local-v1",
            "scope": "Supplementary local digits exercise",
            "model_file": model_path.name,
            "model_sha256": sha256(model_path),
            "dataset_source": {
                "original_creators": "E. Alpaydin and C. Kaynak (1998)",
                "doi": "https://doi.org/10.24432/C50P49",
                "license": "CC BY 4.0, as stated by UCI",
                "subset": "scikit-learn copy of original UCI test subset",
                "evaluation_boundary": "New stratified split within that subset; not original UCI protocol or writer-independent evaluation",
            },
            "dataset_sha256": hashlib.sha256(
                data.data.tobytes() + data.target.tobytes()
            ).hexdigest(),
            "split_indices_sha256": sha256(ARTIFACTS / "split_indices.npz"),
            "split_seed": 42,
            "preprocessing": "StandardScaler fitted on training subset only",
            "input_contract": {
                "features": 64,
                "image_shape": [8, 8],
                "min": 0,
                "max": 16,
                "finite_numeric_only": True,
            },
            "source_sha256": code_hashes,
            "source_git_commit": source_commit,
            "source_worktree_clean": not bool(source_status),
            "requirements_lock_sha256": sha256(ROOT / "requirements.lock"),
            "environment": environment,
            "mlflow_run_id": run.info.run_id,
            "tracking_backend": "Local SQLite store; not remote MLflow registry or cloud",
        }
        write_json(ARTIFACTS / "manifest.json", manifest)
        mlflow.log_params(
            {
                "dataset": "load_digits",
                "split_seed": 42,
                "test_fraction": 0.25,
                "C": 1.0,
                "max_iter": 2000,
                "preprocessing": "StandardScaler",
            }
        )
        mlflow.log_metrics({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        for path in [model_path, ARTIFACTS / "manifest.json", EVIDENCE / "metrics.json"]:
            mlflow.log_artifact(str(path))
        write_json(
            EVIDENCE / "mlflow_run.json",
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "tracking_backend": "Local SQLite store",
                "logged_parameters": {
                    "dataset": "load_digits",
                    "split_seed": 42,
                    "test_fraction": 0.25,
                    "C": 1.0,
                    "max_iter": 2000,
                },
                "logged_metrics": {
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                },
                "logged_artifacts": ["digits_pipeline.joblib", "manifest.json", "metrics.json"],
            },
        )
    print(f"Accuracy: {metrics['accuracy']:.6f}; macro-F1: {metrics['macro_f1']:.6f}")
    print(f"MLflow run: {manifest['mlflow_run_id']}")
    print(f"Model SHA-256: {manifest['model_sha256']}")


if __name__ == "__main__":
    main()
