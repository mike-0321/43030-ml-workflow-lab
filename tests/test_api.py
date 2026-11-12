import json
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from learning_service.api import create_app
from learning_service.common import ARTIFACTS, EVIDENCE, sha256

KEY = "non-secret-test-fixture-key"


@pytest.fixture
def app():
    return create_app(api_key=KEY)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def payload():
    return json.loads((EVIDENCE / "example_request.json").read_text())


def test_health_matches_manifest(client):
    response = client.get("/health")
    assert response.status_code == 200
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    assert response.json()["model_sha256"] == manifest["model_sha256"]
    assert response.json()["mlflow_run_id"] == manifest["mlflow_run_id"]
    assert sha256(ARTIFACTS / manifest["model_file"]) == manifest["model_sha256"]


@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_missing_or_wrong_key_never_predicts(app, payload, key):
    spy = Mock(wraps=app.state.engine.predict)
    app.state.engine.predict = spy
    headers = {} if key is None else {"X-API-Key": key}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 401
    spy.assert_not_called()


def test_valid_prediction_agrees_with_direct_pipeline(app, client, payload):
    response = client.post("/predict", json=payload, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    result = response.json()
    direct = int(app.state.engine.pipeline.predict(np.asarray([payload["pixels"]]))[0])
    assert result["prediction"] == direct
    assert 0 <= result["confidence"] <= 1
    assert result["model_sha256"] == app.state.engine.manifest["model_sha256"]


def test_repeated_prediction_is_stable(client, payload):
    results = [
        client.post("/predict", json=payload, headers={"X-API-Key": KEY}).json() for _ in range(3)
    ]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize(
    "pixels",
    [
        [0.0] * 63,
        [0.0] * 65,
        [-1.0] + [0.0] * 63,
        [17.0] + [0.0] * 63,
        ["1"] + [0.0] * 63,
        [True] + [0.0] * 63,
        [float("nan")] + [0.0] * 63,
        [float("inf")] + [0.0] * 63,
    ],
)
def test_invalid_contract_never_predicts(app, pixels):
    spy = Mock(wraps=app.state.engine.predict)
    app.state.engine.predict = spy
    content = json.dumps({"pixels": pixels})
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=content,
            headers={"X-API-Key": KEY, "Content-Type": "application/json"},
        )
    assert response.status_code == 422
    spy.assert_not_called()


def test_unknown_field_rejected(client, payload):
    response = client.post(
        "/predict", json={**payload, "unused": "unexpected"}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 422


def test_malformed_json_rejected(client):
    response = client.post(
        "/predict",
        content="{bad json",
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_body_limit_never_predicts(app):
    spy = Mock(wraps=app.state.engine.predict)
    app.state.engine.predict = spy
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=b"x" * 4097,
            headers={"X-API-Key": KEY, "Content-Type": "application/json"},
        )
    assert response.status_code == 413
    spy.assert_not_called()


def test_scaler_fitted_to_training_subset_only(app):
    data = load_digits()
    indices = np.load(ARTIFACTS / "split_indices.npz")
    expected = data.data[indices["train"]].mean(axis=0)
    np.testing.assert_allclose(app.state.engine.pipeline.named_steps["scale"].mean_, expected)


def test_tampered_model_rejected_before_deserialisation(tmp_path, monkeypatch):
    import learning_service.api as api_module

    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    source = ARTIFACTS / manifest["model_file"]
    (tmp_path / manifest["model_file"]).write_bytes(source.read_bytes() + b"tampered")
    monkeypatch.setattr(api_module, "ARTIFACTS", tmp_path)
    load_spy = Mock()
    monkeypatch.setattr(api_module.joblib, "load", load_spy)
    with pytest.raises(RuntimeError, match="checksum"):
        api_module.Engine()
    load_spy.assert_not_called()


def test_validation_error_does_not_echo_raw_pixels(client):
    response = client.post(
        "/predict", json={"pixels": [17.123456] + [0.0] * 63}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 422
    assert "17.123456" not in response.text
    for error in response.json()["detail"]:
        assert set(error) == {"location", "message", "type"}


@pytest.fixture
def isolated_artifacts(tmp_path):
    import shutil

    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    shutil.copyfile(ARTIFACTS / "manifest.json", tmp_path / "manifest.json")
    shutil.copyfile(ARTIFACTS / manifest["model_file"], tmp_path / manifest["model_file"])
    return tmp_path, manifest


def test_readiness_is_distinct_from_liveness(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"


def test_tamper_blocks_prediction_then_trusted_restore_recovers(isolated_artifacts, payload):
    folder, manifest = isolated_artifacts
    model_path = folder / manifest["model_file"]
    original = model_path.read_bytes()
    app = create_app(api_key=KEY, artifact_dir=folder)
    predictor_spy = Mock(wraps=app.state.engine.pipeline.predict_proba)
    app.state.engine.pipeline.predict_proba = predictor_spy
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        model_path.write_bytes(original + b"controlled-corruption")
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=payload, headers={"X-API-Key": KEY}).status_code == 503
        predictor_spy.assert_not_called()
        model_path.write_bytes(original)
        assert client.get("/ready").status_code == 200
        assert client.post("/predict", json=payload, headers={"X-API-Key": KEY}).status_code == 200
        assert predictor_spy.call_count == 1


def test_missing_artifact_returns_not_ready(isolated_artifacts):
    folder, manifest = isolated_artifacts
    app = create_app(api_key=KEY, artifact_dir=folder)
    (folder / manifest["model_file"]).unlink()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
