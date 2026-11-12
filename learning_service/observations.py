"""Generate real local response timing, shift and schema-comparison records."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from learning_service.api import Engine, PredictionRequest
from learning_service.common import ARTIFACTS, EVIDENCE, ROOT, write_json


def benchmark():
    key = secrets.token_urlsafe(24)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, LAB_API_KEY=key)
    log = (EVIDENCE / "server.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "learning_service.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Uvicorn stopped; inspect evidence/server.log")
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("Uvicorn did not become healthy")
            payload = json.loads((EVIDENCE / "example_request.json").read_text())
            for _ in range(10):
                client.post("/predict", json=payload, headers={"X-API-Key": key}).raise_for_status()
            durations, statuses = [], []
            for _ in range(200):
                start = time.perf_counter_ns()
                response = client.post("/predict", json=payload, headers={"X-API-Key": key})
                durations.append((time.perf_counter_ns() - start) / 1_000_000)
                statuses.append(response.status_code)
            if any(s != 200 for s in statuses):
                raise RuntimeError("Benchmark included failed requests")
            record = {
                "scope": "One local Uvicorn worker on loopback; sequential HTTP requests",
                "warmup_requests": 10,
                "measured_requests": 200,
                "concurrency": 1,
                "http_200": statuses.count(200),
                "errors": len(statuses) - statuses.count(200),
                "median_ms": float(np.median(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
                "min_ms": float(min(durations)),
                "max_ms": float(max(durations)),
                "durations_ms": durations,
                "limitations": "Local demonstration only; no throughput, multi-user, cloud, SLA or production claim",
            }
            write_json(EVIDENCE / "benchmark.json", record)
            write_json(EVIDENCE / "example_response.json", response.json())
    finally:
        process.terminate()
        process.wait(timeout=10)
        log.close()
    return record


def input_shift():
    data = load_digits()
    split = np.load(ARTIFACTS / "split_indices.npz")
    training = data.data[split["train"]]
    held_out = data.data[split["test"]]
    # Controlled stress condition, not a claim about naturally occurring drift.
    shifted = np.clip(held_out + 6.0, 0, 16)
    reference_mean = training.mean(axis=0)
    reference_std = training.std(axis=0)
    informative = reference_std > 1e-8

    def score(batch):
        return float(
            np.abs(
                (batch.mean(axis=0)[informative] - reference_mean[informative])
                / reference_std[informative]
            ).mean()
        )

    engine = Engine()
    threshold = 0.35  # A stated learning heuristic, not a statistically calibrated detector.
    original_score, shifted_score = score(held_out), score(shifted)
    record = {
        "reference": "Training subset feature means and standard deviations",
        "score": "Mean absolute standardised difference in feature means; zero-variance features excluded",
        "threshold": threshold,
        "threshold_status": "Uncalibrated teaching heuristic",
        "batch_samples": int(len(held_out)),
        "held_out_score": original_score,
        "held_out_flag": original_score > threshold,
        "controlled_shift": "Add 6 intensity units to each held-out pixel, clip to [0,16]",
        "shifted_score": shifted_score,
        "shifted_flag": shifted_score > threshold,
        "original_accuracy": float(
            accuracy_score(data.target[split["test"]], engine.pipeline.predict(held_out))
        ),
        "shifted_accuracy": float(
            accuracy_score(data.target[split["test"]], engine.pipeline.predict(shifted))
        ),
        "limitations": "Constructed perturbation with labels retained; not proof of real-world or concept drift detection",
    }
    write_json(EVIDENCE / "input_shift.json", record)
    return record


def contract_comparison():
    class PermissiveDraft(BaseModel):
        pixels: list[float]

    cases = {
        "valid_64": [0.0] * 64,
        "63_pixels": [0.0] * 63,
        "65_pixels": [0.0] * 65,
        "out_of_range": [17.0] + [0.0] * 63,
        "numeric_string": ["1"] + [0.0] * 63,
        "non_finite": [float("nan")] + [0.0] * 63,
    }
    rows = []
    for name, pixels in cases.items():
        outcomes = {}
        for label, schema in [
            ("permissive_draft", PermissiveDraft),
            ("validated_contract", PredictionRequest),
        ]:
            try:
                schema(pixels=pixels)
                outcomes[label] = "accepted"
            except ValidationError:
                outcomes[label] = "rejected"
        rows.append({"case": name, **outcomes})
    record = {
        "scope": "Deliberate local comparison of two schemas, not a claim about an earlier deployed service",
        "cases": rows,
    }
    write_json(EVIDENCE / "contract_comparison.json", record)
    return record


if __name__ == "__main__":
    b, s, c = benchmark(), input_shift(), contract_comparison()
    print(
        f"Local HTTP: {b['http_200']}/{b['measured_requests']} HTTP 200; median {b['median_ms']:.3f} ms; p95 {b['p95_ms']:.3f} ms"
    )
    print(
        f"Input score: held out {s['held_out_score']:.6f}; controlled shift {s['shifted_score']:.6f}"
    )
    print(f"Controlled shifted accuracy: {s['shifted_accuracy']:.6f}")
    print(f"Schema comparison cases: {len(c['cases'])}")
