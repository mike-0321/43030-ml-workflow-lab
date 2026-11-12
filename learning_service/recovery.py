"""Run a real loopback integrity failure and recovery drill on an isolated copy."""

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from learning_service.common import ARTIFACTS, EVIDENCE, ROOT, sha256, write_json


def main() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    payload = json.loads((EVIDENCE / "example_request.json").read_text())
    key = secrets.token_urlsafe(24)
    with tempfile.TemporaryDirectory(prefix="recovery-", dir=ROOT) as scratch:
        folder = Path(scratch)
        shutil.copyfile(ARTIFACTS / "manifest.json", folder / "manifest.json")
        model_path = folder / manifest["model_file"]
        shutil.copyfile(ARTIFACTS / manifest["model_file"], model_path)
        original = model_path.read_bytes()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, LAB_API_KEY=key, LAB_ARTIFACT_DIR=str(folder))
        with (EVIDENCE / "recovery_server.log").open("w") as log:
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
                rows = []
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
                    for _ in range(100):
                        if process.poll() is not None:
                            raise RuntimeError("Recovery server exited before readiness")
                        try:
                            if client.get("/ready").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("Recovery server did not become ready")

                    def observe(stage: str) -> None:
                        health = client.get("/health")
                        ready = client.get("/ready")
                        prediction = client.post(
                            "/predict", json=payload, headers={"X-API-Key": key}
                        )
                        rows.append(
                            {
                                "stage": stage,
                                "health_status": health.status_code,
                                "readiness_status": ready.status_code,
                                "prediction_status": prediction.status_code,
                                "artifact_sha256": sha256(model_path),
                                "prediction_body": prediction.json(),
                            }
                        )

                    observe("healthy")
                    model_path.write_bytes(original + b"controlled-corruption")
                    observe("corrupted_copy")
                    model_path.write_bytes(original)
                    observe("trusted_copy_restored")
                assert [r["readiness_status"] for r in rows] == [200, 503, 200]
                assert [r["prediction_status"] for r in rows] == [200, 503, 200]
                assert [r["health_status"] for r in rows] == [200, 200, 200]
                assert rows[0]["artifact_sha256"] == rows[2]["artifact_sha256"]
                write_json(
                    EVIDENCE / "recovery_drill.json",
                    {
                        "scope": "Actual local Uvicorn HTTP drill on a temporary artifact copy",
                        "model_version": manifest["model_version"],
                        "source_run_id": manifest["mlflow_run_id"],
                        "stages": rows,
                        "passed": True,
                        "limitations": "Known trusted backup restored; no cloud, concurrency, signing, attacker simulation or recovery-time SLA validated",
                    },
                )
                print(json.dumps(rows, indent=2))
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
