"""Start an interactive loopback-only API after running run_demo.py."""
import os
from pathlib import Path
import secrets
import socket

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'artifacts/manifest.json').exists():
    raise SystemExit('First run: .venv/bin/python run_demo.py')
os.environ.pop('LAB_ARTIFACT_DIR', None)
key = secrets.token_urlsafe(24)
os.environ['LAB_API_KEY'] = key
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
print(f'Interactive API: http://127.0.0.1:{port}/docs', flush=True)
print(f'Local temporary demo key (Authorize > APIKeyHeader): {key}', flush=True)
print('Copy evidence/example_request.json into POST /predict. Ctrl+C stops the service.', flush=True)
import uvicorn
uvicorn.run('learning_service.api:create_app', factory=True, host='127.0.0.1', port=port)
