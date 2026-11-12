"""Execute the complete local demonstration and retain this run's real outputs."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'
ENV = dict(os.environ, PYTHONUNBUFFERED='1', NO_PROXY='127.0.0.1,localhost,::1',
           no_proxy='127.0.0.1,localhost,::1')
for name in ['LAB_ARTIFACT_DIR', 'MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID',
             'MLFLOW_EXPERIMENT_NAME', 'MLFLOW_TRACKING_URI']:
    ENV.pop(name, None)
STEPS: list[dict] = []
RUN_LOG = None


def emit(text: str = '') -> None:
    print(text, flush=True)
    if RUN_LOG:
        RUN_LOG.write(text + '\n')
        RUN_LOG.flush()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def stage(number: int, title: str, arguments: list[str], allowed: tuple[int, ...] = (0,)) -> int:
    emit(f'\n[{number}/7] {title}\n$ python ' + ' '.join(arguments))
    started = time.perf_counter()
    log_path = EVIDENCE / f'live_{number:02d}.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, env=ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                emit(line.rstrip('\n'))
                log.write(line)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    STEPS.append({'step': number, 'title': title, 'command': [sys.executable, *arguments],
                  'exit_code': code, 'seconds': round(time.perf_counter() - started, 3),
                  'log': str(log_path.relative_to(ROOT))})
    save(EVIDENCE / 'live_steps.json', STEPS)
    if code not in allowed:
        raise RuntimeError(f'Step {number} returned {code}; see {log_path}')
    return code


def demonstrate_http() -> None:
    import httpx

    emit('\n[4/7] Real local HTTP requests: authentication and input controls')
    started = time.perf_counter()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(24)
    env = dict(ENV, LAB_API_KEY=key)
    records = []
    with (EVIDENCE / 'live_api_server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-m', 'uvicorn',
                                   'learning_service.api:create_app', '--factory',
                                   '--host', '127.0.0.1', '--port', str(port), '--no-access-log'],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5, trust_env=False) as client:
                for _ in range(200):
                    if server.poll() is not None:
                        raise RuntimeError('HTTP server stopped; inspect live_api_server.log')
                    try:
                        if client.get('/ready').status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.05)
                else:
                    raise RuntimeError('HTTP server did not become ready')
                payload = json.loads((EVIDENCE / 'example_request.json').read_text())
                cases = [
                    ('health', 200, lambda: client.get('/health')),
                    ('readiness', 200, lambda: client.get('/ready')),
                    ('missing_key', 401, lambda: client.post('/predict', json=payload)),
                    ('wrong_key', 401, lambda: client.post('/predict', json=payload,
                                                         headers={'X-API-Key': 'wrong-demo-key'})),
                    ('valid_prediction', 200, lambda: client.post('/predict', json=payload,
                                                               headers={'X-API-Key': key})),
                    ('invalid_63_pixels', 422, lambda: client.post('/predict',
                        json={'pixels': [0.0] * 63}, headers={'X-API-Key': key})),
                    ('body_over_4096_bytes', 413, lambda: client.post('/predict',
                        content=b'x' * 4097, headers={'X-API-Key': key})),
                ]
                for name, expected, request in cases:
                    response = request()
                    row = {'case': name, 'expected_status': expected,
                           'actual_status': response.status_code, 'body': response.json()}
                    records.append(row)
                    save(EVIDENCE / 'live_http_requests.json', records)
                    emit(f'{name}: expected {expected}, received {response.status_code}')
                    if name == 'valid_prediction':
                        emit(json.dumps(response.json(), indent=2))
                    if response.status_code != expected:
                        raise RuntimeError(f'Unexpected result: {name}')
        finally:
            server.terminate()
            server.wait(timeout=10)
    STEPS.append({'step': 4, 'title': 'Live HTTP control demonstration', 'exit_code': 0,
                  'seconds': round(time.perf_counter() - started, 3),
                  'record': 'evidence/live_http_requests.json'})
    save(EVIDENCE / 'live_steps.json', STEPS)


def read(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def validate_audit(exit_code: int) -> dict:
    path = EVIDENCE / 'live_dependency_audit.json'
    if not path.exists():
        raise RuntimeError('Dependency audit produced no new JSON result; inspect live_07.log.')
    audit = read(path.name)
    dependencies = audit.get('dependencies')
    if not isinstance(dependencies, list) or not dependencies:
        raise RuntimeError('Dependency audit did not return a non-empty dependency list.')
    if not all(isinstance(d, dict) and isinstance(d.get('vulns'), list) for d in dependencies):
        raise RuntimeError('Dependency audit returned an unexpected result structure.')
    findings = sum(len(d['vulns']) for d in dependencies)
    if exit_code not in (0, 1) or (exit_code == 0) != (findings == 0):
        raise RuntimeError('Dependency audit exit status and returned findings disagree.')
    return audit


def record_status(run_id: str, status: str, detail: str) -> None:
    save(EVIDENCE / 'LIVE_RUN_SUMMARY.json',
         {'run_id': run_id, 'status': status, 'detail': detail, 'steps': STEPS})
    (EVIDENCE / 'LIVE_RUN_SUMMARY.md').write_text(
        f'# Live run: {status}\n\nRun: `{run_id}`\n\n{detail}\n\n'
        'See [this run log](live_run.log) and [completed steps](live_steps.json).\n'
        'Earlier completed runs remain in `runs/`; they are not results for this run.\n')


def summarise(run_id: str, audit_exit: int) -> None:
    metrics, benchmark, recovery = read('metrics.json'), read('benchmark.json'), read('recovery_drill.json')
    shift, audit = read('input_shift.json'), read('live_dependency_audit.json')
    manifest = json.loads((ROOT / 'artifacts/manifest.json').read_text())
    xml = ET.parse(EVIDENCE / 'pytest.xml').getroot()
    suites = list(xml.iter('testsuite'))
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ['tests', 'failures', 'errors', 'skipped']}
    dependencies = audit.get('dependencies', [])
    vulnerabilities = sum(len(d.get('vulns', [])) for d in dependencies)
    summary = {'run_id': run_id, 'status': 'completed',
               'python': sys.version, 'interpreter': sys.executable, 'steps': STEPS,
               'source_commit': manifest['source_git_commit'],
               'source_worktree_clean': manifest['source_worktree_clean'],
               'metrics': metrics, 'pytest': counts, 'http_requests': read('live_http_requests.json'),
               'recovery': recovery, 'benchmark': benchmark, 'input_comparison': shift,
               'audit': {'exit_code': audit_exit, 'dependencies_returned': len(dependencies),
                         'known_findings': vulnerabilities},
               'mlflow_run_id': manifest['mlflow_run_id'], 'model_sha256': manifest['model_sha256']}
    save(EVIDENCE / 'LIVE_RUN_SUMMARY.json', summary)
    lines = [
        '# Live run results', '', f'Run: `{run_id}`', '',
        'These results were generated by executing this separate demonstration project.',
        'These results describe this execution.', '',
        '| Check | Result |', '|---|---|',
        f"| Python | {sys.version.split()[0]} |",
        f"| Training / test samples | {metrics['samples_train']} / {metrics['samples_test']} |",
        f"| Accuracy | {metrics['accuracy']:.4%} |",
        f"| Macro-F1 | {metrics['macro_f1']:.6f} |",
        f"| pytest | {counts['tests']} tests, {counts['failures']} failures, {counts['errors']} errors |",
        '| Quality | Ruff lint + formatting + configured mypy passed |',
        f"| Actual HTTP cases | {len(summary['http_requests'])}, all expected statuses |",
        '| Model recovery | Readiness/prediction: 200 → 503 → 200; health: 200 throughout |',
        f"| HTTP benchmark | {benchmark['http_200']}/{benchmark['measured_requests']} HTTP 200 |",
        f"| Local median / p95 | {benchmark['median_ms']:.3f} / {benchmark['p95_ms']:.3f} ms |",
        f"| Controlled +6 pixel accuracy | {shift['shifted_accuracy']:.4%} |",
        f'| Dependency lookup | {len(dependencies)} packages; {vulnerabilities} known findings |', '',
        f"MLflow run: `{manifest['mlflow_run_id']}`", '',
        f"Source commit: `{manifest['source_git_commit']}`", '',
        f"Recorded source clean: `{manifest['source_worktree_clean']}`", '',
        '## Open the evidence', '',
        '- [All streamed execution output](live_run.log)',
        '- [Test report](pytest.xml)', '- [Live HTTP request results](live_http_requests.json)',
        '- [Recovery stages](recovery_drill.json)', '- [Per-request timings](benchmark.json)',
        '- [Dependency lookup](live_dependency_audit.json)',
        '- [Model and source manifest](../artifacts/manifest.json)', '',
        '## Scope', '',
        'Loopback, one worker, sequential repeated requests; latency varies between runs.',
        'The pixel perturbation is constructed, not validated natural drift detection.',
        'mypy checks API/common only. Dependency lookup covers known advisories only.',
        'The two course quiz scores and ethical case remain in the separate submitted portfolio.',
        'Docker, remote CI, cloud deployment and independent human review were not executed here.',
    ]
    (EVIDENCE / 'LIVE_RUN_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    emit('\n' + '\n'.join(lines[:27]))


def main() -> int:
    global RUN_LOG
    EVIDENCE.mkdir(exist_ok=True)
    lock = EVIDENCE / '.running'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print('Another run may be active. Inspect evidence/.running before trying again.')
        return 2
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    run_id = datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S_%f')
    RUN_LOG = (EVIDENCE / 'live_run.log').open('w')
    try:
        record_status(run_id, 'running', 'The seven-step demonstration is currently running.')
        save(EVIDENCE / 'live_steps.json', STEPS)
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('Use the configured Python 3.12 environment for the pinned dependencies.')
        for package in ['fastapi', 'httpx', 'pytest', 'scikit-learn', 'mlflow-skinny', 'pip-audit', 'ruff', 'mypy']:
            importlib.metadata.version(package)
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip()
        if Path(repo_root).resolve() != ROOT:
            raise RuntimeError('This demo needs its own Git snapshot. Follow README setup instructions.')
        emit(f'COMPLETE LOCAL DEMONSTRATION\nRun (Sydney): {run_id}\nProject: {ROOT}\nPython: {sys.executable}')
        emit('Fresh outputs are generated here; the original submission evidence is unchanged.')
        stage(1, 'Code quality: Ruff and configured mypy', ['scripts/quality.py', 'final'])
        stage(2, 'Train model and record MLflow provenance', ['-m', 'learning_service.train'])
        stage(3, 'Run regression tests', ['-m', 'pytest', '-v', '--junitxml=evidence/pytest.xml'])
        demonstrate_http()
        stage(5, 'Corrupt a temporary copy, refuse predictions, restore trusted copy',
              ['-m', 'learning_service.recovery'])
        stage(6, 'Measure HTTP latency, controlled input change and contract comparison',
              ['-m', 'learning_service.observations'])
        (EVIDENCE / 'live_dependency_audit.json').unlink(missing_ok=True)
        audit_exit = stage(7, 'Look up known dependency advisories', ['-m', 'pip_audit', '--local',
            '--format', 'json', '--progress-spinner', 'off', '--timeout', '15',
            '--output', 'evidence/live_dependency_audit.json'], allowed=(0, 1))
        validate_audit(audit_exit)
        summarise(run_id, audit_exit)
        emit('\nCOMPLETE — Open evidence/LIVE_RUN_SUMMARY.md and evidence/live_run.log in VS Code.')
        emit('To run again: .venv/bin/python run_demo.py')
        return 0
    except KeyboardInterrupt:
        emit('Run cancelled by the user.')
        record_status(run_id, 'cancelled', 'The run was interrupted; it is not a completed result.')
        return 130
    except Exception as error:
        emit(traceback.format_exc())
        record_status(run_id, 'failed', str(error))
        return 1
    finally:
        RUN_LOG.close()
        RUN_LOG = None
        snapshot = ROOT / 'runs' / run_id
        shutil.copytree(EVIDENCE, snapshot / 'evidence', ignore=shutil.ignore_patterns('.running'))
        if (ROOT / 'artifacts/manifest.json').exists():
            (snapshot / 'artifacts').mkdir()
            for name in ['manifest.json', 'digits_pipeline.joblib', 'split_indices.npz']:
                p = ROOT / 'artifacts' / name
                if p.exists():
                    shutil.copy2(p, snapshot / 'artifacts' / name)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
