"""Run local quality checks; this is not a hosted CI execution."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "evidence").mkdir(parents=True, exist_ok=True)
phase = sys.argv[1] if len(sys.argv) > 1 else "final"
records = []
commands = {
    "ruff_check": ["ruff", "check", "learning_service", "tests", "scripts"],
    "ruff_format": ["ruff", "format", "--check", "learning_service", "tests", "scripts"],
    "mypy": ["mypy"],
}
for name, args in commands.items():
    result = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    log = ROOT / "evidence" / f"quality_{phase}_{name}.log"
    log.write_text(result.stdout + result.stderr)
    records.append(
        {
            "check": name,
            "command": ["python", "-m", *args],
            "exit_code": result.returncode,
            "log": str(log.relative_to(ROOT)),
        }
    )
    print(name, result.returncode)
    print(result.stdout + result.stderr)
(ROOT / "evidence" / f"quality_{phase}.json").write_text(json.dumps(records, indent=2) + "\n")
raise SystemExit(1 if any(row["exit_code"] for row in records) else 0)
