"""Redraw this Goal from fixed records. No training, tests or network calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
mod = importlib.import_module('plot_goals_3_4')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GOAL = 4
ORIGIN = 'Saved test pixels and actual predictions, evaluation, raw local HTTP durations and temporary-copy recovery observations from the recorded Live_Demo.'
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
         "axes.labelsize": 10, "text.color": mod.INK, "axes.labelcolor": mod.INK,
         "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
         "ps.fonttype": 42, "figure.facecolor": "white"}

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def completed_summary(data):
    summary = read(data / "LIVE_RUN_SUMMARY.json")
    if summary.get("status") != "completed":
        raise ValueError("A completed saved Live_Demo record is required")
    return summary

def draw(data, out):
    summary = completed_summary(data)
    metrics, benchmark = read(data / "metrics.json"), read(data / "benchmark.json")
    shift, recovery = read(data / "input_shift.json"), read(data / "recovery_drill.json")
    manifest, exported = read(data / "manifest.json"), read(data / "digits_export_info.json")
    if summary["metrics"] != metrics or summary["benchmark"] != benchmark:
        raise ValueError("Summary disagrees with individual measurements")
    if summary["recovery"] != recovery or summary["input_comparison"] != shift:
        raise ValueError("Summary disagrees with recovery / controlled-input records")
    if not (summary["mlflow_run_id"] == manifest["mlflow_run_id"] == recovery["source_run_id"]):
        raise ValueError("Inputs do not belong to the same recorded model run")
    if summary["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("Model identity differs between summary and manifest")
    if exported["csv_sha256"] != digest(data / "digits_test.csv"):
        raise ValueError("CSV checksum differs from its actual export record")
    for key in ["mlflow_run_id", "model_sha256", "source_git_commit"]:
        if exported[key] != manifest[key]:
            raise ValueError("Export identity differs from manifest: " + key)
    return {"goal4_samples_model": mod._samples_model(data, out, summary, metrics, shift, manifest),
            "goal4_latency_recovery": mod._latency_recovery(data, out, summary, benchmark, recovery)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs/figures")
    args = parser.parse_args()
    data, out = HERE / "data", args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        values = draw(data, out)
    value_path = out / f"figure_values_goal{GOAL}.json"
    value_path.write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    sources = {str(p.relative_to(HERE)): {"sha256": digest(p), "bytes": p.stat().st_size}
               for p in sorted(data.iterdir()) if p.is_file()}
    code = {p.name: digest(p) for p in [Path(__file__), HERE / (mod.__name__ + ".py"),
                                       HERE / "requirements.txt"]}
    generated = [p for p in sorted(out.iterdir())
                 if p.is_file() and (p.suffix in [".png", ".pdf"] or p == value_path)]
    manifest = {"goal": GOAL, "operation": "Redraw fixed recorded evidence only",
                "origin": ORIGIN, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["matplotlib", "numpy"]},
                "input_files": sources, "code_sha256": code,
                "outputs": [{"file": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                            for p in generated],
                "limitations": "No new model training, prediction, test execution, measurement, consultation or course completion is performed."}
    (HERE / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Goal {GOAL}: redrew fixed records successfully")
    for p in generated:
        print(p)
    print(HERE / "reproduction_manifest.json")

if __name__ == "__main__":
    main()
