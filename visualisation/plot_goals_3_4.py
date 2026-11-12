"""Plot recorded Goal 3/4 evidence without training, inference or network access.

The input CSV must contain actual saved test predictions, not example images.
All findings, outcomes and measurements are parsed from the supplied records.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np


BLUE = "#3D617C"
TEAL = "#248278"
ORANGE = "#BD6A31"
INK = "#202C35"
GREY = "#62717D"
PALE = "#E7EDF1"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_records(data_dir: Path, names: list[str]) -> list[dict]:
    return [
        {"file": name, "sha256": sha256((data_dir / name).read_bytes()).hexdigest()}
        for name in names
    ]


def _one_match(pattern: str, text: str, name: str) -> re.Match:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} summary, found {len(matches)}")
    return matches[0]


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected a nonnegative integer for {name}: {value!r}")
    return value


def _fraction(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected a fraction in [0,1] for {name}")
    return value


def _save(fig, out_dir: Path, stem: str) -> dict:
    fig.savefig(out_dir / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    return {"png": f"{stem}.png", "pdf": f"{stem}.pdf"}


def _clean_axis(ax, *, y_grid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#B9C5CD")
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", color=PALE, linewidth=0.7)


def _quality(data_dir: Path, out_dir: Path, summary: dict) -> dict:
    initial_lint = (data_dir / "quality_initial_ruff_check.log").read_text()
    initial_type = (data_dir / "quality_initial_mypy.log").read_text()
    lint_n = int(_one_match(r"^Found (\d+) errors?\.$", initial_lint, "Ruff").group(1))
    type_match = _one_match(
        r"^Found (\d+) errors? in (\d+) files? \(checked (\d+) source files?\)$",
        initial_type, "mypy",
    )
    type_n, type_error_files, type_checked_files = map(int, type_match.groups())
    final_rows = _read(data_dir / "quality_final.json")
    final = {row["check"]: row for row in final_rows}
    if len(final) != len(final_rows):
        raise ValueError("Duplicate quality_final check names")
    for name in ["ruff_check", "ruff_format", "mypy"]:
        if final[name]["exit_code"] != 0:
            raise ValueError(f"Final {name} did not pass; do not plot zero findings")

    sources = ["quality_initial_ruff_check.log", "quality_initial_mypy.log",
               "quality_final.json", "LIVE_RUN_SUMMARY.json"]
    # A successful final check is the evidence for zero reported errors.
    # When final log files are supplied, verify their explicit success messages too.
    for filename, pattern in [
        ("quality_final_ruff_check.log", r"^All checks passed!$"),
        ("quality_final_mypy.log", r"^Success: no issues found in (\d+) source files?$"),
    ]:
        if (data_dir / filename).exists():
            _one_match(pattern, (data_dir / filename).read_text(), filename)
            sources.append(filename)

    tests = {k: _count(summary["pytest"][k], f"pytest.{k}")
             for k in ["tests", "failures", "errors", "skipped"]}
    passed = tests["tests"] - sum(tests[k] for k in ["failures", "errors", "skipped"])
    if passed < 0:
        raise ValueError("Invalid pytest outcome totals")
    if (data_dir / "pytest.xml").exists():
        xml = ET.parse(data_dir / "pytest.xml").getroot()
        suites = list(xml.iter("testsuite"))
        parsed = {k: sum(int(s.get(k, "0")) for s in suites) for k in tests}
        if parsed != tests:
            raise ValueError("JUnit outcome totals disagree with LIVE_RUN_SUMMARY")
        sources.append("pytest.xml")

    fig = plt.figure(figsize=(11.7, 5.5))
    grid = fig.add_gridspec(1, 2, left=.075, right=.97, top=.76, bottom=.25,
                          width_ratios=[1.1, 1], wspace=.38)
    ax = fig.add_subplot(grid[0, 0])
    x, width = np.arange(2), .3
    before = ax.bar(x - width / 2, [lint_n, type_n], width, color=BLUE,
                    label="Initial recorded snapshot")
    after = ax.bar(x + width / 2, [0, 0], width, color=TEAL,
                   label="Final recorded snapshot")
    ax.bar_label(before, padding=4, color=INK)
    ax.bar_label(after, padding=4, color=TEAL)
    ax.set_xticks(x, ["Ruff lint findings", "mypy errors"])
    ax.set_ylabel("Reported findings / errors (count)")
    ax.set_ylim(0, max(lint_n, type_n, 1) * 1.35)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("A. Paired development quality records", loc="left", pad=18)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _clean_axis(ax, y_grid=True)

    ax = fig.add_subplot(grid[0, 1])
    labels = ["Passed", "Failures", "Errors", "Skipped"]
    values = [passed, tests["failures"], tests["errors"], tests["skipped"]]
    bars = ax.barh(labels, values, color=[TEAL, ORANGE, ORANGE, GREY], height=.55)
    ax.bar_label(bars, padding=5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values + [1]) * 1.18)
    ax.set_xlabel("Test cases (count)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("B. Latest Live_Demo regression outcomes", loc="left", pad=18)
    _clean_axis(ax)
    fig.text(.055, .94, "Goal 3 | Quality checks and regression evidence",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(.055, .875,
             "Panel A: recorded learning_lab initial/final pair.  "
             f"Panel B: Live_Demo run {summary['run_id']}.", fontsize=9.2, color=GREY)
    fig.text(.055, .125,
             "The execution sources differ. These counts do not measure all possible defects; "
             "mypy checks API/common only.", fontsize=9, color=GREY)
    fig.text(.055, .075,
             "Final zeros mean the configured checks passed, not that the project has zero defects.",
             fontsize=9, color=GREY)
    outputs = _save(fig, out_dir, "goal3_quality_tests")
    return {
        "outputs": outputs,
        "sources": _source_records(data_dir, sources),
        "execution_origins": {
            "quality": "Earlier recorded learning_lab development activity: paired initial/final records",
            "tests": f"Latest separate Live_Demo run {summary['run_id']}",
        },
        "values": {"lint": {"initial": lint_n, "final": 0},
                   "mypy": {"initial": type_n, "final": 0,
                            "initial_files_with_errors": type_error_files,
                            "checked_source_files": type_checked_files},
                   "final_check_exit_codes": {k: v["exit_code"] for k, v in final.items()},
                   "pytest": {**tests, "passed": passed}},
        "units": "Integer counts; quality findings and test outcomes are separate measures",
        "limitations": ["Different execution sources are deliberately labelled",
                        "mypy scope is API/common only; configured checks are not exhaustive",
                        "No inference about independent student effort or semester hours"],
    }


def _load_samples(data_dir: Path, metrics: dict) -> tuple[list[dict], np.ndarray]:
    with (data_dir / "digits_test.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["source_index", "target", "prediction"] + [f"pixel_{i}" for i in range(64)]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("digits_test.csv is missing required sample / pixel columns")
        rows = []
        for row in reader:
            parsed = {k: int(row[k]) for k in ["source_index", "target", "prediction"]}
            pixels = np.asarray([float(row[f"pixel_{i}"]) for i in range(64)])
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 16:
                raise ValueError("CSV contains invalid pixel values")
            if not 0 <= parsed["target"] <= 9 or not 0 <= parsed["prediction"] <= 9:
                raise ValueError("CSV contains an invalid digit class")
            parsed["pixels"] = pixels
            rows.append(parsed)
    if len(rows) != metrics["samples_test"] or len({r["source_index"] for r in rows}) != len(rows):
        raise ValueError("CSV sample count / uniqueness does not match the test record")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    if matrix.shape != (10, 10) or (matrix < 0).any() or matrix.sum() != len(rows):
        raise ValueError("Expected the recorded 10x10 digit confusion counts")
    rebuilt = np.zeros((10, 10), dtype=int)
    for row in rows:
        rebuilt[row["target"], row["prediction"]] += 1
    if not np.array_equal(matrix, rebuilt):
        raise ValueError("CSV predictions do not reproduce the recorded confusion matrix")
    if not math.isclose(float(np.trace(matrix) / matrix.sum()), metrics["accuracy"], abs_tol=1e-12):
        raise ValueError("Confusion counts disagree with recorded accuracy")
    # Deterministic, diverse examples: first encountered sample of each target;
    # preserve CSV order and do not select according to prediction correctness.
    selected, seen = [], set()
    for row in rows:
        if row["target"] not in seen:
            selected.append(row)
            seen.add(row["target"])
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError("Need at least six target classes for the sample panel")
    return selected, matrix


def _samples_model(data_dir: Path, out_dir: Path, summary: dict, metrics: dict,
                   shift: dict, manifest: dict) -> dict:
    samples, matrix = _load_samples(data_dir, metrics)
    original = _fraction(shift["original_accuracy"], "original_accuracy")
    controlled = _fraction(shift["shifted_accuracy"], "shifted_accuracy")
    if not math.isclose(original, metrics["accuracy"], abs_tol=1e-12):
        raise ValueError("Original input-comparison accuracy disagrees with metrics")
    if shift["batch_samples"] != metrics["samples_test"]:
        raise ValueError("Controlled comparison batch differs from the recorded test batch")

    fig = plt.figure(figsize=(13, 7.8))
    grid = fig.add_gridspec(2, 2, left=.055, right=.96, top=.81, bottom=.22,
                          width_ratios=[1.05, 1.15], height_ratios=[1.5, 1],
                          hspace=.48, wspace=.32)
    sample_grid = grid[0, 0].subgridspec(2, 3, wspace=.28, hspace=.62)
    for i, row in enumerate(samples):
        ax = fig.add_subplot(sample_grid[i // 3, i % 3])
        ax.imshow(row["pixels"].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16,
                  interpolation="nearest")
        ax.set_title(f"GT {row['target']} | Pred {row['prediction']}", fontsize=9.5, pad=3)
        ax.set_xlabel(f"source index {row['source_index']}", fontsize=8, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = fig.add_subplot(grid[1, 0])
    bars = ax.bar(["Original held-out", "Controlled +6 pixels"],
                  [100 * original, 100 * controlled], color=[BLUE, ORANGE], width=.58)
    ax.bar_label(bars, labels=[f"{100 * original:.2f}%", f"{100 * controlled:.2f}%"],
                 padding=4, color=INK, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("C. Same labelled test batch, changed inputs", loc="left", fontsize=11, pad=10)
    _clean_axis(ax, y_grid=True)

    ax = fig.add_subplot(grid[:, 1])
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(matrix.max()), 1))
    for actual in range(10):
        for prediction in range(10):
            value = int(matrix[actual, prediction])
            ax.text(prediction, actual, str(value), ha="center", va="center", fontsize=8,
                    color="white" if value > matrix.max() * .52 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit (ground truth)")
    ax.set_title(f"B. Confusion matrix | {matrix.sum()} actual test samples",
                 loc="left", fontsize=11, pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.035, shrink=.82)
    colorbar.set_label("Sample count")
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.text(.055, .955, "Goal 4 | Recorded test samples and model behaviour",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(.055, .91,
             f"Latest Live_Demo run {summary['run_id']} | "
             f"{metrics['samples_train']} training / {metrics['samples_test']} test samples | "
             f"accuracy {100 * metrics['accuracy']:.2f}%", fontsize=10, color=GREY)
    fig.text(.055, .845, "A. Six recorded 8 x 8 test images (intensity 0-16)",
             fontsize=11, fontweight="bold", color=INK)
    fig.text(.055, .14,
             "Examples are the first sample of each of six distinct targets in CSV order; "
             "selection does not use correctness.", fontsize=9, color=GREY)
    fig.text(.055, .098,
             "Controlled comparison: add 6 to each test pixel, clip to [0,16], retain labels. "
             "This is not validated natural drift detection.", fontsize=9, color=GREY)
    fig.text(.055, .056,
             "Public digits / new stratified split within the scikit-learn UCI test subset; "
             "not writer-independent or medical-image performance.", fontsize=9, color=GREY)
    outputs = _save(fig, out_dir, "goal4_samples_model")
    sources = ["digits_test.csv", "metrics.json", "input_shift.json",
               "manifest.json", "LIVE_RUN_SUMMARY.json"]
    if (data_dir / "digits_export_info.json").exists():
        sources.append("digits_export_info.json")
    return {
        "outputs": outputs,
        "sources": _source_records(data_dir, sources),
        "run_id": summary["run_id"], "mlflow_run_id": manifest["mlflow_run_id"],
        "model_sha256": manifest["model_sha256"],
        "values": {"samples_train": metrics["samples_train"], "samples_test": metrics["samples_test"],
                   "selected_samples": [{"source_index": r["source_index"], "target": r["target"],
                                          "prediction": r["prediction"], "pixels": r["pixels"].tolist()}
                                         for r in samples],
                   "confusion_matrix": matrix.tolist(), "class_labels": list(range(10)),
                   "confusion_rows": "Actual digit", "confusion_columns": "Predicted digit",
                   "confusion_total": int(matrix.sum()), "correct_predictions": int(np.trace(matrix)),
                   "original_accuracy_fraction": original,
                   "controlled_accuracy_fraction": controlled,
                   "controlled_change": shift["controlled_shift"]},
        "units": {"pixels": "intensity units, range 0-16", "confusion_matrix": "sample counts",
                  "stored_accuracy": "fraction", "plotted_accuracy": "percent"},
        "limitations": ["Public digits only; no medical-image performance claim",
                        "New split within sklearn's copy of the original UCI test subset",
                        "Six displayed examples are illustrative, not a representative sample estimate",
                        "Controlled pixel perturbation is not validated natural or concept drift"],
    }


def _latency_recovery(data_dir: Path, out_dir: Path, summary: dict, benchmark: dict,
                      recovery: dict) -> dict:
    durations = np.asarray(benchmark["durations_ms"], dtype=float)
    if durations.ndim != 1 or len(durations) != benchmark["measured_requests"]:
        raise ValueError("Benchmark duration count does not match measured_requests")
    if not len(durations) or not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("Benchmark contains invalid durations")
    for key, computed in [("median_ms", np.median(durations)),
                          ("p95_ms", np.percentile(durations, 95))]:
        if not math.isclose(float(benchmark[key]), float(computed), abs_tol=1e-9):
            raise ValueError(f"Benchmark {key} disagrees with raw durations")
    if benchmark["http_200"] + benchmark["errors"] != len(durations):
        raise ValueError("Benchmark HTTP outcome totals disagree")
    expected_stages = ["healthy", "corrupted_copy", "trusted_copy_restored"]
    stages = recovery["stages"]
    if [r["stage"] for r in stages] != expected_stages:
        raise ValueError("Unexpected recovery stage order")
    statuses = np.asarray([[r[k] for k in ["health_status", "readiness_status", "prediction_status"]]
                           for r in stages], dtype=int)
    if not np.isin(statuses, [200, 503]).all():
        raise ValueError("Recovery grid expects recorded 200/503 categories")

    fig = plt.figure(figsize=(12.3, 5.8))
    grid = fig.add_gridspec(1, 2, left=.07, right=.96, top=.75, bottom=.28,
                          width_ratios=[1.12, 1], wspace=.48)
    ax = fig.add_subplot(grid[0, 0])
    counts, edges, _ = ax.hist(durations, bins="fd", color=BLUE, edgecolor="white", linewidth=.8)
    ax.axvline(benchmark["median_ms"], color=ORANGE, linewidth=1.8,
               label=f"Median {benchmark['median_ms']:.3f} ms")
    ax.axvline(benchmark["p95_ms"], color=TEAL, linewidth=1.8, linestyle="--",
               label=f"p95 {benchmark['p95_ms']:.3f} ms")
    ax.set_xlabel("Complete local HTTP request / response time (ms)")
    ax.set_ylabel("Measured requests (count)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"A. Recorded request durations | n = {len(durations)}", loc="left", pad=18)
    _clean_axis(ax, y_grid=True)

    ax = fig.add_subplot(grid[0, 1])
    categories = (statuses == 503).astype(int)
    ax.imshow(categories, cmap=ListedColormap(["#DDEEE7", "#F6DDC9"]), vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(statuses[i, j]), ha="center", va="center", fontsize=16,
                    fontweight="bold", color=INK)
    ax.set_xticks(range(3), ["/health", "/ready", "/predict"])
    ax.set_yticks(range(3), ["Healthy copy", "Corrupted copy", "Trusted copy\nrestored"])
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="both", length=0)
    ax.set_title("B. Actual recovery HTTP status categories", loc="left", pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(facecolor="#DDEEE7", label="200: available"),
                       Patch(facecolor="#F6DDC9", label="503: refused / not ready")],
              loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False,
              fontsize=9, ncol=1)
    fig.text(.055, .94, "Goal 4 | Local latency and integrity recovery",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(.055, .875,
             f"Latest Live_Demo run {summary['run_id']} | one worker | loopback | "
             f"concurrency {benchmark['concurrency']}", fontsize=10, color=GREY)
    fig.text(.055, .16,
             f"Latency uses {len(durations)} measured requests after {benchmark['warmup_requests']} "
             "excluded warmups; repeated input, sequential traffic.", fontsize=9, color=GREY)
    fig.text(.055, .105,
             "The recovery drill modifies only a temporary artifact copy. HTTP codes are "
             "categories; colour does not encode numerical magnitude.", fontsize=9, color=GREY)
    fig.text(.055, .055,
             "These observations do not establish cloud capacity, concurrent-load behaviour, "
             "production reliability or a recovery-time SLA.", fontsize=9, color=GREY)
    outputs = _save(fig, out_dir, "goal4_latency_recovery")
    return {
        "outputs": outputs,
        "sources": _source_records(data_dir, ["benchmark.json", "recovery_drill.json",
                                               "manifest.json", "LIVE_RUN_SUMMARY.json"]),
        "run_id": summary["run_id"], "mlflow_run_id": recovery["source_run_id"],
        "values": {"durations_ms": durations.tolist(), "histogram_counts": counts.astype(int).tolist(),
                   "histogram_bin_edges_ms": edges.tolist(),
                   **{k: benchmark[k] for k in ["warmup_requests", "measured_requests", "concurrency",
                                               "http_200", "errors", "median_ms", "p95_ms", "min_ms", "max_ms"]},
                   "recovery_stage_order": expected_stages,
                   "recovery_endpoint_order": ["/health", "/ready", "/predict"],
                   "recovery_http_statuses": statuses.tolist()},
        "units": {"durations_and_histogram_edges": "milliseconds (ms)",
                  "histogram_counts": "requests", "http_status": "nominal categories, not magnitudes"},
        "limitations": [benchmark["limitations"], recovery["limitations"],
                        "Timing covers HTTP request/response, not isolated model inference",
                        "Known trusted artifact restored; no cryptographic authentication of the manifest"],
    }


def generate(data_dir: Path, out_dir: Path) -> dict:
    """Generate three PNG/PDF pairs plus the exact plotted-value provenance JSON."""
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read(data_dir / "LIVE_RUN_SUMMARY.json")
    if summary.get("status") != "completed":
        raise ValueError("Use a completed Live_Demo run")
    metrics, benchmark = _read(data_dir / "metrics.json"), _read(data_dir / "benchmark.json")
    shift, recovery = _read(data_dir / "input_shift.json"), _read(data_dir / "recovery_drill.json")
    manifest = _read(data_dir / "manifest.json")
    if summary["metrics"] != metrics or summary["benchmark"] != benchmark:
        raise ValueError("Live summary and individual measurement records disagree")
    if not (summary["mlflow_run_id"] == manifest["mlflow_run_id"] == recovery["source_run_id"]):
        raise ValueError("Input records do not belong to the same model run")
    if summary["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("Model identity differs between summary and manifest")
    if (data_dir / "digits_export_info.json").exists():
        exported = _read(data_dir / "digits_export_info.json")
        if exported["csv_sha256"] != sha256((data_dir / "digits_test.csv").read_bytes()).hexdigest():
            raise ValueError("Sample CSV digest differs from its export record")
        for key in ["mlflow_run_id", "model_sha256", "source_git_commit"]:
            if exported[key] != manifest[key]:
                raise ValueError(f"Sample export and model manifest differ: {key}")
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
             "axes.labelsize": 10, "text.color": INK, "axes.labelcolor": INK,
             "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
             "ps.fonttype": 42, "figure.facecolor": "white"}
    with plt.rc_context(style):
        figures = {
            "goal3_quality_tests": _quality(data_dir, out_dir, summary),
            "goal4_samples_model": _samples_model(data_dir, out_dir, summary, metrics, shift, manifest),
            "goal4_latency_recovery": _latency_recovery(data_dir, out_dir, summary, benchmark, recovery),
        }
    result = {"generated_by": Path(__file__).name,
              "scope": "Code-generated displays of supplied recorded evidence; no new training or measurements",
              "latest_live_run_id": summary["run_id"], "figures": figures}
    (out_dir / "figure_values_goal3_4.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "figures")
    arguments = parser.parse_args()
    generate(arguments.data_dir, arguments.out_dir)
