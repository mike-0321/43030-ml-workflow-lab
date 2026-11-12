# Goal 3 quality visualisation

This folder redraws a quality-check figure from fixed historical records. The
left panel uses the recorded initial and final development checks; the right
panel uses the saved, separately identified Live_Demo regression results.
Redrawing does **not** rerun the 22 tests, Ruff, mypy, model training or HTTP
measurements. Run the full application tests from the repository root using
that directory's test instructions.

## Redraw

Use Python 3.12 and an isolated environment. From this directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate_this_goal.py
```

On Windows, use `.venv\Scripts\python.exe`. Installation needs registry
access; drawing the saved records is offline. Run `generate_this_goal.py`,
which calls only the Goal 3 function. The shared plotting module also retains
Goal 4 functions, but their input data is deliberately not included here.

The generated PNG, PDF and plotted-value JSON go into `outputs/figures/`.
`reproduction_manifest.json` records the supplied input hashes, script hashes,
plotting environment and output hashes. Rerunning replaces those generated
outputs; it does not create new learning or measurement records.

## Input records and privacy

Four required inputs are `LIVE_RUN_SUMMARY.json`,
`quality_initial_ruff_check.log`, `quality_initial_mypy.log` and
`quality_final.json`. The final Ruff/mypy logs and `pytest.xml` are included
for the script's optional consistency checks. All seven are in `data/`.

Personal absolute-path metadata in the copied `LIVE_RUN_SUMMARY.json`
was redacted to `<original-run>`. Recorded metrics, completion status, run and
model identifiers, command arguments and results are otherwise unchanged.
The original evidence was not modified. These records document the original
execution; the historical interpreter metadata is not executed by this kit.

The mypy scope is API/common only. Passing configured checks and tests does
not establish the absence of all defects. This visualisation is a reproducible display of recorded evidence.
