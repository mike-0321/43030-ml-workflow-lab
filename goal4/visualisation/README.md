# Goal 4 recorded-results visualisation

This kit redraws two figures from fixed historical records: held-out digits
and recorded predictions, and local HTTP timing and temporary-copy recovery.
It performs no new training, inference, test execution or measurements. For
a new application run, use the instructions at the repository root.

## Redraw

Use Python 3.12 and an isolated environment. From this directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate_this_goal.py
```

On Windows, use `.venv\Scripts\python.exe`. Installation needs package
registry access; drawing is offline. The wrapper calls only the Goal 4
functions in the unchanged shared plotting module.

Eight input records are in `data/`. The script validates the recorded run
and model identities, CSV checksum, confusion counts, accuracy and timing
summaries before drawing. Output PNGs, PDFs and exact plotted values go to
`outputs/figures/`; `reproduction_manifest.json` records input, code and
output hashes.

Personal absolute-path metadata in the copied JSON was replaced with
`<original-run>`. Metrics, completion status, CSV content and source/run/model
identifiers are unchanged; original evidence was not modified. The saved
interpreter metadata is never executed by this redraw.

The experiment uses public digits with a new stratified split. The pixel
perturbation is controlled, and the sequential loopback timing and recovery
checks do not establish clinical performance, natural-drift detection or
cloud capacity.
