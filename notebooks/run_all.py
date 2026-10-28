import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import shutil
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, io, mo, os, shutil, subprocess, sys, zipfile


@app.cell
def _(mo):
    mo.md(r"""
    # Run the whole project, step by step

    Every step is one script of `scripts/`, in the order set in
    `src/digikala_reviews/pipeline.py`. A step is **done** when its output files exist and
    **ready** when the steps it needs are done. The scripts skip work that is already saved,
    so a step that was interrupted is simply started again.

    The same notebook runs in two places:

    - **local**, your own machine with the raw data: open it with
      `uvx marimo edit notebooks/run_all.py` from the project folder, after `uv sync`.
      Every step can run here, the GPU steps only with a CUDA GPU.
    - **molab**: the steps marked `any` or `gpu`. Fill in `COMMIT` and `CACHE_FOLDER_URL`
      and click *Set up* first; it gets the code at that commit, installs the exact library
      versions and downloads the cache. Download the outputs after each step (the machine
      can be reclaimed), then on your machine run
      `uv run python scripts/import_colab_outputs.py <unzipped folder>`.

    One step runs at a time, so two jobs never compete for memory. A step keeps running
    when the tab is closed; *Refresh* shows where it is.
    """)
    return


@app.cell
def _(Path, mo):
    PLACE = "local"  # "local" or "molab"
    COMMIT = ""  # molab: the full commit hash to run
    CACHE_FOLDER_URL = ""  # molab: Drive link of the cache folder, shared as "anyone with the link"
    GPU = False  # molab: True when the machine has a GPU and the Transformer libraries are needed

    REPO_URL = "https://github.com/alidarviish/digikala-review-analysis.git"
    # The versions every Transformer result was made with (see "environment" in its JSON file)
    TORCH = "torch==2.14.1"
    TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
    TRANSFORMER_LIBRARIES = "transformers==5.21.0 tokenizers==0.23.3"

    if PLACE == "local":
        PROJECT = mo.notebook_dir().parent
        PYTHON = PROJECT / ".venv" / "bin" / "python"
        ENV_EXTRA = {}
        ROOTS = {
            "data": PROJECT / "data",
            "cache": PROJECT / "data" / "cache",
            "reports": PROJECT / "reports",
            "artifacts": PROJECT / "artifacts",
        }
    else:
        WORK = Path("/marimo/work")
        PROJECT = WORK / "repo"
        PYTHON = WORK / "venv" / "bin" / "python"
        ENV_EXTRA = {"DIGIKALA_CACHE_DIR": str(WORK / "cache"), "DIGIKALA_OUTPUT_DIR": str(WORK / "outputs")}
        ROOTS = {
            "data": PROJECT / "data",
            "cache": WORK / "cache",
            "reports": WORK / "outputs" / "reports",
            "artifacts": WORK / "outputs" / "artifacts",
        }
    LOGS = ROOTS["reports"] / "logs"
    return (
        CACHE_FOLDER_URL,
        COMMIT,
        ENV_EXTRA,
        GPU,
        LOGS,
        PLACE,
        PROJECT,
        PYTHON,
        REPO_URL,
        ROOTS,
        TORCH,
        TORCH_INDEX,
        TRANSFORMER_LIBRARIES,
    )


@app.cell
def _(subprocess):
    def run(command):
        """Run a shell command; stop with its output if it fails."""
        result = subprocess.run(["bash", "-o", "pipefail", "-c", command], capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(f"failed ({result.returncode}): {command}\n{output}")
        return output

    return (run,)


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="Set up (molab only)")
    setup_button
    return (setup_button,)


@app.cell
def _(
    CACHE_FOLDER_URL,
    COMMIT,
    GPU,
    PLACE,
    PROJECT,
    PYTHON,
    REPO_URL,
    ROOTS,
    TORCH,
    TORCH_INDEX,
    TRANSFORMER_LIBRARIES,
    mo,
    run,
    setup_button,
    shutil,
):
    mo.stop(not setup_button.value)
    assert PLACE == "molab", "on your own machine, run uv sync in the project folder instead"
    assert len(COMMIT) == 40 and CACHE_FOLDER_URL, "fill in COMMIT and CACHE_FOLDER_URL first"
    _lines = [run(f"rm -rf {PROJECT} && git clone -q {REPO_URL} {PROJECT}")]
    _lines.append(run(f"git -C {PROJECT} checkout -q {COMMIT} && git -C {PROJECT} log -1 --oneline"))
    if not ROOTS["reports"].exists():  # start from the results in the repository, so finished work is skipped
        shutil.copytree(PROJECT / "reports", ROOTS["reports"])
    _venv = PYTHON.parent.parent
    run(f"uv venv -q --clear --python 3.13 {_venv}")
    run(f"uv pip install -q --python {PYTHON} -r {PROJECT}/requirements.txt gdown")
    run(f"uv pip install -q --python {PYTHON} --no-deps -e {PROJECT}")
    if GPU:
        run(f"uv pip install -q --python {PYTHON} {TORCH} --index-url {TORCH_INDEX}")
        run(f"uv pip install -q --python {PYTHON} {TRANSFORMER_LIBRARIES}")
        _lines.append(run(f"{PYTHON} -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"))

    _cache = ROOTS["cache"]
    shutil.rmtree(_cache, ignore_errors=True)
    run(f"{_venv}/bin/gdown -q --folder '{CACHE_FOLDER_URL}' -O {_cache}")
    for _file in list(_cache.rglob("*")):  # gdown may put the files in a subfolder
        if _file.is_file() and _file.parent != _cache:
            shutil.move(_file, _cache / _file.name)
    _check = (
        "from pathlib import Path; from digikala_reviews.provenance import check_cache; "
        f"print(check_cache(Path('{_cache}'), ['train_supervised.parquet', 'validation.parquet'])['sha256'])"
    )
    _lines.append(run(f'{PYTHON} -c "{_check}"'))
    mo.plain_text("\n".join(_lines))
    return


@app.cell
def _(PROJECT, mo, refresh_button, sys):
    refresh_button  # so that Refresh, clicked after Set up, loads the code
    mo.stop(not (PROJECT / "src").exists(), mo.md("Click *Set up* first, then *Refresh*."))
    if str(PROJECT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT / "src"))
    from digikala_reviews import pipeline

    return (pipeline,)


@app.cell
def _(mo):
    refresh_button = mo.ui.run_button(label="Refresh")
    refresh_button
    return (refresh_button,)


@app.cell
def _(ROOTS, mo, pipeline, refresh_button):
    refresh_button
    _status = pipeline.status(pipeline.STEPS, ROOTS, pipeline.running_steps(pipeline.STEPS))
    mo.ui.table(
        [
            {
                "Step": _step.number,
                "What": _step.title,
                "Where": _step.place,
                "Status": _status[_step.number],
                "Note": _step.note,
            }
            for _step in pipeline.STEPS
        ],
        selection=None,
        pagination=False,
    )
    return


@app.cell
def _(mo, pipeline):
    step_choice = mo.ui.dropdown(
        {f"{_s.number}. {_s.title}": _s.number for _s in pipeline.STEPS}, label="Step"
    )
    smoke_box = mo.ui.checkbox(label="test run (--smoke)")
    start_button = mo.ui.run_button(label="Start")
    log_button = mo.ui.run_button(label="Show its log")
    mo.hstack([step_choice, smoke_box, start_button, log_button], justify="start")
    return log_button, smoke_box, start_button, step_choice


@app.cell
def _(
    ENV_EXTRA,
    LOGS,
    PLACE,
    PROJECT,
    PYTHON,
    ROOTS,
    mo,
    os,
    pipeline,
    shutil,
    smoke_box,
    start_button,
    step_choice,
):
    mo.stop(not start_button.value)
    mo.stop(step_choice.value is None, mo.md("Choose a step first."))
    _step = pipeline.STEPS[step_choice.value - 1]
    _running = pipeline.running_steps(pipeline.STEPS)
    _state = pipeline.status(pipeline.STEPS, ROOTS, _running)[_step.number]
    if _running:
        _message = f"Step {min(_running)} is running; one step at a time."
    elif smoke_box.value and not _step.smoke:
        _message = f"Step {_step.number} has no test run."
    elif _state != "ready" and not smoke_box.value:  # a test run writes apart, so it may run anytime
        _message = f"Step {_step.number} is {_state}."
    elif _step.place == "local" and PLACE != "local":
        _message = f"Step {_step.number} needs the raw data, so it runs only on your own machine."
    elif _step.place == "gpu" and shutil.which("nvidia-smi") is None:
        _message = f"Step {_step.number} needs a CUDA GPU, and this machine has none."
    else:
        pipeline.start(_step, PYTHON, PROJECT, {**os.environ, **ENV_EXTRA}, LOGS, smoke=smoke_box.value)
        _what = "test run of step" if smoke_box.value else "step"
        _message = f"Started {_what} {_step.number}: {' '.join(_step.command)}"
    mo.md(_message)
    return


@app.cell
def _(LOGS, Path, log_button, mo, pipeline, run, smoke_box, step_choice):
    mo.stop(not log_button.value or step_choice.value is None)
    _step = pipeline.STEPS[step_choice.value - 1]
    _log = LOGS / (_step.log.replace(".log", "_smoke.log") if smoke_box.value else _step.log)
    _tail = run(f"tail -n 30 '{_log}'") if _log.exists() else "(not started)"
    _used = "memory unknown"
    for _path in ["/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory.current"]:
        if Path(_path).exists():  # the first is molab's, the second most Linux machines'
            _used = f"{int(Path(_path).read_text()) / 1024**3:.1f} GB in use"
            break
    mo.plain_text(f"{_tail}\n\n{_used}")
    return


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Prepare the download (molab only)")
    download_button
    return (download_button,)


@app.cell
def _(ENV_EXTRA, PLACE, PROJECT, PYTHON, ROOTS, download_button, io, mo, os, shutil, subprocess, zipfile):
    mo.stop(not download_button.value)
    assert PLACE == "molab", "on your own machine the outputs are already in the project"
    # Only files that are new or changed here; the ones copied from the repository stay out
    _outputs = ROOTS["reports"].parent
    _new = _outputs.parent / "download"
    shutil.rmtree(_new, ignore_errors=True)
    for _file in sorted(_outputs.rglob("*")):
        _relative = _file.relative_to(_outputs)
        if not _file.is_file() or _file.suffix == ".tmp" or _relative.name == "outputs_manifest.json":
            continue
        _in_repo = PROJECT / _relative
        if _in_repo.is_file() and _in_repo.read_bytes() == _file.read_bytes():
            continue
        (_new / _relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_file, _new / _relative)
    subprocess.run(
        [str(PYTHON), "scripts/write_outputs_manifest.py"],
        cwd=PROJECT,
        env={**os.environ, **ENV_EXTRA, "DIGIKALA_OUTPUT_DIR": str(_new)},
        check=True,
    )
    _buffer = io.BytesIO()
    with zipfile.ZipFile(_buffer, "w", zipfile.ZIP_DEFLATED) as _zip:
        for _file in sorted(_new.rglob("*")):
            if _file.is_file():
                _zip.write(_file, _file.relative_to(_new))
    mo.download(data=_buffer.getvalue(), filename="molab_outputs.zip", label="Download molab_outputs.zip")
    return


if __name__ == "__main__":
    app.run()
