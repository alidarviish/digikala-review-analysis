import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import shutil
    import subprocess
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, io, mo, os, shutil, subprocess, zipfile


@app.cell
def _(mo):
    mo.md(
        """
        # Extra TF-IDF experiments (E8 to E12) on molab

        Runs `scripts/experiment_tfidf_extra.py` next to stage 3 on Colab. The plan and the
        expected outcomes are in `reports/tfidf_experiment_plan.md`.

        1. Fill in `COMMIT` and `CACHE_FOLDER_URL` below.
        2. **Set up**: gets the code, installs the exact library versions in a separate
           environment (so the notebook itself is not touched) and downloads the cache.
        3. **Start** both jobs. They run in the background, so a closed tab does not stop them.
        4. **Status** shows the logs; results are saved one by one and a restarted job skips
           what is done.
        5. **Download** the outputs at the end (and now and then, in case the machine is
           reclaimed), then on your machine run
           `uv run python scripts/import_colab_outputs.py <unzipped folder>`.

        Click the buttons one at a time. The cells only run when their button is clicked.
        """
    )
    return


@app.cell
def _(Path):
    REPO_URL = "https://github.com/alidarviish/digikala-review-analysis.git"
    COMMIT = ""  # the full commit hash to run
    CACHE_FOLDER_URL = ""  # Drive link of the cache folder, shared as "anyone with the link"

    WORK = Path("/marimo/work")
    REPO, CACHE, OUTPUTS = WORK / "repo", WORK / "cache", WORK / "outputs"
    PYTHON = WORK / "venv" / "bin" / "python"
    LOGS = OUTPUTS / "reports" / "logs"
    JOBS = {
        "a": ["reference", "E11", "E8"],
        "b": ["E12", "E9_E10"],
    }
    return CACHE, CACHE_FOLDER_URL, COMMIT, JOBS, LOGS, OUTPUTS, PYTHON, REPO, REPO_URL, WORK


@app.cell
def _(Path, subprocess):
    def run(command):
        """Run a shell command; stop with its output if it fails."""
        result = subprocess.run(["bash", "-o", "pipefail", "-c", command], capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(f"failed ({result.returncode}): {command}\n{output}")
        return output

    def running_jobs():
        """Command lines of the experiment processes that are running now."""
        found = []
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                parts = cmdline.read_bytes().decode().split("\0")
            except OSError:
                continue  # the process ended meanwhile
            if any(part.endswith("experiment_tfidf_extra.py") for part in parts):
                found.append(" ".join(p for p in parts if p))
        return found

    return run, running_jobs


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1. Set up")
    setup_button
    return (setup_button,)


@app.cell
def _(CACHE, CACHE_FOLDER_URL, COMMIT, PYTHON, REPO, REPO_URL, WORK, mo, run, setup_button, shutil):
    mo.stop(not setup_button.value)
    assert len(COMMIT) == 40 and CACHE_FOLDER_URL, "fill in COMMIT and CACHE_FOLDER_URL first"
    WORK.mkdir(parents=True, exist_ok=True)
    _lines = [run(f"rm -rf {REPO} && git clone -q {REPO_URL} {REPO} && git -C {REPO} checkout -q {COMMIT}")]
    _lines.append(run(f"git -C {REPO} log -1 --oneline"))
    _venv = PYTHON.parent.parent
    run(f"uv venv -q --python 3.13 {_venv}")
    run(f"uv pip install -q --python {PYTHON} -r {REPO}/requirements.txt")
    run(f"uv pip install -q --python {PYTHON} --no-deps -e {REPO}")
    run(f"uv pip install -q --python {PYTHON} gdown")
    _lines.append(
        run(f"{PYTHON} -c 'import sklearn, pandas; print(sklearn.__version__, pandas.__version__)'")
    )

    shutil.rmtree(CACHE, ignore_errors=True)
    run(f"{_venv}/bin/gdown -q --folder '{CACHE_FOLDER_URL}' -O {CACHE}")
    for _file in list(CACHE.rglob("*")):  # gdown may put the files in a subfolder
        if _file.is_file() and _file.parent != CACHE:
            shutil.move(_file, CACHE / _file.name)
    _check = (
        "from digikala_reviews.provenance import check_cache; "
        "print(check_cache(__import__('pathlib').Path('" + str(CACHE) + "'), "
        "['train_supervised.parquet', 'validation.parquet'])['sha256'])"
    )
    _lines.append(run(f'{PYTHON} -c "{_check}"'))
    _lines.append(run(f"ls -la {CACHE}"))
    mo.plain_text("\n".join(_lines))
    return


@app.cell
def _(mo):
    start_a = mo.ui.run_button(label="2a. Start job a: reference, E11, E8")
    start_b = mo.ui.run_button(label="2b. Start job b: E12, E9 and E10")
    mo.hstack([start_a, start_b], justify="start")
    return start_a, start_b


@app.cell
def _(CACHE, JOBS, LOGS, OUTPUTS, PYTHON, REPO, mo, os, running_jobs, start_a, start_b, subprocess):
    mo.stop(not (start_a.value or start_b.value))
    _name = "a" if start_a.value else "b"
    _steps = JOBS[_name]
    if any(job.endswith("--experiments " + " ".join(_steps)) for job in running_jobs()):
        _message = f"Job {_name} is already running."
    else:
        LOGS.mkdir(parents=True, exist_ok=True)
        _env = {**os.environ, "DIGIKALA_CACHE_DIR": str(CACHE), "DIGIKALA_OUTPUT_DIR": str(OUTPUTS)}
        _command = [str(PYTHON), "scripts/experiment_tfidf_extra.py", "--experiments", *_steps]
        with open(LOGS / f"extra_job_{_name}.log", "a") as _log:
            subprocess.Popen(
                _command, cwd=REPO, env=_env, stdout=_log, stderr=subprocess.STDOUT, start_new_session=True
            )
        _message = f"Job {_name} started: {' '.join(_steps)}"
    mo.md(_message)
    return


@app.cell
def _(mo):
    status_button = mo.ui.run_button(label="3. Status")
    status_button
    return (status_button,)


@app.cell
def _(LOGS, Path, mo, run, running_jobs, status_button):
    mo.stop(not status_button.value)
    _parts = []
    for _name in ["a", "b"]:
        _log = LOGS / f"extra_job_{_name}.log"
        _tail = run(f"tail -n 15 {_log}") if _log.exists() else "(not started)"
        _parts.append(f"--- job {_name} ---\n{_tail}")
    _processes = "\n".join(running_jobs()) or "no job is running"
    _memory = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    _used = f"{int(_memory.read_text()) / 1024**3:.1f} GB in use" if _memory.exists() else "memory unknown"
    mo.plain_text("\n\n".join(_parts) + f"\n\nrunning:\n{_processes}\n\n{_used} (the machine has 32 GB)")
    return


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="4. Prepare the download")
    download_button
    return (download_button,)


@app.cell
def _(CACHE, OUTPUTS, PYTHON, REPO, download_button, io, mo, run, zipfile):
    mo.stop(not download_button.value)
    run(
        f"cd {REPO} && DIGIKALA_CACHE_DIR={CACHE} DIGIKALA_OUTPUT_DIR={OUTPUTS} "
        f"{PYTHON} scripts/write_outputs_manifest.py"
    )
    _buffer = io.BytesIO()
    with zipfile.ZipFile(_buffer, "w", zipfile.ZIP_DEFLATED) as _zip:
        for _file in sorted(OUTPUTS.rglob("*")):
            if _file.is_file() and _file.suffix != ".tmp":
                _zip.write(_file, _file.relative_to(OUTPUTS))
    mo.download(data=_buffer.getvalue(), filename="molab_outputs.zip", label="Download molab_outputs.zip")
    return


if __name__ == "__main__":
    app.run()
