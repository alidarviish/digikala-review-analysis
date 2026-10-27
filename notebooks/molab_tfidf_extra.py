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
    run(f"uv venv -q --clear --python 3.13 {_venv}")
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


@app.cell
def _(mo):
    mo.md(
        """
        ## Finishing stage 3 here

        Only needed when stage 3 cannot finish on Colab. Share the Colab output folder
        (`MyDrive/digikala/outputs`) as "anyone with the link" and put its link and the commit
        to run in the two cells below. Finished results are copied in and skipped, so stage 3
        goes on from where Colab stopped. A file that is already here is never overwritten.
        """
    )
    return


@app.cell
def _(mo):
    bring_button = mo.ui.run_button(label="5. Bring in the Colab results")
    start_c = mo.ui.run_button(label="6. Start job c: stage 3")
    status_c = mo.ui.run_button(label="7. Status of job c")
    mo.hstack([bring_button, start_c, status_c], justify="start")
    return bring_button, start_c, status_c


@app.cell
def _(OUTPUTS, PYTHON, Path, WORK, bring_button, mo, run, shutil):
    mo.stop(not bring_button.value)
    _url = ""  # Drive link of the Colab output folder
    _source = WORK / "colab_outputs"
    shutil.rmtree(_source, ignore_errors=True)
    run(f"{PYTHON.parent}/gdown -q --folder '{_url}' -O {_source}")
    _copied, _kept = 0, 0
    for _file in sorted(_source.rglob("*")):
        _parts = _file.relative_to(_source).parts
        _top = next((p for p in ("reports", "artifacts") if p in _parts), None)
        if not _file.is_file() or _top is None:
            continue  # outputs_manifest.json stays out; a new one is written at the end
        _target = OUTPUTS / Path(*_parts[_parts.index(_top) :])
        if _target.exists():
            _kept += 1
            continue
        _target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_file, _target)
        _copied += 1
    _listing = run(
        f"ls {OUTPUTS}/reports/tfidf_runs | wc -l; "
        f"ls {OUTPUTS}/reports/tfidf_experiments {OUTPUTS}/artifacts/tfidf"
    )
    mo.plain_text(f"{_copied} files copied, {_kept} were already here\nstage 1 and 2 runs: {_listing}")
    return


@app.cell
def _(CACHE, LOGS, OUTPUTS, PYTHON, Path, REPO_URL, WORK, mo, os, run, start_c, subprocess):
    mo.stop(not start_c.value)
    _commit = ""  # the full commit hash to run stage 3 with
    _repo = WORK / "repo_c"  # a copy of its own, so jobs a and b keep their code

    def _stage3_running():
        for _cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                _args = _cmdline.read_bytes().decode().split("\0")
            except OSError:
                continue
            if any(_a.endswith("scripts/experiment_tfidf.py") for _a in _args):
                return True
        return False

    if _stage3_running():
        _message = "Job c is already running."
    else:
        assert len(_commit) == 40, "fill in the commit first"
        run(f"rm -rf {_repo} && git clone -q {REPO_URL} {_repo} && git -C {_repo} checkout -q {_commit}")
        _env = {**os.environ, "DIGIKALA_CACHE_DIR": str(CACHE), "DIGIKALA_OUTPUT_DIR": str(OUTPUTS)}
        _env["PYTHONPATH"] = str(_repo / "src")  # its own package code, ahead of the one installed
        _probe = "import digikala_reviews as d; print(d.__file__)"
        _where = run(f"cd {_repo} && PYTHONPATH={_repo}/src {PYTHON} -c '{_probe}'")
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LOGS / "stage3_molab.log", "a") as _log:
            subprocess.Popen(
                [str(PYTHON), "scripts/experiment_tfidf.py"],
                cwd=_repo,
                env=_env,
                stdout=_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _message = f"Job c started with {run(f'git -C {_repo} log -1 --oneline')}\npackage from {_where}"
    mo.plain_text(_message)
    return


@app.cell
def _(LOGS, Path, mo, run, status_c):
    mo.stop(not status_c.value)
    _log = LOGS / "stage3_molab.log"
    _tail = run(f"tail -n 25 {_log}") if _log.exists() else "(not started)"
    _running = False
    for _cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            _args = _cmdline.read_bytes().decode().split("\0")
        except OSError:
            continue
        _running = _running or any(_a.endswith("scripts/experiment_tfidf.py") for _a in _args)
    _memory = run("cat /sys/fs/cgroup/memory/memory.usage_in_bytes || true")
    _used = f"{int(_memory) / 1024**3:.1f} GB in use" if _memory.isdigit() else "memory unknown"
    mo.plain_text(f"{_tail}\n\njob c running: {'yes' if _running else 'no'}\n{_used}")
    return


if __name__ == "__main__":
    app.run()
