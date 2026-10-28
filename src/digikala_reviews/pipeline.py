"""Every step of the project in order, as run by notebooks/run_all.py.

Each step is one script call. A step is done when its output files exist, and it can start
when the steps it needs are done. The scripts themselves skip work that is already saved,
so starting a step again after an interruption goes on from where it stopped.

Where a step runs:
    local  needs the raw data or the test labels, so only on your own machine
    any    any machine with the project installed (CPU is enough)
    gpu    needs a CUDA GPU

Output paths start with the folder they live in: data/, cache/, reports/ or artifacts/.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PILOT_MODELS = [
    "PartAI__TookaBERT-Base",
    "sbunlp__fabert",
    "FacebookAI__xlm-roberta-base",
    "jhu-clsp__mmBERT-base",
    "HooshvareLab__bert-fa-zwnj-base",
    "PartAI__TookaBERT-Large",
]
PILOT_SEED_MODELS = ["PartAI__TookaBERT-Base", "sbunlp__fabert", "PartAI__TookaBERT-Large"]
PILOT_LARGE_MODELS = ["sbunlp__fabert", "PartAI__TookaBERT-Large"]


PILOT_ARGS = ["scripts/pilot_transformers.py"]
PILOT_SEED_ARGS = [
    *PILOT_ARGS,
    "--models",
    ",".join(m.replace("__", "/") for m in PILOT_SEED_MODELS),
    "--seeds",
    "2027,2028",
]
PILOT_LARGE_ARGS = [
    *PILOT_ARGS,
    "--models",
    ",".join(m.replace("__", "/") for m in PILOT_LARGE_MODELS),
    "--train-rows",
    "500000",
]


@dataclass
class Step:
    number: int
    title: str
    command: list[str]  # the script and its arguments, run from the project folder
    place: str  # local, any or gpu
    needs: list[int]  # numbers of the steps that must be done first
    outputs: list[tuple[str, int]]  # (path or pattern, how many files it must match)
    note: str = ""
    smoke: bool = False  # the script has a --smoke test run (a tiny model on a few rows)
    log: str = field(init=False)

    def __post_init__(self):
        self.log = f"step_{self.number:02d}.log"


def files(*paths):
    return [(path, 1) for path in paths]


STEPS = [
    # Data: raw CSVs to the normalized cache
    Step(
        1, "Raw CSV to Parquet", ["scripts/build_parquet.py"], "local", [], files("reports/data_ingest.json")
    ),
    Step(
        2, "Quality checks", ["scripts/run_quality_gate.py"], "local", [1], files("reports/quality_gate.json")
    ),
    Step(3, "Cleaning", ["scripts/build_clean.py"], "local", [1], files("reports/cleaning_report.json")),
    Step(
        4,
        "Normalizer report",
        ["scripts/check_normalization.py"],
        "local",
        [3],
        files("reports/normalization_report.json"),
    ),
    Step(
        5,
        "Split by product",
        ["scripts/build_splits.py"],
        "local",
        [1, 3],
        files("reports/split_report.json", "data/splits/split_assignment.parquet"),
    ),
    Step(
        6,
        "Split balance chart",
        ["scripts/plot_split_balance.py"],
        "local",
        [5],
        files("reports/figures/split_balance.png"),
    ),
    Step(
        7,
        "Standard data files",
        ["scripts/build_standard_files.py"],
        "local",
        [3, 5],
        files(
            *[
                f"data/splits/{name}.csv"
                for name in (
                    "train_supervised",
                    "train_pseudo_unlabeled",
                    "natural_unlabeled",
                    "validation",
                    "test",
                )
            ],
            "data/splits/audit/masked_labels_reference.csv",
            "reports/split_manifest.json",
        ),
    ),
    Step(
        8,
        "Check the standard files",
        ["scripts/verify_standard_files.py"],
        "local",
        [7],
        files("reports/standard_files_check.json"),
    ),
    Step(
        9, "Normalized cache", ["scripts/cache_normalized.py"], "local", [8], files("cache/cache_info.json")
    ),
    Step(
        10,
        "Review years (for E10)",
        ["scripts/build_review_years.py"],
        "local",
        [3, 9],
        files("cache/review_years_info.json"),
    ),
    # TF-IDF
    Step(
        11,
        "TF-IDF stage 1: features",
        ["scripts/tune_tfidf.py", "--stage", "features"],
        "any",
        [9],
        [("reports/tfidf_runs/*__C1.json", 16)],
    ),
    Step(
        12,
        "TF-IDF stage 2: regularization",
        ["scripts/tune_tfidf.py", "--stage", "regularization"],
        "any",
        [11],
        [("reports/tfidf_runs/*__C0.*.json", 4), ("reports/tfidf_runs/*__C[24].json", 4)],
    ),
    Step(
        13,
        "TF-IDF experiments E1 to E7 and E13",
        ["scripts/experiment_tfidf.py"],
        "any",
        [12],
        [("reports/tfidf_experiments/*.json", 20)],
        "plan and expected outcomes: reports/tfidf_experiment_plan.md",
    ),
    Step(
        14,
        "TF-IDF extra: reference, E11, E8",
        ["scripts/experiment_tfidf_extra.py", "--experiments", "reference", "E11", "E8"],
        "any",
        [9],
        [
            ("reports/tfidf_extra/reference__*.json", 2),
            ("reports/tfidf_extra/E11__*.json", 2),
            ("reports/tfidf_extra/E8__*.json", 2),
        ],
    ),
    Step(
        15,
        "TF-IDF extra: E12, E9, E10",
        ["scripts/experiment_tfidf_extra.py", "--experiments", "E12", "E9_E10"],
        "any",
        [10],
        [
            ("reports/tfidf_extra/E12__*.json", 6),
            ("reports/tfidf_extra/E9__*.json", 4),
            ("reports/tfidf_extra/E10__*.json", 2),
        ],
    ),
    Step(
        16,
        "TF-IDF adopted changes together",
        ["scripts/final_tfidf.py"],
        "any",
        [13, 14, 15],
        [("reports/tfidf_final/final__*.json", 2)],
        "the adopted changes are written in the script, from the experiment results",
    ),
    Step(
        17,
        "TF-IDF final model",
        ["scripts/fit_final_tfidf.py"],
        "any",
        [16],
        files(
            "reports/tfidf_final/chosen_model.json", "artifacts/tfidf_final/validation_predictions.parquet"
        ),
        "reference features with offsets, chosen after step 16 (reports/tfidf_experiment_plan.md)",
    ),
    Step(
        18,
        "TF-IDF analyses",
        ["scripts/analyze_tfidf.py"],
        "local",
        [17],
        [("reports/tfidf_analysis/*.json", 9)],
        "reads the masked labels and the product file, which stay on your machine",
    ),
    # Transformer
    Step(
        19,
        "Tokenizer comparison",
        ["scripts/compare_tokenizers.py"],
        "any",
        [9],
        files("reports/tokenizers/compare.json"),
    ),
    Step(
        20,
        "Transformer pilot: six models",
        ["scripts/pilot_transformers.py"],
        "gpu",
        [17, 19],
        files(*[f"reports/transformer_pilot/{m}.json" for m in PILOT_MODELS]),
        "settings and reasons: reports/transformer_model_choice.md",
        smoke=True,
    ),
    Step(
        21,
        "Transformer pilot: two more seeds",
        PILOT_SEED_ARGS,
        "gpu",
        [20],
        files(
            *[
                f"reports/transformer_pilot/{m}__seed{s}__n100000.json"
                for m in PILOT_SEED_MODELS
                for s in (2027, 2028)
            ]
        ),
    ),
    Step(
        22,
        "Transformer pilot: 500,000 reviews",
        PILOT_LARGE_ARGS,
        "gpu",
        [20],
        files(*[f"reports/transformer_pilot/{m}__seed2026__n500000.json" for m in PILOT_LARGE_MODELS]),
    ),
    Step(
        23,
        "Pilot summary",
        ["scripts/summarize_pilot.py"],
        "any",
        [21, 22],
        files("reports/transformer_pilot/summary.json"),
    ),
    Step(
        24,
        "FaBERT on all training reviews",
        ["scripts/train_transformer.py"],
        "gpu",
        [23],
        files("reports/transformer_final/fabert.json", "artifacts/transformer_final/model.pt"),
        "FaBERT was chosen after step 23 (reports/transformer_model_choice.md)",
        smoke=True,
    ),
    Step(
        25,
        "FaBERT against TF-IDF",
        ["scripts/compare_final_models.py"],
        "any",
        [24],
        files("reports/transformer_final/vs_tfidf.json"),
    ),
    Step(
        26,
        "FaBERT analyses",
        ["scripts/analyze_tfidf.py", "--model", "transformer"],
        "local",
        [24],
        [("reports/transformer_analysis/*.json", 6)],
    ),
    # Report
    Step(
        27,
        "Result charts",
        ["scripts/plot_results.py"],
        "any",
        [18, 23, 25, 26],
        files("reports/figures/results_where_it_helps.png"),
    ),
    Step(
        28,
        "Table of all results",
        ["scripts/summarize_results.py"],
        "any",
        [27],
        files("reports/results_summary.md"),
    ),
    # Added after the first results: the pilot runs scored on all of validation, and FaBERT
    # trained for every epoch
    Step(
        29,
        "Pilot on all of validation: six models",
        [*PILOT_ARGS, "--whole-validation"],
        "gpu",
        [19],
        files(*[f"reports/transformer_pilot_whole/{m}.json" for m in PILOT_MODELS]),
        smoke=True,
    ),
    Step(
        30,
        "Pilot on all of validation: two more seeds",
        [*PILOT_SEED_ARGS, "--whole-validation"],
        "gpu",
        [29],
        files(
            *[
                f"reports/transformer_pilot_whole/{m}__seed{s}__n100000.json"
                for m in PILOT_SEED_MODELS
                for s in (2027, 2028)
            ]
        ),
    ),
    Step(
        31,
        "Pilot on all of validation: 500,000 reviews",
        [*PILOT_LARGE_ARGS, "--whole-validation"],
        "gpu",
        [29],
        files(*[f"reports/transformer_pilot_whole/{m}__seed2026__n500000.json" for m in PILOT_LARGE_MODELS]),
    ),
    Step(
        32,
        "Pilot summary on all of validation",
        ["scripts/summarize_pilot.py", "--whole-validation"],
        "local",
        [17, 30, 31],
        files("reports/transformer_pilot_whole/summary.json"),
        "reads the TF-IDF predictions, which stay on your machine",
    ),
    Step(
        33,
        "FaBERT for every epoch (no early stopping)",
        ["scripts/train_transformer.py", "--no-early-stop"],
        "gpu",
        [23],
        files("reports/transformer_final/fabert_no_early_stop.json"),
        smoke=True,
    ),
    Step(
        34,
        "FaBERT for every epoch against the first run",
        ["scripts/compare_fabert_runs.py"],
        "local",
        [24, 33],
        files("reports/transformer_final/no_early_stop_vs_first.json"),
        "reads both runs' predictions; the first run's stay on your machine",
    ),
]


def output_path(pattern: str, roots: dict[str, Path]) -> Path:
    """The pattern with its first folder (data, cache, reports, artifacts) replaced by its root."""
    first, _, rest = pattern.partition("/")
    return roots[first] / rest


def is_done(step: Step, roots: dict[str, Path]) -> bool:
    for pattern, count in step.outputs:
        path = output_path(pattern, roots)
        if len(list(path.parent.glob(path.name))) < count:
            return False
    return True


def status(steps: list[Step], roots: dict[str, Path], running: set[int]) -> dict[int, str]:
    """For each step: running, done, ready, or waiting for the steps it needs."""
    done = {step.number for step in steps if is_done(step, roots)}
    result = {}
    for step in steps:
        missing = [n for n in step.needs if n not in done]
        if step.number in running:
            result[step.number] = "running"
        elif step.number in done:
            result[step.number] = "done"
        elif missing:
            result[step.number] = "waiting for " + ", ".join(str(n) for n in missing)
        else:
            result[step.number] = "ready"
    return result


def running_steps(steps: list[Step]) -> set[int]:
    """Steps whose exact command is running now on this machine (read from /proc)."""
    commands = set()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            parts = [p for p in cmdline.read_bytes().decode().split("\0") if p]
        except OSError:
            continue  # the process ended meanwhile
        for i, part in enumerate(parts):
            if part.endswith(".py") and "scripts/" in part:
                commands.add(tuple(["scripts/" + part.split("scripts/")[-1], *parts[i + 1 :]]))
    return {
        step.number
        for step in steps
        if tuple(step.command) in commands or tuple([*step.command, "--smoke"]) in commands
    }


def start(step: Step, python: Path, project: Path, env: dict, log_dir: Path, smoke: bool = False):
    """Start a step in the background; it keeps running if the notebook tab is closed.

    smoke=True runs the script's --smoke test instead, with a log of its own."""
    assert step.smoke or not smoke, f"step {step.number} has no test run"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = step.log.replace(".log", "_smoke.log") if smoke else step.log
    with open(log_dir / log_name, "a") as log:
        subprocess.Popen(
            [str(python), *step.command, *(["--smoke"] if smoke else [])],
            cwd=project,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
