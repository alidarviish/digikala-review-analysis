"""Is the rating band wrong more often near its edges? (hypothesis H2)

The band comes from the buyer's rate: low below 2.5, medium from 2.5 to below 4.0, high from
4.0 to 5. If a text says little about whether its rate was 3.9 or 4.1, the model should be
wrong more often close to the edges 2.5 and 4.0. This reads only saved validation
predictions; nothing is trained, and the rate is used only to group the reviews, never as an
input. The test set is not read.

Data and checks:
- The rate, product_id and rating band come from data/cache/validation.parquet when its
  schema has them (the cache is the normalized copy of data/splits/validation.csv and is
  checked against the hash written when it was built, and that file against the split
  manifest). A missing column is taken from validation.csv instead, joined by id one to one.
  With --verify-csv the cache columns are also compared with validation.csv for every id.
- Every validation row is kept once (counts checked against reports/split_manifest.json);
  rates must be finite; a row has a band exactly when its rate is in (0, RATE_MAX], and the
  band must follow from the rate and the edges in config.py.
- The predictions are joined by id one to one and must be in the same row order.

Models, on exactly the same validation reviews (every review with a rating band):
- FaBERT, the final run: its saved labels with cross-fitted offsets.
- FaBERT and TF-IDF combined (the kept system, step 60): rebuilt from both models' saved
  probabilities with the same weight and offset search, tuned on one half of validation by
  product and scored on the other. Its Macro F1 and weights are checked against
  reports/ensemble.json, so the labels are the ones step 60 scored.

Groups:
1. whole rates (1, 2, 3, 4, 5) against fractional rates (for example 3.6)
2. the rate itself: each whole value, and fractional rates in steps of 0.25
3. distance to the nearest edge, for each edge and side; for all rates, and for fractional
   rates alone, where the distance is not just another name for the whole star value
4. each edge alone: how often a prediction lands on the other side of that edge
5. medium reviews: mistaken for low or for high, by rate

A group defined by the rate holds a single true class, so its Macro F1 means nothing; such
groups report the share of right answers (the recall of their class) and where the wrong
answers go. Groups with several true classes report Macro F1, precision, recall, F1 and the
confusion matrix.

Intervals: a product-level cluster bootstrap. Reviews of one product are not independent, so
each round draws products with replacement (as many as the group has) and takes all their
reviews in the group; both models are scored on the same draw (paired). 1000 rounds, seed
2026, the 2.5 and 97.5 percentiles. Used for Macro F1, its difference between the models, the
share of right answers, the crossing rate and the medium shares. A group with fewer than
MIN_ROWS reviews or MIN_PRODUCTS products is marked "small" and should not carry a
conclusion.

Writes reports/rating_boundaries.json and prints a summary.

Usage:
    uv run python scripts/analyze_rating_boundaries.py
    uv run python scripts/analyze_rating_boundaries.py --verify-csv   # also compare with validation.csv
"""

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from digikala_reviews.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    RATE_MAX,
    RATING_BAND_CLASSES,
    RATING_BAND_EDGES,
    REPORTS_DIR,
    SPLIT_SEED,
)
from digikala_reviews.ensemble import cross_fitted_blend
from digikala_reviews.experiments import product_half
from digikala_reviews.metrics import classification_report
from digikala_reviews.provenance import check_cache, git_commit
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tuning import save_json

HEAD = "rating_band"
LABELS = RATING_BAND_CLASSES  # low, medium, high = 0, 1, 2
LOW_EDGE, HIGH_EDGE = RATING_BAND_EDGES
RATE_STEP = 0.25  # width of the fractional-rate groups; 2.5 and 4.0 fall on group edges
DISTANCE_BINS = [0.0, 0.1, 0.25, 0.5, 1.0, np.inf]  # distance to the nearest edge
MIN_ROWS = 1000  # smaller groups are reported, but marked "small"
MIN_PRODUCTS = 100  # the same for groups with few independent products
ROUNDS = 1000
SEED = SPLIT_SEED  # 2026
CACHE_FILE = "validation.parquet"
SPLIT_FILE = "validation.csv"
OUTPUT = REPORTS_DIR / "rating_boundaries.json"


# ---------------------------------------------------------------- data


def load_validation(verify_csv: bool) -> tuple[pd.DataFrame, dict]:
    """id, product_id, rate and rating band of every validation row, in the cache's row order,
    with a record of where each column came from and what was checked."""
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    cache_info = json.loads((CACHE_DIR / "cache_info.json").read_text(encoding="utf-8"))
    assert cache_info["files"][CACHE_FILE]["source_sha256"] == manifest["sha256"][SPLIT_FILE], (
        "the cache was not built from the checked validation.csv"
    )
    data = check_cache(CACHE_DIR, [CACHE_FILE])  # the cache file is the one built and checked
    columns = pq.read_schema(CACHE_DIR / CACHE_FILE).names
    wanted = ["id", "product_id", "rate", HEAD]
    present = [c for c in wanted if c in columns]
    missing = [c for c in wanted if c not in columns]
    assert "id" in present, f"{CACHE_FILE} has no id column: {columns}"
    record = {"cache_columns": columns, "taken_from_cache": present, "taken_from_csv": missing, "data": data}
    table = pd.read_parquet(CACHE_DIR / CACHE_FILE, columns=present)
    expected_rows = manifest["row_counts"][SPLIT_FILE]
    assert len(table) == expected_rows, f"{CACHE_FILE}: {len(table):,} rows, manifest {expected_rows:,}"
    assert table["id"].is_unique, f"{CACHE_FILE}: duplicate ids"

    if missing or verify_csv:
        csv = read_split_file(SPLIT_FILE)[wanted]
        assert csv["id"].is_unique, f"{SPLIT_FILE}: duplicate ids"
        both = table[["id"]].merge(csv[["id"]], on="id", how="outer", indicator=True)
        record["join_with_csv"] = {
            "cache_rows": int(len(table)),
            "csv_rows": int(len(csv)),
            "only_in_cache": int((both["_merge"] == "left_only").sum()),
            "only_in_csv": int((both["_merge"] == "right_only").sum()),
        }
        assert record["join_with_csv"]["only_in_cache"] == 0, "ids in the cache but not in the csv"
        assert record["join_with_csv"]["only_in_csv"] == 0, "ids in the csv but not in the cache"
        csv = table[["id"]].merge(csv, on="id", how="left", validate="one_to_one")  # cache order
        assert (csv["id"].to_numpy() == table["id"].to_numpy()).all(), "the join changed the row order"
        for column in missing:
            table[column] = csv[column].to_numpy()
        if verify_csv:
            mismatches = {}
            for column in present:
                if column == "id":
                    continue
                a, b = table[column], csv[column]
                same = (a == b) | (a.isna() & b.isna())
                mismatches[column] = int((~same).sum())
            record["cache_vs_csv_mismatches"] = mismatches
            assert all(v == 0 for v in mismatches.values()), f"cache and csv differ: {mismatches}"
    return table[wanted], record


def band_of(rate):
    """The band a rate belongs to, by the edges in config.py (for rates in (0, RATE_MAX])."""
    rate = np.asarray(rate, dtype=float)
    return np.array(LABELS)[np.searchsorted(np.array(RATING_BAND_EDGES), rate, side="right")]


def check_rates(table: pd.DataFrame, manifest_counts: dict) -> dict:
    """Every row with a valid rate has a band and the band follows from the rate; every row
    without a valid rate has none. Returns the counts."""
    rate = table["rate"].to_numpy(dtype=float)
    band = table[HEAD].to_numpy()
    has_band = pd.notna(band)
    finite = np.isfinite(rate)
    valid = finite & (rate > 0) & (rate <= RATE_MAX)
    counts = {
        "rows": int(len(table)),
        "with_band": int(has_band.sum()),
        "without_band": int((~has_band).sum()),
        "rate_not_finite": int((~finite).sum()),
        "rate_zero": int((finite & (rate == 0)).sum()),
        "rate_out_of_range": int((finite & ((rate < 0) | (rate > RATE_MAX))).sum()),
        "band_counts": {label: int((band == label).sum()) for label in LABELS},
    }
    assert (has_band == valid).all(), "a row has a band exactly when its rate is in (0, RATE_MAX]"
    assert (band_of(rate[has_band]) == band[has_band]).all(), "the bands must follow from the rate"
    for label in LABELS:
        assert counts["band_counts"][label] == manifest_counts[label], f"{label}: count differs from manifest"
    assert counts["without_band"] == manifest_counts["missing"], "rows without a band differ from manifest"
    return counts


# ---------------------------------------------------------------- groups


def nearest_edge(rate):
    """For each rate: the nearest edge, and the signed distance to it (negative = below)."""
    rate = np.asarray(rate, dtype=float)
    edge = np.where(np.abs(rate - LOW_EDGE) <= np.abs(rate - HIGH_EDGE), LOW_EDGE, HIGH_EDGE)
    return edge, rate - edge


def distance_bin(distance) -> np.ndarray:
    """Name of the distance group, for example '[0.1, 0.25)'."""
    distance = np.abs(np.asarray(distance, dtype=float))
    index = np.digitize(distance, DISTANCE_BINS[1:-1], right=False)
    names = [f"[{a:g}, {b:g})" for a, b in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:], strict=True)]
    return np.array(names)[index]


def rate_group(rate) -> np.ndarray:
    """A whole rate is its own group; a fractional one falls in a step of RATE_STEP."""
    rate = np.asarray(rate, dtype=float)
    whole = np.isclose(rate, np.round(rate))
    low = np.minimum(np.floor(rate / RATE_STEP) * RATE_STEP, RATE_MAX - RATE_STEP)
    fractional = np.array([f"fractional [{a:.2f}, {a + RATE_STEP:.2f})" for a in low])
    return np.where(whole, np.array([f"whole {r:.0f}" for r in rate]), fractional)


# ---------------------------------------------------------------- weighted scores


def codes_of(labels) -> np.ndarray:
    """Class codes in the order of LABELS (low 0, medium 1, high 2)."""
    position = {label: i for i, label in enumerate(LABELS)}
    return np.array([position[v] for v in labels], dtype=int)


def weighted_matrix(true_codes, predicted_codes, weights) -> np.ndarray:
    n = len(LABELS)
    return np.bincount(true_codes * n + predicted_codes, weights=weights, minlength=n * n).reshape(n, n)


def macro_f1_of(matrix) -> float:
    """Macro F1 from a (weighted) confusion matrix; a class with nothing true or predicted
    scores 0, as in classification_report."""
    tp = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)  # 2 tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    return float(f1.mean())


def cluster_bootstrap(products, statistic, rounds: int = ROUNDS, seed: int = SEED) -> dict:
    """95% percentile intervals of each value of statistic(weights) over product-level draws.

    Each round draws as many products as the group has, with replacement; a review's weight
    is the number of times its product was drawn. statistic gets those weights and returns a
    dict of numbers, so values that are compared (two models) always see the same draw."""
    _, product_index = np.unique(np.asarray(products), return_inverse=True)
    n_products = int(product_index.max()) + 1
    rng = np.random.default_rng(seed)
    values = {}
    for _ in range(rounds):
        drawn = np.bincount(rng.integers(0, n_products, n_products), minlength=n_products)
        for key, value in statistic(drawn[product_index].astype(float)).items():
            values.setdefault(key, []).append(value)
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in values.items()}


def size(n_rows: int, products) -> dict:
    n_products = int(len(np.unique(products)))
    return {
        "rows": int(n_rows),
        "products": n_products,
        "small": bool(n_rows < MIN_ROWS or n_products < MIN_PRODUCTS),
    }


# ---------------------------------------------------------------- group reports


def mixed_group(y_true, predictions: dict, products) -> dict:
    """A group with several true classes: scores of each model and their paired difference
    (the first model is the reference)."""
    names = list(predictions)
    reference, variant = names[0], names[1]
    result = size(len(y_true), products)
    result["true_class_share"] = {label: float(np.mean(y_true == label)) for label in LABELS}
    result["all_classes_present"] = bool(all(v > 0 for v in result["true_class_share"].values()))
    for name, y_pred in predictions.items():
        with np.errstate(invalid="ignore", divide="ignore"):  # an absent class has no recall
            result[name] = classification_report(y_true, y_pred, LABELS)
    if not result["all_classes_present"]:
        # a missing class scores F1 0 and drags Macro F1 down: not comparable with other groups
        result["note"] = "not every class is in this group: read the per-class scores, not Macro F1"
        return result
    true_codes = codes_of(y_true)
    predicted = {name: codes_of(p) for name, p in predictions.items()}

    def statistic(w):
        found = {name: macro_f1_of(weighted_matrix(true_codes, predicted[name], w)) for name in names}
        return {**found, "difference": found[variant] - found[reference]}

    interval = cluster_bootstrap(products, statistic)
    for name in names:
        result[name]["macro_f1_interval_95"] = interval[name]
    result[f"{variant}_minus_{reference}"] = {
        "macro_f1": result[variant]["macro_f1"] - result[reference]["macro_f1"],
        "macro_f1_interval_95": interval["difference"],
    }
    return result


def single_class_group(y_true, predictions: dict, products) -> dict:
    """A group of one true class: the share of right answers, where the wrong ones go, and
    the paired difference in right answers between the models."""
    classes = pd.unique(y_true)
    assert len(classes) == 1, f"one true class expected, found {list(classes)}"
    names = list(predictions)
    reference, variant = names[0], names[1]
    result = {**size(len(y_true), products), "true_class": str(classes[0])}
    right = {name: (np.asarray(p) == y_true).astype(float) for name, p in predictions.items()}
    for name, y_pred in predictions.items():
        counts = {label: int(np.sum(y_pred == label)) for label in LABELS}
        result[name] = {
            "right": float(right[name].mean()),
            "predicted_counts": counts,
            "predicted_share": {label: counts[label] / len(y_true) for label in LABELS},
        }

    def statistic(w):
        found = {name: float((w * right[name]).sum() / w.sum()) for name in names}
        return {**found, "difference": found[variant] - found[reference]}

    interval = cluster_bootstrap(products, statistic)
    for name in names:
        result[name]["right_interval_95"] = interval[name]
    result[f"{variant}_minus_{reference}"] = {
        "right": result[variant]["right"] - result[reference]["right"],
        "right_interval_95": interval["difference"],
    }
    return result


def crossing(y_true, y_pred, edge: float) -> np.ndarray:
    """True where the prediction lands on the other side of this edge from the true band."""
    above = RATING_BAND_EDGES.index(edge) + 1  # bands at or above this edge have code >= above
    return (codes_of(y_true) >= above) != (codes_of(y_pred) >= above)


def edge_group(y_true, predictions: dict, products, edge: float) -> dict:
    """Share of predictions on the other side of one edge, for each model, and the difference."""
    names = list(predictions)
    reference, variant = names[0], names[1]
    result = size(len(y_true), products)
    crossed = {name: crossing(y_true, p, edge).astype(float) for name, p in predictions.items()}

    def statistic(w):
        found = {name: float((w * crossed[name]).sum() / w.sum()) for name in names}
        return {**found, "difference": found[variant] - found[reference]}

    interval = cluster_bootstrap(products, statistic)
    for name in names:
        result[name] = {"crossed_this_edge": float(crossed[name].mean()), "interval_95": interval[name]}
    result[f"{variant}_minus_{reference}"] = {
        "crossed_this_edge": result[variant]["crossed_this_edge"] - result[reference]["crossed_this_edge"],
        "interval_95": interval["difference"],
    }
    return result


def share_group(predictions: dict, products) -> dict:
    """Share of each predicted class among the reviews of the group (true medium ones)."""
    names = list(predictions)
    result = size(len(next(iter(predictions.values()))), products)
    hit = {(name, label): (p == label).astype(float) for name, p in predictions.items() for label in LABELS}

    def statistic(w):
        return {f"{name}|{label}": float((w * h).sum() / w.sum()) for (name, label), h in hit.items()}

    interval = cluster_bootstrap(products, statistic)
    for name in names:
        result[name] = {
            label: {"share": float(hit[(name, label)].mean()), "interval_95": interval[f"{name}|{label}"]}
            for label in LABELS
        }
    return result


def by_groups(groups, y_true, predictions, products, single_class: bool) -> dict:
    report = single_class_group if single_class else mixed_group
    result = {}
    for group in sorted(pd.unique(groups)):
        inside = groups == group
        result[str(group)] = report(
            y_true[inside], {name: p[inside] for name, p in predictions.items()}, products[inside]
        )
    return result


# ---------------------------------------------------------------- main


def load_predictions(validation: pd.DataFrame) -> tuple[dict, dict]:
    """The saved validation predictions of both final models, joined to validation by id."""
    columns = ["id", f"{HEAD}__predicted_with_offsets"] + [f"{HEAD}__log_proba__{label}" for label in LABELS]
    files = {
        "fabert": ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet",
        "tfidf": ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet",
    }
    tables, joins = {}, {}
    for name, path in files.items():
        absent = [c for c in columns if c not in pq.read_schema(path).names]
        assert not absent, f"{path} lacks {absent}"
        table = pd.read_parquet(path, columns=columns)
        assert table["id"].is_unique, f"{name}: duplicate ids"
        both = validation[["id"]].merge(
            table[["id"]], on="id", how="outer", indicator=True, validate="one_to_one"
        )
        joins[name] = {
            "rows": int(len(table)),
            "only_in_validation": int((both["_merge"] == "left_only").sum()),
            "only_in_predictions": int((both["_merge"] == "right_only").sum()),
        }
        assert joins[name]["only_in_validation"] == 0, f"{name}: validation ids without a prediction"
        assert joins[name]["only_in_predictions"] == 0, f"{name}: predictions for ids not in validation"
        assert (table["id"].to_numpy() == validation["id"].to_numpy()).all(), f"{name}: row order differs"
        tables[name] = table
    return tables, joins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-csv", action="store_true", help="compare the cache columns with validation.csv by id"
    )
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    validation, source = load_validation(args.verify_csv)
    rate_checks = check_rates(validation, manifest["class_counts"][SPLIT_FILE][HEAD])
    tables, joins = load_predictions(validation)

    rows = validation[HEAD].notna().to_numpy()
    rate = validation["rate"].to_numpy(dtype=float)[rows]
    y_true = validation[HEAD].to_numpy()[rows].astype(str)
    products = validation["product_id"].to_numpy()[rows]

    def log_proba(table):
        return table[[f"{HEAD}__log_proba__{label}" for label in LABELS]].to_numpy(dtype=float)[rows]

    blended, weights = cross_fitted_blend(
        log_proba(tables["fabert"]), log_proba(tables["tfidf"]), codes_of(y_true), product_half(products)
    )
    predictions = {
        "fabert": tables["fabert"][f"{HEAD}__predicted_with_offsets"].to_numpy()[rows].astype(str),
        "ensemble": np.array(LABELS)[blended],
    }
    # the rebuilt labels must score exactly what step 60 recorded
    saved = json.loads((REPORTS_DIR / "ensemble.json").read_text(encoding="utf-8"))
    saved = saved["final_fabert_and_tfidf"][HEAD]
    rebuilt = {name: classification_report(y_true, p, LABELS)["macro_f1"] for name, p in predictions.items()}
    assert abs(rebuilt["fabert"] - saved["fabert_macro_f1"]) < 1e-9, "FaBERT labels differ from step 60"
    assert abs(rebuilt["ensemble"] - saved["blend_macro_f1"]) < 1e-9, "combined labels differ from step 60"
    assert weights == saved["fabert_weight_per_half"], "combination weights differ from step 60"

    whole = np.isclose(rate, np.round(rate))
    kind = np.where(whole, "whole", "fractional")
    edge, signed = nearest_edge(rate)
    side = np.where(signed < 0, "below", "at or above")
    distance = distance_bin(signed)
    near = np.array([f"edge {e:g}, {s}, distance {d}" for e, s, d in zip(edge, side, distance, strict=True)])
    fractional = {name: p[~whole] for name, p in predictions.items()}

    result = {
        "reviews": int(rows.sum()),
        "products": int(len(np.unique(products))),
        "fabert_weight_per_half": weights,
        "data_checks": {"source": source, "rates": rate_checks, "prediction_joins": joins},
        "rate_values": {
            "distinct": int(len(np.unique(rate))),
            "whole_share": float(whole.mean()),
            "rows_per_whole_value": {f"{v:.0f}": int(np.sum(rate == v)) for v in np.unique(rate[whole])},
        },
        "overall": mixed_group(y_true, predictions, products),
        "1_whole_vs_fractional": by_groups(kind, y_true, predictions, products, single_class=False),
        "2_by_rate": by_groups(rate_group(rate), y_true, predictions, products, single_class=True),
        "3_by_distance_all_rates": by_groups(near, y_true, predictions, products, single_class=True),
        "3_by_distance_fractional_only": by_groups(
            near[~whole], y_true[~whole], fractional, products[~whole], single_class=True
        ),
        "3_by_distance_mixed_classes": by_groups(distance, y_true, predictions, products, single_class=False),
        "4_each_edge": {},
        "5_medium_mistakes": {},
    }
    for e in RATING_BAND_EDGES:
        mine = edge == e
        part = {}
        for group in sorted(pd.unique(distance[mine])):
            inside = mine & (distance == group)
            part[str(group)] = edge_group(
                y_true[inside], {n: p[inside] for n, p in predictions.items()}, products[inside], e
            )
        result["4_each_edge"][f"edge {e:g}"] = part
    medium = y_true == "medium"
    for group_name, groups in [
        ("all", np.full(medium.sum(), "all medium")),
        ("whole_vs_fractional", kind[medium]),
        ("by_rate", rate_group(rate[medium])),
    ]:
        part = {}
        for group in sorted(pd.unique(groups)):
            inside = groups == group
            part[str(group)] = share_group(
                {n: p[medium][inside] for n, p in predictions.items()}, products[medium][inside]
            )
        result["5_medium_mistakes"][group_name] = part

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "edges": list(RATING_BAND_EDGES),
                "rate_max": RATE_MAX,
                "rate_step": RATE_STEP,
                "distance_bins": [b if np.isfinite(b) else None for b in DISTANCE_BINS],
                "min_rows": MIN_ROWS,
                "min_products": MIN_PRODUCTS,
                "intervals": "product-level cluster bootstrap, paired between the models, "
                "percentiles 2.5 and 97.5",
                "bootstrap_rounds": ROUNDS,
                "bootstrap_seed": SEED,
                "nearest_edge_tie": "a rate exactly between the edges (3.25) counts for 2.5",
                "verify_csv": args.verify_csv,
            },
            **result,
            "minutes": round((time.perf_counter() - start) / 60, 1),
            "code": git_commit(),
        },
    )
    summary(result)
    print(f"saved {OUTPUT} ({(time.perf_counter() - start) / 60:.1f} min)")


def interval_text(interval, sign: bool = False) -> str:
    form = "+.4f" if sign else ".3f"
    return f"[{interval[0]:{form}}, {interval[1]:{form}}]"


def summary(result):
    c = result["data_checks"]
    print(
        f"data: {c['rates']['rows']:,} validation rows, {result['reviews']:,} with a band, "
        f"{result['products']:,} products"
    )
    print(
        f"      from the cache {c['source']['taken_from_cache']}, "
        f"from the csv {c['source']['taken_from_csv']}"
    )
    if "cache_vs_csv_mismatches" in c["source"]:
        print(
            "      cache against validation.csv, values that differ: "
            f"{c['source']['cache_vs_csv_mismatches']}"
        )
    o = result["overall"]
    d = o["ensemble_minus_fabert"]
    print(
        f"overall Macro F1: fabert {o['fabert']['macro_f1']:.4f}, ensemble {o['ensemble']['macro_f1']:.4f}, "
        f"difference {d['macro_f1']:+.4f} {interval_text(d['macro_f1_interval_95'], True)}"
    )
    print("1. whole against fractional rates (Macro F1 [95%]; F1 low / medium / high):")
    for group, g in result["1_whole_vs_fractional"].items():
        for name in ("fabert", "ensemble"):
            f1 = " / ".join(f"{g[name]['per_class'][label]['f1']:.3f}" for label in LABELS)
            print(
                f"   {group:<10} {name:<8} rows {g['rows']:>9,} products {g['products']:>6,}  "
                f"Macro F1 {g[name]['macro_f1']:.4f} "
                f"{interval_text(g[name]['macro_f1_interval_95'])}  F1 {f1}"
            )
    for key, title in [
        ("2_by_rate", "2. by rate (share right [95%]; where the answers go)"),
        ("3_by_distance_all_rates", "3a. by distance to the nearest edge, all rates"),
        ("3_by_distance_fractional_only", "3b. by distance to the nearest edge, fractional rates only"),
    ]:
        print(title + ":")
        for group, g in result[key].items():
            f = g["fabert"]
            shares = ", ".join(f"{label} {f['predicted_share'][label]:.1%}" for label in LABELS)
            d = g["ensemble_minus_fabert"]
            flag = "  (small)" if g["small"] else ""
            print(
                f"   {group:<38} {g['true_class']:<6} rows {g['rows']:>8,} products {g['products']:>6,}  "
                f"fabert right {f['right']:.3f} {interval_text(f['right_interval_95'])}  predicted {shares}  "
                f"ensemble-fabert {d['right']:+.4f} {interval_text(d['right_interval_95'], True)}{flag}"
            )
    print("4. each edge: share of predictions on the other side of it, fabert [95%] / ensemble [95%]:")
    for edge, part in result["4_each_edge"].items():
        for group, g in part.items():
            f, e = g["fabert"], g["ensemble"]
            flag = "  (small)" if g["small"] else ""
            print(
                f"   {edge:<9} distance {group:<12} rows {g['rows']:>8,} products {g['products']:>6,}  "
                f"{f['crossed_this_edge']:.3f} {interval_text(f['interval_95'])} / "
                f"{e['crossed_this_edge']:.3f} {interval_text(e['interval_95'])}{flag}"
            )
    print("5. medium reviews predicted as low / medium / high, fabert [95%]:")
    for part in result["5_medium_mistakes"].values():
        for group, g in part.items():
            s = g["fabert"]
            flag = "  (small)" if g["small"] else ""
            shares = " / ".join(
                f"{s[label]['share']:.3f} {interval_text(s[label]['interval_95'])}" for label in LABELS
            )
            print(f"   {group:<30} rows {g['rows']:>8,} products {g['products']:>6,}  {shares}{flag}")


if __name__ == "__main__":
    main()
