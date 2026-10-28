from digikala_reviews.config import PROJECT_ROOT
from digikala_reviews.pipeline import STEPS, Step, is_done, status

ROOT_NAMES = {"data", "cache", "reports", "artifacts"}


def test_steps_are_numbered_in_order():
    assert [step.number for step in STEPS] == list(range(1, len(STEPS) + 1))


def test_a_step_needs_only_earlier_steps():
    for step in STEPS:
        assert all(n < step.number for n in step.needs), step.title


def test_every_script_exists():
    for step in STEPS:
        assert (PROJECT_ROOT / step.command[0]).is_file(), step.command[0]


def test_places_and_output_folders_are_known():
    for step in STEPS:
        assert step.place in {"local", "any", "gpu"}
        assert step.outputs, step.title
        for pattern, count in step.outputs:
            assert pattern.split("/")[0] in ROOT_NAMES and count >= 1


def test_no_step_is_pointed_at_the_test_set():
    # the test set is scored once at the end, in a step of its own
    for step in STEPS:
        assert not any("test" in part for part in step.command[1:])


def make_roots(tmp_path):
    roots = {name: tmp_path / name for name in ROOT_NAMES}
    for folder in roots.values():
        folder.mkdir()
    return roots


def two_steps():
    first = Step(1, "first", ["scripts/a.py"], "any", [], [("reports/a.json", 1)])
    second = Step(2, "second", ["scripts/b.py"], "any", [1], [("reports/runs/*.json", 2)])
    return [first, second]


def test_status_moves_from_waiting_to_done(tmp_path):
    roots = make_roots(tmp_path)
    steps = two_steps()
    assert status(steps, roots, running=set()) == {1: "ready", 2: "waiting for 1"}

    (roots["reports"] / "a.json").write_text("{}")
    assert status(steps, roots, running=set()) == {1: "done", 2: "ready"}

    (roots["reports"] / "runs").mkdir()
    (roots["reports"] / "runs" / "x.json").write_text("{}")
    assert status(steps, roots, running={2}) == {1: "done", 2: "running"}
    assert not is_done(steps[1], roots)  # one of two files is not enough

    (roots["reports"] / "runs" / "y.json").write_text("{}")
    assert status(steps, roots, running=set()) == {1: "done", 2: "done"}


def test_test_runs_only_where_the_script_has_one(tmp_path):
    import pytest

    from digikala_reviews.pipeline import start

    for step in STEPS:
        has_smoke = "--smoke" in (PROJECT_ROOT / step.command[0]).read_text(encoding="utf-8")
        assert has_smoke or not step.smoke, step.title
    plain = next(step for step in STEPS if not step.smoke)
    with pytest.raises(AssertionError):
        start(plain, tmp_path / "python", tmp_path, {}, tmp_path, smoke=True)
