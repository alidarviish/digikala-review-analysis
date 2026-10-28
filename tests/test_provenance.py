import subprocess

from digikala_reviews.provenance import git_commit


def make_repo(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print(1)\n")
    git("add", ".")
    git("commit", "-q", "-m", "first")
    return tmp_path


def test_clean_code_is_committed(tmp_path):
    repo = make_repo(tmp_path)
    assert git_commit(repo)["uncommitted_changes"] is False


def test_new_result_files_do_not_count(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "reports").mkdir()
    (repo / "reports" / "result.json").write_text("{}")
    (repo / "outputs.zip").write_bytes(b"zip")
    assert git_commit(repo)["uncommitted_changes"] is False


def test_edited_script_counts(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "scripts" / "run.py").write_text("print(2)\n")
    assert git_commit(repo)["uncommitted_changes"] is True


def test_new_script_counts(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "scripts" / "other.py").write_text("print(3)\n")
    assert git_commit(repo)["uncommitted_changes"] is True
