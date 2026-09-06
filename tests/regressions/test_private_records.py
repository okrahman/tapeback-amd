"""Regression tests: transcript-adjacent records are private and symlink-safe.

The resume cache, run records, and session state hold full transcript text or
name the recording files, but were written with `mkdir` + `write_text` and no
mode — world-readable under a default umask, and willing to follow a planted
symlink at the entry path. They now go through `_fs.write_private_text`
(0600 file in a verified 0700 directory, atomic tmp+rename) and refuse
symlinks explicitly.
"""

import json
import stat

from tapeback import _resume, _runlog
from tapeback._fs import write_private_text
from tapeback.models import Segment


def _assert_private_file(path):
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, f"{path} must be 0600"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700, f"{path.parent} must be 0700"


def test_resume_entry_is_private_and_atomic(tmp_path):
    directory = tmp_path / "resume"
    key = _resume.ResumeKey("0" * 32)
    assert (
        _resume.store(key, directory, [Segment(start=0.0, end=1.0, text="secret")], {})
        is not None
    )
    _assert_private_file(directory / key.filename)
    # Re-storing replaces atomically — the entry is still complete JSON afterwards.
    _resume.store(key, directory, [Segment(start=0.0, end=2.0, text="updated")], {})
    payload = json.loads((directory / key.filename).read_text())
    assert payload["segments"][0]["text"] == "updated"


def test_resume_store_refuses_a_planted_symlink(tmp_path):
    directory = tmp_path / "resume"
    directory.mkdir()
    target = tmp_path / "victim"
    target.write_text("do not touch")
    key = _resume.ResumeKey("1" * 32)
    (directory / key.filename).symlink_to(target)

    assert _resume.store(key, directory, [Segment(start=0.0, end=1.0, text="x")], {}) is None
    assert target.read_text() == "do not touch"
    assert _resume.load(key, directory) is None


def test_resume_load_refuses_a_planted_symlink(tmp_path):
    directory = tmp_path / "resume"
    directory.mkdir()
    key = _resume.ResumeKey("2" * 32)
    (directory / key.filename).symlink_to(tmp_path / "nonexistent")

    assert _resume.load(key, directory) is None


def test_run_record_is_private(tmp_path):
    record = _runlog.RunLog(session="s", started_at="2026-01-01T00:00:00", config={})
    path = _runlog.write_run_log(record, tmp_path / "runs")
    assert path is not None
    _assert_private_file(path)


def test_write_private_text_repairs_a_permissive_directory(tmp_path):
    directory = tmp_path / "loose"
    directory.mkdir(mode=0o755)
    target = directory / "entry.json"
    write_private_text(target, "{}")
    _assert_private_file(target)
    # No temp residue is left behind.
    assert [p.name for p in directory.iterdir()] == ["entry.json"]
