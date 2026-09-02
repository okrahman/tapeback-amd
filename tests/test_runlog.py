"""Unit tests for per-run metadata records."""

import json

import pytest
from pydantic import SecretStr

from tapeback._runlog import (
    MAX_RUN_RECORDS,
    RECORDED_SETTINGS,
    RunLog,
    _prune_old_records,
    default_run_log_dir,
    redact_text,
    run_log,
    write_run_log,
)
from tapeback.settings import Settings


def _read_only_record(directory):
    """Return the single JSON record written into directory."""
    records = list(directory.glob("*.json"))
    assert len(records) == 1
    return json.loads(records[0].read_text())


def test_run_log_records_config_events_and_outcome(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    printed: list[str] = []
    with run_log("2026-08-05_12-00-00", settings, printed.append) as report:
        report("Stage 'merge' took 1.0s")
        report("Whisper: large-v3-turbo on cuda/float16")

    # The reporter both prints and captures — neither replaces the other.
    assert printed == ["Stage 'merge' took 1.0s", "Whisper: large-v3-turbo on cuda/float16"]

    record = _read_only_record(tmp_path / "runs")
    assert record["session"] == "2026-08-05_12-00-00"
    assert record["outcome"] == "completed"
    assert record["error"] is None
    assert record["events"] == [
        "Stage 'merge' took 1.0s",
        "Whisper: large-v3-turbo on cuda/float16",
    ]
    assert record["config"]["whisper_model"] == "large-v3-turbo"
    assert record["config"]["chunk_length"] == 30
    assert record["config"]["transcription_backend"] == "faster-whisper"
    assert record["config"]["lemonade_url"] == "http://127.0.0.1:13305"
    assert record["config"]["lemonade_model"] == "Whisper-Large-v3-Turbo"
    assert record["config"]["lemonade_chunk_seconds"] == 300.0
    assert record["config"]["lemonade_overlap_seconds"] == 2.0
    assert record["finished_at"] is not None


def test_run_log_records_lemonade_config(tmp_path):
    settings = Settings(
        vault_path=tmp_path / "vault",
        run_log_dir=tmp_path / "runs",
        transcription_backend="lemonade",
        lemonade_url="http://localhost:8000",
        lemonade_model="custom-whisper",
        lemonade_chunk_seconds=120.0,
        lemonade_overlap_seconds=3.0,
        lemonade_api_key=SecretStr("sk-lemon-secret"),
    )

    with run_log("lemon-session", settings, lambda _m: None) as report:
        report("Lemonade transcription started")

    record = _read_only_record(tmp_path / "runs")
    assert record["config"]["transcription_backend"] == "lemonade"
    assert record["config"]["lemonade_url"] == "http://localhost:8000"
    assert record["config"]["lemonade_model"] == "custom-whisper"
    assert record["config"]["lemonade_chunk_seconds"] == 120.0
    assert record["config"]["lemonade_overlap_seconds"] == 3.0
    assert "lemonade_api_key" not in record["config"]
    assert "sk-lemon-secret" not in json.dumps(record)


def test_run_log_never_records_credentials(tmp_path):
    """P0: a post-mortem file that leaks credentials is worse than no file at all."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        run_log_dir=tmp_path / "runs",
        hf_token=SecretStr("hf_SUPERSECRET"),
        llm_api_key=SecretStr("sk-SUPERSECRET"),
    )

    with run_log("session", settings, lambda _m: None) as report:
        report("working")

    raw = next((tmp_path / "runs").glob("*.json")).read_text()
    assert "SUPERSECRET" not in raw
    assert "hf_token" not in raw
    assert "llm_api_key" not in raw


def test_recorded_settings_contains_no_secret_fields():
    """Guard the allow-list itself: adding a SecretStr field here must fail the build."""
    secret_fields = {
        name
        for name, info in Settings.model_fields.items()
        if info.annotation is not None and "Secret" in str(info.annotation)
    }
    assert secret_fields, "expected Settings to carry at least one SecretStr field"
    assert secret_fields.isdisjoint(RECORDED_SETTINGS)


def test_run_log_marks_keyboard_interrupt_as_aborted_and_reraises(tmp_path):
    """Ctrl+C is exactly the case worth a record — the interrupt must still propagate."""
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with pytest.raises(KeyboardInterrupt), run_log("session", settings, lambda _m: None) as report:
        report("transcribing")
        raise KeyboardInterrupt

    record = _read_only_record(tmp_path / "runs")
    assert record["outcome"] == "aborted"
    assert record["error"] is None
    assert record["events"] == ["transcribing"]


def test_run_log_marks_exception_as_failed_and_reraises(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with (
        pytest.raises(RuntimeError, match="cuBLAS"),
        run_log("session", settings, lambda _m: None) as report,
    ):
        report("transcribing")
        raise RuntimeError("cuBLAS not found")

    record = _read_only_record(tmp_path / "runs")
    assert record["outcome"] == "failed"
    assert record["error"] == "RuntimeError: cuBLAS not found"


def test_run_log_disabled_writes_nothing_and_passes_reporter_through(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs", run_log=False)

    printed: list[str] = []
    with run_log("session", settings, printed.append) as report:
        # Passed straight through — not wrapped in a capturing reporter.
        assert report == printed.append
        report("working")

    assert printed == ["working"]
    assert not (tmp_path / "runs").exists()


def test_default_run_log_dir_honours_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_run_log_dir() == tmp_path / "xdg" / "tapeback" / "runs"


def test_default_run_log_dir_falls_back_to_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("tapeback._runlog.Path.home", lambda: tmp_path / "home")
    assert default_run_log_dir() == tmp_path / "home" / ".local" / "share" / "tapeback" / "runs"


@pytest.mark.parametrize(
    ("existing", "keep", "expected_remaining"),
    [
        # Boundary: exactly at the limit keeps everything.
        (3, 3, 3),
        (2, 3, 2),
        (4, 3, 3),
    ],
)
def test_prune_old_records_boundaries(tmp_path, existing, keep, expected_remaining):
    directory = tmp_path / "runs"
    directory.mkdir()
    for index in range(existing):
        (directory / f"2026-08-0{index}_run.json").write_text("{}")

    _prune_old_records(directory, keep=keep)

    remaining = sorted(p.name for p in directory.glob("*.json"))
    assert len(remaining) == expected_remaining
    # Pruning drops the oldest first, so the newest name always survives.
    assert remaining[-1] == f"2026-08-0{existing - 1}_run.json"


def test_write_run_log_keeps_directory_bounded(tmp_path):
    """write_run_log prunes after writing, so the directory cannot grow forever."""
    directory = tmp_path / "runs"
    directory.mkdir()
    for index in range(MAX_RUN_RECORDS + 5):
        (directory / f"2026-01-{index:04d}_old.json").write_text("{}")

    record = RunLog(session="new", started_at="2099-01-01T00:00:00+00:00", config={})
    written = write_run_log(record, directory)

    assert written is not None
    assert len(list(directory.glob("*.json"))) == MAX_RUN_RECORDS
    # The record just written is the newest and must not be the one pruned.
    assert written.exists()


def test_run_log_defaults_to_unknown_outcome(tmp_path):
    """An exit neither branch classified (SystemExit, killed process) must not read as completed."""
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with pytest.raises(SystemExit), run_log("session", settings, lambda _m: None) as report:
        report("working")
        raise SystemExit(1)

    record = _read_only_record(tmp_path / "runs")
    assert record["outcome"] == "unknown"


def test_write_run_log_returns_none_when_directory_is_unwritable(tmp_path):
    """A failed diagnostic write must never take down the run that produced it."""
    blocker = tmp_path / "runs"
    blocker.write_text("not a directory")

    record = RunLog(session="s", started_at="2026-08-05T00:00:00+00:00", config={})
    assert write_run_log(record, blocker) is None


def test_run_log_redacts_reflected_api_key_from_events_and_error(tmp_path):
    """A remote error can reflect the received Authorization header; the run
    record is the final persistence boundary and must not hold the credential."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        run_log_dir=tmp_path / "runs",
        lemonade_api_key=SecretStr("sk-lemonade-secret"),
        hf_token=SecretStr("hf-supersecret"),
        llm_api_key=SecretStr("sk-llm-secret"),
    )

    printed: list[str] = []
    with pytest.raises(RuntimeError), run_log("session", settings, printed.append) as report:
        report("Lemonade server failure (HTTP 502): Bearer sk-lemonade-secret")
        raise RuntimeError("upstream said: Bearer sk-llm-secret with hf-supersecret")

    record = _read_only_record(tmp_path / "runs")
    raw = json.dumps(record)
    assert "sk-lemonade-secret" not in raw
    assert "sk-llm-secret" not in raw
    assert "hf-supersecret" not in raw
    assert record["events"] == ["Lemonade server failure (HTTP 502): Bearer [redacted]"]
    assert "[redacted]" in record["error"]


def test_run_log_strips_terminal_control_characters(tmp_path):
    """ESC sequences must never survive into a file the user may cat."""
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with pytest.raises(RuntimeError), run_log("session", settings, lambda _m: None) as report:
        report("\x1b]0;evil\x07 status line")
        raise RuntimeError("boom\x1b[31m")

    record = _read_only_record(tmp_path / "runs")
    assert "\x1b" not in json.dumps(record)
    assert "\x07" not in json.dumps(record)
    assert record["events"] == ["]0;evil status line"]


def test_run_log_without_secrets_behaves_as_before(tmp_path):
    """Empty redactions are a no-op — plain status lines are stored verbatim."""
    settings = Settings(vault_path=tmp_path / "vault", run_log_dir=tmp_path / "runs")

    with run_log("session", settings, lambda _m: None) as report:
        report("Stage 'merge' took 1.0s")

    record = _read_only_record(tmp_path / "runs")
    assert record["events"] == ["Stage 'merge' took 1.0s"]


def test_redact_text_overlapping_secrets_longest_first():
    """Prefix and suffix overlapping secrets must be fully redacted without leaving fragments."""
    # Prefix overlap: "sk-live" is a prefix of "sk-live-prod"
    redactions = ("sk-live", "sk-live-prod")
    text = "Authorization: Bearer sk-live-prod then sk-live"
    assert redact_text(text, redactions) == "Authorization: Bearer [redacted] then [redacted]"

    # Suffix overlap: "secret" is a suffix of "super-secret"
    redactions = ("secret", "super-secret")
    text = "got super-secret and secret"
    assert redact_text(text, redactions) == "got [redacted] and [redacted]"

    # Infix / substring overlap + duplicate entries
    redactions = ("key", "my-key-prod", "key", "")
    text = "tokens: my-key-prod, key"
    assert redact_text(text, redactions) == "tokens: [redacted], [redacted]"


def test_run_log_redacts_overlapping_secrets_in_events_and_error(tmp_path):
    """Overlapping configured keys in Settings must not leave reconstructable
    fragments in events or errors.
    """
    settings = Settings(
        vault_path=tmp_path / "vault",
        run_log_dir=tmp_path / "runs",
        lemonade_api_key=SecretStr("sk-live"),
        hf_token=SecretStr("sk-live-prod"),
        llm_api_key=SecretStr("sk-live-prod-extended"),
    )

    with (
        pytest.raises(RuntimeError),
        run_log("session", settings, lambda _m: None) as report,
    ):
        report("Event with sk-live-prod-extended and sk-live-prod and sk-live")
        raise RuntimeError("Failed on sk-live-prod-extended with sk-live prefix")

    record = _read_only_record(tmp_path / "runs")
    raw = json.dumps(record)

    assert "sk-live" not in raw
    assert "-prod" not in raw
    assert "-extended" not in raw
    assert record["events"] == ["Event with [redacted] and [redacted] and [redacted]"]
    assert record["error"] == "RuntimeError: Failed on [redacted] with [redacted] prefix"
