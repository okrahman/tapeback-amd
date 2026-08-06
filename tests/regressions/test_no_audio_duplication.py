"""Regression test: re-processing a file already in the vault must not copy it again."""

from tapeback.settings import Settings
from tapeback.vault import save_audio_to_vault


def test_reprocessing_a_vault_file_does_not_duplicate_it(tmp_path):
    """Transcribing a recording that already lives in the vault copied the audio again.

    `process_file` saves its input to the vault unconditionally, and the vault never
    overwrites — so every re-run of an existing recording left another `_1.wav`,
    `_2.wav` beside it. On the project author's machine this accounted for roughly
    2.2 GB of the 5.5 GB audio directory, in raw 48 kHz stereo at ~11 MB per minute.
    """
    vault = tmp_path / "vault"
    settings = Settings(vault_path=vault)
    attachments = vault / settings.attachments_dir
    attachments.mkdir(parents=True)

    existing = attachments / "2026-08-06_12-00-00.wav"
    existing.write_bytes(b"the recording, already in the vault")

    returned = save_audio_to_vault(existing, settings, "2026-08-06_12-00-00")

    assert returned == existing
    assert sorted(p.name for p in attachments.glob("*.wav")) == ["2026-08-06_12-00-00.wav"]


def test_a_file_from_outside_the_vault_is_still_copied(tmp_path):
    vault = tmp_path / "vault"
    settings = Settings(vault_path=vault)
    source = tmp_path / "elsewhere.wav"
    source.write_bytes(b"recorded into a temp dir")

    returned = save_audio_to_vault(source, settings, "2026-08-06_12-00-00")

    assert returned == vault / settings.attachments_dir / "2026-08-06_12-00-00.wav"
    assert returned.read_bytes() == b"recorded into a temp dir"


def test_a_different_recording_with_a_taken_name_still_gets_a_suffix(tmp_path):
    """The no-overwrite guarantee must survive: distinct audio keeps its own file."""
    vault = tmp_path / "vault"
    settings = Settings(vault_path=vault)
    attachments = vault / settings.attachments_dir
    attachments.mkdir(parents=True)
    (attachments / "2026-08-06_12-00-00.wav").write_bytes(b"first recording")

    source = tmp_path / "second.wav"
    source.write_bytes(b"a genuinely different recording")

    returned = save_audio_to_vault(source, settings, "2026-08-06_12-00-00")

    assert returned.name == "2026-08-06_12-00-00_1.wav"
    assert (attachments / "2026-08-06_12-00-00.wav").read_bytes() == b"first recording"
