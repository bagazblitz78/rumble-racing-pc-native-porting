"""Tests for .cue sheet parsing and bin-file resolution (T1.1)."""
import pytest

from ps2toolkit.disc.cue import CueFile, CueTrack, parse_cue, resolve_bin_path, msf_to_frames

SIMPLE_CUE = """FILE "My Game (USA).bin" BINARY
  TRACK 01 MODE2/2352
    INDEX 01 00:00:00
"""

MULTI_CUE = """FILE "game.bin" BINARY
  TRACK 01 MODE2/2352
    INDEX 00 00:00:00
    INDEX 01 00:02:00
  TRACK 02 AUDIO
    INDEX 00 10:00:00
    INDEX 01 10:02:25
"""


def test_parse_simple_cue(tmp_path):
    p = tmp_path / "game.cue"
    p.write_text(SIMPLE_CUE, encoding="utf-8")
    cues = parse_cue(p)
    assert len(cues) == 1
    cue = cues[0]
    assert cue.filename == "My Game (USA).bin"
    assert cue.file_type == "BINARY"
    assert len(cue.tracks) == 1
    t = cue.tracks[0]
    assert t.number == 1
    assert t.mode == "MODE2/2352"
    assert t.sector_size == 2352
    assert t.indexes[1] == "00:00:00"


def test_parse_multi_track_cue(tmp_path):
    p = tmp_path / "game.cue"
    p.write_text(MULTI_CUE, encoding="utf-8")
    cue = parse_cue(p)[0]
    assert len(cue.tracks) == 2
    assert cue.tracks[0].indexes[0] == "00:00:00"
    assert cue.tracks[0].indexes[1] == "00:02:00"
    assert cue.tracks[1].mode == "AUDIO"
    assert cue.tracks[1].indexes[1] == "10:02:25"


def test_sector_size_from_mode():
    p = None  # covered via parse below
    cue_text = 'FILE "a.bin" BINARY\n  TRACK 01 MODE1/2048\n    INDEX 01 00:00:00\n'
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".cue", delete=False, encoding="utf-8") as f:
        f.write(cue_text)
        name = f.name
    try:
        cue = parse_cue(name)[0]
        assert cue.tracks[0].sector_size == 2048
        assert cue.tracks[0].mode == "MODE1/2048"
    finally:
        os.unlink(name)


def test_msf_to_frames():
    assert msf_to_frames("00:00:00") == 0
    assert msf_to_frames("00:02:00") == 150  # 2 seconds pregap in frames
    assert msf_to_frames("01:02:03") == (1 * 60 + 2) * 75 + 3


def test_resolve_exact_filename(tmp_path):
    binf = tmp_path / "My Game (USA).bin"
    binf.write_bytes(b"\x00" * 16)
    cue = tmp_path / "game.cue"
    cue.write_text(SIMPLE_CUE, encoding="utf-8")
    resolved = resolve_bin_path(cue, "My Game (USA).bin")
    assert resolved == binf


def test_resolve_falls_back_to_sibling_bin(tmp_path):
    # cue references "My Game (USA).bin" but only "My Game.bin" exists
    binf = tmp_path / "My Game.bin"
    binf.write_bytes(b"\x00" * 16)
    cue = tmp_path / "game.cue"
    cue.write_text(SIMPLE_CUE, encoding="utf-8")
    resolved = resolve_bin_path(cue, "My Game (USA).bin")
    assert resolved == binf


def test_resolve_missing_raises(tmp_path):
    cue = tmp_path / "game.cue"
    cue.write_text(SIMPLE_CUE, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_bin_path(cue, "My Game (USA).bin")
