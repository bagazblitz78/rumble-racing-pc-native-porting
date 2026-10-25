"""Tests for disc analysis report generation (T1.6)."""
import pytest

from ps2toolkit.analysis.report import analyze_image, render_markdown

from tests.disc_fixtures import build_disc_with_boot


@pytest.fixture()
def image_path(tmp_path):
    p = tmp_path / "game.bin"
    p.write_bytes(build_disc_with_boot())
    return p


def test_analyze_image_basic(image_path):
    report = analyze_image(image_path)
    assert report["source"] == str(image_path)
    assert report["volume_id"] == "BOOTVOL"
    assert report["sector_size"] == 2352
    assert report["sector_count"] == 26
    names = {f["name"] for f in report["files"]}
    assert "SYSTEM.CNF;1" in names
    assert "SLUS_201.74;1" in names
    assert report["boot_elf"]["name"] == "SLUS_201.74;1"
    assert report["boot_elf"]["offset"] == 0
    assert report["boot_elf"]["size"] > 0


def test_render_markdown(image_path):
    report = analyze_image(image_path)
    md = render_markdown(report)
    assert "BOOTVOL" in md
    assert "SLUS_201.74;1" in md
    assert "SYSTEM.CNF" in md


def test_analyze_image_with_cue(tmp_path):
    from tests.disc_fixtures import build_disc_with_boot
    binp = tmp_path / "renamed.bin"
    binp.write_bytes(build_disc_with_boot())
    cue = tmp_path / "game.cue"
    cue.write_text('FILE "different-name.bin" BINARY\n  TRACK 01 MODE2/2352\n    INDEX 01 00:00:00\n',
                   encoding="utf-8")
    report = analyze_image(cue)
    assert report["sector_size"] == 2352
    assert report["boot_elf"]["name"] == "SLUS_201.74;1"


def test_analyze_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_image(tmp_path / "nope.bin")
