"""Tests for safe file extraction from disc images (T1.4)."""
import hashlib

import pytest

from ps2toolkit.disc.extract import safe_extract, safe_extract_tree, sha256_file
from ps2toolkit.disc.iso9660 import ISO9660, ISOFile
from ps2toolkit.disc.sector import RawDiscImage

from tests.disc_fixtures import HELLO_CONTENT, INNER_CONTENT, build_test_disc


@pytest.fixture()
def iso(tmp_path):
    img = tmp_path / "src" / "test.bin"
    img.parent.mkdir()
    img.write_bytes(build_test_disc())
    return ISO9660(RawDiscImage(img))


def test_extract_single_file(iso, tmp_path):
    dest = tmp_path / "out"
    out = safe_extract(iso, "HELLO.TXT;1", dest)
    assert out == dest / "HELLO.TXT;1"
    assert out.read_bytes() == HELLO_CONTENT


def test_extract_nested_file(iso, tmp_path):
    out = safe_extract(iso, "SUB/INNER.TXT;1", tmp_path / "out")
    assert out.read_bytes() == INNER_CONTENT


def test_extract_rejects_traversal_name(iso, tmp_path):
    evil = ISOFile(name="../../evil.txt", lba=22, size=len(HELLO_CONTENT), is_dir=False)
    from ps2toolkit.disc.extract import _safe_dest_name
    with pytest.raises(ValueError):
        _safe_dest_name(evil.name, tmp_path / "out")


def test_extract_tree(iso, tmp_path):
    dest = tmp_path / "tree"
    written = safe_extract_tree(iso, dest)
    names = {p.name for p in written}
    assert names == {"HELLO.TXT;1", "INNER.TXT;1"}
    hello = dest / "HELLO.TXT;1"
    assert hello.read_bytes() == HELLO_CONTENT
    inner_dir = [p for p in written if p.name == "INNER.TXT;1"][0]
    assert inner_dir.parent.name == "SUB"
    assert inner_dir.read_bytes() == INNER_CONTENT


def test_source_image_unchanged_by_extraction(iso, tmp_path):
    src = iso.disc.path
    before = sha256_file(src)
    safe_extract_tree(iso, tmp_path / "tree2")
    assert sha256_file(src) == before


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()
