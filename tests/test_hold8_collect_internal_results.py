from pathlib import Path

from tools.aaai27_hold8.collect_internal_results import _scoped_hash


def test_scoped_hash_treats_lf_and_crlf_as_equivalent(tmp_path: Path) -> None:
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    (lf / "sample.py").write_bytes(b"print('a')\nprint('b')\n")
    (crlf / "sample.py").write_bytes(b"print('a')\r\nprint('b')\r\n")

    assert _scoped_hash(lf, ["sample.py"]) == _scoped_hash(crlf, ["sample.py"])
