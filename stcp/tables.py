"""Look-up tables: character mappings, word mappings, and the MOE-CIPSC ambiguous list."""
from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _read_opencc(path: Path) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, vals = line.split("\t", 1)
        table[key] = vals.split()
    return table


def load_dzb(path: Path = DATA / "dzb.txt") -> dict[str, list[str]]:
    """MOE-CIPSC 2013 list: simplified char -> candidate traditional chars."""
    table = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        sc, tcs = line.split("=", 1)
        table[sc] = list(dict.fromkeys(tcs))
    return table


def load_char_table(extra: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Simplified -> Traditional character candidates (OpenCC STCharacters ∪ MOE list)."""
    table = _read_opencc(DATA / "opencc" / "STCharacters.txt")
    for sc, tcs in (extra if extra is not None else load_dzb()).items():
        table[sc] = list(dict.fromkeys(table.get(sc, []) + tcs))
    return table


def load_ts_char_table() -> dict[str, str]:
    """Traditional -> Simplified, character level (first candidate)."""
    return {k: v[0] for k, v in _read_opencc(DATA / "opencc" / "TSCharacters.txt").items()}


def load_word_table(mode: str) -> dict[str, list[str]]:
    """Word look-up table keyed by Simplified word.

    mode: "none"   - no word conversion
          "phrase" - OpenCC STPhrases (character disambiguation phrases)
          "tw"     - STPhrases + Taiwan vocabulary (TWPhrases, e.g. 软件 -> 軟體)
    """
    if mode == "none":
        return {}
    table = _read_opencc(DATA / "opencc" / "STPhrases.txt")
    if mode == "tw":
        ts = load_ts_char_table()
        for tc_word, tw_words in _read_opencc(DATA / "opencc" / "TWPhrases.txt").items():
            sc_word = "".join(ts.get(c, c) for c in tc_word)
            table[sc_word] = list(dict.fromkeys(tw_words + table.get(sc_word, [])))
    return table


def load_custom_table(path: str | Path) -> dict[str, list[str]]:
    """User table: one mapping per line, `source<TAB>target1 target2 ...` (OpenCC format)."""
    return _read_opencc(Path(path))
