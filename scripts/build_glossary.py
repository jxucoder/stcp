"""Explanations and examples for the proofreading interface.

For each Traditional candidate of an ambiguous Simplified character: pinyin and
English gloss from Unihan, plus its most frequent two-character contexts in the
training corpus (e.g. 瞭 -> 瞭解, 明瞭).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stcp.tables import DATA, load_char_table  # noqa: E402


def main(unihan: str, max_lines: int = 1_500_000) -> None:
    table = load_char_table()
    targets = {t for cands in table.values() if len(cands) > 1 for t in cands}

    info: dict[str, dict] = defaultdict(dict)
    for line in Path(unihan).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line:
            continue
        cp, field, value = line.split("\t", 2)
        ch = chr(int(cp[2:], 16))
        if ch in targets and field in ("kDefinition", "kMandarin"):
            info[ch][field] = value

    bigrams: dict[str, Counter] = defaultdict(Counter)
    with open(DATA / "corpus" / "train.tc") as f:
        for n, line in enumerate(f):
            if n >= max_lines:
                break
            for a, b in zip(line, line[1:]):
                if a in targets and "一" <= b <= "鿿":
                    bigrams[a][a + b] += 1
                if b in targets and "一" <= a <= "鿿":
                    bigrams[b][a + b] += 1

    out = {
        ch: {
            "pinyin": info[ch].get("kMandarin", ""),
            "gloss": info[ch].get("kDefinition", ""),
            "examples": [w for w, _ in bigrams[ch].most_common(6)],
        }
        for ch in sorted(targets)
    }
    (DATA / "glossary.json").write_text(json.dumps(out, ensure_ascii=False))
    print(len(out), "entries;", out.get("瞭"), out.get("了"))


if __name__ == "__main__":
    main(sys.argv[1])
