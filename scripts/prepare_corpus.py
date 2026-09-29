"""Build a Traditional/Simplified parallel corpus (stand-in for Gigaword CNA, which is LDC-licensed).

1. Keep zhwiki paragraphs that were natively written in Traditional Chinese
   (no simplified-only characters, a solid share of traditional-only characters).
2. Split into sentences; split articles 80/20 into train/test at random.
3. Source side = character-level Traditional -> Simplified conversion (paper §4.1).
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stcp.tables import DATA, _read_opencc, load_ts_char_table  # noqa: E402

OUT = DATA / "corpus"
HAN = re.compile(r"[一-鿿]")
MARKUP = re.compile(r"-\{[^{}]*\}-|\{\{[^{}]*\}\}")
SENT_END = re.compile(r"(?<=[。！？；])")


def main(max_train_chars: int = 300_000_000) -> None:
    st = _read_opencc(DATA / "opencc" / "STCharacters.txt")
    ts = load_ts_char_table()
    sc_only = {k for k, v in st.items() if k not in v}
    tc_only = {k for k, v in ts.items() if k != v}

    rng = random.Random(2017)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {name: (open(OUT / f"{name}.tc", "w"), open(OUT / f"{name}.sc", "w")) for name in ("train", "test")}
    stats = {"train": 0, "test": 0}
    seen: set[int] = set()

    for shard in sorted((DATA / "raw").rglob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(columns=["text"], batch_size=2000):
            for text in batch.column(0).to_pylist():
                split = "test" if rng.random() < 0.2 else "train"
                if split == "train" and stats["train"] >= max_train_chars:
                    continue
                for para in text.split("\n"):
                    hans = HAN.findall(para)
                    if len(hans) < 20:
                        continue
                    if any(c in sc_only for c in hans):
                        continue
                    if sum(c in tc_only for c in hans) < max(3, 0.08 * len(hans)):
                        continue
                    para = MARKUP.sub("", para).replace(" ", "").replace("　", "")
                    for sent in SENT_END.split(para):
                        if len(HAN.findall(sent)) < 8 or len(sent) > 300:
                            continue
                        h = hash(sent)
                        if h in seen:
                            continue
                        seen.add(h)
                        tc_f, sc_f = files[split]
                        tc_f.write(sent + "\n")
                        sc_f.write("".join(ts.get(c, c) for c in sent) + "\n")
                        stats[split] += len(sent)
            print(shard.name, stats, flush=True)
    for a, b in files.values():
        a.close()
        b.close()
    print("done", stats)


if __name__ == "__main__":
    main()
