"""Export everything the static (GitHub Pages) site needs into site/data/.

lm.bin.gz  - backoff n-gram LM (order <= 4) in a compact format read by site/stcp.js:
             "STCPLM1\\n", u32 header length, JSON header, then per order k:
               k = 1: int16 prob[V], int16 bow[V]
               k > 1: varint stream of (Δhi, lo or Δlo) key pairs, int16 prob[n], int16 bow[n] (k < N)
             key of ids w1..wk (16-bit each): lo = last min(2, k-1) ids, hi = the rest.
             Log10 probabilities are stored as round(1000 * x).
             Top-order n-grams that contain no character that can vary between
             hypotheses are dropped: they add the same score to every hypothesis.
tables.json.gz - character table, word table (tw mode), jieba dictionary (freq, total)
glossary.json  - explanations/examples for the proofreading popover

    uv run python scripts/export_static.py models/static/o4p5.arpa
"""
from __future__ import annotations

import gzip
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stcp.tables import DATA, load_char_table, load_word_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data"
SCALE = 1000


def varying_chars(char_table: dict[str, list[str]], word_table: dict[str, list[str]]) -> set[str]:
    """Characters that can differ between two hypotheses (or proofreading alternatives)."""
    chars = {c for cands in char_table.values() if len(cands) > 1 for cand in cands for c in cand}
    for src, cands in word_table.items():
        plain = "".join(char_table.get(c, [c])[0] for c in src)
        for cand in cands + [plain]:
            for alt in cands + [plain]:
                if len(alt) == len(cand):
                    chars.update(a for a, b in zip(cand, alt) if a != b)
                elif alt != cand:
                    chars.update(cand)
    return chars


def varint(values: np.ndarray) -> bytes:
    values = values.astype(np.uint64)
    nbytes = np.maximum(1, (np.floor(np.log2(np.maximum(values, 1))).astype(np.int64) // 7) + 1)
    nbytes[values == 0] = 1
    width = int(nbytes.max())
    out = np.zeros((len(values), width), dtype=np.uint8)
    v = values.copy()
    for i in range(width):
        out[:, i] = (v & np.uint64(0x7F)).astype(np.uint8)
        v >>= np.uint64(7)
        out[:, i] |= np.where(i < nbytes - 1, 0x80, 0).astype(np.uint8)
    mask = np.arange(width)[None, :] < nbytes[:, None]
    return out[mask].tobytes()


def q(x: list[float]) -> bytes:
    return np.clip(np.round(np.array(x) * SCALE), -32767, 32767).astype("<i2").tobytes()


def read_arpa(path: Path):
    orders: dict[int, list] = {}
    k = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("\\") and line.endswith("-grams:"):
                k = int(line[1:line.index("-")])
                orders[k] = []
                continue
            if not line or line.startswith("\\") or k == 0 or line.startswith("ngram "):
                continue
            parts = line.split("\t")
            orders[k].append((float(parts[0]), parts[1].split(" "), float(parts[2]) if len(parts) > 2 else 0.0))
    return orders


def export_lm(arpa: Path, keep_chars: set[str]) -> dict:
    orders = read_arpa(arpa)
    n = max(orders)
    assert n <= 4, "the browser format supports order <= 4"
    vocab = [w for _, (w,), _ in orders[1]]
    assert len(vocab) < 65536
    ids = {w: i for i, w in enumerate(vocab)}
    header = {"order": n, "vocab": vocab, "scale": SCALE, "counts": []}
    chunks = []
    for k in range(1, n + 1):
        rows = orders[k]
        if k == n and n > 1:
            rows = [r for r in rows if any(t in keep_chars for t in r[1])]
        header["counts"].append(len(rows))
        if k == 1:
            chunks += [q([r[0] for r in rows]), q([r[2] for r in rows])]
            continue
        arr = np.array([[ids[t] for t in r[1]] for r in rows], dtype=np.uint64)
        nlo = min(2, k - 1)
        hi = np.zeros(len(arr), dtype=np.uint64)
        for j in range(k - nlo):
            hi = (hi << np.uint64(16)) | arr[:, j]
        lo = np.zeros(len(arr), dtype=np.uint64)
        for j in range(k - nlo, k):
            lo = (lo << np.uint64(16)) | arr[:, j]
        order = np.lexsort((lo, hi))
        hi, lo = hi[order], lo[order]
        prob = np.array([r[0] for r in rows])[order]
        bow = np.array([r[2] for r in rows])[order]
        dhi = np.diff(hi, prepend=np.uint64(0))
        dlo = np.where(dhi == 0, lo - np.concatenate([[np.uint64(0)], lo[:-1]]), lo)
        chunks.append(varint(np.stack([dhi, dlo], axis=1).reshape(-1)))
        chunks.append(q(prob.tolist()))
        if k < n:
            chunks.append(q(bow.tolist()))
    head = json.dumps(header, ensure_ascii=False).encode()
    body = b"".join(struct.pack("<I", len(c)) + c for c in chunks)
    blob = b"STCPLM1\n" + struct.pack("<I", len(head)) + head + body
    (OUT / "lm.bin.gz").write_bytes(gzip.compress(blob, 9))
    return {"order": n, "counts": header["counts"], "raw_mb": len(blob) / 1e6,
            "gz_mb": (OUT / "lm.bin.gz").stat().st_size / 1e6}


def export_tables() -> tuple[dict, dict]:
    import jieba

    char_table = load_char_table()
    word_table = load_word_table("tw")
    tok = jieba.Tokenizer()
    tok.initialize()
    for w in word_table:  # identical to stcp.convert.Converter
        if len(w) > 1:
            tok.add_word(w, freq=max(tok.FREQ.get(w, 0), 3000))
    tables = {
        "char": {k: v for k, v in char_table.items()},
        "word": word_table,
        "dict": {w: f for w, f in tok.FREQ.items() if f > 0},
        "total": tok.total,
    }
    (OUT / "tables.json.gz").write_bytes(gzip.compress(json.dumps(tables, ensure_ascii=False).encode(), 9))
    return char_table, word_table


def main(arpa: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    char_table, word_table = export_tables()
    keep = varying_chars(char_table, word_table)
    info = export_lm(Path(arpa), keep)
    shutil.copy(DATA / "glossary.json", OUT / "glossary.json")
    info.update(varying_chars=len(keep), tables_mb=(OUT / "tables.json.gz").stat().st_size / 1e6, source=Path(arpa).name)
    (OUT / "manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
