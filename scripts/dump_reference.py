"""Python reference outputs for checking the browser port (scripts/verify_static.mjs).

    uv run python scripts/dump_reference.py models/static/o3p5.bin -n 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_test  # noqa: E402
from stcp.convert import Converter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("lm")
ap.add_argument("-n", type=int, default=3000)
args = ap.parse_args()

src, gold = load_test(args.n)
none = Converter(args.lm, word_mode="none")
tw = Converter(args.lm, word_mode="tw")
out = ROOT / "results" / "reference.jsonl"
with open(out, "w") as f:
    for s, g in zip(src, gold):
        f.write(json.dumps({"src": s, "gold": g, "none": none.convert(s), "tw": tw.convert(s)}, ensure_ascii=False) + "\n")
print("wrote", out)
