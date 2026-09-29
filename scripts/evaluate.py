"""Character-conversion evaluation (paper §4.2, task 1 of MOE-CIPSC 2013).

Overall accuracy  = #correctly converted ambiguous chars / #ambiguous chars
Macro-avg accuracy = mean over MOE-CIPSC characters of the per-character accuracy
Ambiguous chars = source positions whose Simplified char is in the MOE-CIPSC list and
whose gold Traditional char is one of that list's candidates.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import opencc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stcp.convert import Converter, MostFrequent  # noqa: E402
from stcp.tables import DATA, load_dzb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_test(n: int, seed: int = 0) -> tuple[list[str], list[str]]:
    sc = (DATA / "corpus" / "test.sc").read_text().splitlines()
    tc = (DATA / "corpus" / "test.tc").read_text().splitlines()
    dzb = load_dzb()
    # keep sentences that contain at least one evaluable ambiguous char
    idx = [i for i in range(len(sc)) if any(s in dzb and t in dzb[s] for s, t in zip(sc[i], tc[i]))]
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:n])
    return [sc[i] for i in idx], [tc[i] for i in idx]


def score(src: list[str], gold: list[str], hyp: list[str], dzb: dict[str, list[str]]) -> dict:
    per = defaultdict(lambda: [0, 0])
    confusions = Counter()
    for s_line, g_line, h_line in zip(src, gold, hyp):
        assert len(s_line) == len(h_line), (s_line, h_line)
        for s, g, h in zip(s_line, g_line, h_line):
            if s in dzb and g in dzb[s]:
                per[s][1] += 1
                per[s][0] += g == h
                if g != h:
                    confusions[f"{s}: {g}->{h}"] += 1
    correct = sum(c for c, _ in per.values())
    total = sum(t for _, t in per.values())
    return {
        "overall": 100 * correct / total,
        "macro": 100 * sum(c / t for c, t in per.values()) / len(per),
        "n_ambiguous": total,
        "n_types": len(per),
        "per_char": {k: [c, t] for k, (c, t) in sorted(per.items())},
        "top_errors": confusions.most_common(15),
    }


def train_freq() -> Counter:
    cache = ROOT / "models" / "char_freq.json"
    if cache.exists():
        return Counter(json.loads(cache.read_text()))
    freq: Counter = Counter()
    with open(DATA / "corpus" / "train.tc") as f:
        for line in f:
            freq.update(line)
    cache.write_text(json.dumps(freq, ensure_ascii=False))
    return freq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20000, help="number of test sentences")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--systems", default="opencc,opencc-tw,mostfreq,stcp2,stcp3,stcp5,stcp5-phrase")
    args = ap.parse_args()

    dzb = load_dzb()
    src, gold = load_test(args.n)
    print(f"{len(src)} test sentences, {sum(map(len, src))} chars", flush=True)

    systems = {
        "opencc": lambda: opencc.OpenCC("s2t").convert,
        "opencc-tw": lambda: opencc.OpenCC("s2tw").convert,
        "mostfreq": lambda: MostFrequent(train_freq()).convert,
        "stcp2": lambda: Converter(ROOT / "models/char2.bin", word_mode="none", beam=args.beam).convert,
        "stcp3": lambda: Converter(ROOT / "models/char3.bin", word_mode="none", beam=args.beam).convert,
        "stcp5": lambda: Converter(ROOT / "models/char5.bin", word_mode="none", beam=args.beam).convert,
        "stcp5-phrase": lambda: Converter(ROOT / "models/char5.bin", word_mode="phrase", beam=args.beam).convert,
    }
    results = {}
    for name in args.systems.split(","):
        fn = systems[name]()
        t0 = time.time()
        hyp = [fn(s) for s in src]
        # only character substitutions are scored; drop sentences where a word table changed length
        keep = [i for i in range(len(src)) if len(hyp[i]) == len(src[i])]
        r = score([src[i] for i in keep], [gold[i] for i in keep], [hyp[i] for i in keep], dzb)
        r["seconds"] = round(time.time() - t0, 1)
        r["dropped_sentences"] = len(src) - len(keep)
        results[name] = r
        print(f"{name:14s} overall {r['overall']:.2f}  macro {r['macro']:.2f}  "
              f"(n={r['n_ambiguous']}, types={r['n_types']}, {r['seconds']}s)", flush=True)
    out = ROOT / "results" / "eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
