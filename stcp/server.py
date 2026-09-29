"""Proofreading web interface (paper §3.2).

    uv run python -m stcp.server --lm models/char5.bin --word-mode tw [--table my_terms.txt]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .convert import Converter
from .tables import DATA

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="STCP")
converter: Converter | None = None
glossary: dict[str, dict] = json.loads((DATA / "glossary.json").read_text()) if (DATA / "glossary.json").exists() else {}


class ConvertRequest(BaseModel):
    text: str


class Annotation(BaseModel):
    source: str
    automatic: str
    corrected: str
    edits: list[dict]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/convert")
def convert(req: ConvertRequest) -> dict:
    t0 = time.time()
    segs = converter.annotate(req.text[:20000])
    chars = {c for s in segs if len(s["cands"]) > 1 for cand in s["cands"] for c in cand}
    return {
        "segments": segs,
        "glossary": {c: glossary[c] for c in chars if c in glossary},
        "ms": round(1000 * (time.time() - t0)),
    }


@app.post("/api/annotations")
def save(ann: Annotation) -> dict:
    path = ROOT / "annotations" / "annotations.jsonl"
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.time(), **ann.model_dump()}, ensure_ascii=False) + "\n")
    return {"saved": str(path.relative_to(ROOT)), "edits": len(ann.edits)}


def main() -> None:
    global converter
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default=str(ROOT / "models" / "char5.bin"))
    ap.add_argument("--word-mode", default="tw", choices=["none", "phrase", "tw"])
    ap.add_argument("--table", action="append", default=[], help="extra word table (OpenCC format)")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    converter = Converter(args.lm, word_mode=args.word_mode, custom_tables=args.table)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
