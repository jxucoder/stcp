// Check the browser port against the Python implementation and the gold data.
//   node scripts/verify_static.mjs            (after scripts/dump_reference.py)
import { readFileSync } from "node:fs";
import { gunzip, NgramLM, Converter } from "../site/stcp.js";

const root = new URL("..", import.meta.url);
const read = (p) => readFileSync(new URL(p, root));
const buf = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

let t0 = Date.now();
const lm = new NgramLM(await gunzip(buf(read("site/data/lm.bin.gz"))));
const tables = JSON.parse(new TextDecoder().decode(await gunzip(buf(read("site/data/tables.json.gz")))));
const conv = new Converter(lm, tables);
console.log(`load ${Date.now() - t0} ms`);

const dzb = {};
for (const line of read("data/dzb.txt").toString().replace(/^﻿/, "").split(/\r?\n/)) {
  if (!line || line.startsWith("#") || !line.includes("=")) continue;
  const [sc, tcs] = line.split("=");
  dzb[sc] = [...tcs];
}

const rows = read("results/reference.jsonl").toString().trim().split("\n").map((l) => JSON.parse(l));
for (const mode of ["none", "tw"]) {
  t0 = Date.now();
  let same = 0, correct = 0, total = 0, pyCorrect = 0;
  const diffs = [];
  for (const r of rows) {
    const js = conv.convert(r.src, mode);
    if (js === r[mode]) same++;
    else if (diffs.length < 5) diffs.push([r.src, r[mode], js]);
    if (mode !== "none") continue;
    const s = [...r.src], g = [...r.gold], h = [...js], p = [...r[mode]];
    s.forEach((c, i) => {
      if (dzb[c] && dzb[c].includes(g[i])) { total++; correct += h[i] === g[i]; pyCorrect += p[i] === g[i]; }
    });
  }
  console.log(`${mode}: ${same}/${rows.length} sentences identical to Python (${Date.now() - t0} ms)`);
  if (total) console.log(`  accuracy JS ${(100 * correct / total).toFixed(2)} vs Python ${(100 * pyCorrect / total).toFixed(2)} (n=${total})`);
  for (const d of diffs) console.log("  diff", d);
}
