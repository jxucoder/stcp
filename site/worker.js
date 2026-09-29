import { gunzip, NgramLM, Converter } from "./stcp.js";

let conv = null;

async function load(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    postMessage({ type: "progress", label, got, total });
  }
  const buf = await new Blob(parts).arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  return head[0] === 0x1f && head[1] === 0x8b ? gunzip(buf) : buf; // server may already have decoded it
}

async function init() {
  const [lmBuf, tabBuf] = await Promise.all([load("data/lm.bin.gz", "lm"), load("data/tables.json.gz", "tables")]);
  postMessage({ type: "progress", label: "build" });
  const lm = new NgramLM(lmBuf);
  const tables = JSON.parse(new TextDecoder().decode(tabBuf));
  conv = new Converter(lm, tables);
  postMessage({ type: "ready", order: lm.order });
}

const ready = init().catch((e) => postMessage({ type: "error", message: String(e) }));

onmessage = async (e) => {
  await ready;
  const { id, text, mode } = e.data;
  const t0 = performance.now();
  const segs = conv.annotate(text, mode);
  postMessage({ type: "result", id, segments: segs, ms: Math.round(performance.now() - t0) });
};
