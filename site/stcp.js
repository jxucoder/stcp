// STCP converter for the browser (and Node): a port of stcp/convert.py.
// t* = argmax_t P(t) over the lattice of word-table and character candidates,
// scored by a character n-gram backoff LM, searched with a recombining beam.

const SENT_SPLIT = /(?<=[。！？；\n])/u;
const RE_HAN = /([一-鿕a-zA-Z0-9+#&._%\-]+)/u; // jieba re_han_default
const RE_ENG = /^[a-zA-Z0-9]$/;

// ---------------------------------------------------------------- LM ----------
export class NgramLM {
  constructor(buffer) {
    const bytes = new Uint8Array(buffer);
    const magic = new TextDecoder().decode(bytes.subarray(0, 8));
    if (magic !== "STCPLM1\n") throw new Error("bad LM file");
    const dv = new DataView(buffer);
    const hlen = dv.getUint32(8, true);
    const head = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + hlen)));
    this.order = head.order;
    this.scale = head.scale;
    this.vocab = head.vocab;
    this.ids = new Map(head.vocab.map((w, i) => [w, i]));
    this.unk = this.ids.get("<unk>");
    this.bos = this.ids.get("<s>");
    this.eos = this.ids.get("</s>");
    let off = 12 + hlen;
    const chunk = () => {
      const n = dv.getUint32(off, true);
      const c = buffer.slice(off + 4, off + 4 + n);
      off += 4 + n;
      return c;
    };
    this.tables = [null];
    for (let k = 1; k <= this.order; k++) {
      const n = head.counts[k - 1];
      if (k === 1) {
        this.tables.push({ prob: new Int16Array(chunk()), bow: new Int16Array(chunk()) });
        continue;
      }
      const keys = new Uint8Array(chunk());
      const hi = new Uint32Array(n), lo = new Uint32Array(n);
      let p = 0, h = 0, l = 0;
      for (let i = 0; i < n; i++) {
        let v = 0, s = 1, b;
        do { b = keys[p++]; v += (b & 127) * s; s *= 128; } while (b & 128);
        let u = 0; s = 1;
        do { b = keys[p++]; u += (b & 127) * s; s *= 128; } while (b & 128);
        if (v === 0) l += u; else { h += v; l = u; }
        hi[i] = h; lo[i] = l;
      }
      const prob = new Int16Array(chunk());
      const bow = k < this.order ? new Int16Array(chunk()) : null;
      this.tables.push({ hi, lo, prob, bow });
    }
  }

  id(ch) { return this.ids.has(ch) ? this.ids.get(ch) : this.unk; }

  // index of n-gram ids[a..b) in its order's table, or -1
  find(ids, a, b) {
    const k = b - a;
    if (k === 1) return ids[a];
    const nlo = Math.min(2, k - 1);
    let hi = 0, lo = 0;
    for (let j = a; j < b - nlo; j++) hi = hi * 65536 + ids[j];
    for (let j = b - nlo; j < b; j++) lo = lo * 65536 + ids[j];
    const t = this.tables[k];
    let L = 0, R = t.hi.length - 1;
    while (L <= R) {
      const m = (L + R) >> 1, mh = t.hi[m];
      if (mh < hi || (mh === hi && t.lo[m] < lo)) L = m + 1;
      else if (mh === hi && t.lo[m] === lo) return m;
      else R = m - 1;
    }
    return -1;
  }

  // log10 P(w | hist), hist = previous ids (only the last order-1 are used)
  logp(hist, w) {
    const ids = hist.length >= this.order ? hist.slice(hist.length - this.order + 1) : hist.slice();
    ids.push(w);
    const end = ids.length;
    let backoff = 0;
    for (let a = 0; a < end; a++) {
      const k = end - a;
      const g = this.find(ids, a, end);
      if (g >= 0) return (backoff + this.tables[k].prob[g]) / this.scale;
      const c = this.find(ids, a, end - 1); // context a..end-1
      if (c >= 0) backoff += this.tables[k - 1].bow[c];
    }
    return backoff / this.scale; // unreachable: unigrams always exist
  }

  // like kenlm.Model.score(sentence, bos=True, eos=True) over characters
  score(chars) {
    const hist = [this.bos];
    let s = 0;
    for (const ch of chars) {
      if (/\s/.test(ch)) continue;
      const w = this.id(ch);
      s += this.logp(hist, w);
      hist.push(w);
    }
    return s + this.logp(hist, this.eos);
  }
}

// ---------------------------------------------------------- segmenter ---------
// jieba.cut(text, HMM=False) with the dictionary exported from Python.
export class Jieba {
  constructor(freq, total) {
    this.freq = new Map(Object.entries(freq));
    this.logTotal = Math.log(total);
    this.prefix = new Set();
    for (const w of this.freq.keys()) {
      const cs = [...w];
      for (let i = 1; i <= cs.length; i++) this.prefix.add(cs.slice(0, i).join(""));
    }
  }

  cutBlock(block) {
    const cs = [...block], n = cs.length;
    const dag = [];
    for (let k = 0; k < n; k++) {
      const ends = [];
      let frag = cs[k];
      for (let i = k; i < n && this.prefix.has(frag); ) {
        if (this.freq.get(frag)) ends.push(i);
        i++;
        frag = cs.slice(k, i + 1).join("");
      }
      dag.push(ends.length ? ends : [k]);
    }
    const route = new Array(n + 1);
    route[n] = [0, 0];
    for (let i = n - 1; i >= 0; i--) {
      let best = null;
      for (const x of dag[i]) {
        const f = this.freq.get(cs.slice(i, x + 1).join("")) || 1;
        const v = Math.log(f) - this.logTotal + route[x + 1][0];
        if (!best || v > best[0] || (v === best[0] && x > best[1])) best = [v, x];
      }
      route[i] = best;
    }
    const out = [];
    let buf = "";
    for (let x = 0; x < n; ) {
      const y = route[x][1] + 1;
      const w = cs.slice(x, y).join("");
      if (RE_ENG.test(w) && y - x === 1) buf += w;
      else { if (buf) { out.push(buf); buf = ""; } out.push(w); }
      x = y;
    }
    if (buf) out.push(buf);
    return out;
  }

  cut(text) {
    const out = [];
    for (const blk of text.split(RE_HAN)) {
      if (!blk) continue;
      if (RE_HAN.test(blk) && blk.match(RE_HAN)[0] === blk) out.push(...this.cutBlock(blk));
      else out.push(...blk); // jieba yields non-Han characters one by one (HMM=False)
    }
    return out;
  }
}

// ---------------------------------------------------------- converter ---------
export class Converter {
  constructor(lm, tables, { beam = 10 } = {}) {
    this.lm = lm;
    this.beam = beam;
    this.charTable = new Map(Object.entries(tables.char));
    this.wordTable = new Map(Object.entries(tables.word));
    this.jieba = new Jieba(tables.dict, tables.total);
  }

  lattice(text, mode) {
    const segs = [];
    const words = mode === "tw" ? this.jieba.cut(text) : [...text];
    for (const w of words) {
      if ([...w].length > 1 && this.wordTable.has(w)) {
        segs.push({ src: w, cands: this.wordTable.get(w), kind: "word" });
        continue;
      }
      for (const c of w) {
        const cands = this.charTable.get(c);
        if (cands && cands.length > 1) segs.push({ src: c, cands, kind: "char" });
        else segs.push({ src: c, cands: cands || [c], kind: "fixed" });
      }
    }
    return segs;
  }

  decode(segs) {
    const lm = this.lm, k = lm.order - 1;
    let hyps = new Map([["", { score: 0, hist: [lm.bos], choices: [] }]]);
    for (const seg of segs) {
      const next = new Map();
      for (const h of hyps.values()) {
        seg.cands.forEach((cand, i) => {
          const hist = h.hist.slice(-k);
          let score = h.score;
          for (const ch of cand) {
            if (/\s/.test(ch)) continue;
            const w = lm.id(ch);
            score += lm.logp(hist, w);
            hist.push(w);
            if (hist.length > k) hist.shift();
          }
          const key = hist.join(",");
          const old = next.get(key);
          if (!old || old.score < score) next.set(key, { score, hist, choices: [...h.choices, i] });
        });
      }
      hyps = next.size > this.beam
        ? new Map([...next.entries()].sort((a, b) => b[1].score - a[1].score).slice(0, this.beam))
        : next;
    }
    let best = null, bestScore = -Infinity;
    for (const h of hyps.values()) {
      const s = h.score + lm.logp(h.hist, lm.eos);
      if (s > bestScore) { bestScore = s; best = h; }
    }
    segs.forEach((seg, i) => (seg.out = seg.cands[best.choices[i]]));
  }

  sentences(text) { return text.split(SENT_SPLIT).filter(Boolean); }

  convert(text, mode = "tw") {
    let out = "";
    for (const part of this.sentences(text)) {
      const segs = this.lattice(part, mode);
      this.decode(segs);
      out += segs.map((s) => s.out).join("");
    }
    return out;
  }

  // segments with posteriors over alternatives, for the proofreading UI
  annotate(text, mode = "tw") {
    const result = [];
    for (const part of this.sentences(text)) {
      const segs = this.lattice(part, mode);
      this.decode(segs);
      for (const s of segs) {
        if (s.kind === "word") {
          const plain = [...s.src].map((c) => (this.charTable.get(c) || [c])[0]).join("");
          if (!s.cands.includes(plain)) s.cands = [...s.cands, plain];
        }
      }
      const outs = segs.map((s) => s.out);
      segs.forEach((s, i) => {
        s.probs = [];
        if (s.cands.length < 2) return;
        const lps = s.cands.map((c) => { outs[i] = c; return this.lm.score(outs.join("")); });
        outs[i] = s.out;
        const m = Math.max(...lps);
        const ps = lps.map((x) => 10 ** (x - m));
        const z = ps.reduce((a, b) => a + b, 0);
        s.probs = ps.map((p) => p / z);
      });
      result.push(...segs);
    }
    return result;
  }
}

// ------------------------------------------------------------- loading --------
export async function gunzip(buffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).arrayBuffer();
}
