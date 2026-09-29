# STCP reproduction

**在线使用 / Live demo: https://jxucoder.github.io/stcp/** (runs entirely in your browser)

Reproduction of *STCP: Simplified-Traditional Chinese Conversion and Proofreading*
(Xu, Ma, Tsai, Hovy, IJCNLP 2017 demos, [I17-3016](https://aclanthology.org/I17-3016.pdf)).

## 中文简介

STCP 是一个简体转繁体的工具，附带人工校对界面。它复现了 2017 年 IJCNLP 的系统演示论文
[STCP: Simplified-Traditional Chinese Conversion and Proofreading](https://aclanthology.org/I17-3016.pdf)。

简转繁最难的是「一简对多繁」：同一个简体字，在不同上下文里要转成不同的繁体字，
比如「发」可以是「發」（发展）或「髮」（头发），「了」可以是「了」或「瞭」（了解）。
STCP 的做法是：

1. 用 jieba 分词，在词表里查找整词转换，比如台湾用语「软件 → 軟體」「笔记本电脑 → 筆記型電腦」；
2. 其余的字列出所有繁体候选；
3. 用字级 n-gram 语言模型给所有组合打分，取概率最高的一句。

自动转换不可能百分之百正确，所以界面会把有歧义的字和词标出来。点一下就能看到每个候选的概率、拼音、释义和常见搭配，
选错了可以直接改，改完的结果可以导出成 JSONL 标注数据。

**在线使用：** https://jxucoder.github.io/stcp/ 。模型在浏览器本地运行，文字不会上传到任何服务器，首次打开需要下载约 13 MB。

### 和论文的对比

| | 论文 OpenCC | 论文 STCP | 本复现 OpenCC | 本复现 STCP |
|---|---|---|---|---|
| 总体准确率 | 98.90 | 99.64 | 93.50 | 97.61 |
| 宏平均准确率 | 91.75 | 95.73 | 92.66 | 96.72 |

论文的核心结论复现成功：STCP 在两项指标上都明显好于规则系统 OpenCC，宏平均的提升幅度（+3.98 对 +4.06）也几乎一样。
总体准确率两边都比论文低，主要因为论文用的台湾中央社新闻语料需要 LDC 授权，这里改用了维基百科繁体文章，
而维基百科的写法不统一（比如「台」和「臺」混用），参考答案本身就有噪声。详见下文 *Deviations from the paper*。

### 本地运行

网站版（静态文件，无需后端）：

```bash
python3 -m http.server 8010 -d site
```

服务器版（5-gram 模型，准确率更高，需要先按下文 *Reproduce* 训练模型）：

```bash
uv run python -m stcp.server --lm models/char5.bin
```

## What is implemented

| Paper | Here |
|---|---|
| §3.1.1 word conversion: jieba + word look-up table | `stcp/convert.py` `Converter.lattice`; tables from OpenCC `STPhrases` (+ `TWPhrases` for Taiwan vocabulary, e.g. 软件→軟體) |
| §3.1.2 character conversion: candidate set T of all char combinations | lattice of OpenCC `STCharacters` ∪ MOE-CIPSC candidates |
| §3.1.3 `t* = argmax P(t)`, char 5-gram KenLM | beam search over the lattice with hypothesis recombination on the last n−1 chars (`Converter.decode`) |
| §3.2 proofreading web interface | `stcp/server.py` + `stcp/static/index.html`: highlights ambiguous chars and mapped words, shows alternatives with LM posteriors, Unihan glosses, and corpus examples; supports edits and saves annotations as JSONL |
| Customization (import tables and LM) | `--lm`, `--word-mode`, `--table my_terms.txt` (OpenCC format) |
| §4 evaluation: MOE-CIPSC task 1, overall and macro accuracy | `scripts/evaluate.py` using the original list `data/dzb.txt` |

Paper example: `我了解云端软件` → `我瞭解雲端軟體` ✔

## Deviations from the paper

- **Corpus.** The paper uses Gigaword 5 CNA (LDC, licensed). Here it is replaced by zhwiki (`wikimedia/wikipedia` 20231101.zh) paragraphs that were
  written natively in Traditional Chinese: no simplified-only characters and at least 8% traditional-only characters. Sentences are deduplicated, and the
  split is 80/20 by article: 232M training chars. As in the paper, the source side is a character-level T→S conversion (OpenCC `TSCharacters`).
- The 5-gram LM prunes singleton 4- and 5-grams (`--prune 0 0 1 1`) to keep it at 593 MB.
- **XMUCC** (jf.cloudtranslation.cc) no longer responds, so it is not compared.
- The test set is 20,000 random held-out sentences (66,385 ambiguous tokens, 132 of the 135 MOE characters occur). Only positions whose gold character is one of the MOE candidates are scored.

## Results (`results/eval.json`)

| System | Overall acc. | Macro-avg acc. | Overall, excl. 台 | Macro, excl. 台 |
|---|---|---|---|---|
| OpenCC s2t | 93.50 | 92.66 | 96.22 | 93.14 |
| OpenCC s2tw | 93.99 | 92.21 | 96.74 | 92.68 |
| Most-frequent candidate | 92.31 | 85.85 | 93.31 | 85.98 |
| STCP, 2-gram LM | 96.90 | 96.15 | 98.11 | 96.35 |
| STCP, 3-gram LM | 97.55 | 96.70 | 98.45 | 96.86 |
| **STCP, 5-gram LM** | **97.61** | **96.72** | **98.54** | **96.87** |
| STCP 5-gram + STPhrases word table | 95.13 | 95.63 | 97.65 | 96.07 |
| *Paper: OpenCC / XMUCC / STCP* | *98.90 / 99.81 / 99.64* | *91.75 / 96.98 / 95.73* | | |

- The paper's main claim holds: the LM system clearly beats rule-based OpenCC on both metrics (+4.1 overall, +4.1 macro).
  The 5-gram LM beats the bigram LM that Li et al. (2010) used.
- Absolute overall accuracy is lower than the paper's because Wikipedia's gold text mixes orthographic conventions within the MOE list (台/臺, 布/佈, 線/綫, 裡/裏).
  Newswire uses one house style. 台 alone is 4% of test tokens, and OpenCC always outputs 臺, so it gets 30.8% on 台.
- The forced STPhrases word table hurts on this data, because it hard-codes one convention (e.g. 台→臺, 周→週). That is why the character evaluation uses the LM without word tables.
  The interface defaults to `tw` mode, where word tables handle regional vocabulary such as 软件→軟體 and 笔记本电脑→筆記型電腦.

## Static site (GitHub Pages)

`site/` is a server-free build of the proofreading interface: the converter is ported to JavaScript
(`site/stcp.js`, run in a Web Worker) and the LM is downloaded once (13 MB total, then cached by the browser).

- LM: character 3-gram, `lmplz -o 3 --prune 0 5 10`, packed by `scripts/export_static.py` into a varint/int16 format.
  Top-order n-grams without any character that can differ between hypotheses are dropped, which does not change any decision.
  On the 20,000-sentence test set it scores 97.37 overall / 96.06 macro, vs 97.61 / 96.72 for the 593 MB 5-gram:

| Pruned LM | Overall | Macro |
|---|---|---|
| 3-gram `--prune 0 5 10` (shipped, 11 MB gz) | 97.37 | 96.06 |
| 3-gram `--prune 0 2 3` | 97.48 | 96.22 |
| 4-gram `--prune 0 5 10 20` | 97.30 | 96.07 |
| 4-gram `--prune 0 2 3 5` | 97.47 | 96.21 |

- Segmentation re-implements `jieba.cut(HMM=False)` with the exported dictionary. `scripts/verify_static.mjs` checks the
  port against Python: 3,000/3,000 test sentences give identical output in both `none` and `tw` modes.
- Annotations are exported as a JSONL download instead of being posted to a server.
- `.github/workflows/pages.yml` publishes `site/` on every push to `main` (Settings → Pages → Source: GitHub Actions).

```bash
lmplz -o 3 --prune 0 5 10 < data/corpus/train.chars > models/static/o3p5.arpa
uv run python scripts/export_static.py models/static/o3p5.arpa
uv run python scripts/dump_reference.py models/static/o3p5.bin -n 3000 && node scripts/verify_static.mjs
python3 -m http.server 8010 -d site     # http://localhost:8010
```

## Reproduce

```bash
uv sync
brew install boost eigen   # for KenLM's lmplz
git clone https://github.com/kpu/kenlm third_party/kenlm && cmake -S third_party/kenlm -B third_party/kenlm/build -DCMAKE_BUILD_TYPE=Release && make -C third_party/kenlm/build -j lmplz build_binary
# data/dzb.txt: http://bj.bcebos.com/cips-upload/dzb.txt ; data/opencc/*.txt: OpenCC data/dictionary
uv run python -c "from huggingface_hub import snapshot_download as s; s('wikimedia/wikipedia', repo_type='dataset', allow_patterns='20231101.zh/*', local_dir='data/raw')"
uv run python scripts/prepare_corpus.py
python3 -c "import sys; [print(' '.join(l.strip())) for l in open('data/corpus/train.tc')]" > data/corpus/train.chars
third_party/kenlm/build/bin/lmplz -o 5 -S 40% --prune 0 0 1 1 < data/corpus/train.chars > models/char5.arpa
third_party/kenlm/build/bin/build_binary trie models/char5.arpa models/char5.bin
uv run python scripts/build_glossary.py path/to/Unihan_Readings.txt
uv run python scripts/evaluate.py -n 20000
uv run python -m stcp.server --lm models/char5.bin      # http://127.0.0.1:8002
```

## Data and third-party resources

- `data/opencc/`: dictionaries from [OpenCC](https://github.com/BYVoid/OpenCC) (Apache-2.0).
- `data/dzb.txt`: MOE-CIPSC 2013 evaluation character list, from `http://bj.bcebos.com/cips-upload/dzb.txt`.
- `data/glossary.json`: pinyin and glosses from the [Unicode Unihan database](https://www.unicode.org/charts/unihan.html) (Unicode License); examples from the training corpus.
- `site/data/tables.json.gz`: includes the [jieba](https://github.com/fxsjy/jieba) dictionary (MIT).
- `site/data/lm.bin.gz`: trained on Chinese Wikipedia text (CC BY-SA 4.0).

## License

Code: [MIT](LICENSE). Bundled data keeps the licenses listed above.

## Contact

Jiarui Xu · [jiarui.c.xu@gmail.com](mailto:jiarui.c.xu@gmail.com)
