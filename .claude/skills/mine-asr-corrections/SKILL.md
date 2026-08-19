---
name: mine-asr-corrections
description: >
  Build and extend the recognizer's error table for whisper_input — a list of
  (correct term ← how it actually got heard ← how often), mined from real
  dictation history by comparing what the user's transcripts produced against
  what Claude wrote back. Use whenever the task touches ASR vocabulary accuracy
  on this repo: 「识别不准」「这个词老听错」「上下文偏置能不能再强一点」「词库」
  「长期积累的词表」「把我说的和 Claude 写的对一下」, adding a new term to the
  biasing list, deciding what `lexicon_corrections` should inject, judging
  whether a mined pair is a real error, or building/extending post-hoc output
  correction. Also use it BEFORE hand-writing any term list or find-replace rule
  for ASR output — the whole point is that the list should come from counted
  evidence, and a hand-written one silently encodes guesses about which words
  are hard, which is the exact mistake this method exists to replace.
---

# Mining the recognizer's error table

## What this produces

One table, accumulated over time, in `verified_corrections.json`:

```json
{"H100": {"heard": {"H 一百": 4, "H一百四": 1}, "k": 17, "verdict": "error"},
 "CPU":  {"heard": {"GPU": 3}, "k": 11, "verdict": "not-an-error"}}
```

`k` is how many distinct Claude sessions used the term — the evidence that it is
real vocabulary rather than a one-off. `heard` is what our own transcripts
produced instead. Rejected pairs are kept with `verdict: not-an-error`, not
deleted, so a later pass does not re-propose them.

Two things consume it:

- **Biasing** — `asr_context.Lexicon.corrections()` ranks which terms get a slot
  in the ASR prompt. Real error counts are a far better ranking signal than the
  string-equality proxy it uses today.
- **Post-hoc correction** — knowing `拉斯 → loss` with evidence lets the output
  be fixed after transcription, which does not depend on the model cooperating.
  Read "Why post-hoc matters more than it looks" below before building this.

## The signal, and why the obvious version of it fails

`asr_context.Lexicon` already stores two halves: `observe()` records what Claude
wrote, `observe_said()` records what our transcripts produced. Subtracting one
from the other looks like it should yield "words the recognizer cannot say".

It does not, because **both halves compare by string equality, and the entire
problem is that the two sides do not look alike.** When `loss` comes out as
「拉斯」, equality concludes "loss was never produced" — true, but it cannot say
what `loss` came out AS. So it cannot count how often the word is actually
wrong, and it cannot fix anything after the fact.

Worse, absence is not evidence. Measured on the real history: taking every
utterance where Claude's reply used a term the transcript lacked gave **~80%
noise** — 「跑怎么样了」 paired with `epoch` because the reply happened to mention
it, when nothing was misheard and the word was simply never spoken.

What makes a pair evidence is that the transcript contains something *shaped
like* the term. That is the whole trick, and it is why stage 2 exists.

## The three stages

### Stage 1 — join (deterministic)

`scripts/mine_corrections.py` joins each dictation transcript to the assistant
reply it provoked, matching on normalized text rather than timestamps: Claude
Code stores the pasted transcript with its own trimming, but the text is ours,
so the join is exact where it lands and simply misses where it does not.

The reply is the clean side. It spells the term correctly, and — the user's own
framing — 相当于 Claude 自己已经给修过一次了. Never learn a spelling from the
transcript side; that is ASR output, and a misheard word written into the store
would be re-suggested forever and look more authoritative each time it recurred.

On the history as of 2026-08-18: 3814 transcripts → **2596 joined pairs**.

### Stage 2 — localize (deterministic, `near_miss()`)

For each candidate term, find the fragment of the transcript that looks like a
mangled version of it. Two families, both dependency-free:

| family | examples | how |
|---|---|---|
| spelled-out Latin | `mclou`/MLClaw, `plug in`/plugin, `Jason`/JSON, `D E T R`/DETR, `I O U`/IoU, `side car`/sidecar | edit distance on the alphanumeric skeleton, over 1–3 token windows (the recognizer loves to insert spaces) |
| digits-as-Chinese | `H 一百`/H100, `e 十三 qh`/e13qh, `S 零零零`/s000 | rewrite 一二三…百千 back to digits **before** skeletonizing, then re-test |

This cut 40 raw candidates to 23, of which ~12 were real. That precision is what
makes stage 3 cheap enough to do carefully.

**Known gap:** a term rendered *entirely* in Chinese characters is invisible here
— `拉斯`/loss, `兀兀兀松`/雾凇, `兰舍达格莫特`/"large language model". Catching
those needs a pinyin table (`pypinyin`), which is a new dependency and has not
been added. Until it is, seed those pairs by hand from the probe set in
`context_replay_eval.py` and say so in the entry.

### Stage 3 — judge (this is the part that needs a model)

Read the surviving candidates with their evidence utterances and decide, per
pair, whether it is a real misrecognition. No mechanical rule separates these,
which is exactly why the stage exists — from the real run:

| verdict | pairs |
|---|---|
| real error | `Claude ← Cloud`, `MLClaw ← M R Claw / mclou`, `Claude Code ← CLAW CODE`, `e13qh ← e 十三`, `H100 ← H 一百`, `SKU ← S K U`, `IoU ← I O U`, `CDN ← C D N`, `anyware-core ← anywhere core`, `Orbbec ← Orbic`, `s000 ← S 零零零`, `SfM ← sim` |
| not an error | `CPU ← GPU`, `mode ← code`, `batch ← Match`, `shot ← short`, `location ← motion`, `never ← server`, `skills ← skill`, `hooks ← hook` |

The pattern in the rejects: the "heard" form is itself a legitimate word the user
says on purpose. Tempting to automate as "reject if the heard form is a
graduated lexicon term" — but `Cloud` is graduated too and *is* a real error, so
that rule would throw away the single most frequent correction in the table. Ask
instead: *could the user have meant the heard form here?* Read the utterance,
not just the pair. `就是拉斯约束` is not a sentence about 拉斯; 「不是 CPU 是
GPU」 plainly is about GPU.

Plural/singular (`skills ← skill`, `hooks ← hook`) and truncation of a compound
(`train-run ← train`) are not recognition errors. Do not record them.

## Why post-hoc matters more than it looks

Measured on 286 clips (`context_replay_eval.py`, 2026-08-18), the biasing route
has a small ceiling on the 0.6B model:

| variant | CER | vs off | hard words | ctx |
|---|---|---|---|---|
| off | 0.0576 | — | 1/15 | 0 |
| corr30 (the then-shipped default) | 0.0602 | +0.0027 | 1/15 | 184 |
| **placebo20** (20 cooking terms) | 0.0601 | +0.0026 | **2/15** | 138 |
| lexcorr20 (persistent error list) | **0.0527** | **−0.0049** | 2/15 | 84 |

Read that table with the placebo row first, because it changes both conclusions.
The persistent list's **CER win is real** — a prompt of irrelevant words made CER
worse, so the content is doing the work, not the mere presence of a prompt. Its
**hard-word gain is not**: the placebo scored 2/15 too. Biasing buys general
accuracy here; it does not fix the specific words that motivated the feature.
`correction_terms: 30` bought nothing a cooking list did not, and was turned off.

`config.yaml` records the sharper version: with the correct word **already in the
prompt**, the 0.6B still emitted 「拉斯」「Jason」「H 一百」 (0/7). A model that
ignores the hint cannot be fixed by a better hint — which is the whole argument
for the post-hoc route.

Whenever this table is re-measured, keep a placebo arm. Without it the 2/15 would
have read as a win, and the error table would have been credited with a fix it
did not make.

A verified pair table does not have that ceiling — `拉斯 → loss` can simply be
applied to the output text. But the CPU/GPU row is the warning: a wrong pair
applied automatically corrupts correct transcripts. Any corrector needs, at
minimum, evidence thresholds and a bias toward leaving text alone.

## The number that decides what is safe to act on

`scripts/build_verified.py` counts each judged pair against the whole dictation
log and writes `verified_corrections.json`. What matters is not the raw miss
count but the **error rate** — misses against times we produced the term right:

| correct | wrong | right | read as |
|---|---|---|---|
| `Claude ← cloud/Cloud/Claw` | 69 | 3 | ~96% wrong — the single biggest hole |
| `SfM ← sim` | 14 | 0 | never once produced correctly |
| `IoU`, `anyware-core`, `Orbbec`, `train-init`, `e13qh`, `s000`, `DETR`, `sidecar` | 1–6 | **0** | 100% — safest to rewrite |
| `JSON ← Jason` | 11 | 7 | ~60% — real but inconsistent |
| `SKU ← S K U` | 6 | 126 | ~5% — mostly fine, low priority |

A 0-right row is the strong case: the correct spelling has never once come out
of the recognizer, so rewriting its known wrong form cannot destroy a correct
transcript. `SKU` is the opposite — 126 correct productions mean a blanket
rewrite would be touching text that is already right 95% of the time.

Count case-sensitively. `Cloud` and `cloud` are two separate observations, and
matching case-insensitively counts the same occurrence under both, doubling
every total — it read 138 before the fix and 69 after.

## Running it

```bash
cd /home/shuai/code/whisper_input
pixi run python .claude/skills/mine-asr-corrections/scripts/mine_corrections.py \
    --terms 150 --examples 4
```

Read `correction_candidates.json`, apply stage 3, then record the verdicts in the
`JUDGED` table at the top of `scripts/build_verified.py` — keeping the rejects —
and run it to count them:

```bash
pixi run python .claude/skills/mine-asr-corrections/scripts/build_verified.py
```

That writes `verified_corrections.json` and prints the ranked table above.

Re-run it when the vocabulary moves (a new project, a new model, a new mic).
The table is cumulative; the point is that it gets better the longer it runs,
which is the thing a hand-written list can never do.
