# Pivot Paraphrase

Generate meaning-preserving **variants of a source sentence** by round-tripping it through a typologically distant language, then rank the candidates on the two axes that actually matter: how far the wording moved, and whether the meaning survived.

Runs entirely against a local [Ollama](https://ollama.com) daemon. Zero third-party dependencies (stdlib `urllib` + a pure-Python chrF2).

```bash
python3 roundtrip_divergence.py --model translategemma:12b \
    --text "The reactor was shut down at midnight after a coolant pump failed." \
    --pivots ja,ko,tr,fi --iterations 1 --variants 3
```

```
==============================================================================
SOURCE  My brother called yesterday to say he had finally found a job.
==============================================================================
  1. [ja #1] div= 38.1 fid= 62.5
     My brother phoned yesterday to tell me he had finally landed a job.

  rejected (4):
    [fr #1] identical-to-source
    [ja #2] added-specification:older
    [ja #3] near-duplicate
    [ko #1] added-specification:apparently; below-fidelity-floor:25<60
```

---

## The mechanism

Back-translation as a paraphrase generator is an old trick. What's usually missing is a principled way to decide which candidates are worth keeping — because the two properties you want are in tension.

**Surface distance.** A variant that comes back nearly identical to the source is useless. The wording has to move.

**Meaning preservation.** A variant that drifts is worse than useless.

So the metrics are used in the opposite direction from a normal MT evaluation: **chrF becomes a diversity score to maximise**, not an error to minimise, while fidelity becomes the gate. Good variants live in the corner where surface distance is high and meaning is intact.

### Pivot distance is the diversity dial

`EN → French → EN` barely moves the wording. French is near-isomorphic to English on every structural category that matters, so nothing forces a rewrite — you tend to get your own sentence back.

`EN → Japanese → EN` cannot come back phrased the same way. Japanese is head-final SOV, postpositional, with internally-headed relative clauses and no obligatory definiteness or number marking. There is no way to carry English structure through it; the wording must be rebuilt from meaning.

That contrast *is* the dial:

| Pivot | Register of rewriting | Use when |
|---|---|---|
| `fr` `es` `de` | Conservative — lexical substitution, light reordering | You want minimal-risk rewordings |
| `fi` `hu` `tr` | Moderate — case morphology forces restructuring | Balanced diversity/fidelity |
| `ja` `ko` `vi` `zh` | Aggressive — full structural rebuild | You want genuinely different phrasings |

Mix them in one run (`--pivots ja,tr,fr`) to get a candidate pool spanning the range, then let the ranking sort it out.

### Pool depth comes from pivots, not iterations

`--iterations` feeds each output back in, but at `temperature=0` the round trip is a **deterministic map** and reaches a fixed point almost immediately — usually after one pass, occasionally two. Measured: `Δprev=100.0` on iterations 2 and 3, i.e. byte-identical output, 35 seconds spent for nothing. The loop now detects this and breaks early.

**Widen `--pivots` instead.** Each pivot language is an independent draw; each extra iteration on the same pivot is almost always the same draw again. Use `--iterations 1` and eight pivots rather than `--iterations 8` and one pivot.

---

## The reject filter

This is the part that distinguishes the tool from `translate | translate`.

Some pivots **force** a distinction the source language leaves unspecified. The translator cannot decline to choose, so it invents information — and the invention survives the return leg.

| Forced choice | Pivots that force it | Example damage |
|---|---|---|
| Sibling seniority | `ja` `ko` `zh` | "my brother" → 兄/弟 → "my **older** brother" |
| Politeness level | `ja` `ko` | Neutral request → deferential or curt |
| Evidentiality | `tr` | Plain assertion → "**apparently** X" |
| Spatial frame | some languages | "to the left of" → "to the **north** of" |
| Definiteness/number | reverse direction, from `ja` `zh` `vi` | "a book" ⇄ "the book", singular ⇄ plural |

Not every failure is a forced category, and the lexicon misses those. Turkish uses the seniority-neutral *kardeşim* and dodged the sibling trap entirely — then substituted an adverb-scope error instead (`yesterday` moved from the calling to the finding). Scope, aspect and quantifier drift raise no flags. Read the output.

These are **not paraphrases**. They assert something the source never did. A variant reading "my older brother" is factually stronger than the input, and if you ship it as a rewording you have silently changed the claim.

Candidates are therefore flagged and rejected on:

- `added-specification:<words>` — words from a forced-category lexicon (`older`, `younger`, `north`, `apparently`, `reportedly`, `male`, ...) present in the variant but absent from the source
- `expansion:<ratio>` / `compression:<ratio>` — length ratio outside 0.75–1.25, which usually means specification was added or content dropped
- `identical-to-source` — no rewriting happened
- `near-duplicate` — chrF ≥ 95 against an already-accepted variant
- `below-fidelity-floor` — meaning retention under `--fidelity-floor`

Pass `--allow-flagged` to keep flagged candidates for inspection rather than dropping them. Useful when you *want* to see what the pivot forced.

---

## Setup

```bash
ollama serve
ollama pull translategemma:12b        # purpose-built translation fine-tune
# or
ollama pull mistral-small3.2:24b      # general instruct model, strong FR
```

Python 3.10+.

---

## Usage

```bash
# three rewordings of one sentence — diversity from pivot breadth
python3 roundtrip_divergence.py --model translategemma:12b \
    --text "She put the book on the table and left without saying a word." \
    --pivots ja,ko,tr,vi,zh,fi,hu --iterations 1 --variants 3

# wide pool across the diversity range, results dumped for later filtering
python3 roundtrip_divergence.py --model mistral-small3.2:24b \
    --text "The patient was discharged before the results came back." \
    --pivots ja,ko,tr,vi,fi,fr --iterations 1 --variants 5 --comet --json variants.json

# looser gate, keep flagged candidates so you can see what the pivot invented
python3 roundtrip_divergence.py --text "My brother called yesterday." \
    --pivots ja,ko,zh --iterations 1 --variants 5 \
    --fidelity-floor 45 --allow-flagged

# diagnostic mode (omit --variants): which constructions survive which pivots
python3 roundtrip_divergence.py --model translategemma:12b \
    --pivots ja,fr --iterations 3
```

### Options

| Flag | Default | Notes |
|---|---|---|
| `--variants N` | off | Generation mode: emit the N best rewordings. Omit for diagnostic mode |
| `--text` | probe set | Repeatable; your source sentence(s) |
| `--pivots` | `ja,fr` | Comma-separated codes — this is the diversity dial |
| `--iterations` | `1` | Candidate pool depth. Use 3–5 for generation |
| `--fidelity-floor` | `60` | Reject below this. Lower for more diversity, at your own risk |
| `--allow-flagged` | off | Keep added-specification / expansion candidates |
| `--force-rank` | off | Rank even when the fidelity axis has collapsed (not advised) |
| `--fidelity-spread-min` | `5` | Withhold ranking below this fidelity spread |
| `--comet` | off | Use COMET for fidelity if `unbabel-comet` is installed |
| `--model` | `translategemma:12b` | Any pulled Ollama model |
| `--style` | inferred | `translategemma` or `instruct` prompt template |
| `--host` | `localhost:11434` | Remote daemon is fine |
| `--seed` | `7` | Pinned; decoding is `temperature=0` |
| `--json` | — | Full results including pivot texts and rejected candidates |

Supported pivots: `ja ko tr vi fi hu zh fr de es` — all within TranslateGemma's 55-language coverage.

---

## How candidates are scored

**Diversity** = `100 − chrF2(variant, source)`. Character n-gram distance, robust to tokenisation and word order. Higher = more thoroughly reworded.

**Fidelity** = COMET if `--comet` is active, otherwise a content-recall proxy: the fraction of source content words still present in the variant, stopwords excluded.

Kept candidates are sorted by **descending diversity** — among variants that passed the fidelity gate, you want the one that moved furthest.

That last sentence hides an assumption, and it fails often enough that the tool now checks it.

### The degenerate-fidelity guard

"Sort by diversity among candidates that passed the gate" is only meaningful if fidelity actually *discriminates* between candidates. On short inputs, the content-recall proxy saturates: every candidate retains the same 7 of 8 content words and scores identically. The fidelity axis collapses, the sort reduces to pure diversity, and **the most-reworded candidate wins whether or not it still means the same thing.**

This is not hypothetical. A real run, four pivots, Mistral Small 3.2:

```
source:  My brother called yesterday to say he had finally found a job.

tr:  My brother called to say he found a job yesterday.        div=34.5  fid=87.5
ja:  My older brother called me yesterday and said he...        div=33.7  fid=87.5
ko:  My older brother called yesterday and said he...           div=27.9  fid=87.5
fi:  My brother called yesterday and said he had finally...     div=15.6  fid=87.5
```

All four scored **87.5**. Spread: zero. And the top-ranked Turkish variant is the worst of the set — "yesterday" detached from *called* and reattached to *found*, so the call is now undated and the job-finding is dated. "Finally" vanished too, dropping the long-search implication. Two meaning changes, no flags raised, ranked first *because* it had drifted furthest.

So when the fidelity spread across gate-passing candidates falls below `--fidelity-spread-min` (default 5 points) and fidelity came from the proxy rather than COMET, the tool **refuses to rank**:

```
  ⚠ NOT RANKED — fidelity spread is 0.0 pts (< 5), so the
    fidelity axis carries no ordering information and the sort
    would reduce to 'most reworded wins'. That promotes drift.
    Candidates below are in neutral order, NOT best-first.
```

Candidates are then listed in neutral order (grouped by pivot, bulleted rather than numbered) because any score-based order would imply a judgement the metrics cannot support. `--force-rank` overrides, and the JSON records `"ranking": "unranked-degenerate-fidelity"` so downstream consumers can tell a real ranking from a withheld one.

COMET scores are exempt from the check: a genuinely small COMET spread means the candidates really are equivalent, which is information rather than the absence of it.

### Be honest about the fidelity proxy

Two independent weaknesses, in order of severity:

1. **It saturates** (above). At single-sentence length the proxy is not merely weak, it is frequently non-functional. The guard turns a silent bad ranking into a visible refusal, which is better, but it does not give you a working metric.
2. **Synonym substitution reads as loss.** "phoned" for "called" costs recall even though that is exactly the rewording you want. The measure is asymmetric on purpose — dropped source content is a real failure while added words may be legitimate — but it cannot tell a good paraphrase from a changed claim.

Therefore:

- **Use `--comet` when fidelity matters.** `pip install unbabel-comet` (pulls in torch). It is a learned metric that models semantic equivalence, and it is the difference between a ranking and a coin flip. With a real metric you can raise `--fidelity-floor` to 75+ meaningfully.
- **Read the variants.** No metric here detects "the round trip quietly changed what the sentence asserts". The flags catch common forced-category cases via a word lexicon — a heuristic, not a proof, and it did not catch the Turkish scope error. The tool narrows dozens of candidates to a shortlist; the last judgement is yours.

Also note that absolute chrF values are unintuitive. A semantically flawless rewording scores around 40 against its source:

```
source : She put the book on the table and left without saying a word.
variant: She placed a book on a table and departed without a word.
chrF   : 40.8   (diversity 59.2)
```

That is a *good* variant, not a broken one.

### A better detector than the lexicon

Observed across two models on the same sentence and the same pivot:

| Model | Japanese pivot | Seniority asserted |
|---|---|---|
| TranslateGemma 12B | 弟 | **younger** brother |
| Mistral Small 3.2 | 兄 | **older** brother |

Same source, same target language, opposite invented facts. The information is not merely added — it is added *arbitrarily*, which is the strongest available evidence that the source was underspecified at that point.

That suggests a detector far better than `ADDED_INFO_MARKERS`: **round-trip through one pivot with two different models, and flag every content word where they disagree.** No lexicon to maintain, generalises to phenomena nobody anticipated, and works for any language pair. Not implemented yet; it is the most valuable open item on the list.

## Diagnostic mode

Omit `--variants` and the tool reports divergence across a probe set instead of generating rewordings. This answers a different question: **which of my source constructions are stable under this pipeline, and which will keep mutating?**

It runs ten probes tagged by phenomenon — `kinship-seniority`, `definiteness-number`, `zero-anaphora`, `register-honorific`, `spatial-frame`, `evidentiality`, `aspect-tense`, `technical-terminology`, `nested-relative-clause`, `idiom` — against a test pivot and a control pivot, and reports the control-relative gap. Keep a control (`fr` by default): absolute divergence has no zero point, only the gap is interpretable.

Useful before a generation run, to learn which of your typical sentence shapes this model can be trusted to reword.

---

## The prompt templates

**`translategemma`** — a rigid fine-tune expecting one exact single-user-message format. The `\n\n\n` before the source text (instruction line, then two blank lines) is **load-bearing**; quality degrades measurably if you "tidy" it.

**`instruct`** — for general models. Adds explicit constraints fine-tunes don't need: output-only, preserve register and sentence count, leave code identifiers and numbers untouched. General models leak scaffolding (`Translation:`, code fences, parenthesised romanisation) despite instructions, so `clean()` strips it before scoring — unstripped scaffolding would otherwise register as diversity.

Decoding is pinned: `temperature=0`, `top_k=1`, fixed seed. If you see run-to-run variation, the runtime is the culprit — try `OLLAMA_NUM_PARALLEL=1 ollama serve`.

Note the deliberate design choice here: diversity comes from the **pivot language**, not from sampling temperature. Temperature-based variation degrades fluency and gives you no control over *what* varies; structural pivoting forces a rebuild while keeping each individual translation as good as the model can make it.

---

## Extending

**Two-model pipelines.** Use a specialist for the forward leg and a general instruct model for the return. Split `run_trial` to take separate models; the asymmetry produces different variant characteristics.

**Domain lexicon protection.** Add a glossary block to the `instruct` template listing terms that must survive verbatim, then check whether flagged rejections drop. This is the practical question for technical text, where you want the prose reworded and the terminology frozen.

**Cross-model disagreement detection.** See above — the highest-value unimplemented item.

**Extend `ADDED_INFO_MARKERS`.** The forced-category lexicon is deliberately small and English-side. Adding markers for your domain (regulatory hedges, clinical qualifiers) makes the reject filter meaningfully sharper.

**Quantify pivot distance properly.** URIEL/`lang2vec` exposes per-language syntactic feature vectors from WALS and related databases. Cosine distance over those vectors would let you predict a pivot's diversity yield instead of discovering it empirically — and would test whether word-order distance or obligatory-category asymmetry dominates. My expectation is the latter.

**Paragraph-level input.** Sentence-level pivoting destroys cross-sentence anaphora. Feed paragraphs and raise `num_ctx` if your variants need to preserve reference chains.

---

## Files

- `roundtrip_divergence.py` — the harness (generation + diagnostic modes)
- `README.md` — this file

MIT licensed.
