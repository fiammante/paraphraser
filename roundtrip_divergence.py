#!/usr/bin/env python3
"""
roundtrip_divergence.py — generate meaning-preserving variants of a source
sentence by pivoting EN -> pivot -> EN through a local Ollama model, and rank
the candidates. Also runs a diagnostic mode measuring which constructions
survive which pivots.

Design notes
------------
* Diversity comes from the PIVOT LANGUAGE, not from sampling temperature.
  EN->fr->EN barely moves the wording (French is near-isomorphic to English).
  EN->ja->EN cannot come back phrased the same way: head-final SOV,
  postpositional, no obligatory definiteness or number. Pivot distance is the
  diversity dial; each individual translation stays as good as the model allows.
* Scoring inverts the usual MT setup: chrF is a DIVERSITY score to maximise,
  while fidelity is the gate. Good variants are far in surface, intact in meaning.
* Iteration feeds each output back in, so one source yields a candidate pool.
  When successive outputs stop changing, the pool is exhausted (fixed point).
* Reject filter: some pivots FORCE a distinction English leaves open — "my
  brother" must become 兄 or 弟 and returns as "my older brother". That is not a
  paraphrase, it asserts something the source did not. Such candidates are
  flagged via ADDED_INFO_MARKERS + length ratio and dropped.
* The default fidelity proxy (content recall) is weak: synonym substitution
  reads as loss. Use --comet when fidelity matters, and read the output.
* Decoding is pinned to temperature 0 + fixed seed. Any residual nondeterminism
  is the runtime (batching, kv-cache), not the sampler.

Usage
-----
    # three rewordings of one sentence
    python roundtrip_divergence.py --text "..." --pivots ja --iterations 4 --variants 3
    # wide pool across the diversity range
    python roundtrip_divergence.py --pivots ja,ko,tr,fr --iterations 3 --variants 5
    # diagnostic mode: which constructions survive which pivots
    python roundtrip_divergence.py --pivots ja,fr --iterations 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, asdict

DEFAULT_HOST = "http://localhost:11434"

# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #

LANGS = {
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "fi": "Finnish",
    "hu": "Hungarian",
    "zh": "Chinese",
    "fr": "French",   # control
    "de": "German",
    "es": "Spanish",
    "en": "English",
}

# --------------------------------------------------------------------------- #
# Probe set — each sentence targets an asymmetric obligatory category, i.e. a
# distinction the pivot FORCES that English leaves unspecified. Those are the
# irreversible ones: the translator must invent information, and the invention
# survives the return leg.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Probe:
    probe: str      # phenomenon under test
    text: str


PROBES = [
    Probe("kinship-seniority",
          "My brother called yesterday to say he had finally found a job."),
    Probe("definiteness-number",
          "She put the book on the table and left without saying a word."),
    Probe("zero-anaphora",
          "When the engineer reviewed the design, he realised that he had "
          "misread the specification, so he rewrote it."),
    Probe("register-honorific",
          "Hey, could you take a look at this when you get a chance? No rush."),
    Probe("spatial-frame",
          "The cup is to the left of the plate, just behind the salt shaker."),
    Probe("evidentiality",
          "The reactor was shut down at midnight, apparently after a coolant "
          "pump failed."),
    Probe("aspect-tense",
          "By the time we arrived, they had been waiting for over three hours."),
    Probe("technical-terminology",
          "The neonatal EEG pipeline resamples each channel to 256 Hz, applies "
          "a zero-phase bandpass filter, then detects burst onsets."),
    Probe("nested-relative-clause",
          "The patient whose recordings the algorithm flagged was the one the "
          "clinician had already discharged."),
    Probe("idiom",
          "Management decided to kick the can down the road rather than bite "
          "the bullet."),
]

# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

# TranslateGemma is a rigid fine-tune. The trailing "\n\n\n" is deliberate:
# the card specifies two blank lines between the instruction and the text.
# Deviating measurably degrades output — do not "tidy" this.
TRANSLATEGEMMA = (
    "You are a professional {src_name} ({src}) to {tgt_name} ({tgt}) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original "
    "{src_name} text while adhering to {tgt_name} grammar, vocabulary, and "
    "cultural sensitivities. Produce only the {tgt_name} translation, without "
    "any additional explanations or commentary. Please translate the following "
    "{src_name} text into {tgt_name}:\n\n\n{text}"
)

INSTRUCT = (
    "Translate the following {src_name} text into {tgt_name}.\n"
    "Rules:\n"
    "- Output the translation only. No preamble, no notes, no quotes, no romanisation.\n"
    "- Preserve register, sentence count and paragraph structure.\n"
    "- Leave code identifiers, units and numbers unchanged.\n\n"
    "{src_name} text:\n{text}\n\n{tgt_name} translation:"
)

TEMPLATES = {"translategemma": TRANSLATEGEMMA, "instruct": INSTRUCT}


def build_prompt(text: str, src: str, tgt: str, style: str) -> str:
    return TEMPLATES[style].format(
        src=src, tgt=tgt,
        src_name=LANGS[src], tgt_name=LANGS[tgt],
        text=text.strip(),
    )


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #

class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 300.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise OllamaError(f"{path} -> HTTP {exc.code}: "
                              f"{exc.read().decode('utf-8', 'replace')[:400]}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach {self.host}: {exc.reason}. "
                              "Is `ollama serve` running?") from exc

    def tags(self) -> list[str]:
        req = urllib.request.Request(f"{self.host}/api/tags")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise OllamaError(f"cannot list models at {self.host}: {exc}") from exc
        return [m["name"] for m in data.get("models", [])]

    def generate(self, model: str, prompt: str, *, seed: int = 7,
                 num_ctx: int = 8192, num_predict: int = 1024) -> tuple[str, float]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Pinned for reproducibility. temperature 0 makes top_p/top_k
                # inert but they are set anyway so the config is self-documenting.
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "repeat_penalty": 1.0,
                "seed": seed,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        t0 = time.perf_counter()
        data = self._post("/api/generate", payload)
        return data.get("response", ""), time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Output cleaning — instruct models leak scaffolding even when told not to.
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_LABEL = re.compile(
    r"^\s*(here (?:is|'s) the )?(translation|traduction|翻訳|번역)\s*[:：]\s*",
    re.IGNORECASE)


def clean(raw: str) -> str:
    out = raw.strip()
    out = _FENCE.sub("", out).strip()
    out = _LABEL.sub("", out).strip()
    if len(out) > 1 and out[0] in "\"'“«" and out[-1] in "\"'”»":
        out = out[1:-1].strip()
    # Some models append a parenthesised gloss or romanisation on its own line.
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) > 1 and lines[-1].strip().startswith("(") and lines[-1].strip().endswith(")"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Metrics (stdlib)
# --------------------------------------------------------------------------- #

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def char_ngrams(text: str, n: int) -> Counter:
    s = re.sub(r"\s+", "", text)
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF-beta over character n-grams. 100 = identical. Robust to word order
    and to tokenisation, which is what you want when comparing paraphrases."""
    hyp, ref = normalise(hyp), normalise(ref)
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h, r = char_ngrams(hyp, n), char_ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = statistics.fmean(precisions), statistics.fmean(recalls)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta ** 2
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def token_edit_ratio(hyp: str, ref: str) -> float:
    """Normalised word-level Levenshtein distance. 0 = identical, 1 = disjoint."""
    a, b = normalise(hyp).split(), normalise(ref).split()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1] / max(len(a), len(b))


STOP = frozenset("""a an the and or but if then than that this these those of to in on at by for
with from as is are was were be been being it its he she they them his her their i you we not no
do does did have has had will would can could may might shall should must""".split())


def content_jaccard(hyp: str, ref: str) -> float:
    a = set(normalise(hyp).split()) - STOP
    b = set(normalise(ref).split()) - STOP
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def content_recall(hyp: str, ref: str) -> float:
    """Fraction of SOURCE content words still present in the candidate.
    Asymmetric on purpose: for paraphrase selection, dropped source content is a
    fidelity failure, whereas added words may be legitimate rewording. Note the
    obvious weakness — a synonym substitution reads as loss. This is a cheap
    proxy; use --comet when fidelity actually matters."""
    src = set(normalise(ref).split()) - STOP
    cand = set(normalise(hyp).split()) - STOP
    return len(src & cand) / len(src) if src else 1.0


def length_ratio(hyp: str, ref: str) -> float:
    ra = len(normalise(ref).split())
    return len(normalise(hyp).split()) / ra if ra else 0.0


# Optional: COMET (reference-based, learned). Far better than chrF but heavy.
def try_comet():
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
        model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        def score(src: str, hyp: str, ref: str) -> float:
            out = model.predict([{"src": src, "mt": hyp, "ref": ref}],
                                batch_size=1, gpus=0, progress_bar=False)
            return float(out["scores"][0])
        return score
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

@dataclass
class Hop:
    iteration: int
    pivot_text: str
    back_text: str
    chrf_vs_source: float
    chrf_vs_previous: float
    edit_ratio: float
    jaccard: float
    length_ratio: float
    comet: float | None = None
    seconds: float = 0.0


@dataclass
class Trial:
    probe: str
    pivot: str
    source: str
    hops: list[Hop] = field(default_factory=list)


def run_trial(client: Ollama, model: str, style: str, probe: Probe, pivot: str,
              iterations: int, comet, verbose: bool) -> Trial:
    trial = Trial(probe=probe.probe, pivot=pivot, source=probe.text)
    current = probe.text
    for k in range(1, iterations + 1):
        fwd, t1 = client.generate(model, build_prompt(current, "en", pivot, style))
        pivot_text = clean(fwd)
        if not pivot_text:
            raise OllamaError(f"empty forward output ({probe.probe}, en->{pivot}, iter {k})")
        back, t2 = client.generate(model, build_prompt(pivot_text, pivot, "en", style))
        back_text = clean(back)
        if not back_text:
            raise OllamaError(f"empty return output ({probe.probe}, {pivot}->en, iter {k})")

        hop = Hop(
            iteration=k,
            pivot_text=pivot_text,
            back_text=back_text,
            chrf_vs_source=chrf(back_text, probe.text),
            chrf_vs_previous=chrf(back_text, current),
            edit_ratio=token_edit_ratio(back_text, probe.text),
            jaccard=content_jaccard(back_text, probe.text),
            length_ratio=length_ratio(back_text, probe.text),
            comet=comet(probe.text, back_text, probe.text) if comet else None,
            seconds=t1 + t2,
        )
        trial.hops.append(hop)
        if verbose:
            print(f"    [{pivot} #{k}] chrF={hop.chrf_vs_source:5.1f}  "
                  f"Δprev={hop.chrf_vs_previous:5.1f}  {hop.seconds:4.1f}s")
            print(f"      pivot: {pivot_text[:110]}")
            print(f"      back : {back_text[:110]}")
        current = back_text  # feed forward: iterate toward the fixed point
    return trial


# --------------------------------------------------------------------------- #
# Variant selection (generation mode)
# --------------------------------------------------------------------------- #

# Words that commonly appear when a pivot FORCES a specification English left
# open. Their presence in a candidate but not the source is a strong hint that
# the round trip asserted something the source did not. Heuristic, not a proof —
# see the fidelity caveats in the README.
ADDED_INFO_MARKERS = frozenset("""older younger elder eldest youngest senior junior
north south east west northern southern eastern western
apparently reportedly allegedly seemingly presumably supposedly
respected honourable honorable esteemed sir madam
male female man woman men women""".split())


@dataclass
class Variant:
    text: str
    pivot: str
    iteration: int
    diversity: float          # 0-100, surface distance from source (higher = more reworded)
    fidelity: float           # 0-100, meaning retention (COMET if available, else proxy)
    fidelity_source: str      # "comet" | "content-recall proxy"
    length_ratio: float
    flags: list[str] = field(default_factory=list)


def variant_flags(text: str, source: str) -> list[str]:
    flags: list[str] = []
    src_words = set(normalise(source).split())
    added = (set(normalise(text).split()) - src_words) & ADDED_INFO_MARKERS
    if added:
        flags.append("added-specification:" + ",".join(sorted(added)))
    lr = length_ratio(text, source)
    if lr > 1.25:
        flags.append(f"expansion:{lr:.2f}")
    elif lr < 0.75:
        flags.append(f"compression:{lr:.2f}")
    return flags


def collect_variants(trials: list[Trial], comet) -> dict[str, list[Variant]]:
    """Turn every intermediate round-trip output into a scored candidate."""
    pools: dict[str, list[Variant]] = {}
    for trial in trials:
        pool = pools.setdefault(trial.source, [])
        for hop in trial.hops:
            fid = (hop.comet * 100.0 if hop.comet is not None
                   else 100.0 * content_recall(hop.back_text, trial.source))
            pool.append(Variant(
                text=hop.back_text,
                pivot=trial.pivot,
                iteration=hop.iteration,
                diversity=100.0 - chrf(hop.back_text, trial.source),
                fidelity=fid,
                fidelity_source="comet" if hop.comet is not None else "content-recall proxy",
                length_ratio=length_ratio(hop.back_text, trial.source),
                flags=variant_flags(hop.back_text, trial.source),
            ))
    return pools


def rank_variants(pool: list[Variant], source: str, *, n: int,
                  fidelity_floor: float, dedupe_at: float = 95.0,
                  allow_flagged: bool = False) -> tuple[list[Variant], list[Variant]]:
    """Return (kept, rejected). Kept variants preserve meaning AND move the
    wording; they are sorted by descending diversity, because a variant that
    reads like the source is not a variant."""
    kept: list[Variant] = []
    rejected: list[Variant] = []

    for v in sorted(pool, key=lambda x: (-x.fidelity, -x.diversity)):
        if normalise(v.text) == normalise(source):
            v.flags.append("identical-to-source")
            rejected.append(v)
            continue
        if v.fidelity < fidelity_floor:
            v.flags.append(f"below-fidelity-floor:{v.fidelity:.0f}<{fidelity_floor:.0f}")
            rejected.append(v)
            continue
        if v.flags and not allow_flagged:
            rejected.append(v)
            continue
        if any(chrf(v.text, k.text) >= dedupe_at for k in kept):
            v.flags.append("near-duplicate")
            rejected.append(v)
            continue
        kept.append(v)

    kept.sort(key=lambda x: -x.diversity)
    return kept[:n], rejected


def report_variants(pools: dict[str, list[Variant]], *, n: int,
                    fidelity_floor: float, allow_flagged: bool) -> dict:
    out: dict[str, list[dict]] = {}
    for source, pool in pools.items():
        kept, rejected = rank_variants(pool, source, n=n,
                                       fidelity_floor=fidelity_floor,
                                       allow_flagged=allow_flagged)
        print("\n" + "=" * 78)
        print(f"SOURCE  {source}")
        print("=" * 78)
        if not kept:
            print("  no candidate cleared the filters — lower --fidelity-floor,")
            print("  raise --iterations, or add a more distant pivot")
        for i, v in enumerate(kept, 1):
            print(f"  {i}. [{v.pivot} #{v.iteration}] div={v.diversity:5.1f} "
                  f"fid={v.fidelity:5.1f}")
            print(f"     {v.text}")
        if rejected:
            print(f"\n  rejected ({len(rejected)}):")
            for v in rejected[:6]:
                print(f"    [{v.pivot} #{v.iteration}] {'; '.join(v.flags) or 'n/a'}")
                print(f"      {v.text[:100]}")
            if len(rejected) > 6:
                print(f"    ... and {len(rejected) - 6} more")
        out[source] = [asdict(v) for v in kept]
    print("\nFidelity is a proxy unless --comet is active. Read the variants "
          "before shipping them.")
    return out


def report(trials: list[Trial], control: str) -> None:
    by_pivot: dict[str, list[Trial]] = {}
    for t in trials:
        by_pivot.setdefault(t.pivot, []).append(t)

    print("\n" + "=" * 78)
    print("FIRST ROUND TRIP — mean over probes (chrF: 100 = identical to source)")
    print("=" * 78)
    print(f"{'pivot':<8}{'chrF':>8}{'edit':>8}{'jaccard':>9}{'len':>7}{'comet':>8}")
    means: dict[str, float] = {}
    for pivot, ts in by_pivot.items():
        first = [t.hops[0] for t in ts if t.hops]
        means[pivot] = statistics.fmean(h.chrf_vs_source for h in first)
        comets = [h.comet for h in first if h.comet is not None]
        print(f"{pivot:<8}{means[pivot]:8.1f}"
              f"{statistics.fmean(h.edit_ratio for h in first):8.3f}"
              f"{statistics.fmean(h.jaccard for h in first):9.3f}"
              f"{statistics.fmean(h.length_ratio for h in first):7.2f}"
              f"{(statistics.fmean(comets) if comets else float('nan')):8.3f}")

    if control in means:
        print(f"\nDivergence relative to control ({control}) — the number that matters:")
        for pivot, m in sorted(means.items(), key=lambda kv: kv[1]):
            if pivot != control:
                print(f"  {control} - {pivot:<4} = {means[control] - m:6.1f} chrF")

    print("\n" + "=" * 78)
    print("CONVERGENCE — chrF of each output vs the previous one")
    print("(rising toward 100 => systematic grammatical drift to a fixed point;")
    print(" flat and low => stochastic noise, check your decoding options)")
    print("=" * 78)
    max_it = max((len(t.hops) for t in trials), default=0)
    if max_it > 1:
        print(f"{'pivot':<8}" + "".join(f"{'it'+str(i):>9}" for i in range(1, max_it + 1)))
        for pivot, ts in by_pivot.items():
            row = ""
            for i in range(max_it):
                vals = [t.hops[i].chrf_vs_previous for t in ts if len(t.hops) > i]
                row += f"{statistics.fmean(vals):9.1f}" if vals else f"{'-':>9}"
            print(f"{pivot:<8}{row}")
    else:
        print("(single iteration — rerun with --iterations 3 to see this)")

    print("\n" + "=" * 78)
    print("PER-PROBE, FIRST ROUND TRIP (chrF vs source)")
    print("=" * 78)
    pivots = list(by_pivot)
    print(f"{'probe':<26}" + "".join(f"{p:>8}" for p in pivots))
    for probe in dict.fromkeys(t.probe for t in trials):
        row = ""
        for p in pivots:
            hit = [t for t in trials if t.probe == probe and t.pivot == p and t.hops]
            row += f"{hit[0].hops[0].chrf_vs_source:8.1f}" if hit else f"{'-':>8}"
        print(f"{probe:<26}{row}")
    print("\nLow scores on kinship-seniority / spatial-frame / register are the")
    print("interesting ones: the pivot forced a distinction English never encoded,")
    print("and the invented information cannot be undone on the return leg.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="translategemma:12b")
    ap.add_argument("--style", choices=sorted(TEMPLATES), default=None,
                    help="prompt template; inferred from model name if omitted")
    ap.add_argument("--pivots", default="ja,fr",
                    help="comma-separated codes; keep a control (default ja,fr)")
    ap.add_argument("--control", default="fr")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--variants", type=int, metavar="N", default=None,
                    help="generation mode: emit the N best meaning-preserving "
                         "rewordings of each source instead of a divergence report")
    ap.add_argument("--fidelity-floor", type=float, default=60.0,
                    help="reject variants scoring below this (default 60)")
    ap.add_argument("--allow-flagged", action="store_true",
                    help="keep variants flagged for added specification or "
                         "expansion/compression instead of rejecting them")
    ap.add_argument("--comet", action="store_true", help="use COMET if installed")
    ap.add_argument("--text", action="append", default=None,
                    help="custom sentence (repeatable); replaces the probe set")
    ap.add_argument("--json", metavar="PATH", help="dump full results")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    style = args.style or ("translategemma" if "translategemma" in args.model.lower()
                           else "instruct")
    pivots = [p.strip() for p in args.pivots.split(",") if p.strip()]
    unknown = [p for p in pivots if p not in LANGS or p == "en"]
    if unknown:
        print(f"unknown pivot(s): {unknown}; known: {sorted(set(LANGS) - {'en'})}",
              file=sys.stderr)
        return 2

    client = Ollama(args.host)
    try:
        available = client.tags()
    except OllamaError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.model not in available and args.model.split(":")[0] not in \
            {n.split(":")[0] for n in available}:
        print(f"model {args.model!r} not pulled. Try:\n  ollama pull {args.model}\n"
              f"available: {', '.join(available) or '(none)'}", file=sys.stderr)
        return 1

    probes = ([Probe("custom", t) for t in args.text] if args.text else PROBES)
    comet = try_comet() if args.comet else None
    if args.comet and comet is None:
        print("COMET unavailable (pip install unbabel-comet); using chrF only.",
              file=sys.stderr)

    print(f"model={args.model}  style={style}  pivots={pivots}  "
          f"iterations={args.iterations}  probes={len(probes)}")

    trials: list[Trial] = []
    for probe in probes:
        if not args.quiet:
            print(f"\n  {probe.probe}: {probe.text[:70]}...")
        for pivot in pivots:
            try:
                trials.append(run_trial(client, args.model, style, probe, pivot,
                                        args.iterations, comet, not args.quiet))
            except OllamaError as exc:
                print(f"    !! {exc}", file=sys.stderr)

    if not trials:
        print("no successful trials", file=sys.stderr)
        return 1

    payload: dict = {"model": args.model, "style": style, "seed": args.seed,
                     "mode": "variants" if args.variants else "diagnose"}
    if args.variants:
        payload["variants"] = report_variants(
            collect_variants(trials, comet), n=args.variants,
            fidelity_floor=args.fidelity_floor, allow_flagged=args.allow_flagged)
    else:
        report(trials, args.control)
        payload["trials"] = [asdict(t) for t in trials]

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
