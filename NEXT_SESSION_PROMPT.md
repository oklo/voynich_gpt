# Restart prompt for the Voynich research project

Paste everything below into a fresh Codex session opened in
`/Users/greglaughlin/Projects/voynich`.

---

We are resuming a long-running, falsification-first investigation of the
Voynich Manuscript.  Work as a skeptical research collaborator, not as an
advocate for a decipherment.  Treat every prior claim, script, number, and
interpretation as potentially mistaken until it survives code inspection,
negative controls, held-out evaluation, and reproducible checks.

## Repository state to verify first

- Working directory: `/Users/greglaughlin/Projects/voynich`
- Remote: `https://github.com/oklo/voynich_gpt.git`
- Expected branch: `main`
- Expected minimum analytical checkpoint:
  `2d7c61c Export token-level residual traces`.  HEAD may be a newer
  documentation-only commit containing this restart prompt; inspect it rather
  than resetting back to `2d7c61c`.
- Immediately run `git status --short --branch` and `git log -5 --oneline`.
  Do not discard or overwrite any unexpected user changes.
- Read `HANDOFF.md`, `MEMORY.md`, and the relevant portion of
  `RESEARCH_REPORT.md` before proposing or changing anything.  The nested
  decomposition begins near the heading “Nested decomposition of the residual
  links”; the token export is under “Token-level trace for spatial graphics.”
- Run `python3 -m unittest discover -s tests -v` before substantive edits.
  The last verified suite had 59 passing standard-library tests.  Ruff was not
  installed in the prior environment, so do not imply that Ruff ran unless it
  actually does.

## What this repository now is

It began as a primitive nanoGPT exploration, but it is now a dependency-light,
held-out statistical audit of Voynich claims.  The canonical transcription is
`IT2a-n.txt`, parsed with the IVTFF-aware code in `scripts/audit_voynich.py`.
The conservative corpus is 225 pages and 34,411 certain paragraph-text tokens.
Never parse it with generic whitespace splitting: IVTFF `.`, `,`, `<->`, and
`<~>` are apparent spaces, and uncertain glyph/locus syntax requires the
parser.

No translation or decipherment has been established.  Do not claim proof of
nonsense, a semantic-information upper bound, or a demonstrated Hebrew
translation.  EVA is a transcription rather than ground-truth glyph identity,
and its marked spaces may not be linguistic word boundaries.

## Results that currently survive audit

The strongest current statement is:

> Under EVA's marked spaces, Voynichese has real held-out local sequence
> information, but its tested composition differs sharply from the prose
> controls.  Cross-space edge-glyph transitions dominate the joint model,
> while exact predecessor identities and latent lexical classes are much
> weaker.  This is more compatible with constrained form generation or an
> unusual encoding than ordinary lexical syntax, but it does not prove
> nonlanguage or nonsense.

Important background results:

- The normalized 208,733-unit stream has a reproducible prequential code of
  394,854 bits, but this is a surface code length, not semantic information.
- A broad raw entropy/code-length comparison did not put Voynich below all
  meaningful language.  Formulaic Pali can be at least as predictable.
- The primary common-word block-shuffle experiment finds approximately zero
  Voynich order gain, while all 294 modern and 24 historical comparison
  samples were numerically higher.  This is strong evidence against ordinary
  surface-word prose, not a universal proof of nonlanguage.
- Weak Voynich word order itself is prior art, especially Reddy and Knight
  (2011).  The contribution here is the held-out coding framework, broad
  controls, nuisance-matched nulls, and decomposition of the residual.
- Hebrew is unusually close under an anagram-insensitive word-shape feature,
  especially for Currier-B herbal material and period Hebrew medical/botanical
  comparators.  That feature is exactly reproduced by meaningless
  pattern-matched pseudotext.  Held-out sequence, formula, zodiac-crib, and
  fitted-substitution tests do not support simple Hebrew plaintext or anagram
  decipherment.  “Hebrew or Hebrew-mediated source” remains only a narrow lead
  for future explicit, penalized mechanisms.

## The narrow exact-neighbor experiment

`scripts/residual_sequence_information.py` holds out whole quires and supplies
target length, Currier, topic, physical line position, and the morphology of
the preceding same-line and aligned previous-line tokens.  It then predicts
one of 511 frequent exact word types plus UNKNOWN.

At vocabulary 512 and strength 20:

- procedural baseline: 4.3927 bit/word;
- exact-neighbor mixture: 4.4030 bit/word;
- net exact gain: -0.0103 bit/word;
- real links over matched links: +0.0673 bit/word.

This established a real but weak association that did not pay for that narrow
model's predictive complexity.  It motivated the broader decomposition below.

## Corrected nested residual-link decomposition

Implementation: `scripts/decompose_residual_links.py`

Tests: `tests/test_decompose_residual_links.py`

The model uses five whole-quire outer folds and learns convex mixture weights
only from three inner-quire out-of-fold prediction sets.  Outer quires remain
untouched until final scoring.  Its experts are:

- paragraph state;
- previous-final to target-initial grouped-EVA boundary transition;
- tight and loose normalized same-line copy/edit channels;
- training-only 8- and 16-class PPMI distributional word classes;
- edit and class channels pooled over the complete preceding physical line;
- exact previous-word identity; and
- exact aligned-previous-line identity.

All conditional experts are proper conditional-to-morphology-marginal density
ratios applied to the same procedural baseline.  This matters because an early
implementation allowed paragraph/exact experts to act as alternate generic
smoothers and spuriously gave word-shuffled *Picatrix* +0.065 bit/word.  Those
results were discarded.  The corrected shuffled control is +0.0020 at
morphology depth 0 and -0.0002 at depth 1.

The matched null preserves the target, Currier, topic, line position,
same-line source morphology, and every previous-line source morphology and
normalized slot.  The aligned token remains the physically aligned member of
the permuted previous-line pool.  It changes 99.1% of depth-0 Voynich records.

Primary held-out gains over the procedural baseline, in bit/word:

| Corpus | Boundary | Exact identity | Latent class | Previous-line pool | Full |
|---|---:|---:|---:|---:|---:|
| Voynich | 0.0630 | 0.0193 | 0.0066 | 0.0567 | 0.1239 |
| Hebrew Wikipedia | 0.0338 | 0.1075 | 0.0220 | 0.0338 | 0.1305 |
| Latin Wikipedia | 0.0316 | 0.0807 | 0.0321 | 0.1066 | 0.1442 |
| Spanish *Picatrix* | 0.1235 | 0.3506 | 0.1846 | 0.0291 | 0.3686 |
| word-shuffled *Picatrix* | -0.0005 | -0.0007 | -0.0009 | 0.0028 | 0.0020 |

Voynich actual links beat 49 strict matched permutations by 0.118014297
bit/word jointly (`p=0.02`, the attainable floor).  The advantage is positive
in all five outer folds, 0.0947–0.1379 bit/word; the full net gain is also
positive in all folds, 0.1004–0.1658.

Mean joint mixture weights expose the contrast:

- Voynich: boundary 0.583, loose previous-line edit 0.132, exact previous word
  0.126, paragraph 0.106;
- Hebrew: exact previous word 0.845, loose previous-line edit 0.096, boundary
  0.044;
- Latin: exact previous word 0.624, loose previous-line edit 0.135, line-class
  experts 0.173 combined, boundary 0.032;
- *Picatrix*: exact previous word 0.908, class-16 0.041.

Therefore the total finite-family sequence gain is not unusually low for
Voynich; it is close to Hebrew and Latin.  What differs is its anatomy:
boundary mechanics carry what exact identity or stable class carries in the
controls.  Say “real, boundary-dominated surface sequence,” not “no word
order,” and do not claim Voynich falls comfortably below all language.

Morphology sensitivity:

| Depth | Supplied target morphology | Baseline | Full gain | Actual over null |
|---:|---|---:|---:|---:|
| 0 | grouped-EVA length | 4.3927 | 0.1239 | 0.1180 |
| 1 | length + first/last unit | 1.3206 | 0.0103 | 0.0051 |
| 2 | length + first/last two units | 0.3327 | -0.0015 | 0.0013 |

Do not overinterpret that collapse.  Depth-1 full gains also collapse to
0.0037 for Hebrew, 0.0044 for Latin, and 0.0211 for *Picatrix*, versus -0.0002
for shuffled *Picatrix*.  Edge letters identify ordinary words efficiently as
well.  Depth 0 is the fair cross-script anatomy comparison; depths 1–2 are
sensitivity bounds.

## Completed token trace

Commit `2d7c61c` added `--token-output` to
`scripts/decompose_residual_links.py`.  Schema
`voynich-residual-token-trace-v1` contains one row per outer-held-out token:

- page, zero-based page/physical-line/word coordinates, normalized horizontal
  midpoint, quire, Currier, topic, line position, and paragraph state;
- exact target/source forms and morphology;
- previous-final and target-initial grouped-EVA units;
- all expert and family log losses and gains over baseline;
- the full mixture's fitted weights, per-token posterior responsibilities, and
  named dominant expert; and
- each expert/family's mean matched-null log loss, actual-link advantage, and
  per-token source-changed fraction.

Positive matched residual means the real source link codes that token better
than its average matched source.  Rows exactly reconstruct aggregate actual,
null, residual, and changed-fraction results; regression tests enforce this.

The previously generated full artifact was:

- `/tmp/voynich_depth0_token_trace.jsonl.gz`
- 34,411 rows, about 31 MB compressed
- SHA-256:
  `a23ede848bf684aa222d0f4241fa5c9082e75401927c72f14b4ad3f22e84dbd7`
- summary: `/tmp/voynich_depth0_trace_summary.json`

Because this is under `/tmp`, assume it may no longer exist in the new session.
Check it and its hash.  If absent, regenerate it without changing settings:

```bash
python3 scripts/decompose_residual_links.py \
  --outer-folds 5 --inner-folds 3 \
  --morphology-depth 0 --vocabulary 512 --strength 20 \
  --permutations 49 \
  --output /tmp/voynich_depth0_trace_summary.json \
  --token-output /tmp/voynich_depth0_token_trace.jsonl.gz
```

The verified trace re-aggregated to:

- actual full loss: 4.268812764 bit/word;
- matched-null full loss: 4.386827062 bit/word;
- actual-link advantage: 0.118014297 bit/word.

Do not commit the 31 MB generated trace unless explicitly asked; the exporter,
summary conventions, regeneration command, and hash are committed.

## The task when we resume

Begin with a discussion and design pass for novel data graphics.  Do not rush
straight into polished plotting code.  We want graphics that distinguish a
manuscript-wide encoding/orthographic rule from a page-local production
procedure and that remain honest about model uncertainty and correlated
experts.

The first two candidates are:

### 1. Boundary-transition residual atlas

Use rows with non-null `previous_final_unit` and `target_initial_unit`.
Aggregate the boundary expert's
`matched_null.expert_actual_advantage_bits.boundary` by source-final ×
target-initial grouped-EVA pair.  Consider showing:

- cell color: mean actual-over-null boundary advantage;
- cell area or an adjacent panel: held-out token count/support;
- row/column ordering by grouped-EVA family or clustering, but preserve a
  readable glyph label order as a sensitivity;
- small multiples by Currier A/B, topic, quire, paragraph state, or manuscript
  region; and
- a difference panel (for example Currier B minus A) only with explicit
  support and quire-aware uncertainty.

Important statistical guardrails:

- The trace stores each token's average across 49 matched nulls, not every null
  draw.  A cell heatmap is descriptive.  For uncertainty, resample whole
  pages/quires, not individual tokens, or extend the exporter to retain
  aggregate-per-permutation cell values.
- Show or threshold support; rare pairs can have extreme average residuals.
- Do not select the most dramatic cells and then report uncorrected p-values.
- Compare mean residual (effect per token) and total residual (where the corpus
  stores bits) rather than silently choosing one.
- Check whether the result is driven by a few quires, Currier strata, or line
  positions.
- Since the key signal crosses EVA spaces, eventually repeat under alternative
  defensible boundary interpretations.

Questions this atlas should answer:

1. Is there one stable edge grammar across the manuscript?
2. Does it split cleanly by Currier/topic/quire?
3. Is the advantage concentrated in a few high-frequency transitions?
4. Does it follow physical adjacency and paragraph/line position more than
   section identity?

### 2. Folio sequence/evidence map

Sort rows by `page_index_zero_based`, `physical_line_index_zero_based`, and
`word_index_zero_based`.  Render each page or folio as its physical text-line
geometry.  Candidate encodings:

- hue: expert/family with the strongest positive evidence lift over base;
- saturation: magnitude of that lift;
- opacity or outline: full-model actual-over-matched residual;
- optional glyph/marker: whether the dominant contribution is same-line,
  previous-line, paragraph, or boundary-linked;
- page margin summaries: total residual bits, mean residual, mixture-weight
  fold, Currier/topic/quire, and token count.

Do **not** naïvely color by `dominant_full_expert`.  Posterior responsibility
contains the fitted prior mixture weight.  An expert with the largest global
weight can be labeled dominant even when it makes exactly the base prediction
(for example at a line-initial token with no previous word).  Better primary
quantities are:

- `expert_gain_over_base_bits` or `family_gain_over_base_bits`;
- matched-null actual advantage for the chosen expert/family; or
- a responsibility lift such as posterior responsibility divided by fitted
  mixture weight, with a clearly explained definition.

Use responsibility only as secondary routing information, never as an
additive share of semantic information.  Gray out tokens for which no expert
improves over base or for which the relevant source was unchanged in most
permutations.

Questions this map should answer:

1. Are boundary advantages diffuse or clustered into runs/pages?
2. Do previous-line effects align vertically or occur around paragraph starts?
3. Are expert regimes associated with illustrations/topics or with physical
   scribal layout?
4. Are there abrupt within-quire transitions suggestive of hand/procedure
   changes?

## Other graphics worth discussing after those two

- Morphology-depth retention slopegraph: depth 0→1→2 for Voynich and controls,
  displaying both baseline uncertainty and retained sequence gain.
- Signal-anatomy matrix: family-only held-out gains beside jointly selected
  weights for Voynich, Hebrew, Latin, *Picatrix*, and shuffled *Picatrix*.
  Explicitly label weights as correlated-expert reliance, not information
  shares.
- Quire/page beeswarm or ridgeline of per-token actual-over-null residuals,
  with page/quire bootstrap intervals.
- A transition-network view only if it adds something the matrix cannot show;
  avoid a visually impressive but statistically unreadable hairball.

## Likely next experiment after graphics

Once the atlas localizes the signal, test transfer rather than adding a still
larger global model:

- learn boundary tables/classes on Currier A and score B, then reverse;
- analogous topic or quire-block transfer;
- compare manuscript-wide versus page-local boundary tables with honest
  held-out description length; and
- repeat under alternative EVA space/glyph grouping assumptions.

A boundary rule that transfers across Currier/topic is more compatible with a
stable encoding or orthographic process.  A signal that is page-local and
tracks immediate physical layout is more compatible with procedural
generation.  Metadata are confounded, so do not equate Currier/topic/quire
effects with independent causes.

## Non-negotiable methodological rules

- Hold out whole quires or pages; never randomly split neighboring tokens.
- Freeze vocabularies, classes, keys, directions, and hyperparameters before
  scoring held-out data.
- Use predictive log loss/prequential code, not plug-in entropy or in-sample
  neural loss.
- Pair every signal with a nuisance-preserving null and a positive control.
- Distinguish (1) actual-link advantage over matched links, (2) net gain over a
  simpler held-out code, and (3) total/semantic information.  They are not the
  same.
- Treat word-shuffled *Picatrix* generated by `--shuffled-control` as the true
  negative control.  The old Capote “wordscramble” file mostly shuffled
  passages and retained local syntax.
- Do not call corpus rank a p-value.
- Do not treat mixture weights or responsibilities as additive information or
  semantic shares.
- State representation and tokenization sensitivity instead of choosing the
  favorable setting after inspecting results.
- Preserve unexpected user changes and use `apply_patch` for edits.

## Desired first response in the resumed session

After verifying the repository and trace, summarize in a compact but concrete
way:

1. whether the expected commit/tests/artifact are present;
2. what the boundary atlas and folio map can each establish;
3. the most defensible encodings and uncertainty treatment for each;
4. which graphic should be built first and why; and
5. any additional export or aggregation needed before plotting.

Then discuss the designs with me.  Do not implement the final graphics until
we have agreed on their scientific question, encodings, and guardrails.

---
