# Voynich research handoff

## State of the project

This repository is no longer primarily a nanoGPT experiment.  It is a
dependency-light, falsification-first audit of claims about the Voynich
Manuscript, with the original notebooks and logs retained as provenance.

The main narrative is `RESEARCH_REPORT.md`.  The branch is `main`, the remote
is `https://github.com/oklo/voynich_gpt.git`, and the previous pushed checkpoint
is `4a32e24`.  The nested residual-link decomposition described below is the
current checkpoint.

At the time of writing:

- all 57 standard-library unit tests pass;
- `git diff --check` passes;
- Ruff is not installed in the environment, so no Ruff result is claimed;
- all new analysis code is Python-standard-library only; and
- the working corpus is `IT2a-n.txt`, parsed as IVTFF rather than as generic
  whitespace text.

## Bottom line

No translation or decipherment has been established.  The strongest current
result is structural:

> Under EVA's marked spaces, Voynichese has real held-out local sequence
> information, but it is allocated very differently from the tested prose.
> Cross-space edge-glyph transitions dominate the joint model, while exact
> predecessor identities and latent lexical classes are much weaker.  This is
> more consistent with constrained form generation than ordinary lexical
> syntax, without proving nonlanguage or nonsense.

This is evidence against ordinary plaintext and stable simple substitution.
It is not proof of nonsense.  Altered boundaries, homophonic/verbose/lossy
ciphers, nomenclators, low-rate payloads, and nonlinguistic notation remain
live because they can suppress stable surface lexical identity.

Do not call weak Voynich word order a new discovery.  Reddy and Knight (2011)
already reported it directly.  The contribution here is the stronger
prequential/held-out benchmark, the broad controls, the conditional residual
test, and the explicit separation of real association from useful predictive
code length.

## New residual-sequence experiment

Implementation: `scripts/residual_sequence_information.py`

Tests: `tests/test_residual_sequence_information.py`

The five outer folds hold out whole quires.  For each target token, the decoder
is supplied with its morphology class, Currier label, illustration/topic
class, physical line position, and the morphology of two causally prior
neighbors:

1. the preceding token on the same physical line;
2. the nearest token by normalized horizontal midpoint on the preceding line.

The target is then coded as one of the 511 most frequent exact types plus an
unknown bucket.  Exact neighbor identities are compared with permutations
matched independently on Currier/topic/position and the relevant neighbor
morphology.  Natural-language controls are reflowed onto the exact Voynich
page/line template, so layout and held-out sample size are identical.

Primary run (`morphology-depth=0`, vocabulary 512, strength 20):

| Corpus | Procedural code | Exact-neighbor code | Net gain | Real links over matched permutation |
|---|---:|---:|---:|---:|
| Voynich | 4.3927 | 4.4030 | **-0.0103** | **0.0673** |
| English | 3.9823 | 3.6645 | +0.3178 | 0.3350 |
| Hebrew Wikipedia | 2.7660 | 2.5799 | +0.1861 | 0.1477 |
| Latin Wikipedia | 2.5014 | 2.3341 | +0.1673 | 0.1181 |
| Spanish *Picatrix* | 3.3620 | 3.0121 | +0.3500 | 0.3492 |
| Middle English *Cirurgie* | 4.3342 | 3.7855 | +0.5488 | 0.5411 |
| Word-shuffled *Picatrix* | 3.3216 | 3.3421 | -0.0205 | -0.0015 |

The Voynich matched-link advantage decomposes into approximately 0.0959
bit/word for the previous same-line identity and 0.0180 bit/word for the
previous-line identity.  These components are separate experts and do not add
to the equal-mixture value.  At 99 permutations, all ordered corpora attained
the minimum `p=0.01`; shuffled *Picatrix* had `p=0.69`.  A 199-permutation
Voynich-only run gave `p=0.005`.

The apparent tension was important: real links scored better than matched fake
links, but the exact-identity model scores worse than the procedural baseline.
The first comparison cancels model complexity and detects a small association;
the second asks whether the association is strong enough to be a useful
held-out code.  For Voynich it is not, at the primary setting.  All ordered
controls gain substantially.

Sensitivity results for Voynich:

- vocabulary 128/512/2,048: net gains `+0.0094/-0.0103/-0.0725` bit/word;
- the same settings' matched-link advantages: `0.0397/0.0673/0.0681`;
- Bayesian strength 5/20/100: net gains `-0.0487/-0.0103/+0.0029`;
- strength 5/20/100 matched-link advantages: `0.1057/0.0673/0.0304`;
- adding first/last grouped-EVA units to the target morphology class leaves
  only `0.0016` bit/word over the matched shuffle.

The fine morphology setting is not the primary cross-language comparison:
first/last characters identify control-language words at different rates and
can condition almost all their lexical uncertainty away.

## Nested residual-link decomposition

Implementation: `scripts/decompose_residual_links.py`

Tests: `tests/test_decompose_residual_links.py`

This is the completed version of the previous handoff's “best next research
step.”  It adds paragraph state, final-to-initial grouped-EVA boundary
transitions, two normalized copy/edit channels, training-only 8/16-way PPMI
word classes, whole-previous-line edit/class pools, and corrected exact
identity channels.  Five whole-quire outer folds are untouched while mixture
weights are learned from three inner-quire out-of-fold sets.  Every feature
expert is a proper conditional/morphology-marginal density-ratio distribution.

The matched null preserves Currier, topic, target line position, same-line
source morphology, every previous-line source morphology and normalized slot,
and the target itself.  It changes 99.1% of depth-0 Voynich records.

Primary depth-0 results (held-out gain over the same procedural baseline,
bits/word):

| Corpus | Boundary | Exact identity | Latent class | Previous-line pool | Full |
|---|---:|---:|---:|---:|---:|
| **Voynich** | **0.0630** | 0.0193 | 0.0066 | 0.0567 | **0.1239** |
| Hebrew Wikipedia | 0.0338 | 0.1075 | 0.0220 | 0.0338 | 0.1305 |
| Latin Wikipedia | 0.0316 | 0.0807 | 0.0321 | 0.1066 | 0.1442 |
| Spanish *Picatrix* | 0.1235 | 0.3506 | 0.1846 | 0.0291 | 0.3686 |
| Word-shuffled *Picatrix* | -0.0005 | -0.0007 | -0.0009 | 0.0028 | 0.0020 |

Voynich actual links beat 49 matched permutations by 0.1180 bit/word jointly
(`p=0.02`, the resolution floor).  The advantage is positive in all outer
folds (0.0947--0.1379); net full gain is also positive in all folds
(0.1004--0.1658).  Mean joint weights expose the main contrast:

- Voynich: boundary 0.583, previous-line loose edit 0.132, exact previous word
  0.126, paragraph 0.106;
- Hebrew: exact previous word 0.845, previous-line loose edit 0.096, boundary
  0.044;
- Latin: exact previous word 0.624, previous-line loose edit 0.135, line-class
  experts 0.173 combined, boundary 0.032; and
- *Picatrix*: exact previous word 0.908 and class-16 0.041.

The total Voynich gain is therefore comparable with Hebrew and Latin in this
finite suite.  Do not describe it as “no sequence” or as comfortably below all
language.  The defensible novelty is its composition: boundary mechanics carry
what exact word identity or class carries in the controls.

The negative control caught and corrected a real implementation mistake.  An
initial version allowed paragraph and exact-identity experts to double as
alternate generic smoothers and gave shuffled *Picatrix* +0.065 bit/word.
Those results were discarded.  Conditional/marginal density-ratio experts
reduce it to +0.0020 at depth 0 and -0.0002 at depth 1.

Morphology sensitivity:

| Depth | Supplied target form | Baseline | Full gain | Actual over null |
|---:|---|---:|---:|---:|
| 0 | grouped-EVA length | 4.3927 | 0.1239 | 0.1180 |
| 1 | length + first/last unit | 1.3206 | 0.0103 | 0.0051 |
| 2 | length + first/last two units | 0.3327 | -0.0015 | 0.0013 |

This collapse is not uniquely Voynich.  At depth 1, full gains are 0.0037
Hebrew, 0.0044 Latin, 0.0211 *Picatrix*, and -0.0002 shuffled *Picatrix*.
Edge characters overcondition ordinary words too.  Use depth 0 for the fair
cross-script anatomy comparison and depths 1--2 only as sensitivity bounds.

## Reproduction

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

Clone the public comparison corpus at the recorded commit:

```bash
git clone https://github.com/chirila/Voynich-public.git \
  /tmp/voynich-public-entropy
git -C /tmp/voynich-public-entropy checkout \
  decc4caaa6515b86e42a219d1da8d81114736f2e
```

Run the public-data residual comparison:

```bash
python3 scripts/residual_sequence_information.py \
  --morphology-depth 0 \
  --vocabulary 512 \
  --strength 20 \
  --permutations 99 \
  --control 'Hebrew Wikipedia=/tmp/voynich-public-entropy/Corpora/Wikipedia_texts/full/Hebrew' \
  --control 'Latin Wikipedia=/tmp/voynich-public-entropy/Corpora/Wikipedia_texts/full/Latin' \
  --control 'Spanish Picatrix=/tmp/voynich-public-entropy/Corpora/Historical_texts/Picatrix' \
  --control 'Middle English Cirurgie=/tmp/voynich-public-entropy/Corpora/Historical_texts/Cirurgie' \
  --shuffled-control 'Spanish Picatrix shuffled=/tmp/voynich-public-entropy/Corpora/Historical_texts/Picatrix'
```

Run the nested decomposition for Voynich and then the controls without
recomputing Voynich:

```bash
python3 scripts/decompose_residual_links.py \
  --outer-folds 5 --inner-folds 3 \
  --morphology-depth 0 --vocabulary 512 --strength 20 \
  --permutations 49

python3 scripts/decompose_residual_links.py \
  --skip-voynich \
  --outer-folds 5 --inner-folds 3 \
  --morphology-depth 0 --vocabulary 512 --strength 20 \
  --permutations 0 \
  --control 'Hebrew Wikipedia=/tmp/voynich-public-entropy/Corpora/Wikipedia_texts/full/Hebrew' \
  --control 'Latin Wikipedia=/tmp/voynich-public-entropy/Corpora/Wikipedia_texts/full/Latin' \
  --control 'Spanish Picatrix=/tmp/voynich-public-entropy/Corpora/Historical_texts/Picatrix' \
  --shuffled-control 'Spanish Picatrix shuffled=/tmp/voynich-public-entropy/Corpora/Historical_texts/Picatrix'
```

The depth-1 control run changes only `--morphology-depth 1`; Voynich depth-1
and depth-2 runs used `--permutations 19`.  Full runs are CPU-heavy because
every mixture weight is selected from inner-quire out-of-fold predictions.

The `--shuffled-control` option performs a true deterministic word shuffle in
memory.  Do not use `data/capote_wordscramble_char/clean_wordshuffled_incoldblood.txt`
as a negative control: its old preparation script split on periods and in fact
shuffled lines/passages while leaving local prose order intact.

See the reproduction section of `RESEARCH_REPORT.md` for the remaining audits
and required external Hebrew sources.

## Important assumptions and traps

1. **Use the IVTFF parser.** IVTFF `.`, `,`, `<->`, and `<~>` all encode
   apparent spaces; uncertain glyphs and locus metadata require explicit
   handling.  A generic `split()` silently corrupts the corpus.
2. **EVA is a transcription, not ground truth.** Common multicodepoint forms
   (`ch`, `sh`, gallows benches) and uncertain spaces make entropy and shape
   results representation-dependent.
3. **Do not equate code length with meaning.** The 394,854-bit lossless code is
   an upper bound for one normalized surface string, not an estimate of
   semantic content.
4. **Do not use in-sample entropy as evidence.** Splits are by quire or page;
   model choice and cipher keys must be frozen before held-out scoring.
5. **Do not treat corpus rank as a p-value.** The 294 Wikipedia samples are not
   independent exchangeable draws, and some contain repeated templates.
6. **Metadata are confounded.** Currier, hand, illustration class, quire, and
   manuscript order overlap.  The residual experiment conditions on declared
   fields but cannot condition unknown production state away.
7. **The Hebrew result is shape-only.** Hebrew ranks unusually close under an
   anagram-insensitive decomposition feature, especially in Currier B and the
   herbal section, but the same feature gives the same answer for meaningless
   pattern-matched pseudotext.  Sequence, zodiac, formula, and fitted-key tests
   do not support a simple Hebrew translation.
8. **A significant matched-link result is not syntax.** Boundary glyph rules,
   self-citation, copy/edit production, or line construction can all produce
   it.  Net held-out compression and stable decoded classes are the harder
   criteria.

## File map

- `RESEARCH_REPORT.md`: authoritative narrative, numerical results, caveats,
  and reproduction commands.
- `MEMORY.md`: compact durable facts and decisions for future sessions.
- `scripts/audit_voynich.py`: parser and foundational layout/order audits.
- `scripts/audit_language_hypotheses.py`: Hebrew/abjad feature reproduction and
  adversarial negative control.
- `scripts/compare_medical_hebrew.py`: period medical Hebrew comparison.
- `scripts/compare_hebrew_herbal_structure.py`: herbal formula-sequence test.
- `scripts/zodiac_hebrew_anchors.py`: fixed-vocabulary zodiac crib test.
- `scripts/compare_mechanisms.py`: held-out Hebrew-substitution versus native
  morphology/layout/copy tournament.
- `scripts/estimate_information_bounds.py`: prequential surface code and broad
  word-order benchmark.
- `scripts/residual_sequence_information.py`: conditional residual identity
  experiment and matched permutations.
- `scripts/decompose_residual_links.py`: nested boundary/edit/class/paragraph/
  previous-line decomposition with density-ratio experts and strict matched
  nulls.

## Best next research step

Localize the boundary-dominated result instead of fitting a still larger global
mixture.  Export per-token expert log losses and aggregate them by quire,
Currier, topic, paragraph line, and source/target edge pair.  Then test transfer:
learn boundary tables and latent classes on Currier A (or one topic/hand proxy)
and score Currier B, and reverse the direction.  A manuscript-wide stable
boundary grammar is more compatible with an encoding or orthographic rule; a
page-local effect that follows physical adjacency is more compatible with a
generation procedure.  Pair this with alternative EVA space treatments,
because a cross-space glyph rule is precisely the result most vulnerable to
incorrect token boundaries.

For visualization, prioritize a morphology-depth retention slopegraph, a
family-gain/joint-weight “signal anatomy” matrix, and a quire-by-quire boundary
residual atlas.  Do not make mixture weights look additive: they measure model
reliance among correlated experts, not shares of semantic information.
