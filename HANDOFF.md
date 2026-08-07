# Voynich research handoff

## State of the project

This repository is no longer primarily a nanoGPT experiment.  It is a
dependency-light, falsification-first audit of claims about the Voynich
Manuscript, with the original notebooks and logs retained as provenance.

The main narrative is `RESEARCH_REPORT.md`.  The branch is `main`, the remote
is `https://github.com/oklo/voynich_gpt.git`, and commits through `b00c864`
were pushed before this handoff was written.  The residual-sequence experiment
described below is the next checkpoint and should be committed and pushed with
this file.

At the time of writing:

- all 50 standard-library unit tests pass;
- `git diff --check` passes;
- Ruff is not installed in the environment, so no Ruff result is claimed;
- all new analysis code is Python-standard-library only; and
- the working corpus is `IT2a-n.txt`, parsed as IVTFF rather than as generic
  whitespace text.

## Bottom line

No translation or decipherment has been established.  The strongest current
result is structural:

> Under EVA's marked spaces and stable exact surface-word identity, Voynichese
> is a poor match for ordinary communicative prose.  Its dominant predictable
> structure is word-internal and layout-conditioned.  Exact neighboring word
> identities have a real but weak association after those variables are
> supplied, and they do not improve held-out compression.

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

The apparent tension is important: real links score better than matched fake
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

## Best next research step

Explain, rather than merely detect, the remaining 0.067-bit/word link signal.
Build a nested-quire model that adds the following one at a time:

1. cross-boundary final-to-initial EVA transition;
2. normalized edit/copy family between adjacent forms;
3. latent word classes learned on training quires only;
4. paragraph-start and within-paragraph state; and
5. previous-line pools beyond the single aligned token.

Every component should be selected on inner training folds, then scored on
untouched outer quires against the same matched permutations.  If boundary and
copy/edit features absorb the 0.067 bits and latent identity classes add no
net compression, the residual is best understood as a production trace.  If a
stable latent class model adds control-like held-out gain and transfers across
Currier/hand/topic strata, that would be the first serious evidence here for a
hidden lexical or grammatical payload.
