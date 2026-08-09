# Durable project memory

## Corpus and parsing

- Primary source: `IT2a-n.txt` (IVTFF EVA transcription).
- Current conservative scope: certain paragraph-text tokens only.
- Parsed corpus: 225 pages with paragraph text and 34,411 certain tokens.
- Raw-EVA mechanism model sees 21 codepoints; grouped EVA is always labeled as
  a sensitivity representation, not asserted as the true glyph inventory.
- Apparent IVTFF spaces include `.`, `,`, `<->`, and `<~>`.
- Hold out whole quires for predictive experiments; never randomly split
  neighboring tokens across train and test.

## Conclusions that survived audit

- There is no justified translation or decipherment.
- Voynichese has strong word-internal morphology, page/section variation, and
  extreme physical line-position effects.
- Ordinary exact surface-word order is exceptionally weak under EVA spaces.
- Weak word order itself is prior art (especially Reddy & Knight 2011); the new
  work is a stronger coding benchmark and conditional residual test.
- The full 208,733-unit normalized stream has a reproducible prequential code
  of 394,854 bits (49,357 bytes), but that is not semantic information.
- Broad raw code-length ranges do not prove nonlanguage: meaningful formulaic
  Pali is at least as predictable as Voynich under that character code.
- Primary common-word block shuffle: approximately zero Voynich order gain;
  all 294 modern and 24 historical comparison samples are numerically higher,
  with documented segmentation/short-text sensitivities.
- Conditional residual test: after target length, Currier, topic, line
  position, and neighbor shapes are supplied, exact Voynich neighbor identity
  has a real matched-link advantage (~0.067 bit/word) but no net held-out code
  gain (-0.010 bit/word at vocabulary 512/strength 20).  Ordered controls gain
  +0.167 to +0.549 bit/word; shuffled *Picatrix* gains -0.021.
- The residual Voynich association is mostly previous-word (~0.096 bit/word
  versus matched links), with a smaller previous-line component (~0.018).
- A corrected nested decomposition supersedes the narrow model as the best
  residual result.  At length-only morphology, Voynich gains 0.1239 bit/word
  jointly and actual links beat strict morphology-matched nulls by 0.1180
  (`p=0.02` with 49 permutations); both are positive in all five quire folds.
- The aggregate gain is comparable with Hebrew (0.1305) and Latin (0.1442), so
  do not say Voynich sequence is comfortably below language.  Its composition
  differs: joint weight is 0.583 boundary and only 0.126 exact predecessor,
  versus 0.624--0.908 exact predecessor in the ordered controls.
- Family-only Voynich gains are boundary 0.0630, prior-line pool 0.0567, exact
  identity 0.0193, copy/edit 0.0106, paragraph 0.0074, latent class 0.0066.
- Supplying first/last grouped-EVA units collapses Voynich full gain to 0.0103,
  but also collapses Hebrew/Latin to 0.0037/0.0044; this is an overconditioning
  sensitivity, not a unique nonlanguage diagnostic.  Depth 2 gives Voynich
  -0.0015 net.
- Therefore say “real, boundary-dominated surface sequence,” not “no word
  order” and not ordinary control-like lexical sequencing.
- The best tested simple mechanism remains native layout-conditioned
  morphology.  This favors procedural/pseudotext or nonordinary encoding over
  plaintext, but does not prove nonsense.

## Hebrew and botanical lead

- Hebrew is genuinely nearest under Hauer/Kondrak-style
  decomposition-pattern shape distance for Currier B, across two
  transcriptions and grouped/raw representations.
- The herbal Currier-B subsection is especially close to the materia-medica
  book of a 1491 Hebrew *Canon* and to a small circa-1500 Hebrew herbal catalog
  transcription.
- This feature is invariant to within-word order and is exactly reproduced by
  pattern-matched pseudotext; it is not language identification.
- Held-out sequence tests, formula tests, zodiac cribs, and stable-substitution
  fits do not support a simple Hebrew plaintext/anagram translation.
- Retain “Hebrew or Hebrew-mediated botanical source” only as a narrow lead for
  more expressive, penalized cipher models.

## Methodological decisions

- Use predictive log loss/prequential code, not plug-in entropy or in-sample
  neural loss.
- Freeze vocabularies, keys, directions, and hyperparameters on training data.
- Report representation and tokenization sensitivities rather than choosing
  the favorable one after seeing results.
- Pair every claimed signal with a null that preserves the relevant nuisance
  structure and with a positive control showing estimator power.
- Distinguish:
  1. real-link advantage over a fixed matched permutation;
  2. net gain over a simpler held-out code; and
  3. total or semantic information, which these tests do not estimate.
- Do not claim a universal upper bound on payload from surface statistics.
- Density-ratio feature experts are required here.  An initial nested model let
  paragraph/exact experts act as alternate smoothers and spuriously gave
  shuffled *Picatrix* +0.065 bit/word; the corrected negative control is
  +0.0020 (depth 0) and -0.0002 (depth 1).

## Known trap

`data/capote_wordscramble_char/clean_wordshuffled_incoldblood.txt` is not a
true word shuffle.  Its historical script split prose on periods, so it mostly
shuffled lines/passages and retained local syntax.  Use the
`--shuffled-control NAME=PATH` option in
`scripts/residual_sequence_information.py` for a deterministic genuine word
shuffle.

## Reproducibility state

- Comparison corpus: `https://github.com/chirila/Voynich-public`, recorded at
  commit `decc4caaa6515b86e42a219d1da8d81114736f2e`.
- Main remote: `https://github.com/oklo/voynich_gpt.git`.
- Existing pushed checkpoints before the residual experiment:
  - `3c5c140 Add reproducible Voynich hypothesis audits`
  - `b00c864 Add information bound and word order benchmarks`
- Test command: `python3 -m unittest discover -s tests -v`.
- Current suite: 57 passing tests.
- Ruff was unavailable in the August 2026 environment; do not imply it ran.

## Next priority

Export per-token expert log losses and localize the boundary-dominated signal
by quire, Currier, topic, paragraph position, and glyph pair.  Then test
cross-Currier/topic transfer and alternative EVA space treatments.  Preferred
graphics: morphology-depth retention slopegraph, family-gain versus joint-
weight anatomy matrix, and a quire/page boundary-residual atlas.  Mixture
weights are reliance among correlated experts, not additive information shares.
