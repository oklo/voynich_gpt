# Voynich GPT, audited

This repository now contains a reproducible, falsification-first follow-up to
the original exploratory notebooks.

- [Research report](RESEARCH_REPORT.md)
- `scripts/audit_voynich.py`: IVTFF-aware corpus parser, permutation tests,
  representation sensitivity, matched natural-language controls, and held-out
  sequence models
- `scripts/audit_language_hypotheses.py`: independent reproduction and
  adversarial audit of the Hebrew/abjad decomposition-pattern claim
- `scripts/compare_medical_hebrew.py`: period- and domain-matched comparison
  with Avicenna's 1491 Hebrew *Canon* and a catalog-transcribed circa-1500 Hebrew
  herbal
- `scripts/compare_hebrew_herbal_structure.py`: sequence-level test of the
  formulaic recipe prose predicted by that Hebrew herbal comparison
- `scripts/zodiac_hebrew_anchors.py`: preregistered page-level Hebrew
  zodiac/month crib test with global-key, held-out, and shuffled-sign controls
- `scripts/compare_mechanisms.py`: held-out quire likelihood tournament between
  a fitted Hebrew substitution and Voynich-native layout/copy generators
- `scripts/estimate_information_bounds.py`: prequential universal-code bounds,
  294-language/historical reference ranges, and local word-order ablations
- `scripts/prepare_hebrew_htr_adaptation.py`: auditable PAGE-XML preparation for
  a small handwriting-recognition adaptation pilot on BnF Hébreu 1199
- `tests/`: parser, corpus-integrity, distance, and negative-control regression
  tests

The statistical audit and comparison scripts use only the Python standard
library.  The optional HTR pilot requires Kraken and an external recognition
model.  See the report for commands, results, limitations, and the next
experiments that could genuinely discriminate decipherment from structured
pseudotext.
