from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from zodiac_hebrew_anchors import (  # noqa: E402
    Constraint,
    Label,
    ZODIAC_TARGETS,
    candidate_forms,
    derive_ordered_mapping,
    extract_zodiac_labels,
    fit_exact_mappings,
    heldout_scores,
    local_anagram_key_count,
    normalize_hebrew,
    ordered_constraints,
)


class ZodiacHebrewAnchorTests(unittest.TestCase):
    def test_final_forms_can_be_collapsed(self):
        self.assertEqual(
            normalize_hebrew("דגים", collapse_final_forms=True), "דגימ"
        )
        self.assertEqual(
            normalize_hebrew("דגים", collapse_final_forms=False), "דגים"
        )

    def test_candidate_forms_are_declared_before_matching(self):
        aries = next(target for target in ZODIAC_TARGETS if target.key == "aries")
        forms = candidate_forms(aries, collapse_final_forms=False)
        self.assertIn("טלה", forms)
        self.assertIn("מזלטלה", forms)
        self.assertIn("ניסן", forms)
        self.assertIn("חודשניסן", forms)

    def test_ordered_mapping_rejects_inconsistent_repeats(self):
        self.assertIsNone(
            derive_ordered_mapping(list("aba"), "אבג", injective=False)
        )
        self.assertIsNone(
            derive_ordered_mapping(list("abc"), "אאב", injective=True)
        )
        self.assertEqual(
            derive_ordered_mapping(list("abc"), "אאב", injective=False),
            (("a", "א"), ("b", "א"), ("c", "ב")),
        )

    def test_fit_requires_one_global_mapping(self):
        cache = {
            ("p1", "one"): (
                Constraint((("a", "א"), ("b", "ב")), "p1.1", "ab", "אב", False),
            ),
            ("p2", "two"): (
                Constraint((("a", "א"), ("c", "ג")), "p2.1", "ac", "אג", False),
            ),
        }
        fit = fit_exact_mappings(
            [("p1", "one"), ("p2", "two")],
            cache,
            injective=True,
            beam_size=100,
        )
        self.assertEqual(fit.matched_pages, 2)
        self.assertEqual(dict(fit.example_mapping), {"a": "א", "b": "ב", "c": "ג"})

    def test_heldout_requires_symbols_to_have_been_learned(self):
        cache = {
            ("p", "right"): (
                Constraint((("a", "א"),), "p.1", "a", "א", False),
            ),
            ("p", "wrong"): (
                Constraint((("b", "ב"),), "p.2", "b", "ב", False),
            ),
        }
        scored = heldout_scores(
            "p", "right", ((("a", "א"),),), ["right", "wrong"], cache
        )
        self.assertEqual(scored["correct_vote"], 1.0)
        self.assertEqual(scored["votes"]["wrong"], 0.0)

    def test_anagram_local_key_count(self):
        self.assertEqual(local_anagram_key_count(list("abca"), "אבגא"), 2)
        self.assertEqual(local_anagram_key_count(list("aabc"), "אבגד"), 0)

    def test_real_source_has_all_twelve_diagrams(self):
        root = Path(__file__).parents[1]
        labels = extract_zodiac_labels(root / "IT2a-n.txt")
        expected = {page for target in ZODIAC_TARGETS for page in target.pages}
        self.assertEqual(set(labels), expected)
        self.assertTrue(all(labels.values()))

    def test_constraint_reversal_is_explicit(self):
        constraints = ordered_constraints(
            [Label("p", "p.1", "abc")],
            ["גבא"],
            grouped_eva=False,
            injective=True,
            allow_reversal=True,
        )
        self.assertTrue(any(constraint.reversed_eva for constraint in constraints))


if __name__ == "__main__":
    unittest.main()
