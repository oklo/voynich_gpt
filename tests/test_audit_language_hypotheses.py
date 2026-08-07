import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from audit_language_hypotheses import (  # noqa: E402
    bhattacharyya_distance,
    conditional_pattern_distance,
    decomposition_pattern,
    pattern_distribution,
    pattern_matched_pseudotext,
    unicode_words,
)


class LanguageHypothesisTests(unittest.TestCase):
    def test_decomposition_pattern_matches_definition(self):
        self.assertEqual(decomposition_pattern("seems"), (2, 2, 1))
        self.assertEqual(decomposition_pattern("beams"), (1, 1, 1, 1, 1))

    def test_bhattacharyya_distance(self):
        left = {(1,): 0.5, (2,): 0.5}
        self.assertAlmostEqual(bhattacharyya_distance(left, left), 0.0)
        self.assertTrue(math.isinf(bhattacharyya_distance(left, {(3,): 1.0})))

    def test_pattern_pseudotext_is_exact_negative_control(self):
        words = ["seems", "beams", "banana", "letter"]
        pseudo = pattern_matched_pseudotext(words, grouped_eva=False, seed=4)
        self.assertEqual(pattern_distribution(words), pattern_distribution(pseudo))
        self.assertNotEqual(words, pseudo)

    def test_conditional_pattern_distance_removes_length_mix(self):
        target = ["ab", "cd", "aab"]
        # The common length has the same repeat structure; absent target lengths
        # still contribute a penalty rather than being silently renormalized.
        self.assertGreater(conditional_pattern_distance(target, ["xy"], grouped_eva=False), 0)

    def test_unicode_word_extraction(self):
        self.assertEqual(unicode_words("שלום, עוֹלם! Égalité"), ["שלום", "עולם", "égalité"])


if __name__ == "__main__":
    unittest.main()
