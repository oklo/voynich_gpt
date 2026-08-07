import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from estimate_information_bounds import (  # noqa: E402
    BOUNDARY,
    Corpus,
    conditional_entropy,
    nonoverlapping_windows,
    normalized_words,
    prequential_mixture_bits,
    shuffled_within_blocks,
    words_to_stream,
    word_order_gain,
)


class InformationBoundTests(unittest.TestCase):
    def test_normalization_keeps_letters_and_attached_marks(self):
        self.assertEqual(
            normalized_words("Été, אבָ 42!") ,
            ("été", "אבָ"),
        )

    def test_word_stream_has_one_decodable_boundary_per_word(self):
        self.assertEqual(
            words_to_stream((("a", "b"), ("c",))),
            ("a", "b", BOUNDARY, "c", BOUNDARY),
        )

    def test_prequential_code_is_finite_and_rewards_repetition(self):
        repeated = tuple("ab" * 200)
        irregular = tuple(("a", "b", "c", "d") * 100)
        repeated_bits, weights = prequential_mixture_bits(
            repeated, vocabulary_size=4, max_order=4
        )
        irregular_bits, _ = prequential_mixture_bits(
            irregular, vocabulary_size=4, max_order=4
        )
        self.assertTrue(math.isfinite(repeated_bits))
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertLess(repeated_bits, irregular_bits)

    def test_constant_stream_approaches_zero_bits_per_unit(self):
        bits, _ = prequential_mixture_bits(
            ("a",) * 10_000, vocabulary_size=2, max_order=4
        )
        self.assertLess(bits / 10_000, 0.01)

    def test_conditional_entropy_detects_deterministic_alternation(self):
        self.assertAlmostEqual(conditional_entropy(tuple("abababab")), 0.0)

    def test_windows_are_equal_and_nonoverlapping(self):
        tokens = tuple(str(index) for index in range(25))
        windows = nonoverlapping_windows(tokens, 5, 4)
        self.assertEqual([len(window) for window in windows], [5, 5, 5, 5])
        self.assertEqual(len(set().union(*(set(window) for window in windows))), 20)

    def test_shuffle_preserves_words_within_each_block(self):
        words = tuple((str(index),) for index in range(12))
        shuffled = shuffled_within_blocks(words, block_size=4, seed=3)
        self.assertEqual(set(shuffled), set(words))
        for start in range(0, 12, 4):
            self.assertEqual(set(shuffled[start : start + 4]), set(words[start : start + 4]))

    def test_word_order_gain_detects_a_repeated_local_grammar(self):
        corpus = Corpus(
            name="synthetic",
            family="test",
            source="test",
            words=tuple(("the", "cat", "sat", "and", "the", "dog", "ran") * 500),
        )
        _, _, gain, _ = word_order_gain(
            corpus,
            maximum_words=len(corpus.words),
            block_size=100,
            shuffles=4,
            vocabulary_limit=16,
            max_order=2,
            concentration=100,
            seed=408,
        )
        self.assertGreater(gain, 0.5)


if __name__ == "__main__":
    unittest.main()
