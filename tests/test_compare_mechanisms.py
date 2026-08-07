from __future__ import annotations

import itertools
import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_mechanisms import (  # noqa: E402
    EOW,
    ConditionalWordModel,
    EditParameters,
    NgramModel,
    SubstitutionObjective,
    WordRecord,
    edit_probability,
    fit_mixture_weight,
    greedy_polish_key,
    line_position,
    reduce_hebrew_alphabet,
)


class MechanismTournamentTests(unittest.TestCase):
    def test_ngram_probabilities_are_normalized(self):
        model = NgramModel(1, ["a", "b", EOW], alpha=0.1)
        model.train([("a", "b"), ("a",)])
        for context in [("a",), ("b",), ("unseen",)]:
            self.assertAlmostEqual(
                sum(model.probability(context, token) for token in model.vocabulary),
                1.0,
            )

    def test_line_position_categories(self):
        self.assertEqual(line_position(0, 1), "single")
        self.assertEqual(line_position(0, 3), "first")
        self.assertEqual(line_position(1, 3), "middle")
        self.assertEqual(line_position(2, 3), "last")

    def test_conditional_model_uses_declared_channel(self):
        records = [
            WordRecord(("a",), ("A", "first"), None),
            WordRecord(("a",), ("A", "first"), None),
            WordRecord(("b",), ("B", "last"), None),
        ]
        model = ConditionalWordModel(
            1, ["a", "b", EOW], alpha=0.1, channel_weight=0.8
        )
        model.train(records)
        self.assertGreater(
            model.log2_word(("a",), ("A", "first")),
            model.log2_word(("b",), ("A", "first")),
        )

    def test_edit_channel_sums_to_one(self):
        parameters = EditParameters(0.1, 0.2, 0.8)
        alphabet = ("a", "b")
        total = 0.0
        # A one-symbol source can generate lengths zero through three.
        for length in range(4):
            for target in itertools.product(alphabet, repeat=length):
                total += edit_probability(("a",), target, parameters, len(alphabet))
        self.assertAlmostEqual(total, 1.0)

    def test_mixture_weight_prefers_edit_when_edit_probability_is_higher(self):
        weight, likelihood = fit_mixture_weight([(10, 0.01, 0.5)])
        self.assertGreater(weight, 0.9)
        self.assertTrue(math.isfinite(likelihood))

    def test_greedy_key_polish_never_reduces_training_likelihood(self):
        model = NgramModel(1, ["א", "ב", EOW], alpha=0.1)
        model.train([("א", "ב", "א"), ("א", "ב")])
        objective = SubstitutionObjective([("x", "y", "x")] * 4, model, ["x", "y"])
        initial = {"x": "ב", "y": "א"}
        initial_score = sum(objective.contributions(initial))
        _, polished_score, _ = greedy_polish_key(objective, initial)
        self.assertGreaterEqual(polished_score, initial_score)

    def test_hebrew_alphabet_reduction_is_external_and_21_way(self):
        training = ["אבגדה", "אבג", "אב"]
        train, calibration, rare_pair, alphabet = reduce_hebrew_alphabet(
            training, ["אב"]
        )
        self.assertEqual(len(alphabet), 21)
        self.assertEqual(len(rare_pair), 2)
        self.assertEqual(len(train), len(training))
        self.assertEqual(len(calibration), 1)


if __name__ == "__main__":
    unittest.main()
