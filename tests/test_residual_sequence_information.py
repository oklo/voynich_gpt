from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from residual_sequence_information import (  # noqa: E402
    EncodedRecord,
    PageSequence,
    ResidualIdentityModel,
    SequenceRecord,
    aligned_word,
    matched_neighbor_permutation,
    morphology_signature,
    records_from_sequences,
    reflow_control,
)


def record(
    word: str,
    morphology: tuple[str, ...],
    *,
    previous_word: str | None,
    previous_morphology: tuple[str, ...] | None,
) -> SequenceRecord:
    return SequenceRecord(
        word=word,
        morphology=morphology,
        currier="A",
        topic="H",
        position="middle",
        page="f1r",
        quire="A",
        previous_word=previous_word,
        previous_morphology=previous_morphology,
        previous_line_word=None,
        previous_line_morphology=None,
    )


class ResidualSequenceInformationTests(unittest.TestCase):
    def test_morphology_signature_groups_declared_eva_units(self):
        self.assertEqual(
            morphology_signature("qokedy", depth=1, grouped_eva=True),
            ("6", "q", "|", "y"),
        )
        self.assertEqual(
            morphology_signature("chedy", depth=2, grouped_eva=True),
            ("4", "ch", "e", "|", "d", "y"),
        )

    def test_previous_line_alignment_uses_normalized_midpoints(self):
        previous = ("a", "b", "c", "d")
        self.assertEqual(aligned_word(previous, index=0, line_length=2), "a")
        self.assertEqual(aligned_word(previous, index=1, line_length=2), "c")
        self.assertIsNone(aligned_word(None, index=0, line_length=2))

    def test_records_keep_same_line_and_previous_line_contexts_separate(self):
        page = PageSequence(
            "f1r",
            "A",
            "A",
            "H",
            (("aa", "bb"), ("cc", "dd")),
        )
        records = records_from_sequences(
            [page], morphology_depth=0, grouped_eva=False
        )
        self.assertIsNone(records[0].previous_word)
        self.assertEqual(records[1].previous_word, "aa")
        self.assertEqual(records[2].previous_line_word, "aa")
        self.assertEqual(records[3].previous_line_word, "bb")

    def test_reflow_preserves_layout_and_metadata(self):
        page = PageSequence("f1r", "A", "B", "H", (("x", "y"), ("z",)))
        result = reflow_control([page], ["one", "two", "three", "extra"])
        self.assertEqual(result[0].lines, (("one", "two"), ("three",)))
        self.assertEqual(result[0].quire, "A")
        self.assertEqual(result[0].currier, "B")

    def test_each_probability_layer_is_normalized(self):
        shape = ("2", "|" )
        training = [
            record("aa", shape, previous_word="xx", previous_morphology=shape),
            record("bb", shape, previous_word="yy", previous_morphology=shape),
            record("aa", shape, previous_word="xx", previous_morphology=shape),
        ]
        model = ResidualIdentityModel(
            training, vocabulary_limit=8, alpha=0.5, strength=2.0
        )
        encoded = model.encode(training[0])
        for model_name in model.model_names:
            self.assertAlmostEqual(model.probability_mass(encoded, model_name), 1.0)

    def test_exact_predecessor_predicts_identity_within_same_shape(self):
        shape = ("2", "|")
        training = []
        for _ in range(20):
            training.append(
                record("aa", shape, previous_word="xx", previous_morphology=shape)
            )
            training.append(
                record("bb", shape, previous_word="yy", previous_morphology=shape)
            )
        model = ResidualIdentityModel(
            training, vocabulary_limit=8, alpha=0.5, strength=2.0
        )
        matching = model.encode(training[0])
        mismatching = EncodedRecord(
            **{
                **matching.__dict__,
                "previous_identity": "yy",
            }
        )
        self.assertGreater(
            model.probabilities(matching)["previous_word_identity"],
            model.probabilities(mismatching)["previous_word_identity"],
        )

    def test_matched_permutation_preserves_nuisance_fields(self):
        shape = ("2", "|")
        records = [
            EncodedRecord(
                target=f"target-{index}",
                morphology=shape,
                currier="A",
                topic="H",
                position="middle",
                page="f1r",
                quire="A",
                previous_identity=f"previous-{index}",
                previous_morphology=shape,
                previous_line_identity=f"line-{index}",
                previous_line_morphology=shape,
            )
            for index in range(8)
        ]
        permuted, changed = matched_neighbor_permutation(records, seed=7)
        self.assertGreater(changed, 0)
        self.assertEqual([item.target for item in permuted], [item.target for item in records])
        self.assertEqual(
            [item.morphology for item in permuted],
            [item.morphology for item in records],
        )
        self.assertCountEqual(
            [item.previous_identity for item in permuted],
            [item.previous_identity for item in records],
        )
        self.assertCountEqual(
            [item.previous_line_identity for item in permuted],
            [item.previous_line_identity for item in records],
        )


if __name__ == "__main__":
    unittest.main()
