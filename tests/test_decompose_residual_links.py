from __future__ import annotations

import gzip
import io
import json
import statistics
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from decompose_residual_links import (  # noqa: E402
    DecompositionRecord,
    ExpertSuite,
    StructuredLine,
    StructuredPage,
    audit_corpus,
    fit_mixture_weights,
    matched_source_permutation,
    nearest_previous_line_index,
    open_token_trace,
    reflow_structured,
)
from residual_sequence_information import morphology_signature  # noqa: E402


SHAPE = ("2", "|")


def record(
    word: str,
    *,
    previous_word: str | None,
    previous_line_word: str | None,
    previous_line_words: tuple[str, ...],
    index: int = 0,
) -> DecompositionRecord:
    return DecompositionRecord(
        word=word,
        morphology=SHAPE,
        currier="A",
        topic="H",
        position="middle",
        paragraph_state="1:open",
        page="f1r",
        page_index=0,
        quire="A",
        line_index=0,
        word_index=1,
        line_length=3,
        previous_word=previous_word,
        previous_morphology=SHAPE if previous_word is not None else None,
        previous_line_word=previous_line_word,
        previous_line_morphology=(
            SHAPE if previous_line_word is not None else None
        ),
        previous_line_words=previous_line_words,
        previous_line_morphologies=tuple(SHAPE for _ in previous_line_words),
        previous_line_index=(index if previous_line_word is not None else None),
    )


class ResidualLinkDecompositionTests(unittest.TestCase):
    def test_previous_line_alignment_uses_normalized_midpoints(self):
        previous = StructuredLine(("a", "b", "c", "d"), 0, True, False)
        self.assertEqual(
            nearest_previous_line_index(previous, index=0, current_length=2), 0
        )
        self.assertEqual(
            nearest_previous_line_index(previous, index=1, current_length=2), 2
        )
        self.assertIsNone(
            nearest_previous_line_index(None, index=0, current_length=2)
        )

    def test_reflow_preserves_paragraph_and_page_metadata(self):
        template = [
            StructuredPage(
                "f1r",
                "A",
                "B",
                "H",
                (
                    StructuredLine(("x", "y"), 0, True, False),
                    StructuredLine(("z",), 1, False, True),
                ),
            )
        ]
        result = reflow_structured(template, ["one", "two", "three", "extra"])
        self.assertEqual(result[0].lines[0].words, ("one", "two"))
        self.assertEqual(result[0].lines[1].words, ("three",))
        self.assertEqual(result[0].lines[1].paragraph_line, 1)
        self.assertTrue(result[0].lines[1].paragraph_end)
        self.assertEqual(result[0].quire, "A")
        self.assertEqual(result[0].currier, "B")

    def test_every_probability_expert_is_normalized(self):
        training: list[DecompositionRecord] = []
        for _ in range(12):
            training.extend(
                (
                    record(
                        "aa",
                        previous_word="xx",
                        previous_line_word="yy",
                        previous_line_words=("xx", "yy"),
                    ),
                    record(
                        "bb",
                        previous_word="yy",
                        previous_line_word="xx",
                        previous_line_words=("yy", "xx"),
                        index=1,
                    ),
                )
            )
        suite = ExpertSuite(
            training,
            vocabulary_limit=16,
            alpha=0.5,
            strength=2.0,
            grouped_eva=False,
        )
        for mass in suite.probability_masses(training[0]).values():
            self.assertAlmostEqual(mass, 1.0)

    def test_density_ratio_is_neutral_when_condition_matches_marginal(self):
        training = [
            record(
                "aa",
                previous_word="xx",
                previous_line_word="yy",
                previous_line_words=("xx", "yy"),
            ),
            record(
                "bb",
                previous_word="yy",
                previous_line_word="xx",
                previous_line_words=("yy", "xx"),
                index=1,
            ),
        ]
        suite = ExpertSuite(
            training,
            vocabulary_limit=16,
            alpha=0.5,
            strength=2.0,
            grouped_eva=False,
        )
        base = {"aa": 0.6, "bb": 0.3, "<UNKNOWN>": 0.1}
        counts = Counter({"aa": 1, "bb": 1, "<UNKNOWN>": 1})
        reweighted = suite.reweight_from_counts(base, counts, counts)
        for target, probability in base.items():
            self.assertAlmostEqual(reweighted[target], probability)

    def test_mixture_em_prefers_the_better_expert(self):
        rows = [{"base": 0.2, "signal": 0.8} for _ in range(20)]
        weights = fit_mixture_weights(rows, ("base", "signal"))
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["signal"], 0.999)

    def test_matched_permutation_preserves_all_morphology_profiles(self):
        records = []
        for index in range(20):
            suffix = chr(65 + index)
            pool = (f"x{suffix}", f"y{suffix}")
            aligned_index = index % 2
            records.append(
                record(
                    "aa" if index % 2 else "bb",
                    previous_word=f"p{suffix}",
                    previous_line_word=pool[aligned_index],
                    previous_line_words=pool,
                    index=aligned_index,
                )
            )
        permuted, changed = matched_source_permutation(records, seed=7)
        self.assertGreater(changed, 0)
        self.assertEqual([item.word for item in permuted], [item.word for item in records])
        self.assertEqual(
            [item.previous_morphology for item in permuted],
            [item.previous_morphology for item in records],
        )
        self.assertEqual(
            [item.previous_line_morphologies for item in permuted],
            [item.previous_line_morphologies for item in records],
        )
        self.assertCountEqual(
            [item.previous_word for item in permuted],
            [item.previous_word for item in records],
        )
        self.assertCountEqual(
            [item.previous_line_word for item in permuted],
            [item.previous_line_word for item in records],
        )
        for source_index in (0, 1):
            self.assertCountEqual(
                [item.previous_line_words[source_index] for item in permuted],
                [item.previous_line_words[source_index] for item in records],
            )
            self.assertEqual(
                [
                    morphology_signature(
                        item.previous_line_words[source_index],
                        depth=0,
                        grouped_eva=False,
                    )
                    for item in permuted
                ],
                [item.previous_line_morphologies[source_index] for item in records],
            )

    def test_shape_fixture_matches_declared_length_only_morphology(self):
        self.assertEqual(
            morphology_signature("aa", depth=0, grouped_eva=False), SHAPE
        )

    def test_token_trace_reconstructs_actual_and_matched_scores(self):
        pages = [
            StructuredPage(
                f"f{index + 1}r",
                quire,
                "A",
                "H",
                (
                    StructuredLine(("aa", "bb", "aa"), 0, True, False),
                    StructuredLine(("bb", "aa", "bb"), 1, False, True),
                ),
            )
            for index, quire in enumerate(("A", "B", "C", "D"))
        ]
        trace = io.StringIO()
        result = audit_corpus(
            "synthetic",
            pages,
            outer_folds=2,
            inner_folds=2,
            morphology_depth=0,
            grouped_eva=False,
            vocabulary_limit=16,
            alpha=0.5,
            strength=2.0,
            permutations=3,
            seed=7,
            token_trace=trace,
        )
        rows = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual(len(rows), result["heldout_scored_words"])
        self.assertEqual(rows[0]["schema"], "voynich-residual-token-trace-v1")
        self.assertIn(rows[0]["dominant_full_expert"], ExpertSuite.expert_names)
        self.assertAlmostEqual(
            sum(rows[0]["full_mixture_responsibility"].values()), 1.0
        )
        self.assertAlmostEqual(
            statistics.fmean(row["family_log_loss_bits"]["full"] for row in rows),
            result["bits_per_word"]["full"],
        )
        self.assertAlmostEqual(
            statistics.fmean(
                row["matched_null"]["family_mean_log_loss_bits"]["full"]
                for row in rows
            ),
            result["matched_permutation"]["families"]["full"][
                "permuted_bits_per_word_mean"
            ],
        )
        self.assertAlmostEqual(
            statistics.fmean(
                row["matched_null"]["family_actual_advantage_bits"]["full"]
                for row in rows
            ),
            result["matched_permutation"]["families"]["full"][
                "actual_gain_over_permutation_bits_per_word"
            ],
        )
        self.assertAlmostEqual(
            statistics.fmean(
                row["matched_null"]["source_changed_fraction"] for row in rows
            ),
            result["matched_permutation"]["mean_changed_fraction"],
        )
        linked = next(row for row in rows if row["previous_word"] is not None)
        self.assertEqual(linked["previous_final_unit"], "a")
        self.assertEqual(linked["target_initial_unit"], "b")
        self.assertGreaterEqual(
            linked["matched_null"]["source_changed_fraction"], 0.0
        )
        self.assertLessEqual(
            linked["matched_null"]["source_changed_fraction"], 1.0
        )

    def test_token_trace_gzip_suffix_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl.gz"
            handle = open_token_trace(path)
            handle.write('{"ok":true}\n')
            handle.close()
            with gzip.open(path, "rt", encoding="utf-8") as compressed:
                self.assertEqual(json.loads(compressed.read()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
