from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_voynich import (  # noqa: E402
    levenshtein_at_most_one,
    parse_ivtff,
    tokenize_ivtff,
)


class TokenizerTests(unittest.TestCase):
    def test_all_ivtff_word_spaces_split_tokens(self) -> None:
        tokens, certain = tokenize_ivtff("<%>daiin.chol,shol<->qokaiin<~>chedy<$>")
        expected = ("daiin", "chol", "shol", "qokaiin", "chedy")
        self.assertEqual(tokens, expected)
        self.assertEqual(certain, expected)

    def test_uncertain_token_is_retained_but_not_certain(self) -> None:
        tokens, certain = tokenize_ivtff("[d:t]aiin.qo?ey.{ch}ol")
        self.assertEqual(tokens, ("daiin", "qo?ey", "chol"))
        self.assertEqual(certain, ("daiin", "chol"))


class ParserTests(unittest.TestCase):
    def test_metadata_locus_types_and_text_tags(self) -> None:
        fixture = """#=IVTFF EvaT 2.0 M 3
<f1r> <! $Q=A $I=H $L=A $H=@>
<f1r.1,@P0> <%><@H=1>daiin<->chol<$>
<f1r.2,@Lf> plant
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.ivtff"
            path.write_text(fixture, encoding="utf-8")
            pages = parse_ivtff(path)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].loci[0].tokens, ("daiin", "chol"))
        self.assertEqual(pages[0].loci[0].locus_type, "P0")
        self.assertEqual(pages[0].loci[0].metadata["H"], "1")
        self.assertEqual(pages[0].lines(), [["daiin", "chol"]])

    def test_repository_corpus_boundary_accounting(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pages = parse_ivtff(root / "IT2a-n.txt")
        loci = [locus for page in pages for locus in page.loci]
        parsed_count = sum(len(locus.tokens) for locus in loci)
        drawing_spaces = sum(
            locus.raw_text.count("<->") + locus.raw_text.count("<~>") for locus in loci
        )
        legacy_text = (root / "data/voynich_char/clean_taka.txt").read_text(encoding="utf-8")
        legacy_count = sum(
            bool(token)
            for line in legacy_text.splitlines()
            for token in line.split(".")
        )
        self.assertEqual(parsed_count, 37_919)
        self.assertEqual(drawing_spaces, 875)
        self.assertEqual(legacy_count, 37_044)
        self.assertEqual(parsed_count - legacy_count, drawing_spaces)
        label_words = [
            word
            for locus in loci
            if locus.locus_type.startswith("L")
            for word in locus.certain_tokens
        ]
        self.assertEqual(len(label_words), 1_023)
        self.assertEqual(len(set(label_words)), 748)


class EditDistanceTests(unittest.TestCase):
    def test_one_edit(self) -> None:
        self.assertTrue(levenshtein_at_most_one("chol", "chor"))
        self.assertTrue(levenshtein_at_most_one("chol", "chool"))
        self.assertTrue(levenshtein_at_most_one("chool", "chol"))
        self.assertFalse(levenshtein_at_most_one("chol", "chol"))
        self.assertFalse(levenshtein_at_most_one("chol", "daiin"))


if __name__ == "__main__":
    unittest.main()
