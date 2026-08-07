from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from prepare_hebrew_htr_adaptation import (  # noqa: E402
    levenshtein_alignment,
    normalize_hebrew,
    transfer_line_boundaries,
)


class HebrewHTRAdaptationTests(unittest.TestCase):
    def test_normalization_keeps_only_hebrew_words(self):
        self.assertEqual(normalize_hebrew("אבג, דה! 12"), "אבג דה")
        self.assertEqual(normalize_hebrew("בג''ה ט״ו"), "בגה טו")

    def test_alignment_maps_source_prefixes(self):
        mapping, distance = levenshtein_alignment("abc", "axbc")
        self.assertEqual(distance, 1)
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[-1], 4)

    def test_line_boundary_transfer(self):
        lines, report = transfer_line_boundaries(["אבג דה", "וז חט"], "אבג דה וז חט")
        self.assertEqual(lines, ["אבג דה", "וז חט"])
        self.assertEqual(report["global_character_error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
