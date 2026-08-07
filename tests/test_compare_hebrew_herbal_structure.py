from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_hebrew_herbal_structure import (  # noqa: E402
    canonical_anagram_word,
    extract_bnf_entries,
    extract_page_xml_lines,
    matched_entry_bootstrap,
)


class HebrewHerbalStructureTests(unittest.TestCase):
    def test_catalog_entry_state_machine(self):
        fixture = """<p>F. 67v :  <blockquote>
<p>אנטולא מינור</p>
<p>לרפואת כל מני פצעים קח העשב</p>
<p>סטלאריאה</p>
<p>מעלת להריון מאוד קח מזה העשב</p>
<p>בג''ה [בינה, גבורה, הוד] ואם לא תהר בזה כי היא עקרה</p>
<p>F. 68 r :</p>
<p>קנלריטאס רומנא למי שהוא דקור קח עלה זה העשב</p>
<p>רינא</p>
</blockquote>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.html"
            path.write_text(fixture, encoding="utf-8")
            entries = extract_bnf_entries(path)
        self.assertEqual([entry.heading for entry in entries], [
            "אנטולא מינור",
            "סטלאריאה",
            "קנלריטאס רומנא",
        ])
        self.assertIn("בגה", entries[1].words)
        self.assertNotIn("בינה", entries[1].words)

    def test_anagram_key_preserves_multiset_not_order(self):
        self.assertEqual(
            canonical_anagram_word("אבבג", grouped_eva=False),
            canonical_anagram_word("בגאב", grouped_eva=False),
        )

    def test_page_xml_subsets_are_merged_in_physical_order(self):
        fixture = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
<Page><TextRegion>
<TextLine id="later"><Coords points="0,20 10,20"/><TextEquiv><Unicode>גד</Unicode></TextEquiv></TextLine>
<TextLine id="earlier"><Coords points="0,10 10,10"/><TextEquiv><Unicode>אב</Unicode></TextEquiv></TextLine>
</TextRegion></Page></PcGts>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.xml"
            path.write_text(fixture, encoding="utf-8")
            lines = extract_page_xml_lines([path, path])
        self.assertEqual(lines, [["אב"], ["גד"]])
        self.assertNotEqual(
            canonical_anagram_word("אבבג", grouped_eva=False),
            canonical_anagram_word("אבגג", grouped_eva=False),
        )

    def test_matched_bootstrap_reports_reference_advantage(self):
        reference = [["take", "this", "herb", "take", "this", "herb"]] * 4
        target = [[f"w{row}-{column}" for column in range(12)] for row in range(4)]
        result = matched_entry_bootstrap(reference, target, iterations=10, seed=7)
        self.assertEqual(result["outcomes"]["2"]["reference_higher"], 10)
        self.assertEqual(result["outcomes"]["3"]["reference_higher"], 10)


if __name__ == "__main__":
    unittest.main()
