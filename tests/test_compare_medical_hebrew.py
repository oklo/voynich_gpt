from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_medical_hebrew import (  # noqa: E402
    extract_bnf_herbal_transcript,
    extract_canon_pages,
    normalize_hebrew,
)


class MedicalHebrewTests(unittest.TestCase):
    def test_final_letter_normalization(self):
        self.assertEqual(normalize_hebrew("מלך", True), "מלכ")
        self.assertEqual(normalize_hebrew("מלך", False), "מלך")

    def test_confidence_and_section_extraction(self):
        fixture = """<?xml version="1.0" encoding="UTF-8"?>
<DjVuXML><BODY>
<OBJECT width="100" height="100">
<PARAM name="PAGE" value="sample_0150.djvu"/>
<HIDDENTEXT><PAGECOLUMN><REGION><PARAGRAPH><LINE>
<WORD coords="1,2,3,4" x-confidence="90">תרופה</WORD>
<WORD coords="4,2,6,4" x-confidence="20">שגיאה</WORD>
</LINE></PARAGRAPH></REGION></PAGECOLUMN></HIDDENTEXT>
</OBJECT></BODY></DjVuXML>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.xml"
            path.write_text(fixture, encoding="utf-8")
            result = extract_canon_pages(
                path, minimum_confidence=50, normalize_finals=True
            )
        self.assertEqual(result["book_II_materia_medica"], [["תרופה"]])

    def test_bnf_herbal_transcript_extraction(self):
        fixture = "<p>F. 67v :</p><blockquote><p>תרופה מלך</p></blockquote>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.html"
            path.write_text(fixture, encoding="utf-8")
            result = extract_bnf_herbal_transcript(path, normalize_finals=True)
        self.assertEqual(result, ["תרופה", "מלכ"])


if __name__ == "__main__":
    unittest.main()
