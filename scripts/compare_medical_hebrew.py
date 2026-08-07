#!/usr/bin/env python3
"""Compare Voynich word shapes with medieval Hebrew medical literature.

The primary control is the 1491-92 Naples Hebrew edition of Avicenna's
``Canon of Medicine``.  Its Hebrew translation was completed by Nathan
ha-Me'ati in 1279; Book II covers simple drugs and materia medica.  Internet
Archive item ``4072969.med.yale.edu`` contains Yale's scan and OCR derivatives.
An optional second control extracts the BnF catalog transcription of folios
67v--68v from the circa-1500 northern-Italian Hebrew herbal Hébreu 1199.

The comparison deliberately uses only features invariant to monoalphabetic
substitution and within-word anagramming.  It cannot translate a token or prove
that the source language is Hebrew.  OCR confidence, Hebrew final-letter
normalization, Canon section, Voynich illustration type, and Currier language
are all exposed rather than silently pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from audit_language_hypotheses import (
    bhattacharyya_distance,
    conditional_pattern_distance,
    pattern_distribution,
    word_length_distribution,
)
from audit_voynich import page_words, parse_ivtff


ARCHIVE_ITEM = "4072969.med.yale.edu"
DJVU_XML_URL = (
    "https://archive.org/download/4072969.med.yale.edu/"
    "4072969.med.yale.edu_djvu.xml"
)
EXPECTED_DJVU_XML_SHA256 = "bb50fc34ee63a9cac0c7c097a5cb64ba10937cdda01a71cf73a5f137e7f5b1b5"
BNF_HERBAL_CATALOG_URL = "https://archivesetmanuscrits.bnf.fr/ark:/12148/cc8082r"

# Conservative, boundary-page-excluding samples identified from headings in
# this exact Internet Archive scan.  Book III and IV are size-matched samples,
# not the entirety of those long books.
CANON_RANGES = {
    "book_I_general_medicine": (20, 148),
    "book_II_materia_medica": (150, 280),
    "book_III_organ_diseases_sample": (294, 423),
    "book_IV_general_diseases_sample": (668, 800),
    "book_V_compound_formulary": (863, 935),
}

ILLUSTRATION_NAMES = {
    "A": "astronomical",
    "B": "biological_balneological",
    "C": "cosmological",
    "H": "herbal",
    "P": "pharmaceutical",
    "S": "marginal_stars",
    "T": "text_only",
    "Z": "zodiac",
}

HEBREW_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
PAGE_NUMBER_RE = re.compile(r"_(\d+)\.djvu$")
HEBREW_TOKEN_RE = re.compile(r"[א-ת]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hebrew(word: str, normalize_finals: bool) -> str:
    return word.translate(HEBREW_FINALS) if normalize_finals else word


def extract_canon_pages(
    path: Path,
    *,
    minimum_confidence: int,
    normalize_finals: bool,
) -> dict[str, list[list[str]]]:
    """Extract Hebrew OCR tokens by Canon section from Internet Archive XML."""

    result: dict[str, list[list[str]]] = defaultdict(list)
    for _, obj in ET.iterparse(path, events=("end",)):
        if obj.tag != "OBJECT":
            continue
        page_parameter = obj.find("PARAM[@name='PAGE']")
        page_name = page_parameter.attrib.get("value", "") if page_parameter is not None else ""
        page_match = PAGE_NUMBER_RE.search(page_name)
        page_number = int(page_match.group(1)) if page_match else -1
        section = next(
            (
                name
                for name, (first_page, last_page) in CANON_RANGES.items()
                if first_page <= page_number <= last_page
            ),
            None,
        )
        if section is not None:
            words: list[str] = []
            for element in obj.iter("WORD"):
                confidence = int(element.attrib.get("x-confidence", "0"))
                if confidence < minimum_confidence:
                    continue
                for token in HEBREW_TOKEN_RE.findall(element.text or ""):
                    token = normalize_hebrew(token, normalize_finals)
                    # Longer strings are almost entirely column-merging OCR failures.
                    if 1 <= len(token) <= 20:
                        words.append(token)
            result[section].append(words)
        obj.clear()
    return dict(result)


def extract_bnf_herbal_transcript(
    path: Path, *, normalize_finals: bool
) -> list[str]:
    """Extract the catalog's human transcription of BnF Hébreu 1199.

    The catalog transcribes the continuous Hebrew text on folios 67v--68v
    inside a blockquote. It is only a short sample of the manuscript, not a
    transcription of the 132 illustrated plant pages on folios 1--66v.
    """

    text = path.read_text(encoding="utf-8")
    marker = text.find("<p>F. 67v")
    start = text.find("<blockquote", marker)
    end = text.find("</blockquote>", start)
    if marker < 0 or start < 0 or end < 0:
        raise ValueError("Could not find the F. 67v--68v transcript block")
    return [
        normalize_hebrew(word, normalize_finals)
        for word in HEBREW_TOKEN_RE.findall(text[start:end])
    ]


def voynich_groups(source: Path) -> dict[str, list[list[str]]]:
    """Return page-blocked Voynich paragraph words by illustration and Currier."""

    groups: dict[str, list[list[str]]] = defaultdict(list)
    for page in parse_ivtff(source):
        words = page_words(page)
        if not words:
            continue
        illustration = page.metadata.get("I", "?")
        currier = page.metadata.get("L", "?")
        groups["all"].append(words)
        groups[f"currier_{currier}"].append(words)
        groups[f"{ILLUSTRATION_NAMES.get(illustration, illustration)}_currier_{currier}"].append(words)
    return dict(groups)


def flatten(pages: Sequence[Sequence[str]]) -> list[str]:
    return [word for page in pages for word in page]


def comparison(target: Sequence[str], reference: Sequence[str]) -> dict[str, float]:
    return {
        "decomposition_distance": bhattacharyya_distance(
            pattern_distribution(target, grouped_eva=True), pattern_distribution(reference)
        ),
        "length_only_distance": bhattacharyya_distance(
            word_length_distribution(target, grouped_eva=True),
            word_length_distribution(reference),
        ),
        "repetition_given_length_distance": conditional_pattern_distance(
            target, reference, grouped_eva=True
        ),
    }


def bootstrap_book_ranking(
    target: Sequence[str],
    books: dict[str, Sequence[str]],
    *,
    sample_size: int,
    iterations: int,
    rng: random.Random,
) -> dict:
    size = min(sample_size, len(target), *(len(words) for words in books.values()))
    winners: Counter[str] = Counter()
    book_distances: dict[str, list[float]] = {name: [] for name in books}
    for _ in range(iterations):
        target_sample = rng.sample(list(target), size)
        target_distribution = pattern_distribution(target_sample, grouped_eva=True)
        distances = {}
        for name, words in books.items():
            reference_sample = rng.sample(list(words), size)
            distance = bhattacharyya_distance(
                target_distribution, pattern_distribution(reference_sample)
            )
            distances[name] = distance
            book_distances[name].append(distance)
        winners[min(distances, key=distances.get)] += 1
    return {
        "sample_size": size,
        "iterations": iterations,
        "winner_counts": dict(winners.most_common()),
        "distance_summary": {
            name: {
                "mean": statistics.fmean(values),
                "q05": sorted(values)[max(0, int(0.05 * iterations) - 1)],
                "median": statistics.median(values),
                "q95": sorted(values)[min(iterations - 1, int(0.95 * iterations))],
            }
            for name, values in book_distances.items()
        },
        "warning": "token bootstrap measures ranking stability, not historical causation",
    }


def bootstrap_voynich_group_ranking(
    reference: Sequence[str],
    groups: dict[str, Sequence[str]],
    *,
    sample_size: int,
    iterations: int,
    rng: random.Random,
) -> dict:
    size = min(sample_size, len(reference), *(len(words) for words in groups.values()))
    winners: Counter[str] = Counter()
    for _ in range(iterations):
        reference_sample = rng.sample(list(reference), size)
        reference_distribution = pattern_distribution(reference_sample)
        distances = {
            name: bhattacharyya_distance(
                pattern_distribution(rng.sample(list(words), size), grouped_eva=True),
                reference_distribution,
            )
            for name, words in groups.items()
        }
        winners[min(distances, key=distances.get)] += 1
    return {
        "sample_size": size,
        "iterations": iterations,
        "winner_counts": dict(winners.most_common()),
        "warning": "token bootstrap measures ranking stability, not historical causation",
    }


def audit(
    source: Path,
    canon_xml: Path,
    *,
    minimum_confidence: int,
    normalize_finals: bool,
    bootstrap_iterations: int,
    bootstrap_sample_size: int,
    seed: int,
    direct_herbal_catalog_html: Path | None = None,
) -> dict:
    canon_pages = extract_canon_pages(
        canon_xml,
        minimum_confidence=minimum_confidence,
        normalize_finals=normalize_finals,
    )
    canon_words = {name: flatten(pages) for name, pages in canon_pages.items()}
    missing = set(CANON_RANGES) - set(canon_words)
    if missing:
        raise ValueError(f"Missing expected Canon sections: {sorted(missing)}")

    grouped_pages = voynich_groups(source)
    grouped_words = {name: flatten(pages) for name, pages in grouped_pages.items()}
    eligible_groups = {
        name: words for name, words in grouped_words.items() if len(words) >= 1_000
    }
    comparisons = {
        group_name: {
            book_name: comparison(target, reference)
            for book_name, reference in canon_words.items()
        }
        for group_name, target in eligible_groups.items()
    }

    rng = random.Random(seed)
    book_rankings = {
        group_name: bootstrap_book_ranking(
            target,
            canon_words,
            sample_size=bootstrap_sample_size,
            iterations=bootstrap_iterations,
            rng=rng,
        )
        for group_name, target in eligible_groups.items()
    }

    materia = canon_words["book_II_materia_medica"]
    currier_a = {
        name: words
        for name, words in eligible_groups.items()
        if name.endswith("_currier_A") and not name.startswith("currier_")
    }
    currier_b = {
        name: words
        for name, words in eligible_groups.items()
        if name.endswith("_currier_B") and not name.startswith("currier_")
    }
    group_rankings = {
        "within_currier_A": bootstrap_voynich_group_ranking(
            materia,
            currier_a,
            sample_size=bootstrap_sample_size,
            iterations=bootstrap_iterations,
            rng=rng,
        ),
        "within_currier_B": bootstrap_voynich_group_ranking(
            materia,
            currier_b,
            sample_size=bootstrap_sample_size,
            iterations=bootstrap_iterations,
            rng=rng,
        ),
    }

    canon_hash = sha256(canon_xml)
    result = {
        "provenance": {
            "voynich_source": str(source),
            "canon_ocr_xml": str(canon_xml),
            "canon_ocr_xml_sha256": canon_hash,
            "expected_sha256": EXPECTED_DJVU_XML_SHA256,
            "sha256_matches_expected": canon_hash == EXPECTED_DJVU_XML_SHA256,
            "internet_archive_item": ARCHIVE_ITEM,
            "download_url": DJVU_XML_URL,
            "minimum_ocr_confidence": minimum_confidence,
            "normalized_hebrew_final_letters": normalize_finals,
            "seed": seed,
        },
        "canon": {
            name: {
                "leaf_range": CANON_RANGES[name],
                "pages": len(canon_pages[name]),
                "tokens": len(words),
                "types": len(set(words)),
                "mean_word_length": statistics.fmean(map(len, words)),
                "one_letter_fraction": sum(len(word) == 1 for word in words) / len(words),
            }
            for name, words in canon_words.items()
        },
        "voynich_groups": {
            name: {"pages": len(grouped_pages[name]), "tokens": len(words)}
            for name, words in eligible_groups.items()
        },
        "comparisons": comparisons,
        "bootstrap_book_rankings": book_rankings,
        "bootstrap_materia_medica_voynich_group_rankings": group_rankings,
        "interpretive_limits": [
            "Decomposition similarity does not identify words or translate text.",
            "The Canon OCR has substantial character and word-boundary errors.",
            "Voynich illustration type, Currier language, hand, and manuscript order are confounded.",
            "Internal Canon sections share translator, typography, scanner, and OCR, making them useful controls but not independent corpora.",
            "Bootstrap stability is not a p-value for Hebrew authorship or botanical meaning.",
        ],
    }

    if direct_herbal_catalog_html is not None:
        direct_herbal = extract_bnf_herbal_transcript(
            direct_herbal_catalog_html, normalize_finals=normalize_finals
        )
        if not direct_herbal:
            raise ValueError("The BnF herbal transcript contains no Hebrew words")
        direct_comparisons = {
            name: comparison(words, direct_herbal)
            for name, words in eligible_groups.items()
        }
        direct_rankings = {}
        for currier in ("A", "B"):
            illustration_groups = {
                name: words
                for name, words in eligible_groups.items()
                if name.endswith(f"_currier_{currier}")
                and not name.startswith("currier_")
            }
            direct_rankings[f"within_currier_{currier}"] = (
                bootstrap_voynich_group_ranking(
                    direct_herbal,
                    illustration_groups,
                    sample_size=min(800, bootstrap_sample_size),
                    iterations=bootstrap_iterations,
                    rng=rng,
                )
            )
        result["direct_bnf_hebrew_herbal"] = {
            "provenance": {
                "catalog_url": BNF_HERBAL_CATALOG_URL,
                "catalog_html": str(direct_herbal_catalog_html),
                "catalog_html_sha256": sha256(direct_herbal_catalog_html),
                "shelfmark": "BnF Hébreu 1199",
                "date_and_place": "circa 1500, northern Italy",
                "transcribed_folios": "67v--68v",
            },
            "tokens": len(direct_herbal),
            "types": len(set(direct_herbal)),
            "mean_word_length": statistics.fmean(map(len, direct_herbal)),
            "comparisons": direct_comparisons,
            "bootstrap_voynich_group_rankings": direct_rankings,
            "limits": [
                "The catalog supplies only 880 tokens from folios 67v--68v, not a full manuscript transcription.",
                "The resampling treats tokens as exchangeable and measures ranking stability, not an authorship probability.",
                "The control is close in date and domain but is a different manuscript and may represent a different textual subgenre.",
            ],
        }

    return result


def markdown_report(result: dict) -> str:
    finals = result["provenance"]["normalized_hebrew_final_letters"]
    lines = [
        "# Medieval Hebrew medical comparison",
        "",
        "Distances use grouped EVA and "
        + ("final-letter-normalized" if finals else "unmodified")
        + " Hebrew.",
        "Lower is closer; none of these distances is a translation score.",
        "",
        "| Voynich group | Canon section | decomposition | length only | repetition given length |",
        "|---|---|---:|---:|---:|",
    ]
    for group, books in result["comparisons"].items():
        for book, values in sorted(
            books.items(), key=lambda item: item[1]["decomposition_distance"]
        ):
            lines.append(
                f"| {group} | {book} | {values['decomposition_distance']:.5f} | "
                f"{values['length_only_distance']:.5f} | "
                f"{values['repetition_given_length_distance']:.5f} |"
            )
    lines.extend(["", "## Bootstrap winner counts", ""])
    for group, summary in result["bootstrap_book_rankings"].items():
        winners = ", ".join(f"{name}: {count}" for name, count in summary["winner_counts"].items())
        lines.append(f"- {group}: {winners}")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {warning}" for warning in result["interpretive_limits"])
    if "direct_bnf_hebrew_herbal" in result:
        direct = result["direct_bnf_hebrew_herbal"]
        lines.extend(
            [
                "",
                "## Direct BnF Hebrew herbal control",
                "",
                f"Human-transcribed sample: {direct['tokens']} tokens from "
                f"{direct['provenance']['shelfmark']} folios "
                f"{direct['provenance']['transcribed_folios']}.",
                "",
                "| Voynich group | decomposition | length only | repetition given length |",
                "|---|---:|---:|---:|",
            ]
        )
        for group, values in sorted(
            direct["comparisons"].items(),
            key=lambda item: item[1]["decomposition_distance"],
        ):
            lines.append(
                f"| {group} | {values['decomposition_distance']:.5f} | "
                f"{values['length_only_distance']:.5f} | "
                f"{values['repetition_given_length_distance']:.5f} |"
            )
        lines.extend(["", "Bootstrap group winners:", ""])
        for stratum, summary in direct["bootstrap_voynich_group_rankings"].items():
            winners = ", ".join(
                f"{name}: {count}" for name, count in summary["winner_counts"].items()
            )
            lines.append(f"- {stratum}: {winners}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument("--canon-xml", type=Path, required=True)
    parser.add_argument(
        "--direct-herbal-catalog-html",
        type=Path,
        help="optional downloaded BnF Hébreu 1199 catalog page",
    )
    parser.add_argument("--minimum-confidence", type=int, default=50)
    parser.add_argument("--keep-final-forms", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--bootstrap-sample-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1491)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    result = audit(
        args.source,
        args.canon_xml,
        minimum_confidence=args.minimum_confidence,
        normalize_finals=not args.keep_final_forms,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_sample_size=args.bootstrap_sample_size,
        seed=args.seed,
        direct_herbal_catalog_html=args.direct_herbal_catalog_html,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(markdown_report(result))


if __name__ == "__main__":
    main()
