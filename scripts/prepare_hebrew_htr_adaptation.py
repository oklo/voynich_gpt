#!/usr/bin/env python3
"""Prepare line-level ground truth for adapting HTR to BnF Hébreu 1199.

The BnF catalog transcribes folio 67v as continuous paragraphs, while Kraken
detects physical lines.  This tool globally aligns a baseline HTR reading to
the catalog text, transfers physical line boundaries to the catalog text, and
writes disjoint PAGE-XML training and evaluation files.

The alignment is bootstrapping infrastructure, not a scholarly edition.  It
removes editorial bracket expansions and all non-Hebrew punctuation before
alignment and reports every exclusion and error rate.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence


HEBREW_RE = re.compile(r"[א-ת]+")
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NS = {"p": PAGE_NS}
ET.register_namespace("", PAGE_NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


def normalize_hebrew(text: str) -> str:
    """Return NFKC Hebrew words with one canonical inter-word space."""

    text = unicodedata.normalize("NFKC", text)
    # Hebrew scribal abbreviations put geresh/gershayim inside a token. They
    # should not create false word boundaries when punctuation is discarded.
    text = re.sub(r"['\"׳״’]", "", text)
    return " ".join(HEBREW_RE.findall(text))


def catalog_folio(path: Path, folio: str) -> str:
    """Extract the part of the catalog transcript visible on one Gallica side.

    The image begins with the abbreviation ``בג''ה`` and the continuation
    ``ואם לא תהר``.  The catalog supplies an editorial expansion in brackets;
    it is absent from the manuscript and is removed here.
    """

    source = path.read_text(encoding="utf-8")
    if folio == "67v":
        start = source.find("<p>בג''ה [בינה, גבורה, הוד]. ואם")
        end = source.find("<p>F. 68 r", start)
    elif folio == "68r":
        marker = source.find("<p>F. 68 r")
        start = source.find("<p>קנלריטאס רומנא", marker)
        # The catalog provides no subsequent folio marker, but the Gallica
        # image ends after the אנגליס entry. סירפינטינא heads the next side.
        end = source.find("<p>סירפינטינא</p>", start)
    else:
        raise ValueError(f"Unsupported catalog folio: {folio}")
    if start < 0 or end < 0:
        raise ValueError(f"Could not locate the catalog's folio {folio} transcript")
    fragment = source[start:end]
    fragment = re.sub(r"\[[^]]*\]", "", fragment)
    fragment = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    return normalize_hebrew(fragment)


def text_line_text(line: ET.Element) -> str:
    text_equiv = line.find("p:TextEquiv", NS)
    if text_equiv is None:
        return ""
    unicode_element = text_equiv.find("p:Unicode", NS)
    return unicode_element.text or "" if unicode_element is not None else ""


def page_lines(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//p:TextLine", NS)


def levenshtein_alignment(source: str, target: str) -> tuple[list[int], int]:
    """Map every source-prefix position to a target-prefix position."""

    width = len(target) + 1
    previous = list(range(width))
    directions: list[bytearray] = [bytearray([2]) * width]
    directions[0][0] = 0
    for source_index, source_character in enumerate(source, 1):
        current = [source_index]
        direction = bytearray(width)
        direction[0] = 1
        for target_index, target_character in enumerate(target, 1):
            diagonal = previous[target_index - 1] + (
                source_character != target_character
            )
            up = previous[target_index] + 1
            left = current[target_index - 1] + 1
            if diagonal <= up and diagonal <= left:
                current.append(diagonal)
                direction[target_index] = 0
            elif up <= left:
                current.append(up)
                direction[target_index] = 1
            else:
                current.append(left)
                direction[target_index] = 2
        previous = current
        directions.append(direction)

    source_index = len(source)
    target_index = len(target)
    states = [(source_index, target_index)]
    while source_index or target_index:
        move = directions[source_index][target_index]
        if move == 0:
            source_index -= 1
            target_index -= 1
        elif move == 1:
            source_index -= 1
        else:
            target_index -= 1
        states.append((source_index, target_index))
    states.reverse()

    mapping: list[int | None] = [None] * (len(source) + 1)
    for source_index, target_index in states:
        mapping[source_index] = target_index
    last = 0
    for index, value in enumerate(mapping):
        if value is None:
            mapping[index] = last
        else:
            last = value
    return [int(value) for value in mapping], previous[-1]


def levenshtein_distance(left: Sequence, right: Sequence) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def transfer_line_boundaries(
    ocr_lines: Sequence[str], ground_truth: str
) -> tuple[list[str], dict]:
    normalized_lines = [normalize_hebrew(line) for line in ocr_lines]
    source = " ".join(normalized_lines)
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for index, line in enumerate(normalized_lines):
        start = cursor
        cursor += len(line)
        boundaries.append((start, cursor))
        if index < len(normalized_lines) - 1:
            cursor += 1

    mapping, edit_distance = levenshtein_alignment(source, ground_truth)
    transferred = [
        ground_truth[mapping[start] : mapping[end]].strip()
        for start, end in boundaries
    ]
    report = {
        "baseline_characters": len(source),
        "ground_truth_characters": len(ground_truth),
        "global_edit_distance": edit_distance,
        "global_character_error_rate": edit_distance / len(ground_truth),
        "global_word_error_rate": levenshtein_distance(
            source.split(), ground_truth.split()
        )
        / len(ground_truth.split()),
    }
    return transferred, report


def set_line_text(line: ET.Element, text: str) -> None:
    text_equiv = line.find("p:TextEquiv", NS)
    if text_equiv is None:
        text_equiv = ET.SubElement(line, f"{{{PAGE_NS}}}TextEquiv")
    text_equiv.set("conf", "1.0")
    unicode_element = text_equiv.find("p:Unicode", NS)
    if unicode_element is None:
        unicode_element = ET.SubElement(text_equiv, f"{{{PAGE_NS}}}Unicode")
    unicode_element.text = text


def subset_tree(
    tree: ET.ElementTree,
    transferred: Sequence[str],
    retained_indices: set[int],
) -> ET.ElementTree:
    result = copy.deepcopy(tree)
    lines = page_lines(result.getroot())
    index_by_identity = {id(line): index for index, line in enumerate(lines)}
    for parent in result.getroot().iter():
        for child in list(parent):
            if child.tag != f"{{{PAGE_NS}}}TextLine":
                continue
            index = index_by_identity[id(child)]
            if index not in retained_indices:
                parent.remove(child)
            else:
                set_line_text(child, transferred[index])
    return result


def prepare(
    page_xml: Path,
    catalog_html: Path,
    train_xml: Path,
    evaluation_xml: Path,
    *,
    folio: str,
    evaluation_modulus: int,
    minimum_line_characters: int,
    maximum_line_character_error_rate: float,
    all_xml: Path | None,
) -> dict:
    tree = ET.parse(page_xml)
    lines = page_lines(tree.getroot())
    ocr_lines = [text_line_text(line) for line in lines]
    ground_truth = catalog_folio(catalog_html, folio)
    transferred, report = transfer_line_boundaries(ocr_lines, ground_truth)

    eligible = []
    line_character_error_rates = {}
    for index, (ocr, truth) in enumerate(zip(ocr_lines, transferred)):
        normalized_ocr = normalize_hebrew(ocr)
        character_error_rate = levenshtein_distance(normalized_ocr, truth) / max(
            1, len(truth)
        )
        line_character_error_rates[index] = character_error_rate
        if (
            len(normalized_ocr) >= minimum_line_characters
            and len(truth) >= minimum_line_characters
            and 0.4 <= len(truth) / len(normalized_ocr) <= 2.5
            and character_error_rate <= maximum_line_character_error_rate
        ):
            eligible.append(index)
    evaluation = {
        index for position, index in enumerate(eligible) if position % evaluation_modulus == 0
    }
    training = set(eligible) - evaluation
    if not training or not evaluation:
        raise ValueError("The requested split produced an empty partition")

    train_xml.parent.mkdir(parents=True, exist_ok=True)
    evaluation_xml.parent.mkdir(parents=True, exist_ok=True)
    subset_tree(tree, transferred, training).write(
        train_xml, encoding="utf-8", xml_declaration=True
    )
    subset_tree(tree, transferred, evaluation).write(
        evaluation_xml, encoding="utf-8", xml_declaration=True
    )
    if all_xml is not None:
        all_xml.parent.mkdir(parents=True, exist_ok=True)
        subset_tree(tree, transferred, set(eligible)).write(
            all_xml, encoding="utf-8", xml_declaration=True
        )

    report.update(
        {
            "detected_lines": len(lines),
            "folio": folio,
            "eligible_lines": len(eligible),
            "training_lines": len(training),
            "evaluation_lines": len(evaluation),
            "excluded_line_indices": sorted(set(range(len(lines))) - set(eligible)),
            "evaluation_line_indices": sorted(evaluation),
            "maximum_line_character_error_rate": maximum_line_character_error_rate,
            "minimum_line_characters": minimum_line_characters,
            "eligible_line_character_error_rate_mean": sum(
                line_character_error_rates[index] for index in eligible
            )
            / len(eligible),
            "train_page_xml": str(train_xml),
            "evaluation_page_xml": str(evaluation_xml),
            "all_page_xml": str(all_xml) if all_xml is not None else None,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-xml", type=Path, required=True)
    parser.add_argument("--catalog-html", type=Path, required=True)
    parser.add_argument("--train-xml", type=Path, required=True)
    parser.add_argument("--evaluation-xml", type=Path, required=True)
    parser.add_argument("--all-xml", type=Path)
    parser.add_argument("--folio", choices=("67v", "68r"), default="67v")
    parser.add_argument("--evaluation-modulus", type=int, default=5)
    parser.add_argument("--minimum-line-characters", type=int, default=8)
    parser.add_argument(
        "--maximum-line-character-error-rate", type=float, default=0.6
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.page_xml,
                args.catalog_html,
                args.train_xml,
                args.evaluation_xml,
                folio=args.folio,
                evaluation_modulus=args.evaluation_modulus,
                minimum_line_characters=args.minimum_line_characters,
                maximum_line_character_error_rate=args.maximum_line_character_error_rate,
                all_xml=args.all_xml,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
