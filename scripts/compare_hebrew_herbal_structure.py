#!/usr/bin/env python3
"""Test Voynich herbal text against a period Hebrew recipe-text prediction.

BnF Hébreu 1199 is a circa-1500 northern-Italian Hebrew member of the
Alchemical Herbals tradition.  The BnF catalog contains a human transcription
of sixteen complete entries on folios 67v onward.  Those entries provide a
small but unusually close control for the textual *form* expected of a Hebrew
or Hebrew-mediated practical herbal: a plant heading followed by repeated
indication, preparation, efficacy, and habitat formulae.

This audit asks whether Voynich herbal pages preserve the repeated word
sequences that such prose predicts.  It reports exact-token recurrence and an
anagram-invariant sensitivity in which every word is replaced by its sorted
multiset of characters (grouped EVA units for Voynich).  A monoalphabetic
substitution and a consistent within-word anagram preserve both statistics.
Context-dependent homophones, nulls, changing keys, or incorrect word
boundaries need not preserve them, so this is a test of a deliberately simple
hypothesis rather than of every possible Hebrew cipher.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from audit_voynich import (
    eva_units,
    line_position_mi,
    page_words,
    parse_ivtff,
    permutation_summary,
    repeated_ngram_token_rate,
    shannon_entropy,
    shuffled_lines,
)
from prepare_hebrew_htr_adaptation import (
    NS,
    normalize_hebrew,
    page_lines,
    text_line_text,
)


BNF_HERBAL_CATALOG_URL = "https://archivesetmanuscrits.bnf.fr/ark:/12148/cc8082r"
PROSE_STARTERS = {"למי", "מי", "לרפואת", "לרפאת", "לעשות"}


@dataclass(frozen=True)
class HerbalEntry:
    heading: str
    words: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paragraphs_in_transcript(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    marker = source.find("<p>F. 67v")
    start = source.find("<blockquote", marker)
    end = source.find("</blockquote>", start)
    if marker < 0 or start < 0 or end < 0:
        raise ValueError("Could not find the F. 67v herbal transcript block")
    result = []
    for raw in re.findall(r"<p>(.*?)</p>", source[start:end], flags=re.DOTALL):
        # Editorial expansions are not manuscript text.  Geresh and gershayim
        # within abbreviations are handled by normalize_hebrew.
        raw = re.sub(r"\[[^]]*\]", "", raw)
        result.append(html.unescape(re.sub(r"<[^>]+>", " ", raw)))
    return result


def extract_bnf_entries(path: Path) -> list[HerbalEntry]:
    """Extract complete heading/description entries from the catalog sample.

    Most catalog paragraphs alternate between a short heading and a long
    description.  One description crosses the folio marker and the first 68r
    heading shares a paragraph with its description.  The state machine makes
    those two layout decisions explicit and drops the final heading whose
    description lies outside the catalog block.
    """

    entries: list[HerbalEntry] = []
    heading: str | None = None
    description_parts: list[str] = []
    after_folio_marker = False

    def finish() -> None:
        nonlocal heading, description_parts
        words = tuple(
            word
            for part in description_parts
            for word in normalize_hebrew(part).split()
        )
        if heading is not None and words:
            entries.append(HerbalEntry(heading=heading, words=words))
        heading = None
        description_parts = []

    for paragraph in _paragraphs_in_transcript(path):
        if "F. 68 r" in paragraph:
            finish()
            after_folio_marker = True
            continue
        normalized = normalize_hebrew(paragraph)
        words = normalized.split()
        if not words:
            continue
        if len(words) <= 3:
            finish()
            heading = " ".join(words)
            continue
        if heading is None and after_folio_marker:
            split = next(
                (index for index, word in enumerate(words[1:4], 1) if word in PROSE_STARTERS),
                None,
            )
            if split is None:
                raise ValueError(
                    "Could not separate the combined 68r heading and description"
                )
            heading = " ".join(words[:split])
            description_parts.append(" ".join(words[split:]))
            after_folio_marker = False
            continue
        if heading is None:
            raise ValueError("Description encountered without a plant heading")
        description_parts.append(paragraph)
    finish()
    return entries


def canonical_anagram_word(word: str, *, grouped_eva: bool) -> str:
    units = eva_units(word, grouped=True) if grouped_eva else list(word)
    # Unit separator prevents an accidental collision between multi-character
    # EVA units and sequences of one-character units.
    return "\x1f".join(sorted(units))


def canonical_anagram_sequences(
    sequences: Sequence[Sequence[str]], *, grouped_eva: bool
) -> list[list[str]]:
    return [
        [canonical_anagram_word(word, grouped_eva=grouped_eva) for word in sequence]
        for sequence in sequences
    ]


def _top_ngram_counts(
    sequences: Sequence[Sequence[str]], order: int, limit: int = 10
) -> list[dict[str, object]]:
    counts = Counter(
        tuple(sequence[index : index + order])
        for sequence in sequences
        for index in range(len(sequence) - order + 1)
    )
    return [
        {"tokens": list(ngram), "count": count}
        for ngram, count in counts.most_common(limit)
    ]


def sequence_summary(sequences: Sequence[Sequence[str]]) -> dict[str, object]:
    words = [word for sequence in sequences for word in sequence]
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    return {
        "sequences": len(sequences),
        "tokens": len(words),
        "types": len(counts),
        "hapax_types": sum(count == 1 for count in counts.values()),
        "word_entropy_bits": shannon_entropy(words),
        "top_word": top_word,
        "top_word_count": top_count,
        "top_word_fraction": top_count / len(words),
        "repeated_ngram_token_rates": {
            str(order): repeated_ngram_token_rate(sequences, order)
            for order in (2, 3, 4)
        },
        "top_ngrams": {
            str(order): _top_ngram_counts(sequences, order) for order in (2, 3, 4)
        },
    }


def extract_page_xml_lines(paths: Sequence[Path]) -> list[list[str]]:
    """Merge disjoint PAGE-XML subsets and return lines in physical y order."""

    records: dict[str, tuple[int, list[str]]] = {}
    for path in paths:
        root = ET.parse(path).getroot()
        for line in page_lines(root):
            coordinates = line.find("p:Coords", NS)
            points = coordinates.attrib.get("points", "") if coordinates is not None else ""
            y_values = [
                int(point.split(",")[1])
                for point in points.split()
                if "," in point
            ]
            if not y_values:
                raise ValueError(f"PAGE-XML line in {path} has no usable coordinates")
            key = line.attrib.get("id", points)
            words = normalize_hebrew(text_line_text(line)).split()
            if words:
                records[key] = (min(y_values), words)
    return [words for _, words in sorted(records.values())]


def line_position_test(
    lines: Sequence[Sequence[str]], *, permutations: int, seed: int
) -> dict[str, object]:
    observed = line_position_mi(lines)
    result = permutation_summary(
        observed,
        lambda rng: line_position_mi(shuffled_lines(lines, rng)),
        iterations=permutations,
        seed=seed,
    )
    result.update({"lines": len(lines), "tokens": sum(map(len, lines))})
    return result


def _sample_matching_lengths(
    sequences: Sequence[Sequence[str]], lengths: Sequence[int], rng: random.Random
) -> list[list[str]]:
    """Draw disjoint target entries and contiguous spans of requested lengths."""

    result: list[list[str]] = []
    available = list(range(len(sequences)))
    # Longest first prevents an early short request from consuming one of the
    # few pages capable of satisfying a later long request.
    for length in sorted(lengths, reverse=True):
        eligible = [index for index in available if len(sequences[index]) >= length]
        if not eligible:
            raise ValueError("Target has too few entries of the required lengths")
        index = rng.choice(eligible)
        available.remove(index)
        start = rng.randrange(len(sequences[index]) - length + 1)
        result.append(list(sequences[index][start : start + length]))
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def matched_entry_bootstrap(
    reference: Sequence[Sequence[str]],
    target: Sequence[Sequence[str]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Compare half-samples with entry counts and lengths matched exactly."""

    if len(reference) < 4:
        raise ValueError("At least four reference entries are required")
    sample_entries = len(reference) // 2
    rng = random.Random(seed)
    differences: dict[int, list[float]] = {order: [] for order in (2, 3, 4)}
    outcomes: dict[int, Counter[str]] = {
        order: Counter() for order in (2, 3, 4)
    }
    token_budgets = []
    for _ in range(iterations):
        reference_sample = rng.sample(list(reference), sample_entries)
        token_budget = sum(map(len, reference_sample))
        target_sample = _sample_matching_lengths(
            target, [len(entry) for entry in reference_sample], rng
        )
        token_budgets.append(token_budget)
        for order in (2, 3, 4):
            reference_rate = repeated_ngram_token_rate(reference_sample, order)
            target_rate = repeated_ngram_token_rate(target_sample, order)
            difference = reference_rate - target_rate
            differences[order].append(difference)
            if difference > 0:
                outcomes[order]["reference_higher"] += 1
            elif difference < 0:
                outcomes[order]["target_higher"] += 1
            else:
                outcomes[order]["tie"] += 1
    return {
        "iterations": iterations,
        "reference_entries_per_iteration": sample_entries,
        "target_entries_per_iteration": sample_entries,
        "individual_entry_lengths_matched": True,
        "mean_token_budget": statistics.fmean(token_budgets),
        "outcomes": {str(order): dict(outcomes[order]) for order in (2, 3, 4)},
        "reference_minus_target_rate": {
            str(order): {
                "mean": statistics.fmean(differences[order]),
                "q05": _quantile(differences[order], 0.05),
                "median": statistics.median(differences[order]),
                "q95": _quantile(differences[order], 0.95),
            }
            for order in (2, 3, 4)
        },
        "warning": (
            "This matched resampling describes effect stability; it is not a "
            "p-value for language, provenance, or meaning."
        ),
    }


def audit(
    source: Path,
    catalog_html: Path,
    *,
    bootstrap_iterations: int,
    seed: int,
    hebrew_page_xml: Sequence[Path] = (),
    line_permutations: int = 999,
) -> dict[str, object]:
    entries = extract_bnf_entries(catalog_html)
    hebrew_sequences = [list(entry.words) for entry in entries]
    pages = parse_ivtff(source)
    voynich = {
        f"herbal_currier_{currier}": [
            page_words(page)
            for page in pages
            if page.metadata.get("I") == "H"
            and page.metadata.get("L") == currier
            and page_words(page)
        ]
        for currier in ("A", "B")
    }

    exact = {"bnf_hebrew": sequence_summary(hebrew_sequences)}
    anagram_invariant = {
        "bnf_hebrew": sequence_summary(
            canonical_anagram_sequences(hebrew_sequences, grouped_eva=False)
        )
    }
    bootstrap: dict[str, object] = {}
    for offset, (name, target) in enumerate(voynich.items()):
        exact[name] = sequence_summary(target)
        target_anagrams = canonical_anagram_sequences(target, grouped_eva=True)
        anagram_invariant[name] = sequence_summary(target_anagrams)
        bootstrap[name] = {
            "exact_tokens": matched_entry_bootstrap(
                hebrew_sequences,
                target,
                iterations=bootstrap_iterations,
                seed=seed + offset,
            ),
            "anagram_invariant": matched_entry_bootstrap(
                canonical_anagram_sequences(hebrew_sequences, grouped_eva=False),
                target_anagrams,
                iterations=bootstrap_iterations,
                seed=seed + 100 + offset,
            ),
        }

    result = {
        "provenance": {
            "voynich_source": str(source),
            "bnf_catalog_html": str(catalog_html),
            "bnf_catalog_sha256": sha256(catalog_html),
            "bnf_catalog_url": BNF_HERBAL_CATALOG_URL,
            "transcribed_folios": "67v--68v",
            "complete_entries_extracted": len(entries),
            "seed": seed,
        },
        "bnf_entries": [
            {"heading": entry.heading, "tokens": len(entry.words)} for entry in entries
        ],
        "exact_tokens": exact,
        "anagram_invariant": anagram_invariant,
        "matched_bootstrap": bootstrap,
        "interpretive_limits": [
            "The BnF control contains only sixteen complete human-transcribed entries.",
            "Catalog punctuation and paragraph divisions are editorial rather than a diplomatic edition.",
            "Illustration type, Currier language, hand, quire, and manuscript order are confounded.",
            "A context-dependent cipher, homophones, nulls, or wrong Voynich word boundaries can suppress phrase recurrence.",
            "Failure to resemble this practical recipe tradition does not prove that Voynich is meaningless or non-Hebrew.",
        ],
    }
    if hebrew_page_xml:
        hebrew_lines = extract_page_xml_lines(hebrew_page_xml)
        line_results = {
            "bnf_hebrew_aligned_lines": line_position_test(
                hebrew_lines, permutations=line_permutations, seed=seed + 200
            )
        }
        for offset, currier in enumerate(("A", "B")):
            target_lines = [
                line
                for page in pages
                if page.metadata.get("I") == "H"
                and page.metadata.get("L") == currier
                for line in page.lines("P")
            ]
            line_results[f"herbal_currier_{currier}"] = line_position_test(
                target_lines,
                permutations=line_permutations,
                seed=seed + 201 + offset,
            )
        result["line_position_comparison"] = {
            "results": line_results,
            "source_page_xml": [str(path) for path in hebrew_page_xml],
            "warning": (
                "The Hebrew lines are catalog-grounded automatic alignments selected "
                "for HTR quality, not a paleographically checked diplomatic layout."
            ),
        }
    return result


def markdown_report(result: dict[str, object]) -> str:
    exact = result["exact_tokens"]
    invariant = result["anagram_invariant"]
    labels = {
        "bnf_hebrew": "BnF Hébreu 1199",
        "herbal_currier_A": "Voynich herbal, Currier A",
        "herbal_currier_B": "Voynich herbal, Currier B",
    }
    lines = [
        "# Hebrew herbal sequence-structure comparison",
        "",
        "Lower recurrence is not intrinsically less meaningful. Here it tests a specific",
        "prediction: a stable substitution or within-word anagram of formulaic Hebrew",
        "recipe prose should preserve repeated token sequences.",
        "",
        "| Corpus | entries/pages | tokens | top-token share | repeated 2-gram | repeated 3-gram | repeated 4-gram | anagram-invariant 3-gram |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("bnf_hebrew", "herbal_currier_A", "herbal_currier_B"):
        summary = exact[key]
        rates = summary["repeated_ngram_token_rates"]
        invariant_rate = invariant[key]["repeated_ngram_token_rates"]["3"]
        lines.append(
            f"| {labels[key]} | {summary['sequences']} | {summary['tokens']} | "
            f"{summary['top_word_fraction']:.4f} | {rates['2']:.4f} | "
            f"{rates['3']:.4f} | {rates['4']:.4f} | {invariant_rate:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Matched half-entry bootstrap",
            "",
        "Counts below are iterations in which BnF recurrence was higher than the",
        "Voynich target after matching the entry count and every entry length exactly.",
            "",
            "| Target | representation | 2-gram | 3-gram | 4-gram | iterations |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("herbal_currier_A", "herbal_currier_B"):
        for representation, title in (
            ("exact_tokens", "exact"),
            ("anagram_invariant", "within-word anagram invariant"),
        ):
            sample = result["matched_bootstrap"][key][representation]
            outcomes = sample["outcomes"]
            lines.append(
                f"| {labels[key]} | {title} | "
                f"{outcomes['2'].get('reference_higher', 0)} | "
                f"{outcomes['3'].get('reference_higher', 0)} | "
                f"{outcomes['4'].get('reference_higher', 0)} | "
                f"{sample['iterations']} |"
            )
    if "line_position_comparison" in result:
        lines.extend(
            [
                "",
                "## Physical-line position",
                "",
                "MI measures association between word identity and first/middle/last",
                "position. The null shuffles words within each physical line.",
                "",
                "| Corpus | lines | tokens | observed MI | null mean | z | permutation p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in ("bnf_hebrew_aligned_lines", "herbal_currier_A", "herbal_currier_B"):
            sample = result["line_position_comparison"]["results"][key]
            label = "BnF Hébreu 1199 aligned lines" if key.startswith("bnf") else labels[key]
            lines.append(
                f"| {label} | {sample['lines']} | {sample['tokens']} | "
                f"{sample['observed']:.5f} | {sample['null_mean']:.5f} | "
                f"{sample['z']:.2f} | {sample['p']:.4f} |"
            )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {warning}" for warning in result["interpretive_limits"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument("--catalog-html", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument(
        "--hebrew-page-xml",
        action="append",
        type=Path,
        default=[],
        help="optional disjoint catalog-grounded PAGE-XML subset; repeat as needed",
    )
    parser.add_argument("--line-permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1199)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    result = audit(
        args.source,
        args.catalog_html,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        hebrew_page_xml=args.hebrew_page_xml,
        line_permutations=args.line_permutations,
    )
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(markdown_report(result))


if __name__ == "__main__":
    main()
