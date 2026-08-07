#!/usr/bin/env python3
"""Stress-test language-identification claims made for the Voynich manuscript.

The first experiment reproduces the decomposition-pattern feature from Hauer
and Kondrak (2016).  A word such as ``seems`` maps to ``(2, 2, 1)``: only the
multiplicities of its distinct symbols remain.  This is invariant to a
monoalphabetic substitution and to within-word anagramming, but it also throws
away character identity, character order, word order, and all sentence-level
evidence.  The script therefore reports the intended ranking together with
transcription and negative-control checks.

No third-party packages are required.  Download NLTK's UDHR corpus with::

    curl -L https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr.zip -o /tmp/udhr.zip
    unzip /tmp/udhr.zip -d /tmp/udhr
    python scripts/audit_language_hypotheses.py --udhr-dir /tmp/udhr/udhr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from audit_voynich import (
    assign_quantile_groups,
    corpus_lm_summary,
    eva_units,
    page_words,
    parse_ivtff,
)


ENCODING_SUFFIXES = {
    "-UTF8": ("utf-8", 4),
    "-Latin1": ("latin-1", 3),
    "-Latin2": ("iso8859-2", 3),
    # Match NLTK's own UdhrCorpusReader encoding table.
    "-Arabic": ("cp1256", 2),
    "-Hebrew": ("iso8859-8", 2),
}


def unicode_words(text: str) -> list[str]:
    """Extract NFC-normalized sequences of Unicode letters."""

    text = unicodedata.normalize("NFC", text).casefold()
    words: list[str] = []
    current: list[str] = []
    for character in text:
        if unicodedata.category(character).startswith("L"):
            current.append(character)
        elif unicodedata.category(character).startswith("M") and current:
            # Ignore vowel/combining marks without splitting the base word.
            continue
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def load_udhr(directory: Path, minimum_words: int) -> dict[str, list[str]]:
    """Load one preferred Unicode/ISO representation per UDHR language."""

    candidates: dict[str, tuple[int, list[str], str]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        matched = next(
            (
                (suffix, encoding, priority)
                for suffix, (encoding, priority) in ENCODING_SUFFIXES.items()
                if path.name.endswith(suffix)
            ),
            None,
        )
        if matched is None:
            continue
        suffix, encoding, priority = matched
        language = path.name[: -len(suffix)]
        try:
            words = unicode_words(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        if len(words) < minimum_words:
            continue
        previous = candidates.get(language)
        # Prefer modern Unicode encodings; break ties deterministically.
        if previous is None or (priority, path.name) > (previous[0], previous[2]):
            candidates[language] = (priority, words, path.name)
    return {language: value[1] for language, value in sorted(candidates.items())}


def decomposition_pattern(word: Sequence[str] | str) -> tuple[int, ...]:
    """Sorted multiplicities of distinct symbols in one word."""

    return tuple(sorted(Counter(word).values(), reverse=True))


def pattern_distribution(
    words: Iterable[str], *, grouped_eva: bool = False
) -> dict[tuple[int, ...], float]:
    counts: Counter[tuple[int, ...]] = Counter()
    for word in words:
        units = eva_units(word, grouped=True) if grouped_eva else list(word)
        if units:
            counts[decomposition_pattern(units)] += 1
    total = sum(counts.values())
    return {pattern: count / total for pattern, count in counts.items()}


def bhattacharyya_distance(
    left: dict[tuple[int, ...], float], right: dict[tuple[int, ...], float]
) -> float:
    coefficient = sum(math.sqrt(value * right.get(key, 0.0)) for key, value in left.items())
    if coefficient <= 0:
        return math.inf
    return -math.log(coefficient)


def rank_languages(
    target_words: Sequence[str],
    references: dict[str, Sequence[str]],
    *,
    grouped_eva: bool,
) -> list[tuple[str, float]]:
    target = pattern_distribution(target_words, grouped_eva=grouped_eva)
    distances = [
        (language, bhattacharyya_distance(target, pattern_distribution(words)))
        for language, words in references.items()
    ]
    return sorted(distances, key=lambda item: (item[1], item[0]))


def word_length_distribution(
    words: Iterable[str], *, grouped_eva: bool = False
) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for word in words:
        units = eva_units(word, grouped=True) if grouped_eva else list(word)
        if units:
            counts[len(units)] += 1
    total = sum(counts.values())
    return {length: count / total for length, count in counts.items()}


def conditional_pattern_distance(
    target_words: Sequence[str],
    reference_words: Sequence[str],
    *,
    grouped_eva: bool,
) -> float:
    """Pattern similarity after conditioning on the target length mixture."""

    target_lengths = word_length_distribution(target_words, grouped_eva=grouped_eva)
    target = pattern_counts_by_length(target_words, grouped_eva=grouped_eva)
    reference = pattern_counts_by_length(reference_words)
    return conditional_pattern_distance_from_counts(target_lengths, target, reference)


def pattern_counts_by_length(
    words: Sequence[str], *, grouped_eva: bool = False
) -> dict[int, Counter]:
    result: dict[int, Counter] = defaultdict(Counter)
    for word in words:
        units = eva_units(word, grouped=True) if grouped_eva else list(word)
        if units:
            result[len(units)][decomposition_pattern(units)] += 1
    return result


def conditional_pattern_distance_from_counts(
    target_lengths: dict[int, float],
    target: dict[int, Counter],
    reference: dict[int, Counter],
) -> float:
    coefficient = 0.0
    for length, length_probability in target_lengths.items():
        if length not in reference:
            continue
        target_total = sum(target[length].values())
        reference_total = sum(reference[length].values())
        local = sum(
            math.sqrt(
                (count / target_total)
                * (reference[length].get(pattern, 0) / reference_total)
            )
            for pattern, count in target[length].items()
        )
        coefficient += length_probability * local
    return -math.log(coefficient) if coefficient > 0 else math.inf


def ablation_rankings(
    target_words: Sequence[str],
    references: dict[str, Sequence[str]],
    *,
    grouped_eva: bool,
) -> dict[str, list[tuple[str, float]]]:
    target_lengths = word_length_distribution(target_words, grouped_eva=grouped_eva)
    target_counts = pattern_counts_by_length(target_words, grouped_eva=grouped_eva)
    length_ranking = sorted(
        (
            (
                language,
                bhattacharyya_distance(target_lengths, word_length_distribution(words)),
            )
            for language, words in references.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    conditional_ranking = sorted(
        (
            (
                language,
                conditional_pattern_distance_from_counts(
                    target_lengths,
                    target_counts,
                    pattern_counts_by_length(words),
                ),
            )
            for language, words in references.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    return {"length_only": length_ranking, "repetition_given_length": conditional_ranking}


def target_words_by_currier(source: Path) -> dict[str, list[str]]:
    pages = parse_ivtff(source)
    result = {
        label: [word for page in pages if page.metadata.get("L") == label for word in page_words(page)]
        for label in ("A", "B")
    }
    result["all"] = [word for page in pages for word in page_words(page)]
    return result


def rank_of(ranking: Sequence[tuple[str, float]], query: str) -> int | None:
    query = query.casefold()
    for index, (language, _) in enumerate(ranking, 1):
        if query in language.casefold():
            return index
    return None


def matched_bootstrap(
    target_words: Sequence[str],
    references: dict[str, Sequence[str]],
    *,
    sample_words: int,
    iterations: int,
    seed: int,
    grouped_eva: bool,
) -> dict:
    """Rank fixed, equal-sized reference prefixes against target resamples."""

    usable = {name: list(words[:sample_words]) for name, words in references.items()}
    reference_distributions = {
        name: pattern_distribution(words) for name, words in usable.items()
    }
    rng = random.Random(seed)
    named = ("Hebrew", "Arabic", "Malay", "Amharic", "Esperanto")
    named_ranks: dict[str, list[int]] = {name: [] for name in named}
    winners: Counter[str] = Counter()
    top_five: Counter[str] = Counter()
    for _ in range(iterations):
        sample = rng.sample(list(target_words), sample_words)
        target = pattern_distribution(sample, grouped_eva=grouped_eva)
        ranking = sorted(
            (
                (name, bhattacharyya_distance(target, distribution))
                for name, distribution in reference_distributions.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        winners[ranking[0][0]] += 1
        top_five.update(name for name, _ in ranking[:5])
        for query in named:
            rank = rank_of(ranking, query)
            if rank is not None:
                named_ranks[query].append(rank)
    return {
        "sample_words": sample_words,
        "iterations": iterations,
        "winner_counts": dict(winners.most_common(10)),
        "top_five_counts": dict(top_five.most_common(15)),
        "named_rank_summary": {
            name: {
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in named_ranks.items()
            if values
        },
    }


def pattern_matched_pseudotext(
    words: Sequence[str], *, grouped_eva: bool, seed: int
) -> list[str]:
    """Create meaningless tokens with exactly the same decomposition patterns.

    Each token receives a fresh random assignment from multiplicities to ASCII
    symbols, so there is deliberately no consistent substitution key or lexicon.
    """

    rng = random.Random(seed)
    alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    result: list[str] = []
    for word in words:
        units = eva_units(word, grouped=True) if grouped_eva else list(word)
        pattern = decomposition_pattern(units)
        symbols = rng.sample(alphabet, len(pattern))
        token = [symbol for count, symbol in zip(pattern, symbols, strict=True) for _ in range(count)]
        rng.shuffle(token)
        result.append("".join(token))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_ranking(ranking: Sequence[tuple[str, float]], limit: int = 12) -> list[dict]:
    return [
        {"rank": index, "language": language, "distance": distance}
        for index, (language, distance) in enumerate(ranking[:limit], 1)
    ]


def audit(
    sources: Sequence[Path],
    udhr_dir: Path,
    minimum_words: int,
    sample_words: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    references = load_udhr(udhr_dir, minimum_words=max(minimum_words, sample_words))
    results: dict[str, dict] = {}
    for source_index, source in enumerate(sources):
        currier = target_words_by_currier(source)
        source_result: dict[str, object] = {
            "sha256": sha256(source),
            "token_counts": {key: len(value) for key, value in currier.items()},
            "representations": {},
        }
        for grouped in (False, True):
            representation = "grouped_eva_sensitivity" if grouped else "raw_eva_codepoints"
            split_results: dict[str, object] = {}
            for split, words in currier.items():
                ranking = rank_languages(words, references, grouped_eva=grouped)
                split_result: dict[str, object] = {
                    "top_languages": compact_ranking(ranking),
                    "hebrew_rank": rank_of(ranking, "Hebrew"),
                    "arabic_rank": rank_of(ranking, "Arabic"),
                }
                ablations = ablation_rankings(words, references, grouped_eva=grouped)
                split_result["ablations"] = {
                    name: {
                        "top_languages": compact_ranking(ablation_ranking),
                        "hebrew_rank": rank_of(ablation_ranking, "Hebrew"),
                        "arabic_rank": rank_of(ablation_ranking, "Arabic"),
                    }
                    for name, ablation_ranking in ablations.items()
                }
                if split == "B" and len(words) >= sample_words:
                    split_result["matched_bootstrap"] = matched_bootstrap(
                        words,
                        references,
                        sample_words=sample_words,
                        iterations=bootstrap_iterations,
                        seed=seed + source_index,
                        grouped_eva=grouped,
                    )
                    pseudo = pattern_matched_pseudotext(
                        words, grouped_eva=grouped, seed=seed + 100 + source_index
                    )
                    pseudo_ranking = rank_languages(pseudo, references, grouped_eva=False)
                    split_result["negative_controls"] = {
                        "word_shuffle_invariant": pattern_distribution(
                            list(reversed(words)), grouped_eva=grouped
                        )
                        == pattern_distribution(words, grouped_eva=grouped),
                        "fresh_random_symbol_pseudotext_invariant": pattern_distribution(pseudo)
                        == pattern_distribution(words, grouped_eva=grouped),
                        "pseudotext_top_languages": compact_ranking(pseudo_ranking),
                        "explanation": (
                            "Every pseudoword uses a fresh random symbol mapping. It has no "
                            "consistent cipher key or message, yet its ranking is identical by construction."
                        ),
                    }
                split_results[split] = split_result
            source_result["representations"][representation] = split_results
        # A language label should ultimately predict sequence structure, not
        # just isolated word shapes.  This comparison is descriptive because
        # the UDHR samples are shorter, but the within-corpus shuffle makes the
        # ordering gain interpretable on a common bits/word scale.
        sequence_rng = random.Random(seed + 1000 + source_index)
        sequence_sanity = [
            corpus_lm_summary(
                "voynich_B_raw",
                assign_quantile_groups(currier["B"]),
                sequence_rng,
            ),
            corpus_lm_summary(
                "voynich_B_grouped",
                assign_quantile_groups(currier["B"]),
                sequence_rng,
                grouped_eva=True,
            ),
        ]
        for query in ("Hebrew", "Arabic", "Farsi_Persian-v2", "English"):
            match = next((name for name in references if query in name), None)
            if match is not None:
                sequence_sanity.append(
                    corpus_lm_summary(
                        match,
                        assign_quantile_groups(references[match]),
                        sequence_rng,
                    )
                )
        source_result["sequence_sanity"] = sequence_sanity
        results[source.name] = source_result
    return {
        "method": {
            "feature": "sorted within-word symbol multiplicities",
            "distance": "Bhattacharyya distance",
            "invariances": [
                "monoalphabetic symbol substitution",
                "within-word symbol permutation",
                "word-order permutation",
                "replacement by pattern-matched pseudowords",
            ],
            "warning": (
                "This is a cipher-family screening feature, not a translation test. "
                "A rank does not establish that any Voynich token maps to a word in that language."
            ),
        },
        "udhr": {
            "directory": str(udhr_dir),
            "languages": len(references),
            "minimum_words": minimum_words,
            "matched_sample_words": sample_words,
        },
        "sources": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--udhr-dir", type=Path, required=True)
    parser.add_argument("--minimum-words", type=int, default=1000)
    parser.add_argument("--sample-words", type=int, default=1000)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=260804)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sources = args.source or [Path("IT2a-n.txt")]
    result = audit(
        sources,
        args.udhr_dir,
        args.minimum_words,
        args.sample_words,
        args.bootstrap_iterations,
        args.seed,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"UDHR languages: {result['udhr']['languages']}")
    for source, source_result in result["sources"].items():
        print(f"\n{source}: {source_result['token_counts']}")
        for representation, splits in source_result["representations"].items():
            print(f"  {representation}")
            for split, split_result in splits.items():
                top = ", ".join(
                    f"{item['language']} ({item['distance']:.4f})"
                    for item in split_result["top_languages"][:5]
                )
                print(
                    f"    {split}: Hebrew rank {split_result['hebrew_rank']}; "
                    f"Arabic rank {split_result['arabic_rank']}; top: {top}"
                )


if __name__ == "__main__":
    main()
