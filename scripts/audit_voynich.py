#!/usr/bin/env python3
"""Reproducible, dependency-free audit of the Voynich GPT corpus.

This module deliberately starts from the IVTFF source rather than from the
derived ``clean_taka.txt`` file.  It implements the word-space rules in IVTFF
2.0 and reports tests against explicit permutation null models.  None of the
tests can, by itself, establish that the manuscript is meaningful; their
purpose is to locate structure and to falsify overly simple claims.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


PAGE_RE = re.compile(r"^<([^.,>]+)>\s+<!\s*(.*?)>")
LOCUS_RE = re.compile(r"^<([^,>]+),([^;>]+)(?:;([^>]+))?>\s*(.*)$")
VARIABLE_RE = re.compile(r"\$([A-Z])=([^\s>]+)")
TEXT_TAG_RE = re.compile(r"<@([A-Z])=([^>]+)>")
FREE_COMMENT_RE = re.compile(r"<!.*?>")
ALTERNATIVE_RE = re.compile(r"\[([^\[\]]+)]")
LIGATURE_RE = re.compile(r"\{([^{}]+)}")
HIGH_ASCII_RE = re.compile(r"@(\d{3});")

# A sensitivity representation, not a claim about the correct grapheme
# inventory.  These common EVA sequences are visually/compositionally special
# and otherwise create deterministic Latin-codepoint transitions.
EVA_GROUPS = ("ckh", "cth", "cph", "cfh", "ch", "sh")


@dataclass(frozen=True)
class Locus:
    page: str
    locus_id: str
    locator: str
    locus_type: str
    transcriber: str | None
    raw_text: str
    tokens: tuple[str, ...]
    certain_tokens: tuple[str, ...]
    paragraph_start: bool
    paragraph_end: bool
    metadata: dict[str, str]


@dataclass
class Page:
    name: str
    metadata: dict[str, str]
    loci: list[Locus] = field(default_factory=list)

    def lines(self, generic_type: str = "P", certain: bool = True) -> list[list[str]]:
        result: list[list[str]] = []
        for locus in self.loci:
            if generic_type and not locus.locus_type.startswith(generic_type):
                continue
            tokens = locus.certain_tokens if certain else locus.tokens
            if tokens:
                result.append(list(tokens))
        return result


def _resolve_bracketed(text: str) -> str:
    """Choose the first IVTFF alternative and expose ligature contents."""

    previous = None
    while previous != text:
        previous = text
        text = ALTERNATIVE_RE.sub(lambda match: match.group(1).split(":", 1)[0], text)
        text = LIGATURE_RE.sub(lambda match: match.group(1), text)
    return text


def tokenize_ivtff(raw_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return all and conservatively certain EVA tokens for one locus.

    IVTFF 2.0 section 6.7 defines '.', ',', '<->', and '<~>' as apparent
    word spaces.  Unknown characters remain in ``tokens`` but cause the entire
    token to be omitted from ``certain_tokens``.
    """

    text = raw_text
    text = text.replace("<->", ".").replace("<~>", ".")
    text = text.replace("<%>", "").replace("<$>", "")
    text = TEXT_TAG_RE.sub("", text)
    text = FREE_COMMENT_RE.sub("", text)
    text = _resolve_bracketed(text)
    text = HIGH_ASCII_RE.sub(lambda match: f"U{match.group(1)}", text)
    text = text.replace(",", ".")
    text = text.replace("/", "")

    tokens = tuple(token.strip() for token in text.split(".") if token.strip())
    certain = tuple(token for token in tokens if re.fullmatch(r"[a-z]+", token))
    return tokens, certain


def parse_ivtff(path: Path) -> list[Page]:
    """Parse a single-transcriber IVTFF file, retaining page/locus metadata."""

    pages: list[Page] = []
    current_page: Page | None = None
    active_metadata: dict[str, str] = {}
    continuation = ""

    for line_number, source_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = source_line.rstrip()
        if not line or line.startswith("#"):
            continue

        if continuation:
            if not line.startswith("/"):
                raise ValueError(f"Expected continuation at {path}:{line_number}")
            continuation += line[1:].strip()
            if continuation.endswith("/"):
                continuation = continuation[:-1]
                continue
            line = continuation
            continuation = ""
        elif line.endswith("/"):
            continuation = line[:-1]
            continue

        page_match = PAGE_RE.match(line)
        if page_match:
            metadata = dict(VARIABLE_RE.findall(page_match.group(2)))
            current_page = Page(name=page_match.group(1), metadata=metadata)
            pages.append(current_page)
            active_metadata = metadata.copy()
            continue

        locus_match = LOCUS_RE.match(line)
        if not locus_match:
            raise ValueError(f"Unrecognized IVTFF line at {path}:{line_number}: {line!r}")
        if current_page is None:
            raise ValueError(f"Locus before page header at {path}:{line_number}")

        locus_id, code, transcriber, raw_text = locus_match.groups()
        if not locus_id.startswith(current_page.name + "."):
            raise ValueError(
                f"Locus {locus_id!r} does not belong to page {current_page.name!r} "
                f"at {path}:{line_number}"
            )
        if len(code) < 2:
            raise ValueError(f"Invalid locus code {code!r} at {path}:{line_number}")

        line_metadata = active_metadata.copy()
        for key, value in TEXT_TAG_RE.findall(raw_text):
            if value == "@":
                line_metadata.pop(key, None)
            else:
                line_metadata[key] = value
        active_metadata = line_metadata.copy()

        tokens, certain_tokens = tokenize_ivtff(raw_text)
        current_page.loci.append(
            Locus(
                page=current_page.name,
                locus_id=locus_id,
                locator=code[0],
                locus_type=code[1:],
                transcriber=transcriber,
                raw_text=raw_text,
                tokens=tokens,
                certain_tokens=certain_tokens,
                paragraph_start="<%>" in raw_text,
                paragraph_end="<$>" in raw_text,
                metadata=line_metadata,
            )
        )

    if continuation:
        raise ValueError(f"Unterminated continuation at end of {path}")
    return pages


def flatten(items: Iterable[Iterable[str]]) -> list[str]:
    return [item for group in items for item in group]


def paragraph_lines(pages: Sequence[Page], certain: bool = True) -> list[list[str]]:
    return [line for page in pages for line in page.lines("P", certain=certain)]


def page_words(page: Page, certain: bool = True) -> list[str]:
    return flatten(page.lines("P", certain=certain))


def shannon_entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def mutual_information(pairs: Iterable[tuple[str, str]]) -> float:
    joint = Counter(pairs)
    total = sum(joint.values())
    if not total:
        return 0.0
    left = Counter()
    right = Counter()
    for (x_value, y_value), count in joint.items():
        left[x_value] += count
        right[y_value] += count
    return sum(
        (count / total)
        * math.log2((count * total) / (left[x_value] * right[y_value]))
        for (x_value, y_value), count in joint.items()
    )


def reduce_vocabulary(lines: Sequence[Sequence[str]], size: int) -> tuple[list[list[str]], set[str]]:
    counts = Counter(flatten(lines))
    vocabulary = {word for word, _ in counts.most_common(size)}
    reduced = [[word if word in vocabulary else "<OTHER>" for word in line] for line in lines]
    return reduced, vocabulary


def adjacent_word_mi(lines: Sequence[Sequence[str]], vocabulary_size: int = 100) -> float:
    reduced, _ = reduce_vocabulary(lines, vocabulary_size)
    pairs = (
        (line[index], line[index + 1])
        for line in reduced
        for index in range(len(line) - 1)
    )
    return mutual_information(pairs)


def line_position_mi(lines: Sequence[Sequence[str]], vocabulary_size: int = 100) -> float:
    reduced, _ = reduce_vocabulary(lines, vocabulary_size)
    pairs: list[tuple[str, str]] = []
    for line in reduced:
        if len(line) == 1:
            pairs.append((line[0], "single"))
            continue
        for index, word in enumerate(line):
            if index == 0:
                position = "first"
            elif index == len(line) - 1:
                position = "last"
            else:
                position = "middle"
            pairs.append((word, position))
    return mutual_information(pairs)


def metadata_mi(
    pages: Sequence[Page], key: str, vocabulary_size: int = 200
) -> tuple[float, int, dict[str, int]]:
    eligible = [page for page in pages if page.metadata.get(key) not in (None, "@")]
    counts = Counter(word for page in eligible for word in page_words(page))
    vocabulary = {word for word, _ in counts.most_common(vocabulary_size)}
    pairs = [
        (word if word in vocabulary else "<OTHER>", page.metadata[key])
        for page in eligible
        for word in page_words(page)
    ]
    label_counts = Counter(page.metadata[key] for page in eligible)
    return mutual_information(pairs), len(eligible), dict(sorted(label_counts.items()))


def shuffled_lines(lines: Sequence[Sequence[str]], rng: random.Random) -> list[list[str]]:
    shuffled: list[list[str]] = []
    for line in lines:
        copy = list(line)
        rng.shuffle(copy)
        shuffled.append(copy)
    return shuffled


def permutation_summary(
    observed: float,
    null_factory: Callable[[random.Random], float],
    iterations: int,
    seed: int,
    alternative: str = "greater",
) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    null_values = [null_factory(rng) for _ in range(iterations)]
    null_mean = statistics.fmean(null_values)
    null_std = statistics.stdev(null_values) if len(null_values) > 1 else 0.0
    z_score = (observed - null_mean) / null_std if null_std else math.inf
    if alternative == "greater":
        extreme = sum(value >= observed for value in null_values)
    elif alternative == "less":
        extreme = sum(value <= observed for value in null_values)
    else:
        distance = abs(observed - null_mean)
        extreme = sum(abs(value - null_mean) >= distance for value in null_values)
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z_score,
        "p": (extreme + 1) / (iterations + 1),
        "iterations": iterations,
        "alternative": alternative,
    }


def levenshtein_at_most_one(left: str, right: str) -> bool:
    """Whether two distinct strings are one insertion/deletion/substitution apart."""

    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    if len(left) == len(right):
        mismatches = sum(a != b for a, b in zip(left, right, strict=True))
        return mismatches == 1
    # right is exactly one character longer
    left_index = right_index = differences = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            left_index += 1
            right_index += 1
        else:
            differences += 1
            right_index += 1
            if differences > 1:
                return False
    return True


def adjacency_rates(lines: Sequence[Sequence[str]]) -> dict[str, float | int]:
    adjacent = [
        (line[index], line[index + 1])
        for line in lines
        for index in range(len(line) - 1)
    ]
    if not adjacent:
        return {"pairs": 0, "exact_repeat_rate": 0.0, "edit_distance_one_rate": 0.0}
    exact = sum(left == right for left, right in adjacent)
    near = sum(levenshtein_at_most_one(left, right) for left, right in adjacent)
    return {
        "pairs": len(adjacent),
        "exact_repeat_rate": exact / len(adjacent),
        "edit_distance_one_rate": near / len(adjacent),
    }


def repeated_ngram_token_rate(lines: Sequence[Sequence[str]], order: int) -> float:
    """Fraction of n-gram instances belonging to a type seen at least twice."""

    counts = Counter(
        tuple(line[index : index + order])
        for line in lines
        for index in range(len(line) - order + 1)
    )
    total = sum(counts.values())
    if not total:
        return 0.0
    return sum(count for count in counts.values() if count >= 2) / total


def structure_metrics(lines: Sequence[Sequence[str]]) -> dict[str, float]:
    rates = adjacency_rates(lines)
    return {
        "adjacent_word_mi_bits": adjacent_word_mi(lines),
        "line_position_mi_bits": line_position_mi(lines),
        "edit_distance_one_rate": float(rates["edit_distance_one_rate"]),
        "exact_repeat_rate": float(rates["exact_repeat_rate"]),
        "repeated_bigram_token_rate": repeated_ngram_token_rate(lines, 2),
        "repeated_trigram_token_rate": repeated_ngram_token_rate(lines, 3),
        "repeated_fourgram_token_rate": repeated_ngram_token_rate(lines, 4),
    }


def compact_structure_test(
    lines: Sequence[Sequence[str]], iterations: int, seed: int
) -> dict[str, dict[str, float | int]]:
    observed = structure_metrics(lines)
    rng = random.Random(seed)
    null_samples: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(iterations):
        sample = structure_metrics(shuffled_lines(lines, rng))
        for key, value in sample.items():
            null_samples[key].append(value)
    result: dict[str, dict[str, float | int]] = {}
    for key, value in observed.items():
        null_mean = statistics.fmean(null_samples[key])
        null_std = statistics.stdev(null_samples[key]) if iterations > 1 else 0.0
        result[key] = {
            "observed": value,
            "null_mean": null_mean,
            "null_std": null_std,
            "z": (value - null_mean) / null_std if null_std else math.inf,
            "iterations": iterations,
        }
    return result


def reshape_like(words: Sequence[str], template: Sequence[Sequence[str]]) -> list[list[str]]:
    needed = sum(len(line) for line in template)
    if len(words) < needed:
        raise ValueError(f"Control has {len(words)} words; {needed} required")
    result: list[list[str]] = []
    offset = 0
    for line in template:
        length = len(line)
        result.append(list(words[offset : offset + length]))
        offset += length
    return result


def eva_units(word: str, grouped: bool) -> list[str]:
    if not grouped:
        return list(word)
    units: list[str] = []
    index = 0
    while index < len(word):
        group = next((candidate for candidate in EVA_GROUPS if word.startswith(candidate, index)), None)
        if group is None:
            units.append(word[index])
            index += 1
        else:
            units.append(group)
            index += len(group)
    return units


def token_stream(words: Sequence[str], grouped_eva: bool = False) -> list[str]:
    stream: list[str] = []
    for word in words:
        stream.extend(eva_units(word, grouped=grouped_eva))
        stream.append("<SPACE>")
    return stream


def empirical_conditional_entropy(stream: Sequence[str], order: int) -> float:
    if order < 0:
        raise ValueError("order must be non-negative")
    if len(stream) <= order:
        return 0.0
    context_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for index in range(order, len(stream)):
        context = tuple(stream[index - order : index])
        context_counts[context][stream[index]] += 1
    total = sum(sum(counts.values()) for counts in context_counts.values())
    return sum(
        (sum(counts.values()) / total) * shannon_entropy(counts.elements())
        for counts in context_counts.values()
    )


def _count_ngrams(streams: Iterable[Sequence[str]], order: int) -> tuple[Counter, Counter]:
    transitions: Counter[tuple[tuple[str, ...], str]] = Counter()
    contexts: Counter[tuple[str, ...]] = Counter()
    for stream in streams:
        for index in range(order, len(stream)):
            context = tuple(stream[index - order : index])
            transitions[(context, stream[index])] += 1
            contexts[context] += 1
    return transitions, contexts


def cross_validated_ngram_bits(
    grouped_streams: Sequence[tuple[str, Sequence[str]]], order: int, alpha: float = 0.1
) -> float:
    """Leave-group-out additive-smoothed character/token n-gram loss."""

    groups = sorted({group for group, _ in grouped_streams})
    vocabulary = sorted({token for _, stream in grouped_streams for token in stream})
    vocabulary_size = len(vocabulary)
    total_log_probability = 0.0
    total_tokens = 0

    for held_out in groups:
        training = [stream for group, stream in grouped_streams if group != held_out]
        testing = [stream for group, stream in grouped_streams if group == held_out]
        transitions, contexts = _count_ngrams(training, order)
        for stream in testing:
            for index in range(order, len(stream)):
                context = tuple(stream[index - order : index])
                numerator = transitions[(context, stream[index])] + alpha
                denominator = contexts[context] + alpha * vocabulary_size
                total_log_probability -= math.log2(numerator / denominator)
                total_tokens += 1
    return total_log_probability / total_tokens


def assign_quantile_groups(words: Sequence[str], groups: int = 10) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    for index in range(groups):
        start = len(words) * index // groups
        end = len(words) * (index + 1) // groups
        result.append((f"chunk-{index:02d}", list(words[start:end])))
    return result


def load_plain_words(path: Path, limit: int) -> list[str]:
    text = path.read_text(encoding="utf-8").casefold()
    words: list[str] = []
    current: list[str] = []
    for character in text:
        if character.isalpha():
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
            if len(words) >= limit:
                break
    if current and len(words) < limit:
        words.append("".join(current))
    return words


def corpus_lm_summary(
    name: str,
    groups: Sequence[tuple[str, Sequence[str]]],
    rng: random.Random,
    grouped_eva: bool = False,
    order: int = 5,
) -> dict[str, float | int | str]:
    words = flatten(words for _, words in groups)
    streams = [(group, token_stream(group_words, grouped_eva)) for group, group_words in groups]
    original_bits = cross_validated_ngram_bits(streams, order=order)

    shuffled_groups: list[tuple[str, list[str]]] = []
    for group, group_words in groups:
        shuffled_words = list(group_words)
        rng.shuffle(shuffled_words)
        shuffled_groups.append((group, shuffled_words))
    shuffled_streams = [
        (group, token_stream(group_words, grouped_eva)) for group, group_words in shuffled_groups
    ]
    shuffled_bits = cross_validated_ngram_bits(shuffled_streams, order=order)
    alphabet = {symbol for _, stream in streams for symbol in stream}
    units_per_word = sum(len(stream) for _, stream in streams) / len(words)
    order_gain = shuffled_bits - original_bits
    return {
        "name": name,
        "words": len(words),
        "types": len(set(words)),
        "alphabet_with_space": len(alphabet),
        "order": order,
        "units_per_word_including_space": units_per_word,
        "cv_bits_per_unit": original_bits,
        "word_shuffled_cv_bits_per_unit": shuffled_bits,
        "word_order_gain_bits_per_unit": order_gain,
        "approx_word_order_gain_bits_per_word": order_gain * units_per_word,
    }


def audit(
    source: Path,
    derived: Path | None,
    controls: dict[str, Path],
    permutations: int,
    seed: int,
) -> dict:
    pages = parse_ivtff(source)
    all_loci = [locus for page in pages for locus in page.loci]
    all_tokens = flatten(locus.tokens for locus in all_loci)
    certain_tokens = flatten(locus.certain_tokens for locus in all_loci)
    words_by_locus_type: dict[str, list[str]] = defaultdict(list)
    for locus in all_loci:
        words_by_locus_type[locus.locus_type[:1]].extend(locus.certain_tokens)
    lines = paragraph_lines(pages, certain=True)
    words = flatten(lines)
    counts = Counter(words)

    drawing_spaces = sum(locus.raw_text.count("<->") + locus.raw_text.count("<~>") for locus in all_loci)
    uncertain_spaces = sum(locus.raw_text.count(",") for locus in all_loci)
    unknown_tokens = len(all_tokens) - len(certain_tokens)

    derived_summary: dict[str, int | str | bool] | None = None
    if derived and derived.exists():
        derived_text = derived.read_text(encoding="utf-8")
        derived_words = [
            token
            for line in derived_text.splitlines()
            for token in line.split(".")
            if token
        ]
        derived_summary = {
            "path": str(derived),
            "characters": len(derived_text),
            "lines": len(derived_text.splitlines()),
            "period_delimited_tokens": len(derived_words),
            "token_deficit_vs_correct_all_loci": len(all_tokens) - len(derived_words),
            "contains_drawing_separator": "<->" in derived_text or "<~>" in derived_text,
        }

    word_mi = adjacent_word_mi(lines)
    position_mi = line_position_mi(lines)
    ordering_tests = {
        "adjacent_word_mi_bits": permutation_summary(
            word_mi,
            lambda rng: adjacent_word_mi(shuffled_lines(lines, rng)),
            iterations=permutations,
            seed=seed,
        ),
        "line_position_mi_bits": permutation_summary(
            position_mi,
            lambda rng: line_position_mi(shuffled_lines(lines, rng)),
            iterations=permutations,
            seed=seed + 1,
        ),
    }

    rates = adjacency_rates(lines)
    ordering_tests["edit_distance_one_adjacency"] = permutation_summary(
        float(rates["edit_distance_one_rate"]),
        lambda rng: float(adjacency_rates(shuffled_lines(lines, rng))["edit_distance_one_rate"]),
        iterations=permutations,
        seed=seed + 2,
    )
    ordering_tests["exact_repeat_adjacency"] = permutation_summary(
        float(rates["exact_repeat_rate"]),
        lambda rng: float(adjacency_rates(shuffled_lines(lines, rng))["exact_repeat_rate"]),
        iterations=permutations,
        seed=seed + 3,
    )
    for order, name in ((2, "repeated_bigram_tokens"), (3, "repeated_trigram_tokens"), (4, "repeated_fourgram_tokens")):
        observed = repeated_ngram_token_rate(lines, order)
        ordering_tests[name] = permutation_summary(
            observed,
            lambda rng, n=order: repeated_ngram_token_rate(shuffled_lines(lines, rng), n),
            iterations=permutations,
            seed=seed + 10 + order,
        )

    metadata = {}
    for key, description in (("I", "illustration_type"), ("L", "currier_language"), ("H", "hand")):
        value, page_count, label_counts = metadata_mi(pages, key)
        metadata[description] = {
            "word_label_mi_bits": value,
            "pages": page_count,
            "page_label_counts": label_counts,
            "warning": "association only; metadata fields are mutually confounded",
        }

    page_groups = [
        (page.metadata.get("Q", f"page:{page.name}"), page_words(page))
        for page in pages
        if page_words(page)
    ]
    lm_rng = random.Random(seed + 10)
    contiguous_groups = assign_quantile_groups(words)
    language_models = [
        corpus_lm_summary("voynich_raw_eva_leave_quire_out", page_groups, lm_rng),
        corpus_lm_summary(
            "voynich_grouped_eva_leave_quire_out", page_groups, lm_rng, grouped_eva=True
        ),
        corpus_lm_summary("voynich_raw_eva_contiguous_10fold", contiguous_groups, lm_rng),
        corpus_lm_summary(
            "voynich_grouped_eva_contiguous_10fold",
            contiguous_groups,
            lm_rng,
            grouped_eva=True,
        ),
    ]
    for name, path in controls.items():
        control_words = load_plain_words(path, limit=len(words))
        groups = assign_quantile_groups(control_words)
        language_models.append(corpus_lm_summary(name, groups, lm_rng))

    structure_comparisons = {
        "voynich": compact_structure_test(lines, min(permutations, 99), seed + 100)
    }
    for index, (name, path) in enumerate(controls.items()):
        control_words = load_plain_words(path, limit=len(words))
        control_lines = reshape_like(control_words, lines)
        structure_comparisons[name] = compact_structure_test(
            control_lines, min(permutations, 99), seed + 101 + index
        )

    raw_stream = token_stream(words, grouped_eva=False)
    grouped_stream = token_stream(words, grouped_eva=True)
    locus_type_summary = {
        locus_type: {
            "certain_tokens": len(locus_words),
            "types": len(set(locus_words)),
            "hapax_types": sum(count == 1 for count in Counter(locus_words).values()),
        }
        for locus_type, locus_words in sorted(words_by_locus_type.items())
    }
    label_words = words_by_locus_type.get("L", [])
    label_counts = Counter(label_words)
    paragraph_vocabulary = set(words_by_locus_type.get("P", []))
    label_summary = {
        "certain_tokens": len(label_words),
        "types": len(label_counts),
        "hapax_types": sum(count == 1 for count in label_counts.values()),
        "tokens_also_attested_in_paragraphs": sum(
            count for word, count in label_counts.items() if word in paragraph_vocabulary
        ),
        "types_also_attested_in_paragraphs": sum(
            word in paragraph_vocabulary for word in label_counts
        ),
        "warning": "labels are sparse and heterogeneous; overlap is not semantic identification",
    }

    return {
        "provenance": {
            "source": str(source),
            "ivttf_header": source.read_text(encoding="utf-8").splitlines()[0],
            "seed": seed,
            "permutations": permutations,
        },
        "corpus": {
            "pages": len(pages),
            "loci": len(all_loci),
            "all_apparent_word_tokens": len(all_tokens),
            "certain_all_locus_tokens": len(certain_tokens),
            "paragraph_lines": len(lines),
            "certain_paragraph_tokens": len(words),
            "certain_paragraph_types": len(counts),
            "hapax_types": sum(value == 1 for value in counts.values()),
            "drawing_implied_spaces": drawing_spaces,
            "uncertain_apparent_spaces": uncertain_spaces,
            "tokens_with_uncertain_or_non_eva_content": unknown_tokens,
            "derived_corpus": derived_summary,
        },
        "entropy_sensitivity": {
            "raw_eva_units": len(raw_stream),
            "grouped_eva_units": len(grouped_stream),
            "raw_eva_alphabet_with_space": len(set(raw_stream)),
            "grouped_eva_alphabet_with_space": len(set(grouped_stream)),
            "raw_empirical_bits": {
                str(order): empirical_conditional_entropy(raw_stream, order) for order in range(6)
            },
            "grouped_empirical_bits": {
                str(order): empirical_conditional_entropy(grouped_stream, order) for order in range(6)
            },
            "warning": "grouping is a sensitivity analysis, not an asserted EVA grapheme inventory",
        },
        "ordering_tests": ordering_tests,
        "adjacency_rates": rates,
        "metadata_associations": metadata,
        "locus_type_summary": locus_type_summary,
        "label_summary": label_summary,
        "matched_structure_comparisons": structure_comparisons,
        "cross_validated_character_models": language_models,
        "interpretive_limits": [
            "Statistical structure is compatible with meaning but does not prove meaning.",
            "A structured text-generation procedure can also produce word-order and metadata associations.",
            "Illustration type, Currier language, hand, quire, and manuscript order are confounded.",
            "EVA is a transliteration convention; codepoint-level entropy is not script-invariant.",
        ],
    }


def markdown_report(result: dict) -> str:
    corpus = result["corpus"]
    lines = [
        "# Voynich corpus audit",
        "",
        "## Corpus integrity",
        "",
        f"- IVTFF pages: {corpus['pages']}",
        f"- All apparent word tokens: {corpus['all_apparent_word_tokens']}",
        f"- Conservative paragraph tokens: {corpus['certain_paragraph_tokens']}",
        f"- Drawing-implied spaces: {corpus['drawing_implied_spaces']}",
        f"- Tokens excluded as uncertain/non-basic: {corpus['tokens_with_uncertain_or_non_eva_content']}",
    ]
    derived = corpus.get("derived_corpus")
    if derived:
        lines.extend(
            [
                f"- Legacy period-delimited tokens: {derived['period_delimited_tokens']}",
                f"- Legacy token deficit: {derived['token_deficit_vs_correct_all_loci']}",
            ]
        )
    lines.extend(["", "## Ordering tests", ""])
    for name, summary in result["ordering_tests"].items():
        lines.append(
            f"- {name}: observed={summary['observed']:.6f}, "
            f"null={summary['null_mean']:.6f}±{summary['null_std']:.6f}, "
            f"z={summary['z']:.2f}, p={summary['p']:.4g}"
        )
    lines.extend(["", "## Cross-validated order-5 models", ""])
    for summary in result["cross_validated_character_models"]:
        lines.append(
            f"- {summary['name']}: {summary['cv_bits_per_unit']:.4f} bits/unit; "
            f"word-shuffled {summary['word_shuffled_cv_bits_per_unit']:.4f}; "
            f"ordering gain {summary['word_order_gain_bits_per_unit']:.4f} bits/unit "
            f"(~{summary['approx_word_order_gain_bits_per_word']:.3f} bits/word)"
        )
    comparisons = result.get("matched_structure_comparisons", {})
    if comparisons:
        lines.extend(["", "## Matched word-structure comparisons", ""])
        for corpus_name, tests in comparisons.items():
            bigram = tests["repeated_bigram_token_rate"]
            trigram = tests["repeated_trigram_token_rate"]
            adjacency = tests["adjacent_word_mi_bits"]
            lines.append(
                f"- {corpus_name}: adjacency-MI excess z={adjacency['z']:.2f}; "
                f"repeated-bigram excess z={bigram['z']:.2f}; "
                f"repeated-trigram excess z={trigram['z']:.2f}"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in result["interpretive_limits"]],
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument("--derived", type=Path, default=Path("data/voynich_char/clean_taka.txt"))
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a plain-text comparison corpus (repeatable).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    controls: dict[str, Path] = {}
    for item in args.control:
        if "=" not in item:
            raise SystemExit(f"Invalid --control {item!r}; expected NAME=PATH")
        name, raw_path = item.split("=", 1)
        controls[name] = Path(raw_path)
    result = audit(args.source, args.derived, controls, args.permutations, args.seed)
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(markdown_report(result))


if __name__ == "__main__":
    main()
