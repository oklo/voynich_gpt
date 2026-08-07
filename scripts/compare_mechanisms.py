#!/usr/bin/env python3
"""Held-out likelihood tournament for explicit Voynich production mechanisms.

The tournament compares four deliberately narrow models on held-out quires:

1. A period/domain-matched Hebrew character model.  A single monoalphabetic
   substitution key is fitted on training Voynich words by simulated annealing
   and then frozen.  Both within-word directions compete on training data only.
2. A Voynich-native character model of isolated word forms.
3. The same model conditioned on Currier stratum and physical line position.
4. A proper-probability mixture of the layout model and a copy/edit channel
   from the preceding word on the same physical line.

All models score exactly the same observed units: raw EVA codepoints plus one
end-of-word event per token.  The Hebrew source alphabet is reduced from 22 to
21 classes by merging the two rarest letters in external training text, making
the raw-EVA substitution a true bijection rather than silently dropping source
probability.  The merged class and Hebrew language model are determined before
Voynich key fitting.  Canon pages held out from source-model training provide a
calibration score on actual Hebrew.

This is predictive likelihood, not a proof of meaning or nonsense.  In
particular it does not charge a code length for the model families themselves,
and it tests a simple stable substitution rather than every possible cipher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from audit_voynich import Page, eva_units, parse_ivtff
from compare_medical_hebrew import (
    EXPECTED_DJVU_XML_SHA256,
    extract_bnf_herbal_transcript,
    extract_canon_pages,
)


BOS = "<BOS>"
EOW = "<EOW>"
HEBREW_RARE = "<HEBREW_RARE>"
HEBREW_ALPHABET = tuple("אבגדהוזחטיכלמנסעפצקרשת")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NgramModel:
    """Additively smoothed fixed-order model over finite token sequences."""

    def __init__(self, order: int, vocabulary: Sequence[str], alpha: float = 0.1):
        self.order = order
        self.vocabulary = tuple(vocabulary)
        self.vocabulary_set = set(vocabulary)
        self.alpha = alpha
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.totals: Counter[tuple[str, ...]] = Counter()

    def train(self, sequences: Iterable[Sequence[str]]) -> None:
        for sequence in sequences:
            context = [BOS] * self.order
            for token in [*sequence, EOW]:
                if token not in self.vocabulary_set:
                    raise ValueError(f"Token {token!r} is outside the model vocabulary")
                key = tuple(context)
                self.counts[key][token] += 1
                self.totals[key] += 1
                context = [*context[1:], token] if self.order else []

    def probability(self, context: Sequence[str], token: str) -> float:
        key = tuple(context[-self.order :]) if self.order else ()
        return (self.counts[key][token] + self.alpha) / (
            self.totals[key] + self.alpha * len(self.vocabulary)
        )

    def log2_sequence(self, sequence: Sequence[str]) -> float:
        total = 0.0
        context = [BOS] * self.order
        for token in [*sequence, EOW]:
            total += math.log2(self.probability(context, token))
            context = [*context[1:], token] if self.order else []
        return total


@dataclass(frozen=True)
class WordRecord:
    units: tuple[str, ...]
    channel: tuple[str, str]
    previous: tuple[str, ...] | None


class ConditionalWordModel:
    """Character model interpolated with Currier/line-position channels."""

    def __init__(
        self,
        order: int,
        vocabulary: Sequence[str],
        *,
        alpha: float,
        channel_weight: float,
    ):
        self.order = order
        self.vocabulary = tuple(vocabulary)
        self.alpha = alpha
        self.channel_weight = channel_weight
        self.pooled = NgramModel(order, vocabulary, alpha)
        self.channels: dict[tuple[str, str], NgramModel] = {}

    def train(self, records: Sequence[WordRecord]) -> None:
        self.pooled.train(record.units for record in records)
        grouped: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
        for record in records:
            grouped[record.channel].append(record.units)
        for channel, sequences in grouped.items():
            model = NgramModel(self.order, self.vocabulary, self.alpha)
            model.train(sequences)
            self.channels[channel] = model

    def log2_word(self, units: Sequence[str], channel: tuple[str, str]) -> float:
        total = 0.0
        context = [BOS] * self.order
        channel_model = self.channels.get(channel)
        for token in [*units, EOW]:
            pooled = self.pooled.probability(context, token)
            if channel_model is None or self.channel_weight <= 0:
                probability = pooled
            else:
                probability = (
                    self.channel_weight * channel_model.probability(context, token)
                    + (1 - self.channel_weight) * pooled
                )
            total += math.log2(probability)
            context = [*context[1:], token] if self.order else []
        return total


def line_position(index: int, line_length: int) -> str:
    if line_length == 1:
        return "single"
    if index == 0:
        return "first"
    if index == line_length - 1:
        return "last"
    return "middle"


def records_from_pages(pages: Sequence[Page]) -> list[WordRecord]:
    records: list[WordRecord] = []
    for page in pages:
        currier = page.metadata.get("L", "?")
        for line in page.lines("P", certain=True):
            previous: tuple[str, ...] | None = None
            for index, word in enumerate(line):
                units = tuple(eva_units(word, grouped=False))
                records.append(
                    WordRecord(
                        units=units,
                        channel=(currier, line_position(index, len(line))),
                        previous=previous,
                    )
                )
                previous = units
    return records


def make_quire_folds(pages: Sequence[Page], folds: int) -> list[set[str]]:
    """Greedily balance whole quires across held-out folds."""

    by_quire: dict[str, list[Page]] = defaultdict(list)
    for page in pages:
        if page.lines("P", certain=True):
            by_quire[page.metadata.get("Q", "?")].append(page)
    weighted = sorted(
        (
            (sum(len(line) for page in group for line in page.lines("P", certain=True)), quire)
            for quire, group in by_quire.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    assignments: list[set[str]] = [set() for _ in range(folds)]
    totals = [0] * folds
    for weight, quire in weighted:
        destination = min(range(folds), key=lambda index: (totals[index], index))
        assignments[destination].add(quire)
        totals[destination] += weight
    return assignments


def reduce_hebrew_alphabet(
    training_words: Sequence[str], calibration_words: Sequence[str]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], tuple[str, str], tuple[str, ...]]:
    counts = Counter(character for word in training_words for character in word)
    rare_pair = tuple(sorted(HEBREW_ALPHABET, key=lambda char: (counts[char], char))[:2])

    def convert(word: str) -> tuple[str, ...]:
        return tuple(HEBREW_RARE if character in rare_pair else character for character in word)

    alphabet = tuple(character for character in HEBREW_ALPHABET if character not in rare_pair) + (
        HEBREW_RARE,
    )
    if len(alphabet) != 21:
        raise AssertionError("Hebrew reduction should yield exactly 21 source classes")
    return (
        [convert(word) for word in training_words],
        [convert(word) for word in calibration_words],
        (rare_pair[0], rare_pair[1]),
        alphabet,
    )


def load_hebrew_source(
    canon_xml: Path,
    catalog_html: Path | None,
    *,
    minimum_confidence: int,
) -> dict[str, object]:
    pages = extract_canon_pages(
        canon_xml,
        minimum_confidence=minimum_confidence,
        normalize_finals=True,
    )["book_II_materia_medica"]
    training_words = [
        word for index, page in enumerate(pages) if index % 5 for word in page
    ]
    calibration_words = [
        word for index, page in enumerate(pages) if index % 5 == 0 for word in page
    ]
    catalog_words: list[str] = []
    if catalog_html is not None:
        catalog_words = extract_bnf_herbal_transcript(
            catalog_html, normalize_finals=True
        )
        training_words.extend(catalog_words)
    train_units, calibration_units, rare_pair, alphabet = reduce_hebrew_alphabet(
        training_words, calibration_words
    )
    frequencies = Counter(unit for word in train_units for unit in word)
    return {
        "training_words": train_units,
        "calibration_words": calibration_units,
        "rare_pair": rare_pair,
        "alphabet": alphabet,
        "frequencies": frequencies,
        "canon_training_word_count": len(training_words) - len(catalog_words),
        "canon_calibration_word_count": len(calibration_words),
        "catalog_training_word_count": len(catalog_words),
    }


class SubstitutionObjective:
    """Fast train likelihood and local swap deltas for a substitution key."""

    def __init__(
        self,
        words: Sequence[tuple[str, ...]],
        model: NgramModel,
        cipher_alphabet: Sequence[str],
    ):
        self.model = model
        self.cipher_alphabet = tuple(cipher_alphabet)
        word_counts = Counter(words)
        transition_counts: Counter[tuple[tuple[str, ...], str]] = Counter()
        for word, count in word_counts.items():
            context = [BOS] * model.order
            for token in [*word, EOW]:
                transition_counts[(tuple(context), token)] += count
                context = [*context[1:], token] if model.order else []
        self.entries = [
            (context, token, count)
            for (context, token), count in transition_counts.items()
        ]
        self.total_events = sum(count for _, _, count in self.entries)
        index: dict[str, set[int]] = {symbol: set() for symbol in cipher_alphabet}
        for entry_index, (context, token, _) in enumerate(self.entries):
            for symbol in {*context, token} & set(cipher_alphabet):
                index[symbol].add(entry_index)
        self.index = index

    @staticmethod
    def mapped(token: str, mapping: dict[str, str]) -> str:
        return mapping.get(token, token)

    def contribution(self, entry_index: int, mapping: dict[str, str]) -> float:
        context, token, count = self.entries[entry_index]
        mapped_context = tuple(self.mapped(item, mapping) for item in context)
        mapped_token = self.mapped(token, mapping)
        return count * math.log2(self.model.probability(mapped_context, mapped_token))

    def contributions(self, mapping: dict[str, str]) -> list[float]:
        return [self.contribution(index, mapping) for index in range(len(self.entries))]


@dataclass(frozen=True)
class KeyFit:
    mapping: dict[str, str]
    train_bits_per_unit: float
    restart_bits_per_unit: tuple[float, ...]


def greedy_polish_key(
    objective: SubstitutionObjective,
    initial_mapping: dict[str, str],
) -> tuple[dict[str, str], float, int]:
    """Reach a deterministic two-swap local optimum after stochastic search."""

    mapping = dict(initial_mapping)
    contributions = objective.contributions(mapping)
    score = sum(contributions)
    accepted_swaps = 0
    while True:
        best_delta = 1e-9
        best_pair: tuple[str, str] | None = None
        symbols = objective.cipher_alphabet
        for left_index, left in enumerate(symbols):
            for right in symbols[left_index + 1 :]:
                affected = objective.index[left] | objective.index[right]
                mapping[left], mapping[right] = mapping[right], mapping[left]
                delta = sum(
                    objective.contribution(index, mapping) - contributions[index]
                    for index in affected
                )
                mapping[left], mapping[right] = mapping[right], mapping[left]
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        affected = objective.index[left] | objective.index[right]
        mapping[left], mapping[right] = mapping[right], mapping[left]
        replacements = {
            index: objective.contribution(index, mapping) for index in affected
        }
        score += sum(replacements[index] - contributions[index] for index in affected)
        for index, value in replacements.items():
            contributions[index] = value
        accepted_swaps += 1
    return mapping, score, accepted_swaps


def fit_substitution_key(
    words: Sequence[tuple[str, ...]],
    cipher_alphabet: Sequence[str],
    plain_alphabet: Sequence[str],
    plain_frequencies: Counter[str],
    model: NgramModel,
    *,
    seed: int,
    restarts: int,
    steps: int,
) -> KeyFit:
    if len(cipher_alphabet) != len(plain_alphabet):
        raise ValueError("Substitution key requires equal cipher/plain alphabets")
    objective = SubstitutionObjective(words, model, cipher_alphabet)
    cipher_frequencies = Counter(unit for word in words for unit in word)
    ranked_cipher = sorted(cipher_alphabet, key=lambda unit: (-cipher_frequencies[unit], unit))
    ranked_plain = sorted(plain_alphabet, key=lambda unit: (-plain_frequencies[unit], unit))
    frequency_mapping = dict(zip(ranked_cipher, ranked_plain, strict=True))
    rng = random.Random(seed)
    best_mapping: dict[str, str] | None = None
    best_score = -math.inf
    restart_scores: list[float] = []

    for restart in range(restarts):
        mapping = dict(frequency_mapping)
        if restart:
            perturbations = 3 + restart * 2
            for _ in range(perturbations):
                left, right = rng.sample(list(cipher_alphabet), 2)
                mapping[left], mapping[right] = mapping[right], mapping[left]
        contributions = objective.contributions(mapping)
        score = sum(contributions)
        local_best_score = score
        local_best_mapping = dict(mapping)
        for step in range(steps):
            fraction = step / max(1, steps - 1)
            temperature = 0.025 * (0.0002 / 0.025) ** fraction
            left, right = rng.sample(list(cipher_alphabet), 2)
            affected = objective.index[left] | objective.index[right]
            mapping[left], mapping[right] = mapping[right], mapping[left]
            replacements = {
                index: objective.contribution(index, mapping) for index in affected
            }
            delta = sum(replacements[index] - contributions[index] for index in affected)
            average_delta = delta / objective.total_events
            accept = delta >= 0 or rng.random() < math.exp(
                math.log(2) * average_delta / temperature
            )
            if accept:
                score += delta
                for index, value in replacements.items():
                    contributions[index] = value
                if score > local_best_score:
                    local_best_score = score
                    local_best_mapping = dict(mapping)
            else:
                mapping[left], mapping[right] = mapping[right], mapping[left]
        restart_scores.append(-local_best_score / objective.total_events)
        if local_best_score > best_score:
            best_score = local_best_score
            best_mapping = local_best_mapping
    if best_mapping is None:
        raise AssertionError("No key fit was produced")
    best_mapping, best_score, _ = greedy_polish_key(objective, best_mapping)
    return KeyFit(
        mapping=best_mapping,
        train_bits_per_unit=-best_score / objective.total_events,
        restart_bits_per_unit=tuple(restart_scores),
    )


def score_substitution(
    records: Sequence[WordRecord],
    mapping: dict[str, str],
    model: NgramModel,
    *,
    reverse_words: bool,
) -> tuple[float, int]:
    log_probability = 0.0
    units = 0
    for record in records:
        cipher = tuple(reversed(record.units)) if reverse_words else record.units
        plaintext = tuple(mapping[unit] for unit in cipher)
        log_probability += model.log2_sequence(plaintext)
        units += len(record.units) + 1
    return -log_probability, units


def decoded_lexicon_counts(
    records: Sequence[WordRecord],
    mapping: dict[str, str],
    vocabulary: set[tuple[str, ...]],
    *,
    reverse_words: bool,
) -> dict[str, int]:
    counts = {"hits": 0, "tokens": 0, "hits_length_ge_3": 0, "tokens_length_ge_3": 0}
    for record in records:
        cipher = tuple(reversed(record.units)) if reverse_words else record.units
        decoded = tuple(mapping[unit] for unit in cipher)
        counts["tokens"] += 1
        counts["hits"] += decoded in vocabulary
        if len(decoded) >= 3:
            counts["tokens_length_ge_3"] += 1
            counts["hits_length_ge_3"] += decoded in vocabulary
    return counts


@dataclass(frozen=True)
class EditParameters:
    insertion: float
    deletion: float
    copy: float


def edit_probability(
    source: Sequence[str],
    target: Sequence[str],
    parameters: EditParameters,
    alphabet_size: int,
) -> float:
    """Exact probability under a normalized one-insertion-per-gap edit process."""

    dp = [0.0] * (len(target) + 1)
    dp[0] = 1.0
    for source_token in [*source, None]:
        after_gap = [0.0] * (len(target) + 1)
        for target_index, probability in enumerate(dp):
            after_gap[target_index] += probability * (1 - parameters.insertion)
            if target_index < len(target):
                after_gap[target_index + 1] += (
                    probability * parameters.insertion / alphabet_size
                )
        if source_token is None:
            dp = after_gap
            break
        after_source = [0.0] * (len(target) + 1)
        for target_index, probability in enumerate(after_gap):
            after_source[target_index] += probability * parameters.deletion
            if target_index < len(target):
                emission = (
                    parameters.copy
                    if source_token == target[target_index]
                    else (1 - parameters.copy) / (alphabet_size - 1)
                )
                after_source[target_index + 1] += (
                    probability * (1 - parameters.deletion) * emission
                )
        dp = after_source
    return dp[len(target)]


def fit_mixture_weight(
    pairs: Sequence[tuple[int, float, float]], iterations: int = 30
) -> tuple[float, float]:
    weight = 0.25
    total_count = sum(count for count, _, _ in pairs)
    for _ in range(iterations):
        expected = 0.0
        for count, base, edit in pairs:
            denominator = (1 - weight) * base + weight * edit
            expected += count * (weight * edit / denominator)
        weight = min(0.999, max(0.001, expected / total_count))
    log_likelihood = sum(
        count * math.log2((1 - weight) * base + weight * edit)
        for count, base, edit in pairs
    )
    return weight, log_likelihood


def fit_copy_model(
    records: Sequence[WordRecord],
    layout_model: ConditionalWordModel,
    alphabet_size: int,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...], EditParameters], float],
) -> tuple[EditParameters, float, float]:
    pair_counts = Counter(
        (record.previous, record.units, record.channel)
        for record in records
        if record.previous is not None
    )
    grid = [
        EditParameters(insertion, deletion, copy)
        for insertion in (0.02, 0.08)
        for deletion in (0.02, 0.08)
        for copy in (0.75, 0.90)
    ]
    best: tuple[EditParameters, float, float] | None = None
    for parameters in grid:
        pairs: list[tuple[int, float, float]] = []
        for (source, target, channel), count in pair_counts.items():
            base = 2 ** layout_model.log2_word(target, channel)
            key = (source, target, parameters)
            if key not in cache:
                cache[key] = edit_probability(source, target, parameters, alphabet_size)
            pairs.append((count, base, cache[key]))
        weight, log_likelihood = fit_mixture_weight(pairs)
        if best is None or log_likelihood > best[2]:
            best = (parameters, weight, log_likelihood)
    if best is None:
        raise ValueError("No adjacent training pairs for copy model")
    return best


def score_native_model(
    records: Sequence[WordRecord],
    model: ConditionalWordModel,
    *,
    copy_parameters: EditParameters | None = None,
    copy_weight: float = 0.0,
    alphabet_size: int,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...], EditParameters], float],
) -> tuple[float, int]:
    negative_log_probability = 0.0
    units = 0
    for record in records:
        base_log = model.log2_word(record.units, record.channel)
        if copy_parameters is None or record.previous is None:
            probability_log = base_log
        else:
            key = (record.previous, record.units, copy_parameters)
            if key not in cache:
                cache[key] = edit_probability(
                    record.previous, record.units, copy_parameters, alphabet_size
                )
            probability = (1 - copy_weight) * (2**base_log) + copy_weight * cache[key]
            probability_log = math.log2(probability)
        negative_log_probability -= probability_log
        units += len(record.units) + 1
    return negative_log_probability, units


def summarize_fold_values(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def audit(
    source: Path,
    canon_xml: Path,
    catalog_html: Path | None,
    *,
    folds: int,
    order: int,
    alpha: float,
    channel_weight: float,
    minimum_confidence: int,
    key_restarts: int,
    key_steps: int,
    seed: int,
) -> dict[str, object]:
    pages = [page for page in parse_ivtff(source) if page.lines("P", certain=True)]
    all_records = records_from_pages(pages)
    cipher_alphabet = tuple(sorted({unit for record in all_records for unit in record.units}))
    if len(cipher_alphabet) != 21:
        raise ValueError(f"Expected 21 raw EVA codepoints, found {len(cipher_alphabet)}")

    hebrew = load_hebrew_source(
        canon_xml, catalog_html, minimum_confidence=minimum_confidence
    )
    plain_alphabet = hebrew["alphabet"]
    hebrew_model = NgramModel(order, [*plain_alphabet, EOW], alpha)
    hebrew_model.train(hebrew["training_words"])
    calibration_bits = -sum(
        hebrew_model.log2_sequence(word) for word in hebrew["calibration_words"]
    )
    calibration_units = sum(len(word) + 1 for word in hebrew["calibration_words"])
    hebrew_vocabulary = set(hebrew["training_words"])
    calibration_lexicon = {
        "hits": sum(word in hebrew_vocabulary for word in hebrew["calibration_words"]),
        "tokens": len(hebrew["calibration_words"]),
        "hits_length_ge_3": sum(
            word in hebrew_vocabulary for word in hebrew["calibration_words"] if len(word) >= 3
        ),
        "tokens_length_ge_3": sum(len(word) >= 3 for word in hebrew["calibration_words"]),
    }

    fold_quires = make_quire_folds(pages, folds)
    model_names = (
        "hebrew_substitution",
        "voynich_word_morphology",
        "voynich_layout_conditioned",
        "voynich_layout_copy_mutation",
    )
    totals = {name: [0.0, 0] for name in model_names}
    decoded_lexicon = {
        "hits": 0,
        "tokens": 0,
        "hits_length_ge_3": 0,
        "tokens_length_ge_3": 0,
    }
    fold_results: list[dict[str, object]] = []
    edit_cache: dict[
        tuple[tuple[str, ...], tuple[str, ...], EditParameters], float
    ] = {}

    for fold_index, heldout_quires in enumerate(fold_quires):
        training_pages = [
            page for page in pages if page.metadata.get("Q", "?") not in heldout_quires
        ]
        heldout_pages = [
            page for page in pages if page.metadata.get("Q", "?") in heldout_quires
        ]
        training_records = records_from_pages(training_pages)
        heldout_records = records_from_pages(heldout_pages)

        orientation_fits: list[tuple[bool, KeyFit]] = []
        for reverse_words in (False, True):
            training_words = [
                tuple(reversed(record.units)) if reverse_words else record.units
                for record in training_records
            ]
            fit = fit_substitution_key(
                training_words,
                cipher_alphabet,
                plain_alphabet,
                hebrew["frequencies"],
                hebrew_model,
                seed=seed + fold_index * 100 + int(reverse_words) * 10_000,
                restarts=key_restarts,
                steps=key_steps,
            )
            orientation_fits.append((reverse_words, fit))
        reverse_words, key_fit = min(
            orientation_fits, key=lambda item: item[1].train_bits_per_unit
        )
        hebrew_bits, hebrew_units = score_substitution(
            heldout_records,
            key_fit.mapping,
            hebrew_model,
            reverse_words=reverse_words,
        )
        fold_lexicon = decoded_lexicon_counts(
            heldout_records,
            key_fit.mapping,
            hebrew_vocabulary,
            reverse_words=reverse_words,
        )
        for name, value in fold_lexicon.items():
            decoded_lexicon[name] += value

        word_model = ConditionalWordModel(
            order,
            [*cipher_alphabet, EOW],
            alpha=alpha,
            channel_weight=0.0,
        )
        word_model.train(training_records)
        word_bits, word_units = score_native_model(
            heldout_records,
            word_model,
            alphabet_size=len(cipher_alphabet),
            cache=edit_cache,
        )

        layout_model = ConditionalWordModel(
            order,
            [*cipher_alphabet, EOW],
            alpha=alpha,
            channel_weight=channel_weight,
        )
        layout_model.train(training_records)
        layout_bits, layout_units = score_native_model(
            heldout_records,
            layout_model,
            alphabet_size=len(cipher_alphabet),
            cache=edit_cache,
        )
        copy_parameters, copy_weight, _ = fit_copy_model(
            training_records,
            layout_model,
            len(cipher_alphabet),
            edit_cache,
        )
        copy_bits, copy_units = score_native_model(
            heldout_records,
            layout_model,
            copy_parameters=copy_parameters,
            copy_weight=copy_weight,
            alphabet_size=len(cipher_alphabet),
            cache=edit_cache,
        )

        fold_scores = {
            "hebrew_substitution": hebrew_bits / hebrew_units,
            "voynich_word_morphology": word_bits / word_units,
            "voynich_layout_conditioned": layout_bits / layout_units,
            "voynich_layout_copy_mutation": copy_bits / copy_units,
        }
        raw_scores = {
            "hebrew_substitution": (hebrew_bits, hebrew_units),
            "voynich_word_morphology": (word_bits, word_units),
            "voynich_layout_conditioned": (layout_bits, layout_units),
            "voynich_layout_copy_mutation": (copy_bits, copy_units),
        }
        for name, (bits, units) in raw_scores.items():
            totals[name][0] += bits
            totals[name][1] += units
        fold_results.append(
            {
                "fold": fold_index,
                "heldout_quires": sorted(heldout_quires),
                "training_pages": len(training_pages),
                "heldout_pages": len(heldout_pages),
                "training_words": len(training_records),
                "heldout_words": len(heldout_records),
                "bits_per_unit": fold_scores,
                "winner": min(fold_scores, key=fold_scores.get),
                "hebrew_key": {
                    "reverse_within_words": reverse_words,
                    "train_bits_per_unit": key_fit.train_bits_per_unit,
                    "mapping": key_fit.mapping,
                    "restart_bits_per_unit": list(key_fit.restart_bits_per_unit),
                    "other_orientation_train_bits_per_unit": next(
                        fit.train_bits_per_unit
                        for reverse, fit in orientation_fits
                        if reverse != reverse_words
                    ),
                    "heldout_lexicon_token_rate": (
                        fold_lexicon["hits"] / fold_lexicon["tokens"]
                    ),
                    "heldout_lexicon_token_rate_length_ge_3": (
                        fold_lexicon["hits_length_ge_3"]
                        / fold_lexicon["tokens_length_ge_3"]
                    ),
                },
                "copy_model": {
                    "parameters": asdict(copy_parameters),
                    "mixture_weight": copy_weight,
                },
            }
        )

    aggregate = {
        name: {"bits_per_unit": bits / units, "bits": bits, "units": units}
        for name, (bits, units) in totals.items()
    }
    hebrew_bpu = aggregate["hebrew_substitution"]["bits_per_unit"]
    for name, summary in aggregate.items():
        summary["gain_vs_hebrew_bits_per_unit"] = hebrew_bpu - summary["bits_per_unit"]
        summary["fold_summary"] = summarize_fold_values(
            [fold["bits_per_unit"][name] for fold in fold_results]
        )
    decoded_lexicon["token_rate"] = decoded_lexicon["hits"] / decoded_lexicon["tokens"]
    decoded_lexicon["token_rate_length_ge_3"] = (
        decoded_lexicon["hits_length_ge_3"] / decoded_lexicon["tokens_length_ge_3"]
    )
    calibration_lexicon["token_rate"] = (
        calibration_lexicon["hits"] / calibration_lexicon["tokens"]
    )
    calibration_lexicon["token_rate_length_ge_3"] = (
        calibration_lexicon["hits_length_ge_3"]
        / calibration_lexicon["tokens_length_ge_3"]
    )
    fold_mappings = [fold["hebrew_key"]["mapping"] for fold in fold_results]
    pairwise_key_agreements = [
        sum(left[symbol] == right[symbol] for symbol in cipher_alphabet)
        / len(cipher_alphabet)
        for left_index, left in enumerate(fold_mappings)
        for right in fold_mappings[left_index + 1 :]
    ]

    return {
        "scope": (
            "Predictive likelihood for four declared mechanisms; not a universal test of "
            "Hebrew, ciphers, or meaningfulness."
        ),
        "source": str(source),
        "source_sha256": sha256(source),
        "canon_xml": str(canon_xml),
        "canon_xml_sha256": sha256(canon_xml),
        "canon_expected_sha256": EXPECTED_DJVU_XML_SHA256,
        "catalog_html": str(catalog_html) if catalog_html else None,
        "catalog_html_sha256": sha256(catalog_html) if catalog_html else None,
        "configuration": {
            "folds": folds,
            "order": order,
            "alpha": alpha,
            "channel_weight": channel_weight,
            "minimum_ocr_confidence": minimum_confidence,
            "key_restarts": key_restarts,
            "key_steps": key_steps,
            "seed": seed,
            "heldout_unit": "whole IVTFF quire",
            "scored_unit": "raw EVA codepoint plus one EOW per word",
        },
        "corpus": {
            "voynich_pages": len(pages),
            "voynich_words": len(all_records),
            "cipher_alphabet": cipher_alphabet,
            "hebrew_source_alphabet": plain_alphabet,
            "hebrew_merged_rare_letters": hebrew["rare_pair"],
            "canon_training_words": hebrew["canon_training_word_count"],
            "canon_calibration_words": hebrew["canon_calibration_word_count"],
            "catalog_training_words": hebrew["catalog_training_word_count"],
            "hebrew_calibration_bits_per_unit": calibration_bits / calibration_units,
            "hebrew_calibration_lexicon": calibration_lexicon,
        },
        "aggregate": aggregate,
        "hebrew_decoded_lexicon": decoded_lexicon,
        "hebrew_key_pairwise_fold_agreement": summarize_fold_values(
            pairwise_key_agreements
        ),
        "folds": fold_results,
        "interpretation_guardrails": [
            "The Hebrew model gets an external language prior but only a simple stable key.",
            "Voynich-native models get in-domain training but no access to held-out quires.",
            "Layout channels use Currier stratum and first/middle/last/single word position.",
            "The copy channel is a normalized edit process, not an edit-distance score.",
            "No model-family description-length penalty is included.",
        ],
    }


def print_summary(result: dict[str, object]) -> None:
    corpus = result["corpus"]
    print("Held-out Voynich mechanism tournament")
    print(
        f"Voynich: {corpus['voynich_pages']} pages, {corpus['voynich_words']} words; "
        f"Hebrew calibration: {corpus['hebrew_calibration_bits_per_unit']:.4f} bits/unit"
    )
    print(
        "Hebrew merged source letters: "
        + ", ".join(corpus["hebrew_merged_rare_letters"])
    )
    for name, summary in sorted(
        result["aggregate"].items(), key=lambda item: item[1]["bits_per_unit"]
    ):
        print(
            f"- {name}: {summary['bits_per_unit']:.4f} bits/unit; "
            f"gain vs Hebrew {summary['gain_vs_hebrew_bits_per_unit']:+.4f}"
        )
    winners = Counter(fold["winner"] for fold in result["folds"])
    print("fold winners: " + ", ".join(f"{name}={count}" for name, count in winners.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument("--canon-xml", type=Path, required=True)
    parser.add_argument("--catalog-html", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--channel-weight", type=float, default=0.8)
    parser.add_argument("--minimum-confidence", type=int, default=50)
    parser.add_argument("--key-restarts", type=int, default=5)
    parser.add_argument("--key-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.source,
        args.canon_xml,
        args.catalog_html,
        folds=args.folds,
        order=args.order,
        alpha=args.alpha,
        channel_weight=args.channel_weight,
        minimum_confidence=args.minimum_confidence,
        key_restarts=args.key_restarts,
        key_steps=args.key_steps,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
