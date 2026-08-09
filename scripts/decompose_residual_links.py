#!/usr/bin/env python3
"""Decompose residual Voynich sequence information on nested held-out quires.

This experiment starts from the morphology/layout-conditioned exact-token code
in ``residual_sequence_information.py``.  It adds five proper-probability
expert families:

* paragraph position;
* the previous word's final-to-current-initial boundary transition;
* normalized copy/edit channels from the previous word;
* distributional word classes learned only from training-quire transitions;
* copy/edit and class mixtures over every word on the previous physical line.

Mixture weights are learned from three-way inner-quire out-of-fold predictions.
The weights and all component parameters are then frozen before scoring an
untouched outer quire.  Each family competes alone with the procedural baseline
and all families compete jointly, avoiding attribution by an arbitrary
cumulative ordering.  Matched outer-test permutations preserve target words,
paragraph/layout fields, and the declared source morphology strata.

The result is a predictive decomposition for a finite model family, not an
upper bound on semantic content or a claim that the learned classes are parts
of speech.  ``--token-output`` optionally writes held-out token coordinates,
expert losses, mixture responsibilities, and mean matched-null residuals as
JSONL for spatial diagnostics; a ``.gz`` suffix compresses the trace.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

from audit_voynich import Page, eva_units, parse_ivtff
from compare_mechanisms import EditParameters, edit_probability, line_position
from residual_sequence_information import (
    UNKNOWN,
    EncodedRecord,
    Morphology,
    ResidualIdentityModel,
    SequenceRecord,
    control_words,
    derived_seed,
    morphology_signature,
    parse_control,
    parse_shuffled_control,
    quantile,
    sha256,
)


EDIT_PARAMETERS = (
    EditParameters(insertion=0.02, deletion=0.02, copy=0.90),
    EditParameters(insertion=0.08, deletion=0.08, copy=0.75),
)
CLASS_COUNTS = (8, 16)
MIXTURE_ITERATIONS = 200


@dataclass(frozen=True)
class StructuredLine:
    words: tuple[str, ...]
    paragraph_line: int
    paragraph_start: bool
    paragraph_end: bool


@dataclass(frozen=True)
class StructuredPage:
    name: str
    quire: str
    currier: str
    topic: str
    lines: tuple[StructuredLine, ...]


@dataclass(frozen=True)
class DecompositionRecord:
    word: str
    morphology: Morphology
    currier: str
    topic: str
    position: str
    paragraph_state: str
    page: str
    page_index: int
    quire: str
    line_index: int
    word_index: int
    line_length: int
    previous_word: str | None
    previous_morphology: Morphology | None
    previous_line_word: str | None
    previous_line_morphology: Morphology | None
    previous_line_words: tuple[str, ...]
    previous_line_morphologies: tuple[Morphology, ...]
    previous_line_index: int | None

    def base_record(self) -> SequenceRecord:
        return SequenceRecord(
            word=self.word,
            morphology=self.morphology,
            currier=self.currier,
            topic=self.topic,
            position=self.position,
            page=self.page,
            quire=self.quire,
            previous_word=self.previous_word,
            previous_morphology=self.previous_morphology,
            previous_line_word=self.previous_line_word,
            previous_line_morphology=self.previous_line_morphology,
        )


def pages_to_structured(pages: Sequence[Page]) -> list[StructuredPage]:
    result: list[StructuredPage] = []
    for page in pages:
        lines: list[StructuredLine] = []
        paragraph_line = 0
        start_next = True
        for locus in page.loci:
            if not locus.locus_type.startswith("P") or not locus.certain_tokens:
                continue
            paragraph_start = locus.paragraph_start or start_next
            if paragraph_start:
                paragraph_line = 0
            lines.append(
                StructuredLine(
                    words=tuple(locus.certain_tokens),
                    paragraph_line=paragraph_line,
                    paragraph_start=paragraph_start,
                    paragraph_end=locus.paragraph_end,
                )
            )
            start_next = locus.paragraph_end
            if not start_next:
                paragraph_line += 1
        if lines:
            result.append(
                StructuredPage(
                    name=page.name,
                    quire=page.metadata.get("Q", "?"),
                    currier=page.metadata.get("L", "?"),
                    topic=page.metadata.get("I", "?"),
                    lines=tuple(lines),
                )
            )
    return result


def paragraph_state(line: StructuredLine) -> str:
    line_bucket = str(line.paragraph_line) if line.paragraph_line < 3 else "3+"
    ending = "end" if line.paragraph_end else "open"
    return f"{line_bucket}:{ending}"


def nearest_previous_line_index(
    previous_line: StructuredLine | None, *, index: int, current_length: int
) -> int | None:
    if previous_line is None:
        return None
    target_midpoint = (index + 0.5) / current_length
    return min(
        range(len(previous_line.words)),
        key=lambda candidate: (
            abs(
                (candidate + 0.5) / len(previous_line.words)
                - target_midpoint
            ),
            candidate,
        ),
    )


def records_from_structured(
    pages: Sequence[StructuredPage], *, morphology_depth: int, grouped_eva: bool
) -> list[DecompositionRecord]:
    result: list[DecompositionRecord] = []
    for page_index, page in enumerate(pages):
        previous_line: StructuredLine | None = None
        for physical_line_index, line in enumerate(page.lines):
            for index, word in enumerate(line.words):
                previous_word = line.words[index - 1] if index else None
                previous_line_index = nearest_previous_line_index(
                    previous_line, index=index, current_length=len(line.words)
                )
                previous_line_word = (
                    previous_line.words[previous_line_index]
                    if previous_line is not None and previous_line_index is not None
                    else None
                )
                result.append(
                    DecompositionRecord(
                        word=word,
                        morphology=morphology_signature(
                            word,
                            depth=morphology_depth,
                            grouped_eva=grouped_eva,
                        ),
                        currier=page.currier,
                        topic=page.topic,
                        position=line_position(index, len(line.words)),
                        paragraph_state=paragraph_state(line),
                        page=page.name,
                        page_index=page_index,
                        quire=page.quire,
                        line_index=physical_line_index,
                        word_index=index,
                        line_length=len(line.words),
                        previous_word=previous_word,
                        previous_morphology=(
                            morphology_signature(
                                previous_word,
                                depth=morphology_depth,
                                grouped_eva=grouped_eva,
                            )
                            if previous_word is not None
                            else None
                        ),
                        previous_line_word=previous_line_word,
                        previous_line_morphology=(
                            morphology_signature(
                                previous_line_word,
                                depth=morphology_depth,
                                grouped_eva=grouped_eva,
                            )
                            if previous_line_word is not None
                            else None
                        ),
                        previous_line_words=(
                            previous_line.words if previous_line is not None else ()
                        ),
                        previous_line_morphologies=(
                            tuple(
                                morphology_signature(
                                    source,
                                    depth=morphology_depth,
                                    grouped_eva=grouped_eva,
                                )
                                for source in previous_line.words
                            )
                            if previous_line is not None
                            else ()
                        ),
                        previous_line_index=previous_line_index,
                    )
                )
            previous_line = line
    return result


def reflow_structured(
    template: Sequence[StructuredPage], words: Sequence[str]
) -> list[StructuredPage]:
    needed = sum(len(line.words) for page in template for line in page.lines)
    if len(words) < needed:
        raise ValueError(f"Control has {len(words)} words; {needed} are required")
    result: list[StructuredPage] = []
    offset = 0
    for page in template:
        lines: list[StructuredLine] = []
        for line in page.lines:
            length = len(line.words)
            lines.append(
                replace(line, words=tuple(words[offset : offset + length]))
            )
            offset += length
        result.append(replace(page, lines=tuple(lines)))
    return result


def balanced_quire_folds(
    records: Sequence[DecompositionRecord], folds: int
) -> list[set[str]]:
    if folds < 2:
        raise ValueError("At least two folds are required")
    counts = Counter(record.quire for record in records)
    if len(counts) < folds:
        raise ValueError(f"Only {len(counts)} quires are available for {folds} folds")
    assignments = [set() for _ in range(folds)]
    totals = [0] * folds
    for quire, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        destination = min(range(folds), key=lambda index: (totals[index], index))
        assignments[destination].add(quire)
        totals[destination] += count
    return assignments


def sparse_dot(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def normalize_vector(vector: Mapping[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {key: value / norm for key, value in vector.items()}


def distributional_vectors(
    records: Sequence[DecompositionRecord], vocabulary: set[str]
) -> tuple[dict[str, dict[str, float]], Counter[str]]:
    """Build PPMI vectors from training-only left/right exact contexts."""

    joint: dict[str, Counter[str]] = defaultdict(Counter)
    word_frequency: Counter[str] = Counter()
    feature_frequency: Counter[str] = Counter()
    total = 0
    for record in records:
        if record.word in vocabulary:
            word_frequency[record.word] += 1
        if record.word in vocabulary and record.previous_word is not None:
            context = (
                record.previous_word
                if record.previous_word in vocabulary
                else UNKNOWN
            )
            feature = f"L:{context}"
            joint[record.word][feature] += 1
            feature_frequency[feature] += 1
            total += 1
        # The right context is recorded when this word appears as a predecessor.
        if record.previous_word in vocabulary:
            context = record.word if record.word in vocabulary else UNKNOWN
            feature = f"R:{context}"
            joint[record.previous_word][feature] += 1
            feature_frequency[feature] += 1
            total += 1
    vectors: dict[str, dict[str, float]] = {}
    for word in vocabulary:
        row_total = sum(joint[word].values())
        weighted: dict[str, float] = {}
        if row_total and total:
            for feature, count in joint[word].items():
                ratio = (count * total) / (row_total * feature_frequency[feature])
                if ratio > 1:
                    weighted[feature] = math.log2(ratio)
        vectors[word] = normalize_vector(weighted)
    return vectors, word_frequency


def learn_word_classes(
    records: Sequence[DecompositionRecord], vocabulary: set[str], classes: int
) -> dict[str, int]:
    """Deterministic spherical k-means over training-only PPMI contexts."""

    vectors, frequencies = distributional_vectors(records, vocabulary)
    words = sorted(vocabulary, key=lambda word: (-frequencies[word], word))
    usable = [word for word in words if vectors[word]]
    if not usable:
        return {word: 0 for word in vocabulary}
    classes = min(classes, len(usable))
    seeds = [usable[0]]
    while len(seeds) < classes:
        candidate = min(
            (word for word in usable if word not in seeds),
            key=lambda word: (
                max(sparse_dot(vectors[word], vectors[seed]) for seed in seeds),
                -frequencies[word],
                word,
            ),
        )
        seeds.append(candidate)
    centroids = [vectors[word] for word in seeds]
    assignments: dict[str, int] = {}
    for _ in range(12):
        updated = {
            word: max(
                range(classes),
                key=lambda index: (sparse_dot(vectors[word], centroids[index]), -index),
            )
            for word in usable
        }
        if updated == assignments:
            break
        assignments = updated
        sums: list[dict[str, float]] = [defaultdict(float) for _ in range(classes)]
        for word, assignment in assignments.items():
            weight = math.sqrt(max(1, frequencies[word]))
            for feature, value in vectors[word].items():
                sums[assignment][feature] += weight * value
        centroids = [
            normalize_vector(sums[index]) if sums[index] else vectors[seeds[index]]
            for index in range(classes)
        ]
    for word in vocabulary:
        if word not in assignments:
            assignments[word] = min(range(classes), key=lambda index: index)
    return assignments


def normalize_distribution(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("A probability expert produced no mass")
    return {target: value / total for target, value in weights.items()}


class ExpertSuite:
    """Fit all decomposition experts on one training partition."""

    expert_names = (
        "base",
        "paragraph",
        "boundary",
        "edit_tight",
        "edit_loose",
        "class_8",
        "class_16",
        "line_edit_tight",
        "line_edit_loose",
        "line_class_8",
        "line_class_16",
        "exact_previous",
        "exact_aligned",
    )

    family_names = {
        "base": ("base",),
        "paragraph": ("base", "paragraph"),
        "boundary": ("base", "boundary"),
        "copy_edit": ("base", "edit_tight", "edit_loose"),
        "latent_class": ("base", "class_8", "class_16"),
        "previous_line_pool": (
            "base",
            "line_edit_tight",
            "line_edit_loose",
            "line_class_8",
            "line_class_16",
        ),
        "exact_identity": ("base", "exact_previous", "exact_aligned"),
        "full": expert_names,
    }

    def __init__(
        self,
        records: Sequence[DecompositionRecord],
        *,
        vocabulary_limit: int,
        alpha: float,
        strength: float,
        grouped_eva: bool,
    ):
        self.records = tuple(records)
        self.alpha = alpha
        self.strength = strength
        self.grouped_eva = grouped_eva
        self.base_model = ResidualIdentityModel(
            [record.base_record() for record in records],
            vocabulary_limit=vocabulary_limit,
            alpha=alpha,
            strength=strength,
        )
        self.vocabulary = self.base_model.vocabulary
        self._units_cache: dict[str, tuple[str, ...]] = {}
        self._support_cache: dict[Morphology, tuple[str, ...]] = {}
        self._raw_edit_cache: dict[
            tuple[str, str, EditParameters], float
        ] = {}
        self.unit_inventory = {
            unit
            for record in records
            for unit in self.units(record.word)
        }
        self.alphabet_size = max(2, len(self.unit_inventory))
        self.paragraph_counts: dict[
            tuple[Morphology, str], Counter[str]
        ] = defaultdict(Counter)
        self.paragraph_marginals: dict[Morphology, Counter[str]] = defaultdict(
            Counter
        )
        self.boundary_counts: dict[
            tuple[Morphology, Morphology, str], Counter[str]
        ] = defaultdict(Counter)
        self.initial_counts: dict[
            tuple[Morphology, Morphology], Counter[str]
        ] = defaultdict(Counter)
        self.previous_identity_counts: dict[
            tuple[Morphology, Morphology, str], Counter[str]
        ] = defaultdict(Counter)
        self.previous_identity_marginals: dict[
            tuple[Morphology, Morphology], Counter[str]
        ] = defaultdict(Counter)
        self.aligned_identity_counts: dict[
            tuple[Morphology, Morphology, str], Counter[str]
        ] = defaultdict(Counter)
        self.aligned_identity_marginals: dict[
            tuple[Morphology, Morphology], Counter[str]
        ] = defaultdict(Counter)
        self.class_maps = {
            count: learn_word_classes(records, self.vocabulary, count)
            for count in CLASS_COUNTS
        }
        self.class_transitions: dict[
            int, dict[tuple[Morphology, int], Counter[int]]
        ] = {
            count: defaultdict(Counter) for count in CLASS_COUNTS
        }
        self.class_marginals: dict[
            int, dict[Morphology, Counter[int]]
        ] = {
            count: defaultdict(Counter) for count in CLASS_COUNTS
        }
        self.line_class_transitions: dict[
            int, dict[tuple[Morphology, int], Counter[int]]
        ] = {
            count: defaultdict(Counter) for count in CLASS_COUNTS
        }
        self.line_class_marginals: dict[
            int, dict[Morphology, Counter[int]]
        ] = {
            count: defaultdict(Counter) for count in CLASS_COUNTS
        }
        for record in records:
            encoded = self.encode(record)
            self.paragraph_counts[
                (record.morphology, record.paragraph_state)
            ][encoded.target] += 1
            self.paragraph_marginals[record.morphology][encoded.target] += 1
            target_units = self.units(record.word)
            if (
                record.previous_word is not None
                and record.previous_morphology is not None
                and target_units
            ):
                source_units = self.units(record.previous_word)
                if source_units:
                    self.boundary_counts[
                        (
                            record.morphology,
                            record.previous_morphology,
                            source_units[-1],
                        )
                    ][target_units[0]] += 1
                    self.initial_counts[
                        (record.morphology, record.previous_morphology)
                    ][target_units[0]] += 1
            if (
                record.previous_word in self.vocabulary
                and record.previous_morphology is not None
            ):
                identity_key = (
                    record.morphology,
                    record.previous_morphology,
                )
                self.previous_identity_counts[
                    (*identity_key, record.previous_word)
                ][encoded.target] += 1
                self.previous_identity_marginals[identity_key][encoded.target] += 1
            if (
                record.previous_line_word in self.vocabulary
                and record.previous_line_morphology is not None
            ):
                identity_key = (
                    record.morphology,
                    record.previous_line_morphology,
                )
                self.aligned_identity_counts[
                    (*identity_key, record.previous_line_word)
                ][encoded.target] += 1
                self.aligned_identity_marginals[identity_key][encoded.target] += 1
            for count, mapping in self.class_maps.items():
                if (
                    record.previous_word in mapping
                    and record.previous_morphology is not None
                    and record.word in mapping
                ):
                    target_class = mapping[record.word]
                    source_class = mapping[record.previous_word]
                    self.class_transitions[count][
                        (record.previous_morphology, source_class)
                    ][target_class] += 1
                    self.class_marginals[count][record.previous_morphology][
                        target_class
                    ] += 1
                if record.word in mapping:
                    target_class = mapping[record.word]
                    for source, source_morphology in zip(
                        record.previous_line_words,
                        record.previous_line_morphologies,
                        strict=True,
                    ):
                        if source not in mapping:
                            continue
                        source_class = mapping[source]
                        self.line_class_transitions[count][
                            (source_morphology, source_class)
                        ][target_class] += 1
                        self.line_class_marginals[count][source_morphology][
                            target_class
                        ] += 1
        self._base_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self._paragraph_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self._boundary_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self._edit_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self._class_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self._line_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self._exact_cache: dict[tuple[object, ...], dict[str, float]] = {}

    def units(self, word: str) -> tuple[str, ...]:
        if word not in self._units_cache:
            if self.grouped_eva:
                self._units_cache[word] = tuple(eva_units(word, grouped=True))
            else:
                self._units_cache[word] = tuple(word)
        return self._units_cache[word]

    def support(self, morphology: Morphology) -> tuple[str, ...]:
        if morphology not in self._support_cache:
            self._support_cache[morphology] = tuple(
                sorted(self.base_model.support.get(morphology, {UNKNOWN}))
            )
        return self._support_cache[morphology]

    def raw_edit_probability(
        self, source: str, target: str, parameters: EditParameters
    ) -> float:
        key = (source, target, parameters)
        if key not in self._raw_edit_cache:
            self._raw_edit_cache[key] = edit_probability(
                self.units(source),
                self.units(target),
                parameters,
                self.alphabet_size,
            )
        return self._raw_edit_cache[key]

    def encode(self, record: DecompositionRecord) -> EncodedRecord:
        return self.base_model.encode(record.base_record())

    @staticmethod
    def _base_key(encoded: EncodedRecord) -> tuple[object, ...]:
        return (
            encoded.morphology,
            encoded.currier,
            encoded.topic,
            encoded.position,
            encoded.previous_morphology,
            encoded.previous_line_morphology,
        )

    def base_distribution(
        self, record: DecompositionRecord, encoded: EncodedRecord
    ) -> dict[str, float]:
        key = self._base_key(encoded)
        if key not in self._base_cache:
            support = self.support(encoded.morphology)
            self._base_cache[key] = {
                target: self.base_model.probabilities(
                    replace(encoded, target=target)
                )["procedural_morphology"]
                for target in support
            }
        return self._base_cache[key]

    def paragraph_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
    ) -> dict[str, float]:
        key = (*self._base_key(encoded), record.paragraph_state)
        if key not in self._paragraph_cache:
            conditional = self.paragraph_counts[
                (record.morphology, record.paragraph_state)
            ]
            marginal = self.paragraph_marginals[record.morphology]
            self._paragraph_cache[key] = self.reweight_from_counts(
                base, conditional, marginal
            )
        return self._paragraph_cache[key]

    def reweight_from_counts(
        self,
        base: Mapping[str, float],
        conditional: Counter[str] | Counter[int],
        marginal: Counter[str] | Counter[int],
        *,
        category: Mapping[str, int] | None = None,
    ) -> dict[str, float]:
        """Apply a smoothed conditional/marginal density ratio to a base."""

        categories = set(marginal) | set(conditional)
        category_count = max(1, len(categories))
        marginal_total = sum(marginal.values())
        conditional_total = sum(conditional.values())
        weights: dict[str, float] = {}
        for target, base_probability in base.items():
            target_category: str | int | None
            if category is None:
                target_category = target
            else:
                target_category = category.get(target)
            if target_category is None:
                factor = 1.0
            else:
                marginal_probability = (
                    marginal[target_category] + self.alpha
                ) / (marginal_total + self.alpha * category_count)
                conditional_probability = (
                    conditional[target_category]
                    + self.strength * marginal_probability
                ) / (conditional_total + self.strength)
                factor = conditional_probability / marginal_probability
            weights[target] = base_probability * factor
        return normalize_distribution(weights)

    def boundary_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
    ) -> dict[str, float]:
        if record.previous_word is None:
            return dict(base)
        source_units = self.units(record.previous_word)
        if not source_units:
            return dict(base)
        source_final = source_units[-1]
        key = (*self._base_key(encoded), source_final)
        if key not in self._boundary_cache:
            if record.previous_morphology is None:
                return dict(base)
            conditional = self.boundary_counts[
                (record.morphology, record.previous_morphology, source_final)
            ]
            marginal = self.initial_counts[
                (record.morphology, record.previous_morphology)
            ]
            initials = {
                target: self.units(target)[0]
                for target in base
                if target != UNKNOWN
            }
            self._boundary_cache[key] = self.reweight_from_counts(
                base, conditional, marginal, category=initials
            )
        return self._boundary_cache[key]

    def edit_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        source_word: str | None,
        parameters: EditParameters,
    ) -> dict[str, float]:
        if source_word not in self.vocabulary:
            return dict(base)
        key = (
            *self._base_key(encoded),
            source_word,
            parameters,
        )
        if key not in self._edit_cache:
            known = [target for target in base if target != UNKNOWN]
            if not known:
                return dict(base)
            edit_weights = {
                target: self.raw_edit_probability(
                    source_word, target, parameters
                )
                for target in known
            }
            if sum(edit_weights.values()) <= 0:
                self._edit_cache[key] = dict(base)
                return self._edit_cache[key]
            normalized_edit = normalize_distribution(edit_weights)
            unknown_mass = base.get(UNKNOWN, 0.0)
            known_mass = 1.0 - unknown_mass
            distribution = {
                target: known_mass * probability
                for target, probability in normalized_edit.items()
            }
            if UNKNOWN in base:
                distribution[UNKNOWN] = unknown_mass
            self._edit_cache[key] = distribution
        return self._edit_cache[key]

    def class_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        source_word: str | None,
        source_morphology: Morphology | None,
        classes: int,
        *,
        line_context: bool = False,
    ) -> dict[str, float]:
        mapping = self.class_maps[classes]
        if source_word not in mapping or source_morphology is None:
            return dict(base)
        source_class = mapping[source_word]
        key = (
            *self._base_key(encoded),
            source_morphology,
            source_class,
            classes,
            line_context,
        )
        if key not in self._class_cache:
            if line_context:
                transitions = self.line_class_transitions[classes][
                    (source_morphology, source_class)
                ]
                marginal = self.line_class_marginals[classes][source_morphology]
            else:
                transitions = self.class_transitions[classes][
                    (source_morphology, source_class)
                ]
                marginal = self.class_marginals[classes][source_morphology]
            self._class_cache[key] = self.reweight_from_counts(
                base, transitions, marginal, category=mapping
            )
        return self._class_cache[key]

    @staticmethod
    def average_distributions(
        distributions: Sequence[Mapping[str, float]],
        base: Mapping[str, float],
    ) -> dict[str, float]:
        if not distributions:
            return dict(base)
        return {
            target: statistics.fmean(
                distribution.get(target, 0.0) for distribution in distributions
            )
            for target in base
        }

    def line_edit_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        parameters: EditParameters,
    ) -> dict[str, float]:
        sources = tuple(
            word for word in record.previous_line_words if word in self.vocabulary
        )
        key = (*self._base_key(encoded), "line_edit", sources, parameters)
        if key not in self._line_cache:
            self._line_cache[key] = self.average_distributions(
                [
                    self.edit_distribution(record, encoded, base, source, parameters)
                    for source in sources
                ],
                base,
            )
        return self._line_cache[key]

    def line_edit_probability(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        target: str,
        parameters: EditParameters,
    ) -> float:
        sources = [
            word for word in record.previous_line_words if word in self.vocabulary
        ]
        if not sources:
            return base[target]
        return statistics.fmean(
            self.edit_distribution(record, encoded, base, source, parameters)[target]
            for source in sources
        )

    def line_class_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        classes: int,
    ) -> dict[str, float]:
        mapping = self.class_maps[classes]
        sources = tuple(word for word in record.previous_line_words if word in mapping)
        key = (*self._base_key(encoded), "line_class", sources, classes)
        if key not in self._line_cache:
            self._line_cache[key] = self.average_distributions(
                [
                    self.class_distribution(
                        record,
                        encoded,
                        base,
                        source,
                        source_morphology,
                        classes,
                        line_context=True,
                    )
                    for source, source_morphology in zip(
                        record.previous_line_words,
                        record.previous_line_morphologies,
                        strict=True,
                    )
                    if source in mapping
                ],
                base,
            )
        return self._line_cache[key]

    def line_class_probability(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        target: str,
        classes: int,
    ) -> float:
        mapping = self.class_maps[classes]
        values = [
            self.class_distribution(
                record,
                encoded,
                base,
                source,
                source_morphology,
                classes,
                line_context=True,
            )[target]
            for source, source_morphology in zip(
                record.previous_line_words,
                record.previous_line_morphologies,
                strict=True,
            )
            if source in mapping
        ]
        return statistics.fmean(values) if values else base[target]

    def exact_distribution(
        self,
        record: DecompositionRecord,
        encoded: EncodedRecord,
        base: Mapping[str, float],
        channel: str,
    ) -> dict[str, float]:
        if channel == "previous":
            source_word = record.previous_word
            source_morphology = record.previous_morphology
            counts = self.previous_identity_counts
            marginals = self.previous_identity_marginals
        elif channel == "aligned":
            source_word = record.previous_line_word
            source_morphology = record.previous_line_morphology
            counts = self.aligned_identity_counts
            marginals = self.aligned_identity_marginals
        else:
            raise ValueError(f"Unknown identity channel: {channel}")
        if source_word not in self.vocabulary or source_morphology is None:
            return dict(base)
        key = (*self._base_key(encoded), source_word, channel)
        if key not in self._exact_cache:
            marginal_key = (record.morphology, source_morphology)
            self._exact_cache[key] = self.reweight_from_counts(
                base,
                counts[(*marginal_key, source_word)],
                marginals[marginal_key],
            )
        return self._exact_cache[key]

    def probability_row(self, record: DecompositionRecord) -> dict[str, float]:
        encoded = self.encode(record)
        target = encoded.target
        support = self.base_model.support.get(encoded.morphology, {UNKNOWN})
        if target not in support:
            target = UNKNOWN
            encoded = replace(encoded, target=target)
        base = self.base_distribution(record, encoded)
        distributions = {
            "base": base,
            "paragraph": self.paragraph_distribution(record, encoded, base),
            "boundary": self.boundary_distribution(record, encoded, base),
            "edit_tight": self.edit_distribution(
                record, encoded, base, record.previous_word, EDIT_PARAMETERS[0]
            ),
            "edit_loose": self.edit_distribution(
                record, encoded, base, record.previous_word, EDIT_PARAMETERS[1]
            ),
            "class_8": self.class_distribution(
                record,
                encoded,
                base,
                record.previous_word,
                record.previous_morphology,
                CLASS_COUNTS[0],
            ),
            "class_16": self.class_distribution(
                record,
                encoded,
                base,
                record.previous_word,
                record.previous_morphology,
                CLASS_COUNTS[1],
            ),
        }
        row = {
            name: distribution[target] for name, distribution in distributions.items()
        }
        row["line_edit_tight"] = self.line_edit_probability(
            record, encoded, base, target, EDIT_PARAMETERS[0]
        )
        row["line_edit_loose"] = self.line_edit_probability(
            record, encoded, base, target, EDIT_PARAMETERS[1]
        )
        row["line_class_8"] = self.line_class_probability(
            record, encoded, base, target, CLASS_COUNTS[0]
        )
        row["line_class_16"] = self.line_class_probability(
            record, encoded, base, target, CLASS_COUNTS[1]
        )
        row["exact_previous"] = self.exact_distribution(
            record, encoded, base, "previous"
        )[target]
        row["exact_aligned"] = self.exact_distribution(
            record, encoded, base, "aligned"
        )[target]
        return row

    def probability_masses(self, record: DecompositionRecord) -> dict[str, float]:
        """Return each expert's summed mass for a normalization regression test."""

        encoded = self.encode(record)
        base = self.base_distribution(record, encoded)
        rows: dict[str, Mapping[str, float]] = {
            "base": base,
            "paragraph": self.paragraph_distribution(record, encoded, base),
            "boundary": self.boundary_distribution(record, encoded, base),
            "edit_tight": self.edit_distribution(
                record, encoded, base, record.previous_word, EDIT_PARAMETERS[0]
            ),
            "edit_loose": self.edit_distribution(
                record, encoded, base, record.previous_word, EDIT_PARAMETERS[1]
            ),
            "class_8": self.class_distribution(
                record,
                encoded,
                base,
                record.previous_word,
                record.previous_morphology,
                CLASS_COUNTS[0],
            ),
            "class_16": self.class_distribution(
                record,
                encoded,
                base,
                record.previous_word,
                record.previous_morphology,
                CLASS_COUNTS[1],
            ),
            "line_edit_tight": self.line_edit_distribution(
                record, encoded, base, EDIT_PARAMETERS[0]
            ),
            "line_edit_loose": self.line_edit_distribution(
                record, encoded, base, EDIT_PARAMETERS[1]
            ),
            "line_class_8": self.line_class_distribution(
                record, encoded, base, CLASS_COUNTS[0]
            ),
            "line_class_16": self.line_class_distribution(
                record, encoded, base, CLASS_COUNTS[1]
            ),
            "exact_previous": self.exact_distribution(
                record, encoded, base, "previous"
            ),
            "exact_aligned": self.exact_distribution(
                record, encoded, base, "aligned"
            ),
        }
        return {name: sum(distribution.values()) for name, distribution in rows.items()}


def fit_mixture_weights(
    rows: Sequence[Mapping[str, float]],
    expert_names: Sequence[str],
    *,
    iterations: int = MIXTURE_ITERATIONS,
    floor: float = 1e-9,
) -> dict[str, float]:
    """Maximum-likelihood convex weights fitted by mixture EM."""

    if not rows:
        raise ValueError("Cannot fit mixture weights without validation rows")
    names = tuple(expert_names)
    if len(names) == 1:
        return {names[0]: 1.0}
    weights = {name: 1.0 / len(names) for name in names}
    for _ in range(iterations):
        expected = {name: 0.0 for name in names}
        for row in rows:
            denominator = sum(weights[name] * row[name] for name in names)
            if denominator <= 0:
                raise ValueError("Mixture probability is not positive")
            for name in names:
                expected[name] += weights[name] * row[name] / denominator
        updated = {
            name: max(floor, expected[name] / len(rows)) for name in names
        }
        normalizer = sum(updated.values())
        updated = {name: value / normalizer for name, value in updated.items()}
        if max(abs(updated[name] - weights[name]) for name in names) < 1e-9:
            weights = updated
            break
        weights = updated
    return weights


def mixture_probability(
    row: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    return sum(weight * row[name] for name, weight in weights.items())


def score_rows(
    rows: Iterable[Mapping[str, float]], weights: Mapping[str, float]
) -> float:
    return -sum(math.log2(mixture_probability(row, weights)) for row in rows)


def fit_nested_weights(
    training: Sequence[DecompositionRecord],
    *,
    inner_folds: int,
    vocabulary_limit: int,
    alpha: float,
    strength: float,
    grouped_eva: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    fold_quires = balanced_quire_folds(training, inner_folds)
    out_of_fold_rows: list[dict[str, float]] = []
    for heldout_quires in fold_quires:
        inner_training = [
            record for record in training if record.quire not in heldout_quires
        ]
        inner_heldout = [
            record for record in training if record.quire in heldout_quires
        ]
        suite = ExpertSuite(
            inner_training,
            vocabulary_limit=vocabulary_limit,
            alpha=alpha,
            strength=strength,
            grouped_eva=grouped_eva,
        )
        out_of_fold_rows.extend(
            suite.probability_row(record) for record in inner_heldout
        )
    weights = {
        family: fit_mixture_weights(out_of_fold_rows, experts)
        for family, experts in ExpertSuite.family_names.items()
    }
    bits_per_word = {
        family: score_rows(out_of_fold_rows, family_weights)
        / len(out_of_fold_rows)
        for family, family_weights in weights.items()
    }
    return weights, bits_per_word


def normalized_slot_bucket(index: int, length: int) -> str:
    return line_position(index, length)


def matched_source_permutation(
    records: Sequence[DecompositionRecord], *, seed: int
) -> tuple[list[DecompositionRecord], int]:
    """Break source links while retaining target and declared nuisance fields."""

    rng = random.Random(seed)
    result = list(records)
    changed_indices: set[int] = set()

    previous_groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        previous_groups[
            (
                record.currier,
                record.topic,
                record.position,
                record.previous_morphology,
            )
        ].append(index)
    for indices in previous_groups.values():
        values = [records[index].previous_word for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values, strict=True):
            if value != records[index].previous_word:
                changed_indices.add(index)
            result[index] = replace(result[index], previous_word=value)

    # Shuffle each exact source token within its own morphology and normalized
    # source-position stratum.  This leaves every target record with the same
    # ordered prior-line morphology profile while breaking exact-form links.
    # The aligned source remains the physically aligned member of that pool.
    pool_groups: dict[tuple[object, ...], list[tuple[int, int]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        for source_index, source_morphology in enumerate(
            record.previous_line_morphologies
        ):
            pool_groups[
                (
                    record.currier,
                    record.topic,
                    record.position,
                    source_morphology,
                    normalized_slot_bucket(
                        source_index, len(record.previous_line_morphologies)
                    ),
                    source_index == record.previous_line_index,
                )
            ].append((record_index, source_index))
    mutable_pools = [list(record.previous_line_words) for record in records]
    for locations in pool_groups.values():
        values = [records[i].previous_line_words[j] for i, j in locations]
        rng.shuffle(values)
        for (record_index, source_index), value in zip(
            locations, values, strict=True
        ):
            if value != records[record_index].previous_line_words[source_index]:
                changed_indices.add(record_index)
            mutable_pools[record_index][source_index] = value
    for index, pool in enumerate(mutable_pools):
        aligned_index = records[index].previous_line_index
        aligned = pool[aligned_index] if aligned_index is not None else None
        if aligned != records[index].previous_line_word:
            changed_indices.add(index)
        result[index] = replace(
            result[index],
            previous_line_word=aligned,
            previous_line_words=tuple(pool),
        )
    return result, len(changed_indices)


def summarize_weights(
    fold_weights: Sequence[Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    names = sorted({name for weights in fold_weights for name in weights})
    return {
        name: {
            "mean": statistics.fmean(weights.get(name, 0.0) for weights in fold_weights),
            "minimum": min(weights.get(name, 0.0) for weights in fold_weights),
            "maximum": max(weights.get(name, 0.0) for weights in fold_weights),
        }
        for name in names
    }


def source_link_signature(record: DecompositionRecord) -> tuple[object, ...]:
    """Return the exact source state changed by the matched-link null."""

    return (
        record.previous_word,
        record.previous_line_word,
        record.previous_line_words,
    )


def open_token_trace(path: Path) -> TextIO:
    """Open JSONL output, using gzip when requested by suffix."""

    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def token_trace_payload(
    *,
    corpus_name: str,
    fold_index: int,
    record: DecompositionRecord,
    suite: ExpertSuite,
    actual_row: Mapping[str, float],
    weights: Mapping[str, Mapping[str, float]],
    morphology_depth: int,
    permutations: int,
    changed_count: int,
    permuted_expert_loss_sums: Sequence[float] | None,
    permuted_family_loss_sums: Sequence[float] | None,
) -> dict[str, object]:
    """Build one self-describing held-out token record for graphics."""

    expert_names = ExpertSuite.expert_names
    family_names = tuple(ExpertSuite.family_names)
    expert_losses = {
        name: -math.log2(actual_row[name]) for name in expert_names
    }
    family_probabilities = {
        family: mixture_probability(actual_row, weights[family])
        for family in family_names
    }
    family_losses = {
        family: -math.log2(probability)
        for family, probability in family_probabilities.items()
    }
    base_loss = expert_losses["base"]
    full_probability = family_probabilities["full"]
    full_weights = weights["full"]
    responsibilities = {
        name: full_weights.get(name, 0.0) * actual_row[name] / full_probability
        for name in expert_names
    }
    dominant_expert = max(
        expert_names,
        key=lambda name: (responsibilities[name], name),
    )

    target_units = suite.units(record.word)
    previous_units = (
        suite.units(record.previous_word)
        if record.previous_word is not None
        else ()
    )
    encoded = suite.encode(record)
    matched_null: dict[str, object] | None = None
    if permutations:
        if (
            permuted_expert_loss_sums is None
            or permuted_family_loss_sums is None
        ):
            raise ValueError("Token trace is missing matched-null accumulators")
        expert_mean_losses = {
            name: permuted_expert_loss_sums[index] / permutations
            for index, name in enumerate(expert_names)
        }
        family_mean_losses = {
            family: permuted_family_loss_sums[index] / permutations
            for index, family in enumerate(family_names)
        }
        matched_null = {
            "permutations": permutations,
            "source_changed_fraction": changed_count / permutations,
            "expert_mean_log_loss_bits": expert_mean_losses,
            "expert_actual_advantage_bits": {
                name: expert_mean_losses[name] - expert_losses[name]
                for name in expert_names
            },
            "family_mean_log_loss_bits": family_mean_losses,
            "family_actual_advantage_bits": {
                family: family_mean_losses[family] - family_losses[family]
                for family in family_names
            },
        }

    return {
        "schema": "voynich-residual-token-trace-v1",
        "corpus": corpus_name,
        "outer_fold": fold_index,
        "page": record.page,
        "page_index_zero_based": record.page_index,
        "quire": record.quire,
        "currier": record.currier,
        "topic": record.topic,
        "physical_line_index_zero_based": record.line_index,
        "word_index_zero_based": record.word_index,
        "line_length": record.line_length,
        "normalized_word_midpoint": (
            record.word_index + 0.5
        ) / record.line_length,
        "line_position": record.position,
        "paragraph_state": record.paragraph_state,
        "paragraph_start": record.paragraph_state.startswith("0:"),
        "target_word": record.word,
        "target_identity_bucket": encoded.target,
        "target_morphology": record.morphology,
        "morphology_depth": morphology_depth,
        "grouped_eva": suite.grouped_eva,
        "training_vocabulary_size": len(suite.vocabulary),
        "target_initial_unit": target_units[0] if target_units else None,
        "target_final_unit": target_units[-1] if target_units else None,
        "previous_word": record.previous_word,
        "previous_morphology": record.previous_morphology,
        "previous_final_unit": previous_units[-1] if previous_units else None,
        "aligned_previous_line_word": record.previous_line_word,
        "aligned_previous_line_morphology": record.previous_line_morphology,
        "aligned_previous_line_index_zero_based": record.previous_line_index,
        "previous_line_words": record.previous_line_words,
        "previous_line_morphologies": record.previous_line_morphologies,
        "expert_log_loss_bits": expert_losses,
        "expert_gain_over_base_bits": {
            name: base_loss - loss for name, loss in expert_losses.items()
        },
        "family_log_loss_bits": family_losses,
        "family_gain_over_base_bits": {
            family: base_loss - loss for family, loss in family_losses.items()
        },
        "full_mixture_weights": full_weights,
        "full_mixture_responsibility": responsibilities,
        "dominant_full_expert": dominant_expert,
        "matched_null": matched_null,
    }


def audit_corpus(
    name: str,
    pages: Sequence[StructuredPage],
    *,
    outer_folds: int,
    inner_folds: int,
    morphology_depth: int,
    grouped_eva: bool,
    vocabulary_limit: int,
    alpha: float,
    strength: float,
    permutations: int,
    seed: int,
    token_trace: TextIO | None = None,
) -> dict[str, object]:
    records = records_from_structured(
        pages,
        morphology_depth=morphology_depth,
        grouped_eva=grouped_eva,
    )
    outer_quires = balanced_quire_folds(records, outer_folds)
    totals = {family: 0.0 for family in ExpertSuite.family_names}
    permutation_totals = {
        family: [0.0] * permutations for family in ExpertSuite.family_names
    }
    changed_totals = [0] * permutations
    fold_results: list[dict[str, object]] = []
    fold_full_weights: list[dict[str, float]] = []
    fold_family_weights: dict[str, list[dict[str, float]]] = {
        family: [] for family in ExpertSuite.family_names
    }
    total_words = 0

    for fold_index, heldout_quires in enumerate(outer_quires):
        training = [record for record in records if record.quire not in heldout_quires]
        heldout = [record for record in records if record.quire in heldout_quires]
        weights, inner_bits = fit_nested_weights(
            training,
            inner_folds=inner_folds,
            vocabulary_limit=vocabulary_limit,
            alpha=alpha,
            strength=strength,
            grouped_eva=grouped_eva,
        )
        suite = ExpertSuite(
            training,
            vocabulary_limit=vocabulary_limit,
            alpha=alpha,
            strength=strength,
            grouped_eva=grouped_eva,
        )
        actual_rows = [suite.probability_row(record) for record in heldout]
        trace_expert_sums = (
            [[0.0] * len(ExpertSuite.expert_names) for _ in heldout]
            if token_trace is not None and permutations
            else None
        )
        trace_family_names = tuple(ExpertSuite.family_names)
        trace_family_sums = (
            [[0.0] * len(trace_family_names) for _ in heldout]
            if token_trace is not None and permutations
            else None
        )
        trace_changed_counts = (
            [0] * len(heldout) if token_trace is not None else None
        )
        fold_bits: dict[str, float] = {}
        for family, family_weights in weights.items():
            bits = score_rows(actual_rows, family_weights)
            fold_bits[family] = bits
            totals[family] += bits
            fold_family_weights[family].append(family_weights)
        fold_full_weights.append(weights["full"])
        total_words += len(heldout)

        fold_permutation_bits = {
            family: [] for family in ExpertSuite.family_names
        }
        fold_changed: list[int] = []
        for permutation_index in range(permutations):
            permuted, changed = matched_source_permutation(
                heldout,
                seed=seed + fold_index * 1_000_003 + permutation_index,
            )
            permuted_rows = [suite.probability_row(record) for record in permuted]
            for family, family_weights in weights.items():
                bits = score_rows(permuted_rows, family_weights)
                permutation_totals[family][permutation_index] += bits
                fold_permutation_bits[family].append(bits)
            if token_trace is not None:
                if (
                    trace_expert_sums is None
                    or trace_family_sums is None
                    or trace_changed_counts is None
                ):
                    raise ValueError("Token trace accumulators were not initialized")
                for record_index, (
                    actual_record,
                    permuted_record,
                    permuted_row,
                ) in enumerate(zip(heldout, permuted, permuted_rows, strict=True)):
                    if source_link_signature(actual_record) != source_link_signature(
                        permuted_record
                    ):
                        trace_changed_counts[record_index] += 1
                    for expert_index, expert_name in enumerate(
                        ExpertSuite.expert_names
                    ):
                        trace_expert_sums[record_index][expert_index] -= math.log2(
                            permuted_row[expert_name]
                        )
                    for family_index, family in enumerate(trace_family_names):
                        trace_family_sums[record_index][family_index] -= math.log2(
                            mixture_probability(permuted_row, weights[family])
                        )
            changed_totals[permutation_index] += changed
            fold_changed.append(changed)

        if token_trace is not None:
            if trace_changed_counts is None:
                raise ValueError("Token trace change counts were not initialized")
            for record_index, (record, actual_row) in enumerate(
                zip(heldout, actual_rows, strict=True)
            ):
                payload = token_trace_payload(
                    corpus_name=name,
                    fold_index=fold_index,
                    record=record,
                    suite=suite,
                    actual_row=actual_row,
                    weights=weights,
                    morphology_depth=morphology_depth,
                    permutations=permutations,
                    changed_count=trace_changed_counts[record_index],
                    permuted_expert_loss_sums=(
                        trace_expert_sums[record_index]
                        if trace_expert_sums is not None
                        else None
                    ),
                    permuted_family_loss_sums=(
                        trace_family_sums[record_index]
                        if trace_family_sums is not None
                        else None
                    ),
                )
                token_trace.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )

        fold_results.append(
            {
                "fold": fold_index,
                "heldout_quires": sorted(heldout_quires),
                "training_words": len(training),
                "heldout_words": len(heldout),
                "inner_bits_per_word": inner_bits,
                "outer_bits_per_word": {
                    family: bits / len(heldout) for family, bits in fold_bits.items()
                },
                "weights": weights,
                "permuted_bits_per_word": {
                    family: statistics.fmean(values) / len(heldout)
                    for family, values in fold_permutation_bits.items()
                }
                if permutations
                else {},
                "mean_permutation_changed_fraction": (
                    statistics.fmean(fold_changed) / len(heldout)
                    if fold_changed
                    else 0.0
                ),
            }
        )

    bits_per_word = {
        family: bits / total_words for family, bits in totals.items()
    }
    gains = {
        family: bits_per_word["base"] - bits
        for family, bits in bits_per_word.items()
    }
    permutation_summary: dict[str, object] = {
        "iterations": permutations,
        "mean_changed_fraction": (
            statistics.fmean(changed_totals) / total_words if permutations else 0.0
        ),
        "families": {},
    }
    if permutations:
        for family in ExpertSuite.family_names:
            actual = totals[family]
            raw = permutation_totals[family]
            values = [bits / total_words for bits in raw]
            permutation_summary["families"][family] = {
                "actual_gain_over_permutation_bits_per_word": (
                    statistics.fmean(values) - bits_per_word[family]
                ),
                "permuted_bits_per_word_mean": statistics.fmean(values),
                "permuted_bits_per_word_2_5pct": quantile(values, 0.025),
                "permuted_bits_per_word_97_5pct": quantile(values, 0.975),
                "one_sided_p": (1 + sum(bits <= actual for bits in raw))
                / (permutations + 1),
            }
    return {
        "name": name,
        "words": len(records),
        "heldout_scored_words": total_words,
        "bits_per_word": bits_per_word,
        "gain_over_base_bits_per_word": gains,
        "full_weight_summary": summarize_weights(fold_full_weights),
        "family_weight_summaries": {
            family: summarize_weights(values)
            for family, values in fold_family_weights.items()
        },
        "matched_permutation": permutation_summary,
        "folds": fold_results,
    }


def audit(
    source: Path,
    controls: Sequence[tuple[str, Path, bool]],
    *,
    include_voynich: bool,
    outer_folds: int,
    inner_folds: int,
    morphology_depth: int,
    vocabulary_limit: int,
    alpha: float,
    strength: float,
    permutations: int,
    seed: int,
    token_trace: TextIO | None = None,
) -> dict[str, object]:
    source_pages = [page for page in parse_ivtff(source) if page.lines("P", True)]
    template = pages_to_structured(source_pages)
    corpora = []
    if include_voynich:
        corpora.append(
            audit_corpus(
                "Voynich",
                template,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                morphology_depth=morphology_depth,
                grouped_eva=True,
                vocabulary_limit=vocabulary_limit,
                alpha=alpha,
                strength=strength,
                permutations=permutations,
                seed=seed,
                token_trace=token_trace,
            )
        )
    control_metadata: list[dict[str, object]] = []
    for control_name, path, shuffle in controls:
        words = control_words(path)
        if shuffle:
            random.Random(
                derived_seed(seed, "decomposition-shuffle", control_name, str(path))
            ).shuffle(words)
        pages = reflow_structured(template, words)
        corpora.append(
            audit_corpus(
                control_name,
                pages,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                morphology_depth=morphology_depth,
                grouped_eva=False,
                vocabulary_limit=vocabulary_limit,
                alpha=alpha,
                strength=strength,
                permutations=permutations,
                seed=derived_seed(seed, "decomposition", control_name, str(path)),
                token_trace=token_trace,
            )
        )
        control_metadata.append(
            {
                "name": control_name,
                "path": str(path),
                "sha256": sha256(path),
                "available_words": len(words),
                "word_order_shuffled_before_reflow": shuffle,
            }
        )
    return {
        "scope": (
            "Nested held-out decomposition of exact surface-token prediction; "
            "not a universal or semantic information bound."
        ),
        "source": str(source),
        "source_sha256": sha256(source),
        "configuration": {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "heldout_unit": "whole IVTFF quire",
            "morphology_depth": morphology_depth,
            "vocabulary_limit_including_unknown": vocabulary_limit,
            "alpha": alpha,
            "hierarchical_strength": strength,
            "permutations": permutations,
            "seed": seed,
            "edit_parameters": [parameters.__dict__ for parameters in EDIT_PARAMETERS],
            "latent_class_counts": CLASS_COUNTS,
            "weight_selection": "mixture EM on inner-quire out-of-fold predictions",
            "mixture_em_max_iterations": MIXTURE_ITERATIONS,
        },
        "controls": control_metadata,
        "corpora": corpora,
        "interpretation_guardrails": [
            "All component parameters and mixture weights are fitted without outer-quire data.",
            "Target morphology and declared layout are side information, not predicted payload.",
            "Latent classes are distributional clusters, not asserted linguistic categories.",
            "Positive held-out gain is model-family-specific recoverable information.",
            "Matched permutations preserve declared source morphology but not unknown production state.",
            "Per-token mixture responsibilities route probability among correlated experts; they are not information shares.",
        ],
    }


def print_summary(result: dict[str, object]) -> None:
    print("Nested residual-link decomposition")
    for corpus in result["corpora"]:
        gains = corpus["gain_over_base_bits_per_word"]
        permutation = corpus["matched_permutation"]
        full_permutation = permutation["families"].get("full", {})
        parts = ", ".join(
            f"{family}={gains[family]:+.4f}"
            for family in (
                "paragraph",
                "boundary",
                "copy_edit",
                "latent_class",
                "previous_line_pool",
                "exact_identity",
                "full",
            )
        )
        print(f"- {corpus['name']}: {parts}")
        if full_permutation:
            print(
                "  full actual-vs-permuted "
                f"{full_permutation['actual_gain_over_permutation_bits_per_word']:+.4f}; "
                f"p={full_permutation['one_sided_p']:.4f}; "
                f"changed={permutation['mean_changed_fraction']:.1%}"
            )
        weights = corpus["full_weight_summary"]
        selected = sorted(
            (
                (name, summary["mean"])
                for name, summary in weights.items()
                if summary["mean"] >= 0.01
            ),
            key=lambda item: (-item[1], item[0]),
        )
        print(
            "  full mean weights: "
            + ", ".join(f"{name}={weight:.3f}" for name, weight in selected)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument(
        "--control",
        dest="controls",
        action="append",
        default=[],
        type=parse_control,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--shuffled-control",
        dest="controls",
        action="append",
        type=parse_shuffled_control,
        metavar="NAME=PATH",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--morphology-depth", type=int, default=0)
    parser.add_argument("--vocabulary", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=20.0)
    parser.add_argument("--permutations", type=int, default=49)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--token-output",
        type=Path,
        help=(
            "Write one JSONL row per outer-held-out token with expert losses, "
            "mixture responsibilities, and mean matched-null residuals; a .gz "
            "suffix enables gzip compression"
        ),
    )
    parser.add_argument("--skip-voynich", action="store_true")
    args = parser.parse_args()
    if args.output is not None and args.output == args.token_output:
        parser.error("--output and --token-output must be different paths")
    token_trace = (
        open_token_trace(args.token_output)
        if args.token_output is not None
        else None
    )
    try:
        result = audit(
            args.source,
            args.controls,
            include_voynich=not args.skip_voynich,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            morphology_depth=args.morphology_depth,
            vocabulary_limit=args.vocabulary,
            alpha=args.alpha,
            strength=args.strength,
            permutations=args.permutations,
            seed=args.seed,
            token_trace=token_trace,
        )
    finally:
        if token_trace is not None:
            token_trace.close()
    if args.token_output is not None:
        result["token_trace"] = {
            "path": str(args.token_output),
            "sha256": sha256(args.token_output),
            "format": (
                "gzip-compressed newline-delimited JSON"
                if args.token_output.suffix == ".gz"
                else "newline-delimited JSON"
            ),
            "schema": "voynich-residual-token-trace-v1",
            "rows": sum(corpus["heldout_scored_words"] for corpus in result["corpora"]),
            "matched_residual_definition": (
                "mean permuted log loss minus actual log loss; positive favors "
                "the actual source link"
            ),
            "output_order": (
                "outer fold then source-corpus order; sort page_index_zero_based, "
                "physical_line_index_zero_based, word_index_zero_based for "
                "manuscript order"
            ),
        }
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print_summary(result)
        print(f"Wrote {args.output}")
    elif args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_summary(result)
    if args.token_output is not None and not (
        args.json and args.output is None
    ):
        print(f"Wrote {args.token_output}")


if __name__ == "__main__":
    main()
