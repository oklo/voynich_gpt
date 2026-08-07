#!/usr/bin/env python3
"""Measure lexical sequence information after conditioning on form and layout.

The experiment asks a deliberately narrow question: once the decoder is given
the current token's coarse morphology, Currier stratum, illustration/topic
class, and physical line position, do exact identities of causally preceding
tokens still help identify the current token on held-out quires?

The target vocabulary is the most frequent surface types plus an unknown-type
bucket.  A hierarchical Bayesian categorical code predicts target identity in
five stages:

1. morphology alone;
2. morphology plus Currier/topic/line position;
3. those variables plus the morphology of the preceding same-line token and
   the vertically aligned token on the previous physical line;
4. the exact preceding same-line identity;
5. the exact previous-line identity, or an equal mixture of both identity
   experts.

Exact neighbor identities are then permuted independently in held-out data
within strata that preserve Currier, topic, line position, and the relevant
neighbor morphology class.  Target identities and all conditioning variables remain fixed.  The
actual-versus-permuted code-length difference is therefore a held-out estimate
of recoverable exact-identity sequence information beyond the declared
procedural variables.  It is not a universal upper bound on semantic content.

Optional natural-language controls are reflowed onto the exact Voynich
page/line template and inherit its metadata.  They test whether the estimator
can recover ordinary lexical sequence under the same sample size and layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from audit_voynich import Page, eva_units, parse_ivtff
from compare_mechanisms import line_position, make_quire_folds


UNKNOWN = "<UNKNOWN>"
Morphology = tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return seed + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class PageSequence:
    name: str
    quire: str
    currier: str
    topic: str
    lines: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SequenceRecord:
    word: str
    morphology: Morphology
    currier: str
    topic: str
    position: str
    page: str
    quire: str
    previous_word: str | None
    previous_morphology: Morphology | None
    previous_line_word: str | None
    previous_line_morphology: Morphology | None


@dataclass(frozen=True)
class EncodedRecord:
    target: str
    morphology: Morphology
    currier: str
    topic: str
    position: str
    page: str
    quire: str
    previous_identity: str | None
    previous_morphology: Morphology | None
    previous_line_identity: str | None
    previous_line_morphology: Morphology | None


def pages_to_sequences(pages: Sequence[Page]) -> list[PageSequence]:
    return [
        PageSequence(
            name=page.name,
            quire=page.metadata.get("Q", "?"),
            currier=page.metadata.get("L", "?"),
            topic=page.metadata.get("I", "?"),
            lines=tuple(tuple(line) for line in page.lines("P", certain=True)),
        )
        for page in pages
        if page.lines("P", certain=True)
    ]


def length_bucket(length: int) -> str:
    return str(length) if length < 8 else "8+"


def morphology_signature(
    word: str, *, depth: int, grouped_eva: bool
) -> Morphology:
    """Return a declared shape class supplied as side information to the code."""

    if depth < 0:
        raise ValueError("Morphology depth cannot be negative")
    units = tuple(eva_units(word, grouped=True) if grouped_eva else tuple(word))
    prefix = units[:depth]
    suffix = units[-depth:] if depth else ()
    return (length_bucket(len(units)), *prefix, "|", *suffix)


def aligned_word(
    previous_line: Sequence[str] | None,
    *,
    index: int,
    line_length: int,
) -> str | None:
    """Select the previous-line token with the nearest normalized midpoint."""

    if not previous_line:
        return None
    target_midpoint = (index + 0.5) / line_length
    aligned_index = min(
        range(len(previous_line)),
        key=lambda candidate: (
            abs((candidate + 0.5) / len(previous_line) - target_midpoint),
            candidate,
        ),
    )
    return previous_line[aligned_index]


def records_from_sequences(
    pages: Sequence[PageSequence], *, morphology_depth: int, grouped_eva: bool
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for page in pages:
        previous_line: tuple[str, ...] | None = None
        for line in page.lines:
            for index, word in enumerate(line):
                previous_word = line[index - 1] if index else None
                previous_line_word = aligned_word(
                    previous_line, index=index, line_length=len(line)
                )
                records.append(
                    SequenceRecord(
                        word=word,
                        morphology=morphology_signature(
                            word, depth=morphology_depth, grouped_eva=grouped_eva
                        ),
                        currier=page.currier,
                        topic=page.topic,
                        position=line_position(index, len(line)),
                        page=page.name,
                        quire=page.quire,
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
                    )
                )
            previous_line = line
    return records


def control_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").lower()
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text, flags=re.UNICODE)


def reflow_control(
    template: Sequence[PageSequence], words: Sequence[str]
) -> list[PageSequence]:
    needed = sum(len(line) for page in template for line in page.lines)
    if len(words) < needed:
        raise ValueError(f"Control has {len(words)} words; {needed} are required")
    result: list[PageSequence] = []
    offset = 0
    for page in template:
        lines: list[tuple[str, ...]] = []
        for line in page.lines:
            length = len(line)
            lines.append(tuple(words[offset : offset + length]))
            offset += length
        result.append(replace(page, lines=tuple(lines)))
    return result


class ResidualIdentityModel:
    """Hierarchical proper-probability code for exact token identity."""

    model_names = (
        "morphology",
        "layout",
        "procedural_morphology",
        "previous_word_identity",
        "previous_line_identity",
        "neighbor_identity_mixture",
    )

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        *,
        vocabulary_limit: int,
        alpha: float,
        strength: float,
    ):
        if vocabulary_limit < 2:
            raise ValueError("Vocabulary limit must be at least two")
        if alpha <= 0 or strength <= 0:
            raise ValueError("Smoothing parameters must be positive")
        self.alpha = alpha
        self.strength = strength
        self.vocabulary = {
            word
            for word, _ in Counter(record.word for record in records).most_common(
                vocabulary_limit - 1
            )
        }
        encoded = [self.encode(record) for record in records]
        self.support: dict[Morphology, set[str]] = defaultdict(lambda: {UNKNOWN})
        self.morphology_counts: dict[Morphology, Counter[str]] = defaultdict(Counter)
        self.layout_counts: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
        self.procedural_counts: dict[
            tuple[object, ...], Counter[str]
        ] = defaultdict(Counter)
        self.previous_counts: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
        self.previous_line_counts: dict[
            tuple[object, ...], Counter[str]
        ] = defaultdict(Counter)
        for record in encoded:
            self.support[record.morphology].add(record.target)
            self.morphology_counts[record.morphology][record.target] += 1
            self.layout_counts[self.layout_key(record)][record.target] += 1
            self.procedural_counts[self.procedural_key(record)][record.target] += 1
            if record.previous_identity is not None:
                self.previous_counts[self.previous_key(record)][record.target] += 1
            if record.previous_line_identity is not None:
                self.previous_line_counts[self.previous_line_key(record)][record.target] += 1

    def identity(self, word: str | None) -> str | None:
        if word is None:
            return None
        return word if word in self.vocabulary else UNKNOWN

    def encode(self, record: SequenceRecord) -> EncodedRecord:
        return EncodedRecord(
            target=self.identity(record.word) or UNKNOWN,
            morphology=record.morphology,
            currier=record.currier,
            topic=record.topic,
            position=record.position,
            page=record.page,
            quire=record.quire,
            previous_identity=self.identity(record.previous_word),
            previous_morphology=record.previous_morphology,
            previous_line_identity=self.identity(record.previous_line_word),
            previous_line_morphology=record.previous_line_morphology,
        )

    @staticmethod
    def layout_key(record: EncodedRecord) -> tuple[object, ...]:
        return (
            record.morphology,
            record.currier,
            record.topic,
            record.position,
        )

    @classmethod
    def procedural_key(cls, record: EncodedRecord) -> tuple[object, ...]:
        return (
            *cls.layout_key(record),
            record.previous_morphology,
            record.previous_line_morphology,
        )

    @staticmethod
    def previous_key(record: EncodedRecord) -> tuple[object, ...]:
        return (
            record.morphology,
            record.previous_morphology,
            record.previous_identity,
        )

    @staticmethod
    def previous_line_key(record: EncodedRecord) -> tuple[object, ...]:
        return (
            record.morphology,
            record.previous_line_morphology,
            record.previous_line_identity,
        )

    def _posterior(
        self, counts: Counter[str], target: str, prior: float
    ) -> float:
        return (counts[target] + self.strength * prior) / (
            sum(counts.values()) + self.strength
        )

    def probabilities(self, record: EncodedRecord) -> dict[str, float]:
        support = self.support.get(record.morphology, {UNKNOWN})
        target = record.target if record.target in support else UNKNOWN
        morphology_counts = self.morphology_counts[record.morphology]
        morphology_probability = (morphology_counts[target] + self.alpha) / (
            sum(morphology_counts.values()) + self.alpha * len(support)
        )
        layout_probability = self._posterior(
            self.layout_counts[self.layout_key(record)],
            target,
            morphology_probability,
        )
        procedural_probability = self._posterior(
            self.procedural_counts[self.procedural_key(record)],
            target,
            layout_probability,
        )
        if record.previous_identity is None:
            previous_probability = procedural_probability
        else:
            previous_probability = self._posterior(
                self.previous_counts[self.previous_key(record)],
                target,
                procedural_probability,
            )
        if record.previous_line_identity is None:
            previous_line_probability = procedural_probability
        else:
            previous_line_probability = self._posterior(
                self.previous_line_counts[self.previous_line_key(record)],
                target,
                procedural_probability,
            )
        return {
            "morphology": morphology_probability,
            "layout": layout_probability,
            "procedural_morphology": procedural_probability,
            "previous_word_identity": previous_probability,
            "previous_line_identity": previous_line_probability,
            "neighbor_identity_mixture": (
                previous_probability + previous_line_probability
            )
            / 2,
        }

    def score(self, records: Iterable[EncodedRecord]) -> dict[str, float]:
        bits = {name: 0.0 for name in self.model_names}
        for record in records:
            for name, probability in self.probabilities(record).items():
                bits[name] -= math.log2(probability)
        return bits

    def probability_mass(
        self, record: EncodedRecord, model_name: str
    ) -> float:
        """Sum a model's probabilities over its morphology-specific support."""

        total = 0.0
        for target in self.support.get(record.morphology, {UNKNOWN}):
            total += self.probabilities(replace(record, target=target))[model_name]
        return total


def matched_neighbor_permutation(
    records: Sequence[EncodedRecord], *, seed: int
) -> tuple[list[EncodedRecord], int]:
    """Permute neighbor identities while retaining their own morphology strata."""

    rng = random.Random(seed)
    result = list(records)
    changed_indices: set[int] = set()
    for identity_field, morphology_field in (
        ("previous_identity", "previous_morphology"),
        ("previous_line_identity", "previous_line_morphology"),
    ):
        groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            key = (
                record.currier,
                record.topic,
                record.position,
                getattr(record, morphology_field),
            )
            groups[key].append(index)
        for indices in groups.values():
            values = [getattr(records[index], identity_field) for index in indices]
            rng.shuffle(values)
            for index, identity in zip(indices, values, strict=True):
                if identity != getattr(records[index], identity_field):
                    changed_indices.add(index)
                result[index] = replace(result[index], **{identity_field: identity})
    return result, len(changed_indices)


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def audit_corpus(
    name: str,
    pages: Sequence[PageSequence],
    fold_quires: Sequence[set[str]],
    *,
    morphology_depth: int,
    grouped_eva: bool,
    vocabulary_limit: int,
    alpha: float,
    strength: float,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    records = records_from_sequences(
        pages, morphology_depth=morphology_depth, grouped_eva=grouped_eva
    )
    totals = {model: 0.0 for model in ResidualIdentityModel.model_names}
    permuted_models = (
        "previous_word_identity",
        "previous_line_identity",
        "neighbor_identity_mixture",
    )
    permutation_totals = {
        model_name: [0.0] * permutations for model_name in permuted_models
    }
    changed_totals = [0] * permutations
    heldout_total = 0
    vocabulary_cover_tokens = 0
    fold_results: list[dict[str, object]] = []

    for fold_index, heldout_quires in enumerate(fold_quires):
        training = [record for record in records if record.quire not in heldout_quires]
        heldout = [record for record in records if record.quire in heldout_quires]
        model = ResidualIdentityModel(
            training,
            vocabulary_limit=vocabulary_limit,
            alpha=alpha,
            strength=strength,
        )
        encoded = [model.encode(record) for record in heldout]
        scores = model.score(encoded)
        for model_name, bits in scores.items():
            totals[model_name] += bits
        heldout_total += len(encoded)
        vocabulary_cover_tokens += sum(record.word in model.vocabulary for record in heldout)

        fold_permutation_bits = {
            model_name: [] for model_name in permuted_models
        }
        fold_changed: list[int] = []
        for permutation_index in range(permutations):
            permuted, changed = matched_neighbor_permutation(
                encoded,
                seed=seed + fold_index * 1_000_003 + permutation_index,
            )
            permuted_scores = model.score(permuted)
            for model_name in permuted_models:
                bits = permuted_scores[model_name]
                permutation_totals[model_name][permutation_index] += bits
                fold_permutation_bits[model_name].append(bits)
            changed_totals[permutation_index] += changed
            fold_changed.append(changed)
        fold_results.append(
            {
                "fold": fold_index,
                "heldout_quires": sorted(heldout_quires),
                "training_words": len(training),
                "heldout_words": len(heldout),
                "training_vocabulary": len(model.vocabulary),
                "bits_per_word": {
                    model_name: bits / len(heldout)
                    for model_name, bits in scores.items()
                },
                "permuted_bits_per_word": {
                    model_name: {
                        "mean": statistics.fmean(values) / len(heldout),
                        "minimum": min(values) / len(heldout),
                        "maximum": max(values) / len(heldout),
                    }
                    for model_name, values in fold_permutation_bits.items()
                },
                "mean_permutation_changed_fraction": statistics.fmean(fold_changed)
                / len(heldout),
            }
        )

    bits_per_word = {
        model_name: bits / heldout_total for model_name, bits in totals.items()
    }
    permutation_summaries: dict[str, dict[str, float | int]] = {}
    for model_name in permuted_models:
        actual = totals[model_name]
        raw_permutation_bits = permutation_totals[model_name]
        values = [bits / heldout_total for bits in raw_permutation_bits]
        null_better_or_equal = sum(bits <= actual for bits in raw_permutation_bits)
        permutation_summaries[model_name] = {
            "iterations": permutations,
            "actual_bits_per_word": bits_per_word[model_name],
            "bits_per_word_mean": statistics.fmean(values),
            "bits_per_word_2_5pct": quantile(values, 0.025),
            "bits_per_word_97_5pct": quantile(values, 0.975),
            "actual_gain_over_permutation_bits_per_word": (
                statistics.fmean(values) - bits_per_word[model_name]
            ),
            "one_sided_p": (null_better_or_equal + 1) / (permutations + 1),
        }
    return {
        "name": name,
        "words": len(records),
        "heldout_scored_words": heldout_total,
        "vocabulary_coverage": vocabulary_cover_tokens / heldout_total,
        "morphology_classes": len({record.morphology for record in records}),
        "bits_per_word": bits_per_word,
        "gains_bits_per_word": {
            "layout_over_morphology": (
                bits_per_word["morphology"] - bits_per_word["layout"]
            ),
            "neighbor_shapes_over_layout": (
                bits_per_word["layout"]
                - bits_per_word["procedural_morphology"]
            ),
            "exact_neighbors_over_procedural": (
                bits_per_word["procedural_morphology"]
                - bits_per_word["neighbor_identity_mixture"]
            ),
            "previous_word_identity_over_procedural": (
                bits_per_word["procedural_morphology"]
                - bits_per_word["previous_word_identity"]
            ),
            "previous_line_identity_over_procedural": (
                bits_per_word["procedural_morphology"]
                - bits_per_word["previous_line_identity"]
            ),
            "actual_exact_neighbors_over_matched_permutation": (
                permutation_summaries["neighbor_identity_mixture"][
                    "actual_gain_over_permutation_bits_per_word"
                ]
            ),
        },
        "matched_permutation": {
            "iterations": permutations,
            "mean_changed_fraction": statistics.fmean(changed_totals) / heldout_total,
            "models": permutation_summaries,
        },
        "folds": fold_results,
    }


def parse_control(value: str, *, shuffle: bool = False) -> tuple[str, Path, bool]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Controls must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Controls must use nonempty NAME=PATH")
    return name, Path(path), shuffle


def parse_shuffled_control(value: str) -> tuple[str, Path, bool]:
    return parse_control(value, shuffle=True)


def audit(
    source: Path,
    controls: Sequence[tuple[str, Path, bool]],
    *,
    folds: int,
    morphology_depth: int,
    vocabulary_limit: int,
    alpha: float,
    strength: float,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    source_pages = [page for page in parse_ivtff(source) if page.lines("P", True)]
    template = pages_to_sequences(source_pages)
    fold_quires = make_quire_folds(source_pages, folds)
    corpus_results = [
        audit_corpus(
            "Voynich",
            template,
            fold_quires,
            morphology_depth=morphology_depth,
            grouped_eva=True,
            vocabulary_limit=vocabulary_limit,
            alpha=alpha,
            strength=strength,
            permutations=permutations,
            seed=seed,
        )
    ]
    control_metadata: list[dict[str, str]] = []
    for name, path, shuffle in controls:
        words = control_words(path)
        if shuffle:
            random.Random(derived_seed(seed, "shuffle", name, str(path))).shuffle(words)
        pages = reflow_control(template, words)
        corpus_results.append(
            audit_corpus(
                name,
                pages,
                fold_quires,
                morphology_depth=morphology_depth,
                grouped_eva=False,
                vocabulary_limit=vocabulary_limit,
                alpha=alpha,
                strength=strength,
                permutations=permutations,
                seed=derived_seed(seed, "corpus", name, str(path)),
            )
        )
        control_metadata.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256(path),
                "available_words": str(len(words)),
                "word_order_shuffled_before_reflow": str(shuffle),
            }
        )

    return {
        "scope": (
            "Held-out recoverable exact-token identity information after declared "
            "morphology/layout conditioning; not an upper bound on semantic content."
        ),
        "source": str(source),
        "source_sha256": sha256(source),
        "configuration": {
            "folds": folds,
            "heldout_unit": "whole IVTFF quire",
            "morphology": (
                "grouped-EVA length (capped at 8+) plus the first and last "
                f"{morphology_depth} units; controls use Unicode characters"
            ),
            "morphology_depth": morphology_depth,
            "vocabulary_limit_including_unknown": vocabulary_limit,
            "alpha": alpha,
            "hierarchical_strength": strength,
            "permutations": permutations,
            "seed": seed,
            "previous_line_link": "nearest normalized token midpoint on prior physical line",
            "permutation_match": (
                "Currier, illustration/topic, target line position, previous-word "
                "morphology, and previous-line-word morphology"
            ),
        },
        "controls": control_metadata,
        "corpora": corpus_results,
        "interpretation_guardrails": [
            "Target morphology is supplied to the decoder and is not counted as "
            "residual identity information.",
            "Only the declared frequent surface vocabulary plus an unknown bucket is scored.",
            "Positive code-length gain is recoverable information for this model "
            "family, not total mutual information.",
            "The matched shuffle preserves declared nuisance variables but cannot "
            "preserve unknown ones.",
            "Reflowed controls test estimator power, not direct historical or genre equivalence.",
        ],
    }


def print_summary(result: dict[str, object]) -> None:
    print("Residual exact-token sequence information")
    print(
        "Conditioning: morphology + Currier + topic + line position; "
        "matched exact-neighbor permutation"
    )
    for corpus in result["corpora"]:
        gains = corpus["gains_bits_per_word"]
        permutation = corpus["matched_permutation"]
        mixture_permutation = permutation["models"]["neighbor_identity_mixture"]
        previous_permutation = permutation["models"]["previous_word_identity"]
        line_permutation = permutation["models"]["previous_line_identity"]
        bits = corpus["bits_per_word"]
        print(
            f"- {corpus['name']}: residual code {bits['procedural_morphology']:.4f} "
            f"-> {bits['neighbor_identity_mixture']:.4f} bits/word; "
            f"net exact gain {gains['exact_neighbors_over_procedural']:+.4f}; "
            f"net within/line gains "
            f"{gains['previous_word_identity_over_procedural']:+.4f}/"
            f"{gains['previous_line_identity_over_procedural']:+.4f}; "
            f"actual-vs-permuted {gains['actual_exact_neighbors_over_matched_permutation']:+.4f}; "
            f"within/line perm gains "
            f"{previous_permutation['actual_gain_over_permutation_bits_per_word']:+.4f}/"
            f"{line_permutation['actual_gain_over_permutation_bits_per_word']:+.4f}; "
            f"p={mixture_permutation['one_sided_p']:.4f}; "
            f"shuffle changed {permutation['mean_changed_fraction']:.1%}"
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
        help="Shuffle parsed control words deterministically before reflowing them",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--morphology-depth", type=int, default=1)
    parser.add_argument("--vocabulary", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=20.0)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.source,
        args.controls,
        folds=args.folds,
        morphology_depth=args.morphology_depth,
        vocabulary_limit=args.vocabulary,
        alpha=args.alpha,
        strength=args.strength,
        permutations=args.permutations,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
