#!/usr/bin/env python3
"""Test a deliberately narrow Hebrew crib against the Voynich zodiac pages.

This is not a decipherment search.  The central zodiac emblems are known, but
the Voynich-script items on the pages label the surrounding figures/stars; the
readable month names beside the emblems were added later in a Romance language.
The test therefore asks only a weaker page-level question:

    Can one global, exact substitution key recover the page's Hebrew zodiac
    sign or its corresponding Hebrew month somewhere among the Lz labels?

Candidate locations are fixed by the IVTFF ``Lz`` locus type, not selected after
looking at Hebrew words.  Candidate vocabulary is fixed from the conventional
medieval Hebrew zodiac/month sequence.  Model selection is evaluated both by a
two-fold held-out split (alternating signs in zodiac order) and by shuffling the
ten semantic targets among the ten diagram blocks.  Duplicate Aries and Taurus
diagrams remain together as blocks during permutations.

The ordered model is intentionally generous: it ignores Voynich word spaces,
allows either reading direction independently for every label, optionally
groups common EVA sequences, and tests both bijective substitution and a
homophonic relaxation in which several cipher units may represent one Hebrew
letter.  Matches are exact; there are no edit-distance or cherry-picked-token
allowances.

For the within-word-anagram family, short labels admit factorially many local
keys.  We report a necessary-condition/local-key saturation test against the
same shuffled assignments rather than pretending that a locally compatible
anagram supplies a global decipherment key.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from audit_voynich import eva_units, parse_ivtff


FINAL_TO_MEDIAL = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})


@dataclass(frozen=True)
class ZodiacTarget:
    key: str
    english_sign: str
    sign_names: tuple[str, ...]
    month_names: tuple[str, ...]
    pages: tuple[str, ...]


# The order is astronomical, beginning with the first surviving diagram.  The
# spelling variants cover forms visible in medieval sources (e.g. ארי/אריה and
# מאזנים) and common plene/defective spellings.  They are declared here rather
# than generated from the observed Voynich labels.
ZODIAC_TARGETS: tuple[ZodiacTarget, ...] = (
    ZodiacTarget("pisces", "Pisces", ("דגים",), ("אדר",), ("f70v2",)),
    ZodiacTarget("aries", "Aries", ("טלה",), ("ניסן",), ("f70v1", "f71r")),
    ZodiacTarget("taurus", "Taurus", ("שור",), ("אייר", "איר"), ("f71v", "f72r1")),
    ZodiacTarget("gemini", "Gemini", ("תאומים",), ("סיון", "סיוון"), ("f72r2",)),
    ZodiacTarget("cancer", "Cancer", ("סרטן",), ("תמוז",), ("f72r3",)),
    ZodiacTarget("leo", "Leo", ("ארי", "אריה"), ("אב",), ("f72v3",)),
    ZodiacTarget("virgo", "Virgo", ("בתולה",), ("אלול",), ("f72v2",)),
    ZodiacTarget(
        "libra", "Libra", ("מאזנים", "מאזניים"), ("תשרי",), ("f72v1",)
    ),
    ZodiacTarget(
        "scorpio",
        "Scorpio",
        ("עקרב",),
        ("חשון", "חשוון", "מרחשון", "מרחשוון"),
        ("f73r",),
    ),
    ZodiacTarget("sagittarius", "Sagittarius", ("קשת",), ("כסלו",), ("f73v",)),
)


@dataclass(frozen=True)
class Label:
    page: str
    locus_id: str
    eva: str


@dataclass(frozen=True)
class Constraint:
    mapping: tuple[tuple[str, str], ...]
    locus_id: str
    eva: str
    hebrew: str
    reversed_eva: bool


@dataclass(frozen=True)
class FitResult:
    matched_pages: int
    total_pages: int
    optimal_mapping_count: int
    mappings: tuple[tuple[tuple[str, str], ...], ...]
    example_mapping: tuple[tuple[str, str], ...]
    example_matches: tuple[dict[str, object], ...]
    beam_truncated: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hebrew(text: str, *, collapse_final_forms: bool) -> str:
    """Remove Hebrew marks and optionally identify final/medial letter forms."""

    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    if collapse_final_forms:
        normalized = normalized.translate(FINAL_TO_MEDIAL)
    return normalized


def candidate_forms(
    target: ZodiacTarget, *, collapse_final_forms: bool
) -> tuple[str, ...]:
    """Return preregistered bare and title-prefixed one-label candidates."""

    forms: list[str] = []
    forms.extend(target.sign_names)
    forms.extend("מזל" + name for name in target.sign_names)
    forms.extend(target.month_names)
    forms.extend("חדש" + name for name in target.month_names)
    forms.extend("חודש" + name for name in target.month_names)
    return tuple(
        dict.fromkeys(
            normalize_hebrew(form, collapse_final_forms=collapse_final_forms)
            for form in forms
        )
    )


def extract_zodiac_labels(source: Path) -> dict[str, tuple[Label, ...]]:
    """Extract only wholly certain IVTFF labels explicitly typed as ``Lz``."""

    wanted = {page for target in ZODIAC_TARGETS for page in target.pages}
    result: dict[str, list[Label]] = {page: [] for page in wanted}
    for page in parse_ivtff(source):
        if page.name not in wanted:
            continue
        for locus in page.loci:
            if locus.locus_type != "Lz" or not locus.tokens:
                continue
            # Do not silently delete an uncertain token from a multi-token label.
            if locus.tokens != locus.certain_tokens:
                continue
            result[page.name].append(
                Label(page.name, locus.locus_id, "".join(locus.certain_tokens))
            )
    missing = sorted(page for page, labels in result.items() if not labels)
    if missing:
        raise ValueError(f"No wholly certain Lz labels for: {', '.join(missing)}")
    return {page: tuple(labels) for page, labels in result.items()}


def derive_ordered_mapping(
    cipher_units: Sequence[str], plaintext: str, *, injective: bool
) -> tuple[tuple[str, str], ...] | None:
    """Return the exact pairwise mapping, or ``None`` if it is inconsistent."""

    if len(cipher_units) != len(plaintext):
        return None
    forward: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for cipher, plain in zip(cipher_units, plaintext, strict=True):
        if cipher in forward and forward[cipher] != plain:
            return None
        if injective and plain in inverse and inverse[plain] != cipher:
            return None
        forward[cipher] = plain
        inverse[plain] = cipher
    return tuple(sorted(forward.items()))


def ordered_constraints(
    labels: Sequence[Label],
    forms: Sequence[str],
    *,
    grouped_eva: bool,
    injective: bool,
    allow_reversal: bool,
) -> tuple[Constraint, ...]:
    """Build all exact label/form constraints, deduplicated by local key."""

    by_mapping: dict[tuple[tuple[str, str], ...], Constraint] = {}
    for label in labels:
        units = eva_units(label.eva, grouped=grouped_eva)
        for form in forms:
            for reversed_eva in ((False, True) if allow_reversal else (False,)):
                oriented = list(reversed(units)) if reversed_eva else units
                mapping = derive_ordered_mapping(oriented, form, injective=injective)
                if mapping is None:
                    continue
                constraint = Constraint(
                    mapping=mapping,
                    locus_id=label.locus_id,
                    eva=label.eva,
                    hebrew=form,
                    reversed_eva=reversed_eva,
                )
                previous = by_mapping.get(mapping)
                if previous is None or (
                    constraint.locus_id,
                    constraint.hebrew,
                    constraint.reversed_eva,
                ) < (previous.locus_id, previous.hebrew, previous.reversed_eva):
                    by_mapping[mapping] = constraint
    return tuple(
        sorted(
            by_mapping.values(),
            key=lambda item: (item.mapping, item.locus_id, item.hebrew, item.reversed_eva),
        )
    )


def merge_mapping(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
    *,
    injective: bool,
) -> tuple[tuple[str, str], ...] | None:
    mapping = dict(left)
    inverse = {plain: cipher for cipher, plain in left}
    for cipher, plain in right:
        if cipher in mapping and mapping[cipher] != plain:
            return None
        if injective and plain in inverse and inverse[plain] != cipher:
            return None
        mapping[cipher] = plain
        inverse[plain] = cipher
    return tuple(sorted(mapping.items()))


def fit_exact_mappings(
    page_targets: Sequence[tuple[str, str]],
    constraint_cache: dict[tuple[str, str], tuple[Constraint, ...]],
    *,
    injective: bool,
    beam_size: int,
    max_optimal_mappings: int = 20_000,
) -> FitResult:
    """Maximize pages with an exact anchor under one globally consistent key."""

    ordered = sorted(
        page_targets,
        key=lambda item: (len(constraint_cache[item]), item[0], item[1]),
    )
    # State maps a cipher key to one train-only witness per matched page.
    states: dict[
        tuple[tuple[str, str], ...], tuple[tuple[str, Constraint], ...]
    ] = {(): ()}
    truncated = False
    for page, target_key in ordered:
        next_states = dict(states)  # Explicitly permit a page to remain unmatched.
        for mapping, witnesses in states.items():
            for constraint in constraint_cache[(page, target_key)]:
                merged = merge_mapping(mapping, constraint.mapping, injective=injective)
                if merged is None:
                    continue
                candidate = witnesses + ((page, constraint),)
                previous = next_states.get(merged)
                if previous is None or len(candidate) > len(previous) or (
                    len(candidate) == len(previous)
                    and tuple((p, c.locus_id, c.hebrew) for p, c in candidate)
                    < tuple((p, c.locus_id, c.hebrew) for p, c in previous)
                ):
                    next_states[merged] = candidate
        if len(next_states) > beam_size:
            truncated = True
            ranked = sorted(
                next_states.items(),
                key=lambda item: (-len(item[1]), len(item[0]), item[0]),
            )[:beam_size]
            states = dict(ranked)
        else:
            states = next_states

    maximum = max((len(witnesses) for witnesses in states.values()), default=0)
    optimal = sorted(
        (
            (mapping, witnesses)
            for mapping, witnesses in states.items()
            if len(witnesses) == maximum
        ),
        key=lambda item: (len(item[0]), item[0]),
    )
    if len(optimal) > max_optimal_mappings:
        truncated = True
        optimal = optimal[:max_optimal_mappings]
    example_mapping, example_witnesses = optimal[0]
    return FitResult(
        matched_pages=maximum,
        total_pages=len(page_targets),
        optimal_mapping_count=len(optimal),
        mappings=tuple(mapping for mapping, _ in optimal),
        example_mapping=example_mapping,
        example_matches=tuple(
            {
                "page": page,
                "locus_id": constraint.locus_id,
                "eva": constraint.eva,
                "hebrew": constraint.hebrew,
                "reversed_eva": constraint.reversed_eva,
            }
            for page, constraint in example_witnesses
        ),
        beam_truncated=truncated,
    )


def mapping_satisfies(
    mapping: tuple[tuple[str, str], ...], constraint: Constraint
) -> bool:
    learned = dict(mapping)
    return all(learned.get(cipher) == plain for cipher, plain in constraint.mapping)


def heldout_scores(
    page: str,
    correct_target: str,
    mappings: Sequence[tuple[tuple[str, str], ...]],
    target_keys: Sequence[str],
    constraint_cache: dict[tuple[str, str], tuple[Constraint, ...]],
) -> dict[str, object]:
    """Version-space vote over keys tied on the training objective."""

    votes: dict[str, float] = {}
    denominator = max(1, len(mappings))
    for target in target_keys:
        supported = sum(
            any(
                mapping_satisfies(mapping, constraint)
                for constraint in constraint_cache[(page, target)]
            )
            for mapping in mappings
        )
        votes[target] = supported / denominator
    correct = votes[correct_target]
    maximum = max(votes.values(), default=0.0)
    if maximum <= 0.0:
        top1_credit = 0.0
        reciprocal_rank = 0.0
    else:
        tied_top = sum(math.isclose(value, maximum) for value in votes.values())
        top1_credit = (1.0 / tied_top) if math.isclose(correct, maximum) else 0.0
        if correct <= 0.0:
            reciprocal_rank = 0.0
        else:
            greater = sum(value > correct and not math.isclose(value, correct) for value in votes.values())
            tied = sum(math.isclose(value, correct) for value in votes.values())
            reciprocal_rank = 1.0 / (1.0 + greater + 0.5 * (tied - 1))
    return {
        "page": page,
        "correct_target": correct_target,
        "correct_vote": correct,
        "top1_credit": top1_credit,
        "reciprocal_rank": reciprocal_rank,
        "votes": dict(sorted(votes.items(), key=lambda item: (-item[1], item[0]))),
    }


def assignments_for_targets(target_keys: Sequence[str]) -> list[tuple[str, str]]:
    """Assign permuted semantic targets to fixed diagram/page blocks."""

    if len(target_keys) != len(ZODIAC_TARGETS):
        raise ValueError("Expected one semantic target per zodiac diagram block")
    assignments: list[tuple[str, str]] = []
    for geometry, target_key in zip(ZODIAC_TARGETS, target_keys, strict=True):
        assignments.extend((page, target_key) for page in geometry.pages)
    return assignments


def cross_validated_statistic(
    target_keys: Sequence[str],
    constraint_cache: dict[tuple[str, str], tuple[Constraint, ...]],
    *,
    injective: bool,
    beam_size: int,
    include_details: bool,
) -> dict[str, object]:
    """Two folds: even zodiac blocks train odd blocks, then vice versa."""

    all_keys = [target.key for target in ZODIAC_TARGETS]
    page_assignment = dict(assignments_for_targets(target_keys))
    fold_results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    truncated = False
    for train_parity in (0, 1):
        train_pages = [
            page
            for index, target in enumerate(ZODIAC_TARGETS)
            if index % 2 == train_parity
            for page in target.pages
        ]
        test_pages = [
            page
            for index, target in enumerate(ZODIAC_TARGETS)
            if index % 2 != train_parity
            for page in target.pages
        ]
        fit = fit_exact_mappings(
            [(page, page_assignment[page]) for page in train_pages],
            constraint_cache,
            injective=injective,
            beam_size=beam_size,
        )
        truncated = truncated or fit.beam_truncated
        fold_predictions = [
            heldout_scores(
                page,
                page_assignment[page],
                fit.mappings,
                all_keys,
                constraint_cache,
            )
            for page in test_pages
        ]
        predictions.extend(fold_predictions)
        fold_result: dict[str, object] = {
            "train_parity": train_parity,
            "train_pages": train_pages,
            "test_pages": test_pages,
            "train_matched_pages": fit.matched_pages,
            "train_total_pages": fit.total_pages,
            "optimal_mapping_count": fit.optimal_mapping_count,
            "heldout_top1_credit": sum(p["top1_credit"] for p in fold_predictions),
            "heldout_mrr_sum": sum(p["reciprocal_rank"] for p in fold_predictions),
        }
        if include_details:
            fold_result["example_mapping"] = dict(fit.example_mapping)
            fold_result["example_train_matches"] = list(fit.example_matches)
            fold_result["predictions"] = fold_predictions
        fold_results.append(fold_result)
    return {
        "top1_credit": sum(p["top1_credit"] for p in predictions),
        "mrr_sum": sum(p["reciprocal_rank"] for p in predictions),
        "pages": len(predictions),
        "supported_correct_pages": sum(p["correct_vote"] > 0 for p in predictions),
        "beam_truncated": truncated,
        "folds": fold_results if include_details else None,
    }


def permutation_summary(observed: float, null: Sequence[float]) -> dict[str, object]:
    if not null:
        return {"observed": observed, "iterations": 0}
    return {
        "observed": observed,
        "iterations": len(null),
        "null_mean": statistics.fmean(null),
        "null_median": statistics.median(null),
        "null_minimum": min(null),
        "null_maximum": max(null),
        "upper_tail_p": (1 + sum(value >= observed for value in null)) / (len(null) + 1),
    }


def local_anagram_key_count(cipher_units: Sequence[str], plaintext: str) -> int:
    """Count injective local keys after discarding within-label order."""

    if len(cipher_units) != len(plaintext):
        return 0
    cipher_by_count: dict[int, int] = Counter(Counter(cipher_units).values())
    plain_by_count: dict[int, int] = Counter(Counter(plaintext).values())
    if cipher_by_count != plain_by_count:
        return 0
    return math.prod(math.factorial(group_size) for group_size in cipher_by_count.values())


def anagram_page_key_count(
    labels: Sequence[Label], forms: Sequence[str], *, grouped_eva: bool
) -> int:
    return sum(
        local_anagram_key_count(eva_units(label.eva, grouped=grouped_eva), form)
        for label in labels
        for form in forms
    )


def make_permutations(iterations: int, seed: int) -> list[tuple[str, ...]]:
    observed = [target.key for target in ZODIAC_TARGETS]
    rng = random.Random(seed)
    result: list[tuple[str, ...]] = []
    while len(result) < iterations:
        shuffled = observed.copy()
        rng.shuffle(shuffled)
        if shuffled != observed:
            result.append(tuple(shuffled))
    return result


def run_model(
    labels: dict[str, tuple[Label, ...]],
    *,
    grouped_eva: bool,
    injective: bool,
    collapse_final_forms: bool,
    permutations: Sequence[tuple[str, ...]],
    beam_size: int,
) -> dict[str, object]:
    target_by_key = {target.key: target for target in ZODIAC_TARGETS}
    cache = {
        (page, target.key): ordered_constraints(
            labels[page],
            candidate_forms(target, collapse_final_forms=collapse_final_forms),
            grouped_eva=grouped_eva,
            injective=injective,
            allow_reversal=True,
        )
        for page in labels
        for target in ZODIAC_TARGETS
    }
    observed_keys = tuple(target.key for target in ZODIAC_TARGETS)
    observed_assignment = assignments_for_targets(observed_keys)
    full_fit = fit_exact_mappings(
        observed_assignment,
        cache,
        injective=injective,
        beam_size=beam_size,
    )
    cross_validated = cross_validated_statistic(
        observed_keys,
        cache,
        injective=injective,
        beam_size=beam_size,
        include_details=True,
    )

    null_full: list[float] = []
    null_cv_top1: list[float] = []
    null_cv_mrr: list[float] = []
    null_truncated = False
    for permuted_keys in permutations:
        permuted_fit = fit_exact_mappings(
            assignments_for_targets(permuted_keys),
            cache,
            injective=injective,
            beam_size=beam_size,
        )
        permuted_cv = cross_validated_statistic(
            permuted_keys,
            cache,
            injective=injective,
            beam_size=beam_size,
            include_details=False,
        )
        null_full.append(float(permuted_fit.matched_pages))
        null_cv_top1.append(float(permuted_cv["top1_credit"]))
        null_cv_mrr.append(float(permuted_cv["mrr_sum"]))
        null_truncated = (
            null_truncated
            or permuted_fit.beam_truncated
            or bool(permuted_cv["beam_truncated"])
        )

    return {
        "representation": "grouped_eva" if grouped_eva else "raw_eva_codepoints",
        "cipher_mapping": "bijective" if injective else "homophonic_many_to_one",
        "hebrew_final_forms": "collapsed" if collapse_final_forms else "distinct",
        "candidate_form_counts": {
            key: len(candidate_forms(target, collapse_final_forms=collapse_final_forms))
            for key, target in target_by_key.items()
        },
        "full_fit": {
            "matched_pages": full_fit.matched_pages,
            "total_pages": full_fit.total_pages,
            "optimal_mapping_count": full_fit.optimal_mapping_count,
            "example_mapping": dict(full_fit.example_mapping),
            "example_matches": list(full_fit.example_matches),
            "beam_truncated": full_fit.beam_truncated,
            "permutation_test": permutation_summary(full_fit.matched_pages, null_full),
        },
        "heldout": {
            **cross_validated,
            "top1_permutation_test": permutation_summary(
                float(cross_validated["top1_credit"]), null_cv_top1
            ),
            "mrr_permutation_test": permutation_summary(
                float(cross_validated["mrr_sum"]), null_cv_mrr
            ),
        },
        "any_null_beam_truncated": null_truncated,
    }


def run_anagram_gate(
    labels: dict[str, tuple[Label, ...]],
    *,
    collapse_final_forms: bool,
    permutations: Sequence[tuple[str, ...]],
) -> dict[str, object]:
    forms = {
        target.key: candidate_forms(
            target, collapse_final_forms=collapse_final_forms
        )
        for target in ZODIAC_TARGETS
    }
    counts = {
        (page, target.key): anagram_page_key_count(
            labels[page], forms[target.key], grouped_eva=True
        )
        for page in labels
        for target in ZODIAC_TARGETS
    }

    def statistic(target_keys: Sequence[str]) -> tuple[int, float]:
        assigned = assignments_for_targets(target_keys)
        local_counts = [counts[(page, target)] for page, target in assigned]
        return sum(value > 0 for value in local_counts), sum(
            math.log1p(value) for value in local_counts
        )

    observed_keys = tuple(target.key for target in ZODIAC_TARGETS)
    observed_feasible, observed_log_keys = statistic(observed_keys)
    null = [statistic(keys) for keys in permutations]
    return {
        "scope": "necessary local compatibility only; not a global anagram key",
        "representation": "grouped_eva",
        "hebrew_final_forms": "collapsed" if collapse_final_forms else "distinct",
        "feasible_pages": permutation_summary(
            observed_feasible, [value[0] for value in null]
        ),
        "sum_log1p_local_keys": permutation_summary(
            observed_log_keys, [value[1] for value in null]
        ),
        "correct_assignment_local_key_counts": {
            page: counts[(page, target)]
            for page, target in assignments_for_targets(observed_keys)
        },
    }


def audit(
    source: Path,
    *,
    permutation_iterations: int,
    seed: int,
    beam_size: int,
) -> dict[str, object]:
    labels = extract_zodiac_labels(source)
    permutations = make_permutations(permutation_iterations, seed)
    models = [
        run_model(
            labels,
            grouped_eva=grouped,
            injective=injective,
            collapse_final_forms=collapse,
            permutations=permutations,
            beam_size=beam_size,
        )
        for collapse in (True, False)
        for grouped in (True, False)
        for injective in (True, False)
    ]
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "seed": seed,
        "permutation_iterations": permutation_iterations,
        "label_counts": {page: len(page_labels) for page, page_labels in sorted(labels.items())},
        "targets": [asdict(target) for target in ZODIAC_TARGETS],
        "selection_rule": (
            "All wholly certain IVTFF Lz loci; all tokens within a locus are concatenated. "
            "No token is selected using a candidate Hebrew reading."
        ),
        "heldout_rule": (
            "Two folds over alternating semantic blocks in zodiac order; duplicate Aries and "
            "Taurus pages remain in their diagram block."
        ),
        "ordered_models": models,
        "anagram_local_gate": [
            run_anagram_gate(
                labels,
                collapse_final_forms=collapse,
                permutations=permutations,
            )
            for collapse in (True, False)
        ],
        "sources_and_scope": {
            "voynich_sign_mapping": "https://voynich.nu/writing.html#extra",
            "ivtff_locus_definition": "https://www.voynich.nu/software/ivtt/IVTFF_format.pdf",
            "medieval_hebrew_zodiac_list": (
                "Abraham ibn Ezra, Reshit Hokhmah: טלה, שור, תאומים, סרטן, "
                "אריה, בתולה, מאזנים, עקרב, קשת, גדי, דלי, דגים"
            ),
            "interpretation_limit": (
                "Lz identifies labels of zodiac elements, mostly the surrounding figures/stars; "
                "it does not identify a Voynich-script sign-name label beside the central emblem."
            ),
        },
    }


def print_summary(result: dict[str, object]) -> None:
    print("Voynich zodiac / Hebrew anchor test")
    print(f"source: {result['source']} ({result['source_sha256']})")
    print(f"permutations: {result['permutation_iterations']}  seed: {result['seed']}")
    print(f"labels: {sum(result['label_counts'].values())} wholly certain Lz loci")
    for model in result["ordered_models"]:
        full = model["full_fit"]
        heldout = model["heldout"]
        full_null = full["permutation_test"]
        cv_null = heldout["top1_permutation_test"]
        print(
            f"- {model['representation']}, {model['cipher_mapping']}, "
            f"final={model['hebrew_final_forms']}: full {full['matched_pages']}/"
            f"{full['total_pages']} (p={full_null.get('upper_tail_p', float('nan')):.4f}); "
            f"held-out top1 credit {heldout['top1_credit']:.3f}/{heldout['pages']} "
            f"(p={cv_null.get('upper_tail_p', float('nan')):.4f})"
        )
    for gate in result["anagram_local_gate"]:
        feasible = gate["feasible_pages"]
        log_keys = gate["sum_log1p_local_keys"]
        print(
            f"- anagram local gate, final={gate['hebrew_final_forms']}: "
            f"{feasible['observed']}/12 pages locally feasible "
            f"(p={feasible.get('upper_tail_p', float('nan')):.4f}); "
            f"log-key association p={log_keys.get('upper_tail_p', float('nan')):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--beam-size", type=int, default=50_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.source,
        permutation_iterations=args.permutations,
        seed=args.seed,
        beam_size=args.beam_size,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
