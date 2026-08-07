#!/usr/bin/env python3
"""Estimate reproducible upper bounds on the Voynich surface code length.

This script deliberately does *not* call a compression ratio "semantic
information".  It constructs a normalized symbol stream and assigns it a
proper probability with an online Bayesian mixture of character-context
models.  ``-log2(probability)`` is therefore the length of an ideal lossless
code for the observed stream, conditional on the declared normalization and
symbol inventory.  The online model pays for learning each text; there is no
train-on-test plug-in entropy estimate.

The primary comparison uses equal-sized, non-overlapping windows from the
published Lindemann--Bowern corpora (294 modern language samples and a set of
historical texts).  Optional word-order ablations compare each original text
with deterministic within-block word shuffles.  Shuffling within small blocks
preserves vocabulary, word frequencies, word spellings, and broad topic while
destroying much local syntax and discourse order.

These are bounds on the exact normalized inscription, not on meaning.  Low
surface code length can also result from an information-losing orthography,
abbreviation, a verbose code, tables, formulae, or a low-rate payload.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from audit_voynich import eva_units, parse_ivtff


BOUNDARY = "<W>"
UNKNOWN_WORD = "<UNKNOWN_WORD>"


@dataclass(frozen=True)
class Corpus:
    name: str
    family: str
    source: str
    words: tuple[str | tuple[str, ...], ...]

    @property
    def stream(self) -> tuple[str, ...]:
        return words_to_stream(self.words)


@dataclass(frozen=True)
class WindowScore:
    bits_per_unit: float
    marginal_entropy: float
    conditional_entropy: float
    final_order_weights: tuple[float, ...]


@dataclass(frozen=True)
class CorpusScore:
    name: str
    family: str
    source: str
    units: int
    alphabet_size: int
    windows: int
    median_bits_per_unit: float
    minimum_bits_per_unit: float
    maximum_bits_per_unit: float
    median_marginal_entropy: float
    median_conditional_entropy: float
    dominant_order: int


def normalized_words(
    text: str, *, maximum_units: int | None = None
) -> tuple[str, ...]:
    """Return case-folded Unicode letter/mark words from already-clean text."""

    text = unicodedata.normalize("NFC", text.casefold())
    words: list[str] = []
    current: list[str] = []
    units = 0
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L") or (category.startswith("M") and current):
            current.append(character)
        elif current:
            word = "".join(current)
            words.append(word)
            units += len(word) + 1
            current = []
            if maximum_units is not None and units >= maximum_units:
                break
    if current:
        words.append("".join(current))
    return tuple(words)


def words_to_stream(words: Sequence[Sequence[str]]) -> tuple[str, ...]:
    stream: list[str] = []
    for word in words:
        stream.extend(word)
        stream.append(BOUNDARY)
    return tuple(stream)


def voynich_corpus(path: Path, *, grouped: bool = False) -> Corpus:
    words: list[tuple[str, ...]] = []
    for page in parse_ivtff(path):
        for line in page.lines("P", certain=True):
            words.extend(tuple(eva_units(word, grouped=grouped)) for word in line)
    name = "Voynich IVTFF certain grouped" if grouped else "Voynich IVTFF certain raw EVA"
    return Corpus(name=name, family="Voynich", source=str(path), words=tuple(words))


def empirical_entropy(tokens: Sequence[str]) -> float:
    if not tokens:
        return math.nan
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def conditional_entropy(tokens: Sequence[str]) -> float:
    """Plug-in H(X_i | X_{i-1}), reported only as a familiar diagnostic."""

    if len(tokens) < 2:
        return math.nan
    contexts: Counter[str] = Counter(tokens[:-1])
    pairs: Counter[tuple[str, str]] = Counter(zip(tokens[:-1], tokens[1:]))
    total = len(tokens) - 1
    return -sum(
        (count / total) * math.log2(count / contexts[left])
        for (left, _), count in pairs.items()
    )


def prequential_mixture_bits(
    tokens: Sequence[str],
    *,
    vocabulary_size: int,
    max_order: int = 8,
    concentration: float = 2.0,
) -> tuple[float, tuple[float, ...]]:
    """Code a stream with a Bayesian mixture over context orders 0..max_order.

    Order zero uses the Jeffreys/Krichevsky--Trofimov prior.  Higher orders
    interpolate their context counts with the immediately shorter context.
    A Bayesian mixture over all orders pays at most log2(max_order + 1) bits
    relative to its best constituent.  Counts are updated only after scoring
    each symbol, making this a genuine prequential code rather than an in-sample
    maximum-likelihood entropy estimate.
    """

    if vocabulary_size < 1:
        raise ValueError("vocabulary_size must be positive")
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    if not tokens:
        return 0.0, tuple([1.0 / (max_order + 1)] * (max_order + 1))

    unigram: Counter[str] = Counter()
    pair_counts: list[Counter[tuple[tuple[str, ...], str]]] = [
        Counter() for _ in range(max_order + 1)
    ]
    context_totals: list[Counter[tuple[str, ...]]] = [
        Counter() for _ in range(max_order + 1)
    ]
    weights = [1.0 / (max_order + 1)] * (max_order + 1)
    history: list[str] = []
    bits = 0.0

    for index, token in enumerate(tokens):
        base = (unigram[token] + 0.5) / (index + 0.5 * vocabulary_size)
        probabilities = [base] * (max_order + 1)
        probability = base
        available = min(max_order, len(history))
        contexts: list[tuple[str, ...] | None] = [None] * (max_order + 1)
        for order in range(1, available + 1):
            context = tuple(history[-order:])
            contexts[order] = context
            probability = (
                pair_counts[order][(context, token)] + concentration * probability
            ) / (context_totals[order][context] + concentration)
            probabilities[order] = probability
        for order in range(available + 1, max_order + 1):
            probabilities[order] = probability

        mixture_probability = sum(
            weight * prediction for weight, prediction in zip(weights, probabilities)
        )
        if not 0 < mixture_probability <= 1.0000000001:
            raise AssertionError(f"Invalid predictive probability {mixture_probability}")
        bits -= math.log2(mixture_probability)
        weights = [
            weight * prediction / mixture_probability
            for weight, prediction in zip(weights, probabilities)
        ]

        unigram[token] += 1
        for order in range(1, available + 1):
            context = contexts[order]
            assert context is not None
            pair_counts[order][(context, token)] += 1
            context_totals[order][context] += 1
        history.append(token)

    normalizer = sum(weights)
    weights = [weight / normalizer for weight in weights]
    return bits, tuple(weights)


def nonoverlapping_windows(
    tokens: Sequence[str], window_size: int, maximum_windows: int
) -> list[tuple[str, ...]]:
    if window_size <= 0 or maximum_windows <= 0:
        raise ValueError("window size and count must be positive")
    available = len(tokens) // window_size
    count = min(available, maximum_windows)
    if count == 0:
        return []
    if count == 1:
        start = (len(tokens) - window_size) // 2
        return [tuple(tokens[start : start + window_size])]
    slack = len(tokens) - count * window_size
    starts = [
        index * window_size + round(index * slack / (count - 1))
        for index in range(count)
    ]
    return [tuple(tokens[start : start + window_size]) for start in starts]


def score_window(
    tokens: Sequence[str],
    *,
    alphabet_size: int,
    max_order: int,
    concentration: float,
) -> WindowScore:
    bits, weights = prequential_mixture_bits(
        tokens,
        vocabulary_size=alphabet_size,
        max_order=max_order,
        concentration=concentration,
    )
    return WindowScore(
        bits_per_unit=bits / len(tokens),
        marginal_entropy=empirical_entropy(tokens),
        conditional_entropy=conditional_entropy(tokens),
        final_order_weights=weights,
    )


def score_corpus(
    corpus: Corpus,
    *,
    window_size: int,
    maximum_windows: int,
    max_order: int,
    concentration: float,
) -> CorpusScore | None:
    # Keep the character benchmark invariant to whether a longer prefix was
    # retained for the separate word-order analysis.
    stream = corpus.stream[: window_size * maximum_windows]
    windows = nonoverlapping_windows(stream, window_size, maximum_windows)
    if not windows:
        return None
    alphabet_size = len(set(stream))
    scores = [
        score_window(
            window,
            alphabet_size=alphabet_size,
            max_order=max_order,
            concentration=concentration,
        )
        for window in windows
    ]
    average_weights = [
        statistics.fmean(score.final_order_weights[order] for score in scores)
        for order in range(max_order + 1)
    ]
    bpu = [score.bits_per_unit for score in scores]
    return CorpusScore(
        name=corpus.name,
        family=corpus.family,
        source=corpus.source,
        units=len(stream),
        alphabet_size=alphabet_size,
        windows=len(windows),
        median_bits_per_unit=statistics.median(bpu),
        minimum_bits_per_unit=min(bpu),
        maximum_bits_per_unit=max(bpu),
        median_marginal_entropy=statistics.median(
            score.marginal_entropy for score in scores
        ),
        median_conditional_entropy=statistics.median(
            score.conditional_entropy for score in scores
        ),
        dominant_order=max(range(max_order + 1), key=average_weights.__getitem__),
    )


def load_comparison_corpora(
    root: Path, *, maximum_units: int, limit_modern: int | None = None
) -> Iterator[Corpus]:
    """Yield bounded corpora one at a time to keep the 745 MB suite streamable."""

    wikipedia = root / "Wikipedia_texts" / "full"
    modern_paths = [
        path
        for path in sorted(wikipedia.iterdir())
        if path.is_file() and not path.name.startswith(".")
    ]
    if limit_modern is not None:
        modern_paths = modern_paths[:limit_modern]
    for path in modern_paths:
        if not path.is_file() or path.name.startswith("."):
            continue
        words = normalized_words(
            path.read_text(encoding="utf-8"), maximum_units=maximum_units
        )
        yield Corpus(
            name=path.name,
            family="modern-language",
            source=str(path),
            words=words,
        )

    metadata: dict[str, str] = {}
    statistics_path = root / "Historical_texts_statistics.csv"
    if statistics_path.exists():
        with statistics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                metadata[row["langs"]] = row.get("fams", "historical")
    historical = root / "Historical_texts"
    ignored_suffixes = {".csv", ".xlsx"}
    for path in sorted(historical.iterdir()):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.suffix in ignored_suffixes
        ):
            continue
        words = normalized_words(
            path.read_text(encoding="utf-8"), maximum_units=maximum_units
        )
        yield Corpus(
            name=path.name,
            family=f"historical:{metadata.get(path.name, 'unknown')}",
            source=str(path),
            words=words,
        )


def shuffled_within_blocks(
    words: Sequence[Sequence[str]], *, block_size: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    if block_size <= 1:
        raise ValueError("block_size must exceed one")
    rng = random.Random(seed)
    result: list[tuple[str, ...]] = []
    for start in range(0, len(words), block_size):
        block = [tuple(word) for word in words[start : start + block_size]]
        rng.shuffle(block)
        result.extend(block)
    return tuple(result)


def word_order_gain(
    corpus: Corpus,
    *,
    maximum_words: int,
    block_size: int,
    shuffles: int,
    vocabulary_limit: int,
    max_order: int,
    concentration: float,
    seed: int,
) -> tuple[float, float, float, int]:
    """Return original, shuffled, and gain in bits/word under a word model."""

    words = corpus.words[:maximum_words]
    if len(words) < block_size:
        raise ValueError(f"Not enough words in {corpus.name} for order ablation")
    if vocabulary_limit < 2:
        raise ValueError("vocabulary_limit must be at least two")
    retained = {
        word for word, _ in Counter(words).most_common(vocabulary_limit - 1)
    }
    collapsed = tuple(word if word in retained else UNKNOWN_WORD for word in words)
    vocabulary = len(set(collapsed))
    original_bits, _ = prequential_mixture_bits(
        collapsed,
        vocabulary_size=vocabulary,
        max_order=max_order,
        concentration=concentration,
    )
    shuffled_scores: list[float] = []
    for iteration in range(shuffles):
        shuffled = shuffled_within_blocks(
            collapsed, block_size=block_size, seed=seed + iteration
        )
        bits, _ = prequential_mixture_bits(
            shuffled,
            vocabulary_size=vocabulary,
            max_order=max_order,
            concentration=concentration,
        )
        shuffled_scores.append(bits / len(collapsed))
    original = original_bits / len(collapsed)
    shuffled = statistics.fmean(shuffled_scores)
    return original, shuffled, shuffled - original, len(collapsed)


def print_score_table(scores: Sequence[CorpusScore]) -> None:
    print("name\tfamily\tK\twindows\tH0\tH1\tprequential_bpu\trange\torder")
    for score in sorted(scores, key=lambda item: item.median_bits_per_unit):
        print(
            f"{score.name}\t{score.family}\t{score.alphabet_size}\t{score.windows}\t"
            f"{score.median_marginal_entropy:.4f}\t"
            f"{score.median_conditional_entropy:.4f}\t"
            f"{score.median_bits_per_unit:.4f}\t"
            f"[{score.minimum_bits_per_unit:.4f},{score.maximum_bits_per_unit:.4f}]\t"
            f"{score.dominant_order}"
        )


def print_group_summary(voynich: CorpusScore, controls: Sequence[CorpusScore]) -> None:
    print("\nREFERENCE-RANGE SUMMARY")
    for label, selected in [
        ("modern all", [s for s in controls if s.family == "modern-language"]),
        ("historical all", [s for s in controls if s.family.startswith("historical:")]),
        (
            "modern K=15..35",
            [
                s
                for s in controls
                if s.family == "modern-language" and 15 <= s.alphabet_size <= 35
            ],
        ),
    ]:
        values = [score.median_bits_per_unit for score in selected]
        if not values:
            continue
        at_or_below = sum(value <= voynich.median_bits_per_unit for value in values)
        rank_bound = (at_or_below + 1) / (len(values) + 1)
        print(
            f"{label}: n={len(values)}, range=[{min(values):.4f},{max(values):.4f}], "
            f"median={statistics.median(values):.4f}, controls<=Voynich={at_or_below}, "
            f"exchangeability_rank_bound={rank_bound:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ivtff", type=Path, default=Path("IT2a-n.txt"))
    parser.add_argument(
        "--comparison-root",
        type=Path,
        help="Path to the Lindemann--Bowern Corpora directory",
    )
    parser.add_argument("--window-size", type=int, default=50_000)
    parser.add_argument("--maximum-windows", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--concentration", type=float, default=2.0)
    parser.add_argument(
        "--limit-modern",
        type=int,
        help="For a quick diagnostic, score only this many modern samples",
    )
    parser.add_argument(
        "--word-order",
        action="store_true",
        help="Also run matched within-block word-order ablations",
    )
    parser.add_argument(
        "--word-order-all",
        action="store_true",
        help="Run the word-order ablation for every eligible comparison corpus",
    )
    parser.add_argument(
        "--skip-character",
        action="store_true",
        help="Skip character coding (useful with --word-order-all)",
    )
    parser.add_argument("--order-words", type=int, default=30_000)
    parser.add_argument("--order-block-size", type=int, default=100)
    parser.add_argument("--order-shuffles", type=int, default=8)
    parser.add_argument("--order-vocabulary", type=int, default=512)
    parser.add_argument("--word-concentration", type=float, default=100.0)
    parser.add_argument("--word-max-order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=408)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = voynich_corpus(args.ivtff)
    corpora: Iterable[Corpus] = [primary]
    if args.comparison_root is not None:
        corpora = itertools.chain(
            [primary],
            load_comparison_corpora(
                args.comparison_root,
                maximum_units=args.window_size * args.maximum_windows,
                limit_modern=args.limit_modern,
            ),
        )

    scores: list[CorpusScore] = []
    skipped: list[str] = []
    order_results: list[tuple[str, str, int, float, float, float]] = []
    selected_names = {
        primary.name,
        "English",
        "French",
        "Hebrew",
        "Latin",
        "Hawaiian",
        "Picatrix",
        "Cirurgie",
        "Hebrew_Mishneh",
        "Steganographia",
    }
    for corpus in corpora:
        run_order = (args.word_order or args.word_order_all) and (
            args.word_order_all or corpus.name in selected_names
        )
        if run_order:
            available = min(args.order_words, len(corpus.words))
            if available >= args.order_block_size:
                original, shuffled, gain, count = word_order_gain(
                    corpus,
                    maximum_words=available,
                    block_size=args.order_block_size,
                    shuffles=args.order_shuffles,
                    vocabulary_limit=args.order_vocabulary,
                    max_order=args.word_max_order,
                    concentration=args.word_concentration,
                    seed=args.seed,
                )
                order_results.append(
                    (corpus.name, corpus.family, count, original, shuffled, gain)
                )
        if args.skip_character:
            continue
        score = score_corpus(
            corpus,
            window_size=args.window_size,
            maximum_windows=args.maximum_windows,
            max_order=args.max_order,
            concentration=args.concentration,
        )
        if score is None:
            skipped.append(corpus.name)
        else:
            scores.append(score)
    if not args.skip_character:
        print_score_table(scores)
        primary_score = next(score for score in scores if score.name == primary.name)
        print_group_summary(primary_score, [score for score in scores if score != primary_score])
        if skipped:
            print(f"\nSkipped (<{args.window_size} normalized units): {', '.join(skipped)}")

        full_stream = primary.stream
        full_bits, full_weights = prequential_mixture_bits(
            full_stream,
            vocabulary_size=len(set(full_stream)),
            max_order=args.max_order,
            concentration=args.concentration,
        )
        print(
            "\nVOYNICH WHOLE-STREAM UPPER BOUND\n"
            f"units={len(full_stream)} alphabet={len(set(full_stream))} "
            f"bits={full_bits:.1f} bytes={full_bits / 8:.1f} "
            f"bits_per_unit={full_bits / len(full_stream):.6f} "
            f"dominant_order={max(range(len(full_weights)), key=full_weights.__getitem__)}"
        )

    if args.word_order or args.word_order_all:
        print("\nWITHIN-BLOCK WORD-ORDER ABLATION")
        print("name\tfamily\twords\toriginal_bits/word\tshuffled_bits/word\tgain")
        for name, family, count, original, shuffled, gain in order_results:
            print(
                f"{name}\t{family}\t{count}\t{original:.4f}\t{shuffled:.4f}\t{gain:.4f}"
            )
        primary_gain = next(
            gain for name, _, _, _, _, gain in order_results if name == primary.name
        )
        print("\nWORD-ORDER REFERENCE-RANGE SUMMARY")
        for label, prefix in [
            ("modern all", "modern-language"),
            ("historical all", "historical:"),
        ]:
            values = [
                gain
                for _, family, _, _, _, gain in order_results
                if family == prefix or family.startswith(prefix)
            ]
            if values:
                at_or_below = sum(value <= primary_gain for value in values)
                print(
                    f"{label}: n={len(values)}, range=[{min(values):.4f},{max(values):.4f}], "
                    f"median={statistics.median(values):.4f}, "
                    f"controls<=Voynich={at_or_below}, "
                    f"exchangeability_rank_bound={(at_or_below + 1) / (len(values) + 1):.6f}"
                )


if __name__ == "__main__":
    main()
