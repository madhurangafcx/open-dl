"""
WHAT BPE TRAINING IS
--------------------
Byte Pair Encoding was a compression algorithm before it was a tokenizer. The
idea is one line long: **repeatedly replace the most frequent adjacent pair of
symbols with a single new symbol.**

Starting from the 256 byte tokens, on a corpus where "the" is common:

    round 1   most frequent pair is ('t','h')     -> new token 'th'    (id 256)
    round 2   most frequent pair is ('th','e')    -> new token 'the'   (id 257)
    round 3   most frequent pair is ('Ġ','the')   -> new token 'Ġthe'  (id 258)

Each round adds exactly one token and one merge rule. Stop when the vocabulary
reaches the size you asked for. The result is a vocabulary where frequency buys
brevity: common words become single tokens, rare ones stay as fragments, and
anything at all is still representable because the 256 byte tokens never go
away.

WHY THE OUTPUT IS AN ORDERED LIST
---------------------------------
The merges are not a set of rules -- they are a *sequence*. 'the' can only be
built after 'th' exists, so applying them out of order produces different (and
wrong) tokenization. bpe_encoder reconstructs rank from list position, so the
order this function returns is load-bearing. Never sort it.

WHY WORD COUNTS INSTEAD OF RAW TEXT
-----------------------------------
Step 1 collapses the corpus into {piece: count}. A 1 GB corpus might hold only a
few hundred thousand distinct pieces, so every later round scans that instead of
a gigabyte. The counts preserve frequency exactly, so results are identical --
this is purely a speed decision, and it is the difference between minutes and
hours.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .bpe_encoder import merge_pair
from .normalization import DEFAULT_FORM, normalize
from .pretokenizer import PreTokenizer
from .vocabulary import Vocabulary

# Every BPE vocabulary starts from the 256 single-byte tokens, so this is the
# floor for vocab_size and the offset for every merged id.
BASELINE_SIZE = 256

@dataclass
class TrainResult:
    """The two artifacts training produces, and everything needed to use them.

    Pair them permanently: a vocabulary tells you what ids mean, merges tell you
    how to produce those ids. One without the other is useless, which is why
    Tokenizer.save writes both and Tokenizer.load reads both.
    """

    vocabulary: Vocabulary
    merges: list[tuple[str, str]] = field(default_factory=list)

def _count_pieces(corpus, normalization_form, pretokenizer):
    """Collapse the corpus to {piece: frequency}. This is the speed decision.

    Every later round scans this dict rather than the raw text. A gigabyte of
    English holds only a few hundred thousand distinct pieces, and the counts
    preserve frequency exactly -- so the result is identical, just far cheaper.
    """
    counts = Counter()
    for document in corpus:
        counts.update(pretokenizer.split(normalize(document, normalization_form)))
    return counts

def _as_symbol_words(piece_counts, vocabulary):
    """Turn each distinct piece into a tuple of baseline symbols, keeping counts.

    Tuples, not lists, so they can be dict keys. Symbols are the disguised
    characters from byte_encoder -- the same alphabet merges.txt stores, which
    is what lets BPEEncoder look pairs up directly.
    """
    words = Counter()
    for piece, frequency in piece_counts.items():
        symbols = tuple(vocabulary.id_to_token[b] for b in piece.encode("utf-8"))
        words[symbols] += frequency
    return words

def _count_pairs(words):
    """Count every adjacent pair, weighted by how often its word occurs.

    The weighting is the whole point. A pair inside a word that appears 10,000
    times must count 10,000 times, not once -- otherwise a rare word's pairs
    outvote a common one's.

    zip(symbols, symbols[1:]) walks adjacent pairs and never runs past the end,
    so pairs are counted strictly *within* a piece. That is what makes the
    pre-tokenizer's boundaries real walls.
    """
    pairs = Counter()
    for symbols, frequency in words.items():
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += frequency
    return pairs

def train(
    corpus: Iterable[str],
    vocab_size: int,
    *,
    normalization_form: str = DEFAULT_FORM,
    pretokenizer: PreTokenizer | None = None,
) -> TrainResult:
    """Learn `vocab_size - 256` merge rules from `corpus`.

    `corpus` is an iterable of documents, so a generator over files works
    without loading everything into memory.

    `normalization_form` and `pretokenizer` MUST match what the Tokenizer will
    later use. They are parameters rather than hardcoded precisely so that
    mismatch is visible: train under one pre-tokenizer and encode under another
    and the encoder produces pieces the trainer never saw, so the merges
    largely fail to apply and token counts quietly balloon.

    Pairs are counted *within* pieces only, never across them. The
    pre-tokenizer's boundaries are walls -- see pretokenizer.py for why.

    The order of `merges` is the merge ranking. bpe_encoder applies them in
    exactly this order, so it must never be re-sorted.
    """
    if vocab_size < BASELINE_SIZE:
        raise ValueError(
            f"vocab_size must be at least {BASELINE_SIZE} (the byte baseline), "
            f"got {vocab_size}"
        )

    pretokenizer = pretokenizer or PreTokenizer()
    vocabulary = Vocabulary.byte_baseline()
    merges: list[tuple[str, str]] = []

    words = _as_symbol_words(
        _count_pieces(corpus, normalization_form, pretokenizer), vocabulary
    )

    # One merge per round, one new token per merge, so this is also the
    # remaining-merge budget.
    while len(vocabulary) < vocab_size:
        pairs = _count_pairs(words)

        # Every piece is a single symbol already; nothing left to merge.
        if not pairs:
            break

        # Explicit tiebreak: highest count, then the lexicographically largest
        # pair. Without the second key, ties resolve by dict iteration order
        # and two runs on the same corpus yield different vocabularies.
        pair, count = max(pairs.items(), key=lambda item: (item[1], item[0]))

        # A pair seen once teaches nothing -- it just memorizes one word.
        # Small corpora hit this long before vocab_size, and stopping here is
        # correct rather than a failure.
        if count < 2:
            break

        merges.append(pair)
        vocabulary.add_token(pair[0] + pair[1])

        # Rewrite every word with the merge applied. Accumulate with += rather
        # than assigning: two distinct words can collapse to the same tuple
        # once a pair is merged, and their counts must add.
        rewritten = Counter()
        for symbols, frequency in words.items():
            rewritten[tuple(merge_pair(list(symbols), pair))] += frequency
        words = rewritten

    return TrainResult(vocabulary=vocabulary, merges=merges)