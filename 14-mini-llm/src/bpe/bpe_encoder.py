"""
TRAINING VS ENCODING -- THE ASYMMETRY
-------------------------------------
Training and encoding both "do BPE", but they answer different questions:

    bpe_trainer   which pairs are most frequent across the whole corpus?
                  -> runs once, sees everything, WRITES the rule list
    bpe_encoder   which rules apply to this one piece, in what order?
                  -> runs constantly, sees one piece, READS the rule list

Encoding never counts anything. Frequency was already decided at training time
and frozen into the merge order. This is what makes tokenization deterministic:
the same piece always produces the same ids, regardless of what surrounds it.

RANK IS THE WHOLE MECHANISM
---------------------------
Given merges [('t','h'), ('th','e'), ('e','r')], encoding "there" must apply
them in rank order, not in whatever order they happen to appear in the text:

    t h e r e
    ('t','h') is rank 0, ('e','r') is rank 2 -> apply rank 0 first
    th e r e
    ('th','e') is rank 1 -> apply it
    the r e
    ('e','r') no longer occurs; nothing left to merge
    -> ['the', 'r', 'e']

Had ('e','r') been applied first, 'th' + 'er' would never have combined into
'the' and the result would differ. Always pick the lowest-ranked pair present,
never the leftmost.
"""

from collections.abc import Iterable
from .vocabulary import Vocabulary

def merge_pair(symbols: list[str], pair: tuple[str,str]) -> list[str]:
    """Replace every occurrence of `pair` in `symbols` with the joined symbol.

    Module-level and shared: bpe_trainer imports this to rewrite its word
    tuples after choosing a merge. Both sides must apply merges identically,
    and this loop is the fiddly part -- duplicating it would let a fix in one
    place silently miss the other.

    Scanning left to right is not a style choice. On overlaps it decides the
    answer: merging ('a','a') in "aaa" gives ['aa', 'a'], never ['a', 'aa'].
    """
    left,right, = pair
    merged = left + right


    out, index, count = [], 0, len(symbols)
    while index < count:
        if (
            index < count - 1
            and symbols[index] == left
            and symbols[index + 1] == right
        ):
            out.append(merged)
            index += 2
        else:
            out.append(symbols[index])
            index += 1
    return out  

class BPEEncoder:
    """Applies a frozen merge list to individual pieces.

    Constructed once per Tokenizer and reused, because building `ranks` over a
    50,000-entry merge list on every call would dominate the cost of encoding.
    """

    def __init__(self, vocabulary: Vocabulary, merges: Iterable[tuple[str, str]]) -> None:
        self.vocabulary = vocabulary
        self.merges = list(merges)

        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # Real text repeats pieces relentlessly -- "the" arrives thousands of
        # times and always yields the same ids. Keyed by byte ids, so it stays
        # valid for this encoder's lifetime: ranks never change.
        self._cache: dict[tuple[int, ...], tuple[int, ...]] = {}


    def encode_piece(self, byte_ids: list[int]) -> list[int]:
        """Collapse `byte_ids` into the fewest token ids the merges allow.

        Takes byte ids rather than text because pre-tokenization and byte
        encoding have already run -- see Tokenizer.encode. Returns ids again,
        so it is a drop-in replacement for the no-merge passthrough.

        Never merges across pieces -- the pre-tokenizer's boundaries are the
        whole reason it runs first, and this method only ever sees one piece.
        """
        # No adjacent pair exists, so no merge can apply.
        if len(byte_ids) < 2:
            return list(byte_ids)

        key = tuple(byte_ids)
        if key in self._cache:
            return list(self._cache[key])

        # Vocabulary.byte_baseline() assigns id == byte value, so the
        # vocabulary is already the symbol table -- no need for
        # byte_encoder.bytes_to_unicode() here.
        symbols = [self.vocabulary.id_to_token[byte_id] for byte_id in byte_ids]

        while len(symbols) > 1:
            # Lowest rank present, NOT leftmost. Leftmost would apply merges
            # out of training order -- see the "there" trace above.
            best_pair, best_rank = None, None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank

            # No ranked pair left: as merged as the rules allow.
            if best_pair is None:
                break

            symbols = merge_pair(symbols, best_pair)

        try:
            ids = [self.vocabulary.token_to_id[symbol] for symbol in symbols]
        except KeyError as missing:
            raise KeyError(
                f"merge produced token {missing.args[0]!r}, which is not in "
                f"the vocabulary -- merges.txt and vocab.json are almost "
                f"certainly from different training runs"
            ) from None

        self._cache[key] = tuple(ids)
        return ids

