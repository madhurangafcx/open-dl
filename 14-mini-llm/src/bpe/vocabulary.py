"""The token registry: id <-> token <-> bytes, plus disk persistence.

WHAT A VOCABULARY ACTUALLY IS
-----------------------------
Two dictionaries that must never disagree:

    id_to_token   257 -> 'the'      what the model emits -> what it means
    token_to_id   'the' -> 257      what the trainer found -> its id

Plus a third relationship that is computed rather than stored: what raw *bytes*
a token stands for. `bytes_for()` derives it via the byte<->character map in
byte_encoder, which is why token strings look like text but are not.

Every write goes through `_register` so the two dicts cannot drift apart.

THE THREE KINDS OF TOKEN
------------------------
    ids 0..255    the byte baseline. Always present, id == byte value.
    ids 256+      merged tokens learned by bpe_trainer ('th', 'the', 'Ġthe').
    specials      <|endoftext|> and friends. Not text -- control signals that
                  must never be produced by merging, so they are added
                  explicitly and decoded by a separate path in bytes_for().

ON-DISK FORMAT
--------------
Two files, matching what real BPE tokenizers ship:

    vocab.json    {"token": id, ...}   the alphabet
    merges.txt    "left right" lines   the rules, one per line

They are separate because they answer different questions -- "what tokens
exist?" versus "in what order were they built?" -- and merges.txt has to stay
line-ordered, which JSON object ordering does not reliably guarantee.
"""

import json
from pathlib import Path

from .byte_encoder import bytes_to_unicode, unicode_to_bytes

VOCAB_FILENAME = "vocab.json"
MERGES_FILENAME = "merges.txt"

# GPT-2 writes this header as the first line of merges.txt. Kept for
# compatibility; load_merges skips any line starting with '#'.
MERGES_HEADER = "#version: 0.1"


class Vocabulary:
    def __init__(self) -> None:
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}

        # Specials are tracked separately, not because they need different
        # storage, but because bytes_for() has to decode them differently:
        # '<|endoftext|>' is literal text, not disguised bytes.
        self.special_tokens: dict[str, int] = {}

        # Snapshot of the stand-in -> byte table, held per instance so decoding
        # never re-derives it. Cheap: byte_encoder caches the underlying map.
        self._byte_decoder = unicode_to_bytes()

        # High-water mark for id allocation. See add_token for why this exists
        # instead of just using len().
        self._next_id = 0

    @classmethod
    def byte_baseline(cls) -> "Vocabulary":
        """The 256 single-byte tokens every BPE vocabulary starts from.

        Iterating in sorted byte order makes id == byte value, which is the
        property that lets `byte_encoder.encode()` output be used as token ids
        directly, with no lookup, before any merge exists. Break this ordering
        and the untrained tokenizer silently produces wrong ids.
        """
        vocabulary = cls()
        for byte, character in sorted(bytes_to_unicode().items()):
            vocabulary._register(byte, character)
        return vocabulary

    def _register(self, token_id: int, token: str) -> None:
        """The single write path into both dicts. Never bypass it.

        Refuses to overwrite an occupied id. Without this guard a bad id would
        replace `id_to_token[id]` while `token_to_id` still pointed the old
        token at it -- the two dicts disagree, and decode() starts returning a
        different token than the one that was encoded. That corruption is
        silent and surfaces far from its cause, so it fails loudly here instead.
        """
        if token_id in self.id_to_token:
            raise ValueError(
                f"id {token_id} is already taken by "
                f"{self.id_to_token[token_id]!r}; refusing to overwrite it "
                f"with {token!r}"
            )

        self.id_to_token[token_id] = token
        self.token_to_id[token] = token_id
        self._next_id = max(self._next_id, token_id + 1)

    def add_token(self, token: str) -> int:
        """Add `token` if new; return its id either way.

        Idempotent, because bpe_trainer may propose a merge whose result
        already exists and should reuse the same id rather than fail.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        # Allocate from a high-water mark, not len(). Those agree only when ids
        # are gapless. Given ids {0, 1, 3}, len() is 3 -- an id already taken --
        # so the guard in _register would (correctly) reject the write. Using
        # _next_id yields 4 and just works. bpe_trainer calls this thousands of
        # times, so an allocator that depends on gaplessness is a trap.
        token_id = self._next_id
        self._register(token_id, token)
        return token_id

    def add_special(self, token: str) -> int:
        """Register a control token such as '<|endoftext|>'.

        Specials get ordinary ids -- the model sees no difference -- but are
        recorded in special_tokens so bytes_for() decodes them as literal text
        rather than trying to read them as disguised bytes.
        """
        token_id = self.add_token(token)
        self.special_tokens[token] = token_id
        return token_id

    def bytes_for(self, token_id: int) -> bytes:
        """The raw bytes a token id stands for. The only correct way to get them.

        A token string is a sequence of printable stand-ins from
        byte_encoder.bytes_to_unicode, not text. Compare:

            token 'Ġthe'
              .encode('utf-8')  -> b'\\xc4\\xa0the'   WRONG: bytes of the disguise
              bytes_for(id)     -> b' the'            RIGHT: what it represents

        Specials are the one exception: '<|endoftext|>' is literal text that was
        never disguised, so it is encoded directly.
        """
        try:
            token = self.id_to_token[token_id]
        except KeyError:
            # A bare KeyError here says only "999" and gives no hint. The
            # usual cause is ids produced against a different vocab.json than
            # the one loaded, so name that explicitly.
            raise KeyError(
                f"token id {token_id} is not in this vocabulary "
                f"(size {len(self)}); the ids may come from a different "
                f"vocab.json than the one loaded"
            ) from None

        if token in self.special_tokens:
            return token.encode("utf-8")

        return bytes(self._byte_decoder[character] for character in token)

    def __len__(self) -> int:
        """Vocabulary size -- the trainer's stopping condition."""
        return len(self.id_to_token)

    def __contains__(self, token: str) -> bool:
        """Membership is by token string, so `'the' in vocabulary` reads right."""
        return token in self.token_to_id

    def save(self, directory) -> Path:
        """Write vocab.json as {token: id}, ordered by id.

        Token-keyed rather than id-keyed to match GPT-2's format. Sorted by id
        so the file is stable across runs and diffs cleanly when retrained.
        ensure_ascii=False keeps the stand-in characters readable ('Ġthe'
        rather than '\\u0120the').
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / VOCAB_FILENAME
        payload = {
            token: token_id
            for token_id, token in sorted(self.id_to_token.items())
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory) -> "Vocabulary":
        """Rebuild a Vocabulary from vocab.json.

        Inserting in id order keeps _next_id correct and makes a duplicate id
        in a hand-edited file fail on the second insert rather than silently
        clobbering the first. JSON object keys are always strings, hence the
        int() on every id.
        """
        payload = json.loads(
            (Path(directory) / VOCAB_FILENAME).read_text(encoding="utf-8")
        )

        vocabulary = cls()
        for token, token_id in sorted(payload.items(), key=lambda item: item[1]):
            vocabulary._register(int(token_id), token)

            # Specials are recovered by GPT-2's <|...|> naming convention
            # rather than from a separate file. The trade-off: a merged token
            # that happened to look like '<|...|>' would be misread as a
            # special. Byte stand-ins make that essentially impossible, since
            # '<' and '|' only appear in a token if they were merged together.
            if token.startswith("<|") and token.endswith("|>"):
                vocabulary.special_tokens[token] = int(token_id)

        return vocabulary


def save_merges(directory, merges) -> Path:
    """Write merges.txt, one 'left right' pair per line, in rank order.

    Line order IS the merge ranking, so this must never sort. A plain text
    format keeps it readable: opening merges.txt shows exactly what the
    trainer learned, in the order it learned it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / MERGES_FILENAME
    lines = [MERGES_HEADER] + [f"{left} {right}" for left, right in merges]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_merges(directory) -> list[tuple[str, str]]:
    """Read merges.txt back, preserving rank. Line order IS rank -- never sort.

    Returns [] when the file is absent, which is how a byte-only tokenizer
    loads: no merges means Tokenizer skips the BPE stage entirely.
    """
    path = Path(directory) / MERGES_FILENAME
    if not path.exists():
        return []

    merges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # Skips the '#version:' header and any blank trailing line.
        if not line or line.startswith("#"):
            continue

        # Splitting on a literal space is safe *because* of the byte disguise:
        # byte 0x20 is stored as 'Ġ', so no token can contain a real space.
        # Were tokens stored as raw text, this format would be ambiguous the
        # moment a merge produced a token containing a space.
        parts = line.split(" ")
        if len(parts) != 2:
            raise ValueError(f"malformed merge line: {line!r}")

        merges.append((parts[0], parts[1]))

    return merges
