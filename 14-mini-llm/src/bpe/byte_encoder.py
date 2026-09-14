"""Stage 3 of 3: UTF-8 byte encoding, plus the byte<->character map.

WHY BYTES INSTEAD OF CHARACTERS
-------------------------------
A tokenizer needs a finite starting alphabet. Two obvious choices:

  characters -- Unicode has ~150,000 assigned codepoints. Seeding a vocabulary
                with all of them is wasteful, and any character you leave out
                is unrepresentable: you need an <UNK> token, and <UNK> destroys
                information permanently.

  bytes      -- there are exactly 256. Every possible string, in every script,
                including text that did not exist when the tokenizer was
                trained, is *some* sequence of these 256 symbols.

Bytes win: 256 seed tokens and **no <UNK> token is ever needed**. This is what
"byte-level BPE" means. The cost is that non-ASCII characters start out as
several tokens, since UTF-8 is variable-width:

    'a'  -> 1 byte   [97]
    'é'  -> 2 bytes  [195, 169]
    '漢' -> 3 bytes  [230, 188, 162]
    '🎉' -> 4 bytes  [240, 159, 142, 137]

BPE's whole job is to earn those back: frequent byte runs get merged into single
tokens during training, so common words in common scripts end up cheap while
rare text degrades gracefully to bytes instead of failing.

WHY BYTES NEED A CHARACTER DISGUISE
-----------------------------------
See bytes_to_unicode below. Short version: token strings must be printable to
survive a round trip through vocab.json and merges.txt, and 65 of the 256 byte
values are not printable characters.
"""

from functools import lru_cache


def encode(text: str) -> list[int]:
    """Turn text into byte values 0..255, each of which is one starting token.

    These integers double as token ids without any lookup, because
    `Vocabulary.byte_baseline()` deliberately assigns id == byte value. That is
    why this stage can hand its output straight to the model when no merges
    exist yet.
    """
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> bytes:
    """Inverse of `encode`, for raw byte ids only.

    Only valid before any merging: it assumes every id is a byte value below
    256. Once merges exist an id can stand for many bytes, and the real decode
    path is `Vocabulary.bytes_for` (used by decoder.decode), which knows what
    each id expands to.
    """
    return bytes(ids)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map each of the 256 byte values to a distinct *printable* character.

    THE PROBLEM
    -----------
    Tokens are stored as JSON strings in vocab.json and as space-separated text
    in merges.txt. But raw bytes include values with no printable form: 0x00
    (NUL), 0x0A (newline), 0x09 (tab), 0x20 (space). Writing those into a token
    file would produce entries you cannot read, diff, or -- for newline and
    space -- parse back out, since merges.txt is line-based and space-separated.

    THE TRICK
    ---------
    Give every byte a printable stand-in character. 188 byte values already are
    printable, so they map to themselves. The remaining 68 borrow unused
    codepoints from U+0100 to U+0143:

        byte 0x41 -> 'A'  (U+0041)   already printable, maps to itself
        byte 0x7E -> '~'  (U+007E)   already printable
        byte 0xFF -> 'ÿ'  (U+00FF)   already printable
        byte 0x20 -> 'Ġ'  (U+0120)   space          <- the famous one
        byte 0x0A -> 'Ċ'  (U+010A)   newline
        byte 0x09 -> 'ĉ'  (U+0109)   tab
        byte 0x00 -> 'Ā'  (U+0100)   NUL
        byte 0x7F -> 'ġ'  (U+0121)   DEL

    This is where the `Ġ` you see all over real GPT-2 vocabularies comes from:
    it is not a letter, it is byte 0x20 wearing a costume.

    TWO PROPERTIES THE REST OF THE CODE RELIES ON
    ---------------------------------------------
    * The map is injective -- 256 bytes, 256 distinct characters -- so
      unicode_to_bytes() is an exact inverse and nothing is ambiguous.
    * No stand-in is a space. That is precisely what makes it safe for
      `load_merges` to split a merges.txt line on a literal space.

    CONSEQUENCE
    -----------
    A token string is a sequence of these stand-ins, NOT text. Calling
    `token.encode("utf-8")` on one gives you the bytes of the disguise, not the
    bytes the token represents. Always go through `Vocabulary.bytes_for(id)`.

    Cached because the result is a fixed 256-entry table; building it per call
    would be pure waste. maxsize=1 since it takes no arguments.
    """
    # The three ranges of Latin-1 that are safely printable. Deliberately
    # excluded: everything below '!' (control characters plus space), 0x7F
    # (DEL) through 0xA0, plus U+00AD (soft hyphen), which is invisible.
    printable = (
        list(range(ord("!"), ord("~") + 1))     # 0x21-0x7E  ASCII visible
        + list(range(ord("¡"), ord("¬") + 1))   # 0xA1-0xAC  Latin-1 visible
        + list(range(ord("®"), ord("ÿ") + 1))   # 0xAE-0xFF  (skips 0xAD)
    )
    # These 188 bytes need no disguise: byte 0x41 is simply 'A'.
    mapping = {byte: chr(byte) for byte in printable}

    # The other 68 get codepoints from U+0100 upward -- a region of Latin
    # Extended-A that no byte maps to naturally, so no collision is possible.
    # Assigned in ascending byte order, which is what makes this map stable:
    # every run produces the identical table, so a vocab.json written today
    # still loads correctly tomorrow.
    next_codepoint = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(next_codepoint)
            next_codepoint += 1

    return mapping


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    """Exact inverse of bytes_to_unicode: stand-in character -> byte value.

    Used by `Vocabulary.bytes_for` to turn a token's disguised characters back
    into the real bytes it stands for. Safe to invert by simply flipping the
    dict because the forward map is injective.
    """
    return {character: byte for byte, character in bytes_to_unicode().items()}
