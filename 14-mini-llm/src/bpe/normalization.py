"""Stage 1 of 3: Unicode normalization.

WHY THIS STAGE EXISTS
---------------------
Unicode lets the *same-looking* text be stored as different codepoints. The
letter "é" has two legitimate encodings:

    U+00E9                  a single precomposed character      -> 2 UTF-8 bytes
    U+0065 U+0301           plain "e" + a combining acute accent -> 3 UTF-8 bytes

Python considers those two strings unequal, and they produce different byte
sequences, so a tokenizer would assign them *different token ids*. The model
would then have to learn "é" twice. Normalization collapses such variants to one
canonical spelling before anything downstream looks at the text.

THE FOUR FORMS
--------------
Two independent choices, giving four combinations:

  composed vs decomposed    C = combine into single characters (é -> U+00E9)
                            D = split into base + marks (é -> U+0065 U+0301)

  canonical vs compatibility
                            (no K) = only truly equivalent spellings are merged
                            K      = also merge "compatibility" lookalikes,
                                     which LOSES formatting information:
                                         'ﬁ' -> 'fi'     (ligature split apart)
                                         '½' -> '1⁄2'    (one char becomes three)
                                         'Ａ' -> 'A'     (fullwidth to ASCII)
                                         '①' -> '1'      (circled digit)

  NFC  = composed,   canonical       NFD  = decomposed, canonical
  NFKC = composed,   compatibility   NFKD = decomposed, compatibility

WHY NFKC IS THE DEFAULT
-----------------------
Composed (C rather than D) keeps accented characters as one codepoint, so "é"
costs 2 bytes instead of 3 -- fewer byte tokens for the BPE stage to merge back
together. Compatibility (K) folds lookalikes so 'ﬁ' and 'fi' share tokens. This
is the same choice GPT-2-era tokenizers and SentencePiece make.

The K forms are lossy on purpose: '½' really does become three characters. If
you ever need to reproduce the input exactly, keep the original text around --
`EncodeResult.original_text` does exactly that, alongside the normalized form.

HISTORICAL NOTE
---------------
An earlier version of main.py chained all four forms, NFC -> NFD -> NFKC -> NFKD.
Each call overwrites the previous result, so the net effect was simply NFKD.
Pass form="NFKD" if you want that old output back.
"""

import unicodedata

# All four normalization forms Python's unicodedata accepts. Used to reject
# typos like "NFKZ" up front instead of letting them reach unicodedata, whose
# own error is less obvious.
FORMS = ("NFC", "NFD", "NFKC", "NFKD")

# Shared by Tokenizer and bpe_trainer so training and encoding always normalize
# identically. A vocabulary trained under one form and used under another would
# silently mis-tokenize: the trainer would never have seen the byte sequences
# the encoder produces.
DEFAULT_FORM = "NFKC"


def normalize(text: str, form: str = DEFAULT_FORM) -> str:
    """Return `text` rewritten in a single canonical Unicode form.

    Applying one form is enough. Normalization is idempotent -- normalizing an
    already-normalized string changes nothing -- so chaining forms only means
    the last one wins.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")

    return unicodedata.normalize(form, text)
