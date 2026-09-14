"""Stage 2 of 3: pre-tokenization.

WHY THIS STAGE EXISTS
---------------------
BPE merges whichever adjacent pair is most frequent, over and over. Left to run
on raw text it would happily learn nonsense spanning natural boundaries -- a
single token for "dog." or ", the" or even "end of sentence. The" -- because
those sequences genuinely are frequent. The vocabulary fills up with
punctuation-glued variants of the same word, and "dog" alone becomes rare.

Pre-tokenization draws hard walls first. Text is chopped into pieces, and BPE is
then run *inside each piece independently*. A merge can never span a wall, so no
token can ever contain a piece boundary. This is why `Tokenizer.encode` calls
`BPEEncoder.encode_piece` once per piece instead of once per document.

The pattern you choose therefore decides what the model is even *able* to learn.
It is not a preprocessing detail; it is a modelling decision.

WHAT THE DEFAULT PATTERN DOES
-----------------------------
Four alternatives, tried left to right at each position:

    "Don't stop 42 times!"
    -> ['Don', "'", 't', ' ', 'stop', ' ', '42', ' ', 'times', '!']

Letters group with letters and digits with digits, so "abc123" splits into
'abc' + '123' -- a token can never mix the two. Each punctuation mark is its own
single-character piece (note the missing `+`), so "!!!" becomes three pieces and
no "!!!" token can ever be learned. Whitespace runs group together.

HOW GPT-2'S PATTERN DIFFERS, AND WHY IT MATTERS
-----------------------------------------------
    "Don't stop 42 times!"
    -> ['Don', "'t", ' stop', ' 42', ' times', '!']

Two real differences:

  1. A leading space is glued onto the following word (' stop', not ' ' + 'stop').
     So GPT-2 learns " stop" -- word-with-preceding-space -- as one token. This
     is why token counts for English prose are so much lower there, and why
     leading-space tokens dominate real GPT-2 vocabularies.
  2. Punctuation runs stay together ('...' is one piece) and contractions are
     kept whole ("'t").

Under DEFAULT_PATTERN, whitespace is its own piece, so **no merge can ever span
a space**: " stop" is not learnable, and every word costs at least one extra
token for the space in front of it. Swap in GPT2_PATTERN to compare -- but note
a vocabulary trained under one pattern is invalid under the other, since the
pieces the trainer saw would no longer be the pieces the encoder produces.
"""

import regex  # not `re`: only this module supports \p{...} Unicode categories

# The pattern main.py used inline before the split into modules.
#
# \p{L} and \p{N} are Unicode *category* classes -- "any letter" and "any
# number" in any script -- so this works on Sinhala, Greek, and CJK without
# listing ranges. The stdlib `re` module does not support them, which is why
# the `regex` package is a dependency.
DEFAULT_PATTERN = r"""
    \p{L}+              # Match sequences of letters
    |\p{N}+             # Match sequences of numbers
    |[^\s\p{L}\p{N}]    # Punctuation / symbols
    |\s+                # Whitespace
"""

# GPT-2's pattern, provided for comparison. Not the default.
#
# Written with [ ] rather than a bare space on purpose: these patterns are
# compiled with regex.VERBOSE, which strips unescaped whitespace so that
# comments and indentation are possible. A literal " ?" would have its space
# stripped and become just "?" -- a quantifier with nothing to quantify, which
# is a compile error. [ ] is a character class, so it survives VERBOSE.
GPT2_PATTERN = r"""
    '(?:s|t|re|ve|m|ll|d)   # common contractions
    |[ ]?\p{L}+             # optional leading space, then letters
    |[ ]?\p{N}+             # optional leading space, then digits
    |[ ]?[^\s\p{L}\p{N}]+   # optional leading space, then punctuation
    |\s+(?!\S)              # trailing whitespace run
    |\s+                    # any remaining whitespace
"""


class PreTokenizer:
    """Splits text into the pieces BPE may not merge across.

    A class rather than a function so the pattern is compiled once at startup
    instead of on every request. `regex` does cache compiled patterns
    internally, but holding it explicitly also lets a Tokenizer carry its own
    pattern -- which is what makes swapping in GPT2_PATTERN a one-liner.
    """

    def __init__(self, pattern: str = DEFAULT_PATTERN) -> None:
        self.pattern = pattern
        self._compiled = regex.compile(pattern, regex.VERBOSE)

    def split(self, text: str) -> list[str]:
        """Return `text` as a list of pieces.

        findall (not split) because the pattern describes what to *keep*, not
        what to cut on. Every alternative in the pattern matches content, and
        the four alternatives together cover every possible character, so
        joining the result reproduces the input exactly -- nothing is dropped.
        """
        return self._compiled.findall(text)
