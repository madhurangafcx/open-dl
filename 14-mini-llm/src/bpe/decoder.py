"""Decoding: token ids back to text. The inverse of the whole pipeline.

WHAT DECODING CAN AND CANNOT RECOVER
------------------------------------
Decoding reverses byte encoding and BPE merging exactly -- those stages are
lossless. It cannot reverse normalization, which is lossy by design ('ﬁ' became
'fi' and there is no way back). So the guarantee is:

    decode(encode(text)) == normalize(text)     always
    decode(encode(text)) == text                only if text was already normalized

That is why `EncodeResult` keeps both `original_text` and `normalized_text`. If
a round-trip test fails, compare against the normalized form first -- comparing
against the raw input is the more common mistake.
"""

from collections.abc import Iterable

from .vocabulary import Vocabulary


def decode(ids: Iterable[int], vocabulary: Vocabulary, errors: str = "replace") -> str:
    """Turn token ids back into text.

    JOIN FIRST, DECODE ONCE -- THIS ORDER IS NOT OPTIONAL
    -----------------------------------------------------
    A single token can hold a *fragment* of a character. '漢' is three UTF-8
    bytes, and nothing stops those bytes from landing in two different tokens.
    Decoding token-by-token would hand `bytes.decode` an incomplete sequence
    and mangle every such character:

        per-token:   b'\\xe6\\xbc' -> '\\ufffd'  +  b'\\xa2' -> '\\ufffd'   two replacements
        joined:      b'\\xe6\\xbc\\xa2'                     -> '漢'          correct

    So every token's bytes are concatenated into one buffer, and UTF-8 decoding
    happens exactly once at the end.

    WHY errors="replace"
    --------------------
    A *slice* of a valid token stream can still cut a character in half -- the
    obvious case being a partially generated model output. Rather than raise, an
    incomplete tail becomes U+FFFD ('\\ufffd'), so streaming and truncated
    output stay usable. Pass errors="strict" when you specifically want a
    malformed stream to fail loudly.
    """
    # bytes_for() is what makes this work for merged tokens as well as raw
    # bytes: it expands each id to however many bytes that token represents.
    raw = b"".join(vocabulary.bytes_for(token_id) for token_id in ids)
    return raw.decode("utf-8", errors=errors)
