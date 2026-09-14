"""The public facade. Composes the stages; knows nothing about HTTP or SSE.

    from app.tokenizer import Tokenizer

This is the only module app/main.py imports from the tokenizer side. Everything
below is plain synchronous Python -- no async, no FastAPI, no awareness that a
web server exists. That boundary is deliberate: it means the whole tokenizer can
be driven from a script or a test with no server running, and the streaming in
main.py stays a presentation concern.

THE PIPELINE
------------
    text
      -> normalization.normalize          one canonical Unicode spelling
      -> PreTokenizer.split               pieces BPE may not merge across
      -> byte_encoder.encode              each piece becomes byte ids 0..255
      -> BPEEncoder.encode_piece          merges applied, IF merges exist
      -> EncodeResult

and back:

    ids -> decoder.decode -> text

THE SEAM THIS MODULE EXISTS TO PROTECT
--------------------------------------
`self._bpe` is None when there are no merges, and `encode` then passes byte ids
straight through. That single branch is the entire integration point for BPE:
once bpe_trainer produces merges, it wakes up and *no other module changes*.
Implement the trainer and encoder against this seam rather than reshaping the
pipeline around them.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from . import byte_encoder, decoder
from .bpe_encoder import BPEEncoder
from .normalization import DEFAULT_FORM, normalize
from .pretokenizer import PreTokenizer
from .vocabulary import Vocabulary, load_merges, save_merges


@dataclass
class Piece:
    """One pre-token and the token ids it became.

    Keeping the source text alongside the ids is what lets the SSE stream show
    *which* text produced which tokens, and makes "why did this cost 7 tokens?"
    answerable by inspection.
    """

    text: str
    token_ids: list[int]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass
class EncodeResult:
    """Everything one encode() call produced, structured for inspection.

    Richer than a bare list of ids because the point of this project is to
    *watch* tokenization: main.py needs the pieces and their boundaries, not
    just the final ids.

    Both texts are kept because normalization is lossy -- see decoder.py.
    `original_text` is exactly what came in; `normalized_text` is what the
    tokenizer actually worked on and what a round-trip will reproduce.

    The derived values are properties rather than stored fields so they can
    never fall out of sync with `pieces`. They recompute on every access, so
    bind them to a local when looping (main.py does this with `total`).
    """

    original_text: str
    normalized_text: str
    pieces: list[Piece] = field(default_factory=list)

    @property
    def pre_tokens(self) -> list[str]:
        """Just the piece texts -- the pre-tokenizer's output."""
        return [piece.text for piece in self.pieces]

    @property
    def token_ids(self) -> list[int]:
        """All ids, flattened across pieces. This is what a model would consume."""
        return [
            token_id for piece in self.pieces for token_id in piece.token_ids
        ]

    @property
    def pre_token_count(self) -> int:
        return len(self.pieces)

    @property
    def total_tokens(self) -> int:
        """The number that matters: what this text costs in tokens.

        Equals pre_token_count only when every piece is a single token. Before
        training, it is the UTF-8 byte length of the normalized text.
        """
        return sum(piece.token_count for piece in self.pieces)


class Tokenizer:
    def __init__(
        self,
        *,
        vocabulary: Vocabulary | None = None,
        merges: Iterable[tuple[str, str]] | None = None,
        normalization_form: str = DEFAULT_FORM,
        pretokenizer: PreTokenizer | None = None,
    ) -> None:
        """Every stage is injectable, and all four arguments travel together.

        Keyword-only because `Tokenizer(vocab, merges)` reads ambiguously and
        the pair must not be mixed up.

        A vocabulary is only meaningful alongside the exact normalization form
        and pre-tokenizer it was trained with. Mismatch any of them and the
        encoder feeds the merge table pieces the trainer never saw: no error,
        just merges that fail to apply and token counts that quietly balloon.
        """
        # Default: the 256 byte tokens. Enough to represent any text at all,
        # which is why an untrained tokenizer still works.
        self.vocabulary = vocabulary or Vocabulary.byte_baseline()
        self.normalization_form = normalization_form
        self.pretokenizer = pretokenizer or PreTokenizer()

        # Copied into a list so the caller cannot mutate our merges afterwards,
        # and because rank depends on position -- see BPEEncoder.
        self.merges = list(merges or [])

        # No merges -> pure byte-level tokenization, exactly what main.py
        # did inline. Once bpe_trainer produces merges, this wakes up and
        # nothing else in the pipeline changes.
        self._bpe = BPEEncoder(self.vocabulary, self.merges) if self.merges else None

    @property
    def vocab_size(self) -> int:
        """256 before training; 256 + len(merges) after, ignoring specials."""
        return len(self.vocabulary)

    def encode(self, text: str) -> EncodeResult:
        """Run the full pipeline. Synchronous and deterministic.

        Note the loop body: BPE is applied *per piece*, never to the whole
        stream. That is what enforces the pre-tokenizer's boundaries -- a merge
        physically cannot span two pieces because encode_piece never sees more
        than one.
        """
        normalized = normalize(text, self.normalization_form)

        pieces = []
        for pre_token in self.pretokenizer.split(normalized):
            byte_ids = byte_encoder.encode(pre_token)
            token_ids = (
                self._bpe.encode_piece(byte_ids) if self._bpe else byte_ids
            )
            pieces.append(Piece(text=pre_token, token_ids=token_ids))

        return EncodeResult(
            original_text=text,
            normalized_text=normalized,
            pieces=pieces,
        )

    def decode(self, ids: Iterable[int]) -> str:
        """Ids back to text. Returns the *normalized* text, not the original."""
        return decoder.decode(ids, self.vocabulary)

    def save(self, directory) -> None:
        """Write vocab.json and merges.txt into `directory`.

        Both, always, so the pair can never be separated. Note what is NOT
        saved: normalization_form and the pre-tokenizer pattern. Whoever loads
        this has to supply matching ones -- see load().
        """
        self.vocabulary.save(directory)
        save_merges(directory, self.merges)

    @classmethod
    def load(cls, directory, **kwargs) -> "Tokenizer":
        """Rebuild a Tokenizer from a saved directory.

        `**kwargs` forwards normalization_form and pretokenizer, because those
        are not persisted and must match what the vocabulary was trained with.
        Passing nothing gives the defaults, which is correct only if training
        used them too.
        """
        return cls(
            vocabulary=Vocabulary.load(directory),
            merges=load_merges(directory),
            **kwargs,
        )
