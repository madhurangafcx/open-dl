# Vendored: byte-level BPE tokenizer

Copied from the `tokenizer-from-scrach` repo, `text_pipeline/app/`.

| | |
|---|---|
| Source commit | `80cf449840b8705b4c628d28d34bbef97bfa7288` |
| Commit date | 2026-08-24 |
| Commit subject | feat: implement BPE trainer and encoder; update docs |

## Why vendored rather than imported across projects

`tokenizer-from-scrach` is a separate git repository and is gitignored by
`open-dl`, so a cross-project import left `14-mini-llm` unable to run from a
fresh clone. Copying the modules in makes this project self-contained.

The trade: this is a fork, not a link. Improvements made to the original do
not flow here on their own.

## What was deliberately left behind

`main.py` and `stages.py` — the FastAPI/SSE presentation layer. Nothing in the
nine modules here imports them, and excluding them drops fastapi, uvicorn,
pydantic, and sse-starlette from this project's dependencies. `regex` is the
only third-party package the tokenizer needs.

That last claim narrowed later: `src/api.py` serves the trained model over
HTTP, so fastapi and uvicorn are back in `requirements.txt`. They are this
project's dependencies, not the tokenizer's — nothing in this directory imports
them, and the sentence above still holds for everything under `src/bpe/`.

## Re-syncing

The nine `.py` files are byte-identical to the source, so an update is a plain
copy — no local edits to reapply:

```sh
SRC=../../../tokenizer-from-scrach/text_pipeline/app
for f in __init__.py tokenizer.py bpe_encoder.py bpe_trainer.py byte_encoder.py \
         decoder.py normalization.py pretokenizer.py vocabulary.py; do
  cp "$SRC/$f" "$f"
done
```

Do not add `main.py` or `stages.py` back, and keep this directory edit-free so
the copy stays trivial. Update the commit hash above when you re-sync.
