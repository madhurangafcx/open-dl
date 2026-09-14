# 14-mini-llm

A decoder-only transformer language model built from scratch in NumPy, then
mirrored in PyTorch so the two can be checked against each other before any
training happens.

The build order is the table in [`src/order.md`](src/order.md): ten steps, each
with a property that says whether it works. Every step's check is an assertion
in the code, not a claim in this file — if a step is listed as done below, its
check runs and passes.

## The model

| | |
| --- | --- |
| Vocabulary | 256 (byte-level; no BPE merges trained yet) |
| Model width `d_model` | 64 |
| Attention heads | 4, each `head_dim` 16 |
| Layers | 2 |
| Feed-forward width `d_ff` | 256 (4 × `d_model`) |
| Context window `max_seq_len` | 64 |
| Trainable parameters | 132,608 |

Decoder-only, pre-norm, with fixed sinusoidal positions and a tanh-approximate
GELU. Positions are a 4,096-value buffer, not a parameter — they move with the
model to a device and ride along in the checkpoint, but the optimizer never
touches them.

Where the parameters live:

| Component | Count | Share |
| --- | ---: | ---: |
| 2 × transformer block | 99,456 | 75.0% |
| Vocabulary projection `W_vocab` + bias | 16,640 | 12.5% |
| Token embedding table | 16,384 | 12.4% |
| Final LayerNorm | 128 | 0.1% |

Each block is 49,728: attention 16,384 (4 heads × Q/K/V at 64×16, plus a 64×64
output projection), feed-forward 33,088, two LayerNorms 256.

## Forward pass

Token IDs `(B, T)` index the embedding table into `(B, T, 64)`, and the
positional encoding for the first `T` positions is added.

Each block then applies two residual branches in pre-norm order — normalize,
transform, add:

```
h = x + attention(norm1(x))
y = h + ffn(norm2(h))
```

Attention projects each position to a query, key, and value, scores every query
against every key, divides by `sqrt(head_dim)` to hold the dot-product variance
near 1, masks the strict upper triangle to `-inf`, and takes a softmax over key
positions. The mask is what makes the model causal: position *t* can only ever
attend to positions ≤ *t*. Four heads run on the same input, concatenate back to
64 features, and mix through `Wo`.

The feed-forward network expands 64 → 256, applies GELU, and projects back. It
sees one position at a time, so it adds nonlinearity without moving information
between positions — attention is the only thing that does that.

After the last block a final LayerNorm runs, then `x @ W_vocab + b_vocab`
produces `(B, T, 256)`: one score per vocabulary token, at every position.

Every position predicts the *next* token, so a `T`-token input needs `T + 1`
tokens of text — inputs are `tokens[:T]` and targets are `tokens[1:T+1]`.

## Loss and gradients

Mean cross-entropy over all `B × T` predictions, computed as a stable
log-softmax (subtract the row max before exponentiating) rather than
`log(softmax(x))`.

Two reference points make the number readable: a uniform distribution scores
`ln(256) = 5.545`, and this model starts at `6.228` — slightly worse than
uniform, which is what random initialization should look like.

`step_08_gradients.py` derives ∂L/∂logits analytically. Softmax and
cross-entropy collapse into `p − one_hot(target)`, averaged over the number of
predictions. Two properties fall out and are both asserted: the gradients at each
position sum to zero (because `sum(p) − 1 = 0`), and each one matches a central
finite difference to eight decimal places. Everything behind the logits is
handled by autograd.

## Verification

Each step proves a property rather than just producing output.

| Step | Check | How |
| --- | --- | --- |
| 1 | Targets are inputs shifted by one | asserts `input_ids[:, 1:] == target_ids[:, :-1]` |
| 2 | No attention flows backward in time | upper triangle of the weights is exactly 0; rows sum to 1 |
| 3 | Heads recombine to the model width | output shape equals input shape |
| 4 | Positions are independent | perturb position 5, assert every other position is unchanged |
| 5 | A block runs, and normalization is safe | a constant feature vector normalizes to 0 instead of dividing by 0 |
| 6 | One score per vocabulary token | logits are `(1, 16, 256)` |
| 7 | Better predictions cost less | boosting the correct logits lowers the loss |
| 8 | The model can memorize a batch | loss 6.228 → 0.023, accuracy 100% |
| 9 | Prediction and appending loop correctly | determinism, context cropping, batch independence, rejection cases |
| 10 | An application can call the model | reloaded weights reproduce the saved loss and the memorized text |

Causality is re-checked at three levels — single head, multi-head with its
output projection, and the full stacked model — by changing the last input token
and asserting every earlier output is bit-identical. It is the property most
easily broken by a reshape, and the one that silently destroys a language model.

Step 8 also checks the NumPy and PyTorch models against each other before
training: logits to `3e-5`, loss to `1e-5`, ∂L/∂logits to `1e-7`, and that every
parameter receives a finite gradient. Two details make that match possible —
PyTorch's GELU is called with `approximate="tanh"` to match the NumPy
approximation, and the Torch LayerNorm uses population variance to match NumPy's
default `ddof=0`.

## Running it

Uses the shared virtual environment at the repository root. Run from this
directory, in this order:

```sh
../.venv/bin/pip install -r requirements.txt

../.venv/bin/python src/step_01_embeddings.py   # steps 1-8, NumPy assertions
../.venv/bin/python src/step_08_train.py        # parity check, then trains
../.venv/bin/python src/step_09_generate.py     # generation loop
../.venv/bin/python src/step_10_app.py          # loads checkpoint, completes text
```

`step_08_train.py` accepts `--device mps` and `--steps`; `step_10_app.py`
accepts `--device`, `--prompt`, and `--max-new-tokens`.

`step_09_generate.py` deliberately runs against a *freshly initialized* model.
Everything it checks is a property of the loop, not of the weights, so it works
before a checkpoint exists.

Training writes to `artifacts/step08/` — a checkpoint plus the tokenizer's
`vocab.json` and `merges.txt`. That directory is gitignored; rerun
`step_08_train.py` to regenerate it.

## HTTP API

`src/api.py` serves the trained model. It is a delivery surface, not another
step: every request is one `MiniLLMApp.complete()` call — tokenizer, model,
tokenizer — with validation in front and JSON around it.

```sh
../.venv/bin/python src/api.py            # http://127.0.0.1:8000
```

```sh
curl -s localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "Transformers", "max_new_tokens": 15}'
```

```json
{
  "prompt": "Transformers",
  "completion": " learrrrrrrrrrr",
  "text": "Transformers learrrrrrrrrrr",
  "prompt_tokens": 12,
  "generated_tokens": 15,
  "elapsed_ms": 6.35
}
```

`GET /health` reports the loaded device, vocabulary, context window, parameter
count, and the training loss the checkpoint was saved at. `GET /docs` is
generated from the request models.

Request fields: `prompt`, `max_new_tokens` (1–256), `temperature` (0–5, where 0
is greedy and reproducible), `top_k`, and `seed` for repeatable sampling.
Out-of-range values are rejected with 422 before the model runs; a prompt that
encodes to zero tokens, or a `top_k` above the vocabulary size, returns 400.

Two behaviors worth knowing. The checkpoint is loaded and verified **once at
startup**, so a missing checkpoint stops the service from starting rather than
failing the first request. And generation is serialized behind a lock —
endpoints are synchronous, so FastAPI runs them in a threadpool, and the
generation loop briefly switches the shared module into eval mode.

Configuration is environment-based, so there are no flags to keep in sync with
Step 10: `MINI_LLM_ARTIFACTS`, `MINI_LLM_DEVICE`, `MINI_LLM_HOST`,
`MINI_LLM_PORT`.

Because the model memorized one 16-token window, sampling barely matters at
normal temperatures — the top token carries about 95% of the probability mass,
so different seeds usually produce the same output. Raise the temperature
toward 5 and seeds diverge.

## Files

| File | Purpose |
| --- | --- |
| `src/order.md` | The ten-step build order and its acceptance criteria |
| `src/positions.py` | Sinusoidal positional encoding |
| `src/step_01_embeddings.py` | Steps 1–8 end to end, with every assertion |
| `src/step_02_attention.py` | One causal attention head |
| `src/step_03_multihead.py` | Four heads plus the output projection |
| `src/step_04_feedforward.py` | GELU feed-forward network |
| `src/step_05_transformer_block.py` | LayerNorm and one pre-norm block |
| `src/step_06_language_model.py` | Stacked blocks and the vocabulary projection |
| `src/step_07_loss.py` | Stable cross-entropy |
| `src/step_08_gradients.py` | ∂L/∂logits, derived by hand |
| `src/step_08_torch_model.py` | The same architecture in PyTorch, weights copied |
| `src/step_08_train.py` | Parity check, training loop, checkpoint |
| `src/step_09_generate.py` | Autoregressive generation |
| `src/step_10_app.py` | Checkpoint loading and the application API |
| `src/api.py` | FastAPI service wrapping the application API |
| `src/bpe/` | Vendored byte-level BPE tokenizer — see `src/bpe/VENDORED.md` |

## Scope

Two limits are deliberate, and both follow from step 8's criterion being
"memorize a tiny training batch":

**The model is trained on one 16-token sequence.** It reaches 100% accuracy on
that window in 25 steps and reproduces it exactly during generation. Ask for
more than 16 tokens and the output degrades into repetition. That is the correct
behavior for a model that has seen one example, not a bug to chase. Training on
real text needs a dataset, batching, and a validation split — none of which
exist here.

**The tokenizer is untrained.** `bpe_trainer.py` is vendored but never called,
so `merges.txt` holds only a version header and the vocabulary is the 256 raw
bytes. Every byte is its own token; nothing is merged into subwords. The
tokenizer works correctly in this mode, and the model is indifferent to it —
training merges would change `vocab_size` and nothing else.

Backpropagation through the blocks is autograd's, not hand-derived. Step 8 asks
that the model train and memorize a batch, and that is what the PyTorch path
delivers; the hand-derived gradient stops at the logits.

## Scaling up

The architecture here is complete. What separates it from a usable model is the
"large" in Large Language Model, plus training — numbers to turn up and data to
feed in, not components to invent.

The clearest evidence is a parameter count. Every weight in this model falls out
of four numbers:

```
params = 2·V·d + V + L·(12·d² + 9·d) + 2·d
```

which reproduces this model's 132,608 exactly. Evaluated at GPT-2 small's
dimensions it gives **162,264,145** — and 123,616,512 once the embedding and
output matrices are tied, against GPT-2 small's actual ~124M. This architecture,
given GPT-2's numbers, *is* GPT-2 small. The gap between those two figures names
the single structural difference: GPT-2 reuses its embedding table as the output
projection, while `token_embedding` and `W_vocab` here are separate.

### What each dimension costs

| Config | Parameters | vs. this model |
| --- | ---: | ---: |
| `V=256, d=64, L=2` (current) | 132,608 | 1× |
| `V=256, d=128, L=2` | 461,568 | 3× |
| `V=256, d=64, L=8` | 430,976 | 3× |
| `V=256, d=128, L=8` | 1,648,128 | 12× |
| `V=8000, d=256, L=6` | 8,836,928 | 67× |
| `V=50257, d=768, L=12` | 162,264,145 | 1224× |

Width is the expensive dimension: attention matrices are `d × d` and the
feed-forward is `d × 4d`, so parameters grow with `d²`. Depth is linear. Context
length costs no parameters at all — the positions are fixed sinusoids — but
attention is O(T²) in compute, so a 16× longer window is 256× the attention work.

### Three tiers of work

**Config only.** `d_model`, `num_heads`, `num_layers`, and `max_seq_len` are
entries in the `config` dict in `step_08_train.py`. `MiniLLM` already loops over
its blocks and sizes the positional encoding from `max_seq_len`, so nothing else
changes. Two constraints the code enforces: `d_model` must be even, and
divisible by `num_heads`.

**Train the tokenizer.** `src/bpe/bpe_trainer.py` is vendored and ready but never
called. Training merges on a corpus raises `vocab_size` above 256 and packs more
text into the same context window — `"Transformers"` is 12 byte tokens today and
would become one or two. `vocab_size` flows into the config from
`tokenizer.vocab_size` automatically, so the model needs no edit.

**Write a real training loop.** This is the only genuinely missing code. The
current loop feeds one fixed batch forever; real training needs a dataset, random
batch sampling, a train/validation split, and a learning-rate schedule with
warmup. Note that `step_08_train.py` raises when it fails to reach 100% accuracy
at loss < 0.05 — a correct check for memorizing a batch, and one that will always
fire on real data, where neither is reachable or desirable.

### Order

Data first, then vocabulary, then size. Scaling the model without more data makes
results worse, not better: a larger model on 16 tokens simply memorizes them
faster. The step after that is a validation split, because once the model can no
longer memorize the training set, the training loss stops telling you whether it
is learning.
