| Step | Implement                                          | How you know it works                               |
| ---- | -------------------------------------------------- | --------------------------------------------------- |
| 1    | Token IDs, training targets, embeddings, positions | Tensor shapes and shifted targets are correct       |
| 2    | One causal attention head                          | Future positions receive zero attention             |
| 3    | Multiple attention heads                           | Heads combine into the original embedding dimension |
| 4    | Feed-forward network                               | Each position is transformed independently          |
| 5    | Layer normalization and residual connections       | One complete transformer block runs                 |
| 6    | Stack blocks and add vocabulary projection         | Output has one score per vocabulary token           |
| 7    | Cross-entropy loss                                 | Correct predictions produce lower loss              |
| 8    | Backpropagation and training                       | Model can memorize a tiny training batch            |
| 9    | Autoregressive generation                          | Model repeatedly predicts and appends tokens        |
| 10   | Save/load and an application API                   | Your application can call the trained model         |
