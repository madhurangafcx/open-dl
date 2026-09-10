Standard Learning Workflow

Every project follows this sequence:

1. Mathematics
   ↓
2. NumPy Implementation
   ↓
3. Understand Forward Pass
   ↓
4. Understand Loss
   ↓
5. Derive Gradients
   ↓
6. Implement Backpropagation
   ↓
7. Train Model
   ↓
8. Debug & Analyze
   ↓
9. PyTorch Implementation
   ↓
10. Compare Results

For every project, document:

What problem is being solved?
What mathematical concepts are required?
What happens during the forward pass?
What loss function is used and why?
How are gradients calculated?
How does backpropagation update parameters?
What causes training failures?
How does the NumPy implementation compare with PyTorch?
What did I actually learn?

What you should know by the end

You should be able to look at:

model = nn.Sequential(
nn.Conv2d(3, 32, 3),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Flatten(),
nn.Linear(32 _ 15 _ 15, 10)
)

and immediately understand:

what every layer receives, what tensor shape it produces, what mathematical operation occurs, and why that operation exists.

Then move to:

ANN
↓
CNN
↓
RNN
↓
LSTM/GRU
↓
Attention
↓
Transformer
↓
LLM architecture
