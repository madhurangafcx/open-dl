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

03 — MNIST ANN

Goal: Apply an ANN to a real classification problem.

Learn
Dataset preparation
Normalization
Train/validation/test split
Multiclass classification
Accuracy
Confusion matrix
Implement
28×28 Image
↓
Flatten
↓
ANN
↓
10 Classes
Focus

Understand why flattening an image loses spatial information.
