# Step 1 — Mathematics (on paper)

Goal: understand exactly how `y = f(Wx + b)` works, before writing any code.

---

## Q1 — Neuron formula for 2 inputs

weighted sum = z = x1*w1 + x2*w2 + b

y = f(z)

---

## Q2 — One example by hand

inputs `[1, 2]`, weights `[0.5, -1]`, bias `0.1`

z = (1*0.5) + (2*-1) + 0.1

z = -1.4

---

## Q3 — From one example to a batch

### Shapes

```
X.shape = (4, 2)
w.shape = (2,)
z.shape = (4,)
```

because (4,2) @ (2,) → (4,)

for (2,) @ (4,2) the dimensions don't match: 2 ≠ 4, multiplication breaks

### Row by row

Weights: w = [0.5, -1]
Bias: b = 0.1

Input rows:

```
[0, 0]
[0, 1]
[1, 0]
[1, 1]
```

Row 1 `[0, 0]`

```
z₁ = (0 * 0.5) + (0 * -1) + 0.1
   = 0 + 0 + 0.1
   = 0.1
```

Row 2 `[0, 1]`

```
z₂ = (0 * 0.5) + (1 * -1) + 0.1
   = 0 - 1 + 0.1
   = -0.9
```

Row 3 `[1, 0]`

```
z₃ = (1 * 0.5) + (0 * -1) + 0.1
   = 0.5 + 0 + 0.1
   = 0.6
```

Row 4 `[1, 1]`

```
z₄ = (1 * 0.5) + (1 * -1) + 0.1
   = 0.5 - 1 + 0.1
   = -0.4
```

$$
Xw = \begin{bmatrix} 0 \\ -1 \\ 0.5 \\ -0.5 \end{bmatrix}
$$

z = Xw + b

really means:

$$
z = \begin{bmatrix} 0 \\ -1 \\ 0.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.1 \\ 0.1 \\ 0.1 \end{bmatrix}
$$

giving:

z = [0.1, −0.9, 0.6, −0.4]

---

## Q4 — Activations

### Identity

$$ f(z) = z $$

At $z = 0$ you get $f(0) = 0$

Range: $(-\infty, \infty)$

### ReLU

$$ f(z) = \max(0, z) $$

At $z = 0$ you get $f(0) = 0$

Examples:

```
z = -5 → 0
z =  2 → 2
```

So its range is: $[0, \infty)$

### Sigmoid

$$ f(z) = \frac{1}{1 + e^{-z}} $$

At $z = 0$:

$$ f(0) = \frac{1}{1 + e^0} $$

Since $e^0 = 1$, then $f(0) = \frac{1}{2}$

Therefore: $\boxed{f(0) = 0.5}$

Its range is: $\boxed{(0, 1)}$

Notice the parentheses. It approaches 0 and 1 but never actually reaches them.

This is why sigmoid can be used to represent a probability.

### tanh

$$ f(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} $$

At $z = 0$: the numerator is $1 - 1 = 0$, so $f(0) = \tanh(0) = 0$

For $z = 2$:

$$ f(2) \approx 0.964 $$

For $z = -2$:

$$ f(-2) \approx -0.964 $$

so it becomes: Range of tanh = $(-1, 1)$

### Which activation gives a probability?

A legal probability must satisfy:

$$ 0 \le p \le 1 $$

The key conclusion:

Only sigmoid guarantees that its output is a valid probability:
`
$$ \text{sigmoid}(z) \in (0, 1) $$

---

## Summary table

| Activation | f(0) | Range   | Valid probability? |
| ---------- | ---- | ------- | ------------------ |
| identity   | 0    | (−∞, ∞) | no                 |
| ReLU       | 0    | [0, ∞)  | no                 |
| sigmoid    | 0.5  | (0, 1)  | yes                |
| tanh       | 0    | (−1, 1) | no                 |
