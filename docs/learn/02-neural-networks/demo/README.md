# Module 2 demo — a tiny network that learns XOR, from scratch

This is the runnable capstone of Module 2: a 2-input neural network that teaches itself the
**XOR** rule — *"output 1 only when the two inputs differ"* — using nothing but the four steps of
the training loop, written by hand in pure Python (no libraries).

## Run it

```
python xor_from_scratch.py
```

(Any Python 3 works. On Windows you can also use `py xor_from_scratch.py`.)

## What you should see

```
  epoch      0   average loss = 0.8098     <- random weights: basically guessing
  epoch   2000   average loss = 0.0025     <- the loop has driven the weights downhill
  ...
  epoch  20000   average loss = 0.0002     <- loss has flattened: trained

  input [0,0] -> 0.000  (true 0)  [OK]
  input [0,1] -> 1.000  (true 1)  [OK]
  input [1,0] -> 1.000  (true 1)  [OK]
  input [1,1] -> 0.000  (true 0)  [OK]
```

The network starts knowing **nothing** (random dials) and, purely by descending its loss valley,
discovers the XOR rule. The loss falling from ~0.81 to ~0.0002 *is* the training loop working.

## Why XOR?

XOR cannot be solved by a single linear layer — you **need** a hidden layer plus a non-linear
**bend** (the sigmoid). So a working XOR network is living proof of the Module 2 lesson that the
activation's bend is what gives a network real power (see
[`../01-what-a-neuron-is.md`](../01-what-a-neuron-is.md)).

## Where each concept lives in the code

| Concept | In `xor_from_scratch.py` |
|---|---|
| **Neuron** = weighted sum + bias | the `z = b + Σ w·x` lines inside `forward()` |
| **Activation (the bend)** | `sigmoid()` |
| **Layer / forward pass** | hidden layer (2 neurons) → output neuron, in `forward()` |
| **Loss (cross-entropy)** | `loss()` = `−[y·log(p) + (1−y)·log(1−p)]` |
| **Backprop (blame flowing backward)** | `dz_out` handed back to `dz_h` — the baton |
| **Gradient descent (the nudge)** | `W -= LEARNING_RATE * slope` |
| **Training loop (epochs)** | the `for epoch …: for x, y in DATA:` nesting |

**Bias:** one extra always-on number added to each neuron's weighted sum — a neuron's built-in
"lean." A small, natural extension of the weighted sum; XOR needs it.

## Things to try (to build intuition)

- Set `LEARNING_RATE = 5.0` — too big a step; watch the loss bounce or stall (overshooting the
  valley). Then try `0.01` — too small; watch it crawl.
- Remove the hidden layer (predict straight from inputs) — it can **never** learn XOR, because
  there's no bend. This is the collapse lesson, made real.
- Change `random.seed(1)` to another number — different random start, same final success.

## The big takeaway

This is **exactly** how an LLM is trained, too: same four steps, same loop. An LLM just has
billions of weights instead of 9, text instead of XOR, and runs the loop over enormous data.
