# Layers & the forward pass — neurons working together, data flowing through

## In one sentence

A **layer** is several neurons side by side that all read the *same* input vector (each with its
own weights) and together turn it into a new vector; the **forward pass** is data flowing through
a stack of such layers, input to output, to produce the network's answer.

## Why it matters

One neuron gives one opinion — not enough. Stacking neurons into **layers**, and layers into a
network, is what gives a model its power. Every model you'll build, including the transformer, is
layers chained together, and **running the model = doing a forward pass**. This is also where the
abstract Module 1 idea "the model's weights are matrices" becomes concrete: *a layer's weights
are exactly one matrix.*

## The intuition

### Part 1 — a layer is a stack of neurons

A **layer** is **several neurons sitting side by side, all reading the same input vector at the
same time** — but each neuron has its *own* weight vector, so each forms its *own* opinion.

```
                   ┌─ neuron A (weights wA) ─► score → activation ─► out_A
   inputs  x  ───► ├─ neuron B (weights wB) ─► score → activation ─► out_B
   (one vector)    └─ neuron C (weights wC) ─► score → activation ─► out_C

   one input vector  →  THREE outputs  →  a new vector [out_A, out_B, out_C]
```

Two key things:

1. **Same input, different weights.** Every neuron sees the *exact same* input `x`; what differs
   is each neuron's weights. Neuron A might ask "does this argue for *cat*?", B for "*dog*?", C
   for "*bird*?" — same evidence, three different questions.
2. **A layer turns a vector into a new vector.** In goes one vector (e.g. 4 numbers), out comes a
   new vector (3 numbers, one per neuron). The layer **re-expresses** the input in terms of the
   questions its neurons ask.

**The whole layer is one matrix × vector, then activation.** Stack each neuron's weights as the
*rows* of a matrix `W`. Then `W × x` is "a stack of dot products" — exactly all the neurons'
weighted sums computed at once. Apply the activation to each entry and you have the layer's
output:

```
        ┌ wA ┐                    ┌ score_A ┐         ┌ out_A ┐
   W =  │ wB │   ,   W × x  =     │ score_B │  ──act──►│ out_B │
        └ wC ┘                    └ score_C ┘         └ out_C ┘
     (matrix =                 (one matrix-vector      (the layer's
   stack of neurons)            multiply does the       output vector)
                                whole layer's sums)
```

So a layer = **`W × x`** (all weighted sums in one shot) **then activation** on each entry. That
matrix `W` *is* the layer's weights.

**The shape rule.** A layer with `3` neurons each taking a `4`-number input is a **3×4 matrix**
(3 rows, 4 columns): it eats a 4-vector, produces a 3-vector. **Rows = neurons = outputs;
columns = inputs.** Each neuron's weight vector has as many numbers as the input (one weight per
input slot, so the dot product lines up).

### Part 2 — the forward pass

Take a layer's output vector and **feed it as the input to the next layer.** Repeat. That chain
— a vector flowing in, getting re-expressed, flowing to the next layer, from raw data to final
answer — is the **forward pass** (a.k.a. forward propagation). It is literally *what running the
model means.*

```
  data        Layer 1            Layer 2            Layer 3        answer
  ┌──┐      W1×x → bend       W2×h1 → bend       W3×h2 → bend     ┌──┐
  │x │ ───►  ┌────────┐  h1  ┌────────┐   h2   ┌────────┐  ───►  │ŷ │
  │  │       │ 4 → 6  │ ───► │ 6 → 6  │  ────►  │ 6 → 2  │        └──┘
  └──┘       └────────┘      └────────┘        └────────┘
            (6 neurons)      (6 neurons)       (2 neurons)
```

- **Input** `x` = the data as a vector.
- Each **hidden layer** output (`h1`, `h2`) is an in-between vector — "hidden" = neither the
  input nor the final answer; it lives inside the network.
- **Final output** `ŷ` ("y-hat") = the model's answer.

Three things to pin down:

1. **Shapes must chain.** Output size of one layer = input size of the next. A layer's matrix is
   **(neurons) × (inputs) = (chosen) × (inherited)**: the *input* count is inherited from the
   previous layer's neuron count; the *output* count is chosen (for the **final** layer, chosen
   to match the number of possible answers — e.g. 3 outputs → a 3-way question).
2. **This is exactly where the activation earns its keep.** Without the bend between layers,
   `W3 × W2 × W1 × x` would collapse into one matrix — the forward pass is literally the place
   the [collapse](01-what-a-neuron-is.md) would happen if we let it.
3. **Each layer re-expresses meaning.** Early layers find simple patterns; later layers combine
   them into complex ones (raw numbers → edges → shapes → "cat vs dog"). That stacking of
   vocabulary is *why depth matters*.

No learning happens in a forward pass — it's just the model **running** to produce a guess.
Learning comes next: measure how wrong `ŷ` is (**loss**), then adjust the weights.

### Width, depth, and hyperparameters (how big to make a layer)

- **Final-layer width is fixed by the task** — 2 answers → 2 neurons, 1,000 words → 1,000
  neurons. Not a quality choice; just match the question.
- **Hidden-layer width is a capacity dial (a tradeoff).** Too few neurons → a **bottleneck**, the
  network **underfits** (too simple to capture the patterns). Too many → slow, needs more data,
  and risks **overfitting** (memorizing training examples instead of learning the general rule).
  The sweet spot **matches the complexity of the problem**.
- **You don't calculate the sweet spot — you search for it.** Hold back a **validation set**
  (examples never trained on) and judge sizes by their validation score: bad even on training →
  widen; great on training but poor on validation → narrow. Start from known-good sizes (often
  powers of 2) and adjust.
- The neuron count is a **hyperparameter** — a setting *you choose before training* (like
  [learning rate](../01-foundations/03-functions-and-gradients.md)), as opposed to **weights**,
  which training *learns*. (Overfitting & validation get full treatment in Module 5.)

## Questions we worked through *(so far)*

- **Q: A layer has 3 neurons, each reading the same 4-number input. How many weights per neuron,
  and why?**
  A: 4 — a weight must pair with each input number for the dot product, so the weight vector
  length matches the input length.
- **Q: How many numbers come out of the layer, and why?**
  A: 3 — each neuron produces one number, and there are 3 neurons, so a 3-number output vector.
- **Q: Why does Layer 3 take 6 inputs but produce 2 outputs?**
  A: Input 6 is *inherited* (Layer 2 had 6 neurons → sent 6 numbers, so each Layer-3 neuron needs
  6 weights); output 2 is *chosen* (I put 2 neurons in, matching a 2-answer task). Its matrix is
  2×6.
- **Q: What does "put 2 neurons in a layer" mean concretely?**
  A: Give the layer's matrix 2 rows — 2 weight vectors — so it outputs 2 numbers and has 2 sets
  of dials for training to tune. A design-time architecture choice; values are learned later.
- **Q: 2 vs 6 neurons — which is better?**
  A: For the *final* layer, neither — match the task. For a *hidden* layer it's a tradeoff: too
  few underfits (bottleneck), too many is wasteful and overfits; best = matches the problem's
  complexity, found by validation.
- **Q: Network is input 5 → L1 (8) → L2 (8) → L3 (3). Output size after each layer, and what
  does the final 3 tell you?**
  A: 8, then 8, then 3. Final size 3 → the task has 3 possible answers (e.g. yes/no/maybe).

## What's next / depends on

- **Depends on:** [What a neuron is](01-what-a-neuron-is.md) (weighted sum + activation),
  [Matrices & matrix multiplication](../01-foundations/02-matrices-and-matrix-multiplication.md)
  (matrix × vector = a stack of dot products).
- **Next:** *Loss — measuring "wrong"*: now that the forward pass produces a guess `ŷ`, we need
  a single number for how far off it is — the bridge to training.
