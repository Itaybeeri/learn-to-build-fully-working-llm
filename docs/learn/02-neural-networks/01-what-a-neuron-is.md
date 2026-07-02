# What a neuron is — the weighted sum that makes a decision

## In one sentence

A **neuron** is the tiny unit a network is built from: it takes several inputs, multiplies
each by a **weight** (how much that input matters), and sums them into **one score** — the
exact dot product you already know — then passes that score through a small shaping step.

## Why it matters

Everything in a neural network — every layer, the whole transformer — is just neurons, stacked
and repeated. A neuron is where the abstract pieces finally meet: **vectors** (the inputs),
the **dot product** (the weighted sum), **weights** (the learned dials), and soon **gradients**
(which tune those weights) and **probability** (what the final scores become). Understand one
neuron and the rest of the network is that idea copied many times.

## The intuition

### Part 1 — a neuron is a weighted sum

A neuron takes several **inputs**, decides **how much each matters** (a **weight** per input),
and combines them into **one output number** by multiplying matching slots and adding them up
— a **dot product**. Toy "should I go outside?" neuron:

```
 inputs  x = [ temperature=22, is_raining=0, is_weekend=1 ]
 weights w = [    0.3,            -5,            2         ]

 score = (22 × 0.3) + (0 × −5) + (1 × 2) = 6.6 + 0 + 2 = 8.6
```

The weights are the neuron's **opinion** (warmth helps a little, rain kills it, weekend helps);
the dot product **applies** that opinion to a situation. The output `8.6` is the neuron's
**score** — how strongly the inputs argue "yes." Make it rain (`is_raining = 1`) and the score
falls to `3.6`. A matrix is just **a stack of neurons** — each row one neuron with its own
weights.

You can also read *why*: the per-slot products (`+6.6`, `0`, `+2`) show which input drove the
verdict.

### Where inputs and weights come from (they differ — this is the key idea)

- **Inputs** come from the **data** — a description of the situation, written as numbers
  (yes/no facts become `1`/`0`). They are given to the neuron from outside and **change every
  example**. In a real network the first layer's inputs are the actual data (e.g. a word's
  embedding vector).
- **Weights** are the neuron's own **tunable dials** — the same `weights/parameters` from
  Module 0. Nobody sets them by hand; they start **random** and **training** (the
  guess → error → gradient → nudge loop) drifts them toward good values. They are **shared
  across every example** and only change **when training nudges them**.

```
 INPUTS  → from DATA; change per example; the question being judged.
 WEIGHTS → the model's knobs; shared across examples; change per nudge; what the model "knows".
```

Two different clocks: inputs change per *example*, weights change per *nudge*. Training is the
process of getting the weights right.

### Part 2 — the activation function (the shaping "switch")

The weighted sum gives a raw **score** that can be anything (`8.6`, `−2`, `−500`). After the
sum, a neuron passes that score through a small fixed **activation function** that bends it
into a more useful shape. It earns its keep **two** ways:

**1. Fences (interpretability).** A raw score has no ceiling, so you can't read it — `1`, `500`,
and `1,000,000` are all just "somewhere on the line," with no marker for where *no* becomes
*maybe* becomes *yes*. The activation builds those fences. The bluntest version is a hard light
switch (the historical **step function**): `score ≥ 0 → 1` (fire), else `0` (silent) — a neuron
either **fires** or stays **silent**, like a brain cell crossing a threshold. But a switch is
too blunt (`0.001` and `500` both become `1`), so we usually want a **dimmer** that slides
smoothly from "definitely no" through "unsure" to "definitely yes."

**2. Bend (the real reason — it keeps depth from collapsing).** A weighted sum is "multiply and
add" — the technical word is **linear**. The key fact: *a chain of linear steps is still just one
linear step.* Tiny proof, one input `x`, two plain layers:

```
 layer1 = 3 × x
 layer2 = 2 × (layer1) = 2 × (3 × x) = 6 × x      ← just ONE multiply, by 6
```

Two layers folded into one. So a 100-layer network of *plain* weighted sums has the same power
as 1 layer — all the depth is wasted, and it can only draw straight-line relationships (never
something bumpy like "warm is good, but *too* hot is bad again"). The activation is a **bend**
placed between layers (e.g. "negatives become 0", or "squash smoothly toward 0/1"). Because a
bend is **not** linear, `2 × bend(3 × x)` will **not** fold flat — depth finally buys power.
Without activations, deep learning wouldn't exist.

**The sigmoid — a smooth 0-to-1 dimmer.** The most famous activation, `σ(z) = 1/(1 + e^{−z})`,
takes any score `z` and slides it into the range 0 to 1 along an S-curve: very negative → ~0,
exactly 0 → 0.5, very positive → ~1. (Here `e` ≈ 2.718 is a fixed constant; the `e^{−z}` engine
is tiny when `z` is big-positive, so `1/(1+tiny) ≈ 1`, and huge when `z` is big-negative, so
`1/(1+huge) ≈ 0`.) A number between 0 and 1 *is* a probability, so the sigmoid lets a neuron say
*"I'm 0.88 confident it's yes"* — connecting straight back to
[probability](../01-foundations/04-probability-basics.md).

**What makes a bend "non-linear" (the subtle part).** *Linear* means **one single straight line**
describes the whole function — the same step everywhere. Non-linearity is a property of the
*whole* shape, not of a piece of it:

- **ReLU** (negative → 0, else unchanged) has two straight pieces glued at a **corner**. Each
  piece is straight, but no single ruler fits both, so the whole is non-linear. Concretely:
  score `5→6` raises output by 1, but score `−5→−4` raises it by 0 — *same +1, different effect*.
  That non-uniformity is the bend. ReLU is the simplest and most-used activation.
- **Sigmoid** has no corner but is **steep in the middle, flat at the edges**: adding 1 to the
  score moves the output a lot near 0 but barely near the extremes (it's pinned to the 0/1
  ceiling and has nowhere to climb). Same action, different effect across regions → non-linear.

Either way the lesson is the same: a bend is **anywhere the relationship changes**, and that is
exactly what a straight line is forbidden to do.

## A picture in words

```
   inputs ──► [ weighted sum ] ──► raw score ──► [ activation ] ──► output
              (dot product,        (any number)   (fences + bend)   (bounded,
               Part 1)                                               readable)
```

Straight line (no activation) vs. the two bends:

```
 plain line        ReLU                sigmoid
   |    /            |    /              1|     ___---
   |  /              |   /                |   _-
   |/                |__/               0 |_--
  (climbs forever)  (flat, then climbs) (flat → steep → flat)
```

## Check yourself

- A neuron = **weighted sum** (one raw score) **+ activation** (shaping). Why both?
- Give the **two** jobs of an activation function. (Fences = readable range; bend = stops layers
  collapsing and lets the net learn curves.)
- Why would a deep stack of *plain* weighted sums (no activation) have the same power as a
  single layer? (A chain of linear steps is one linear step.)
- ReLU's right side is a straight line — so why is ReLU non-linear? (Linearity is about the
  *whole* function; the corner means the same +1 to the score has different effects in
  different regions.)

## Questions we worked through *(so far)*

- **Q: Compute the neuron's score for `x = [10, 1, 0]` with `w = [0.3, −5, 2]`; better or worse
  than 8.6, and which input hurt most?**
  A: `(10×0.3)+(1×−5)+(0×2) = 3 − 5 + 0 = −2.0`. Much worse (now negative) — rain (`−5`) hurt
  most.
- **Q: Where do the inputs `[10,1,0]` and weights `[0.3,−5,2]` come from?**
  A: Inputs describe the situation and come from the data; weights are learned knobs — faked
  here for illustration, but set by training in a real model.
- **Q: Moving from a rainy example to a sunny one, which changes — inputs or weights — and
  which is training trying to get right?**
  A: Inputs change (per example); weights stay (shared), changing only when training nudges
  them. Training is trying to get the weights right.
- **Q: Why can't a neuron just output its raw weighted-sum score?**
  A: The score is unbounded — `1`, `500`, `1,000,000` all sit on an endless line with no marker
  for where *no* becomes *maybe* becomes *yes*. (That's reason 1, fences. Reason 2 is the bend.)
- **Q: Does stacking plain weighted sums make the system more powerful, or stay the same kind of
  thing — and why?**
  A: Same kind. A chain of linear (multiply-and-add) steps simplifies to a single linear step
  (`2 × (3 × x) = 6 × x`), so 100 plain layers = 1 layer's power. The activation's bend prevents
  this collapse.
- **Q: For a large positive score, what does the sigmoid output and what does it mean?**
  A: ~1 (strong "yes"/high confidence). The `e^{−z}` term → ~0, so `1/(1+0) ≈ 1`.
- **Q: ReLU's positive side is a straight line — what makes it non-linear?**
  A: Linearity is a property of the *whole* function; ReLU's corner means the same +1 to the
  score raises the output by 1 on the right but by 0 on the left — different effects in different
  regions, which a straight line can't do.
- **Q: The sigmoid has no corner — what makes *it* non-linear?**
  A: Its steepness changes: +1 to the score moves the output a lot in the steep middle but barely
  at the flat edges. Same action, different effect → non-linear.

## What's next / depends on

- **Depends on:** [Matrices & matrix multiplication](../01-foundations/02-matrices-and-matrix-multiplication.md)
  (the dot product / weighted sum), [Functions & gradients](../01-foundations/03-functions-and-gradients.md)
  (how weights get tuned), [How an LLM learns](../00-orientation/02-how-llms-learn-and-where-knowledge-lives.md)
  (weights = the dials).
- **Next:** *Layers & the forward pass* — stack these neurons into a layer (a matrix of
  weights) and watch data flow through, layer to layer, with the bend protecting the depth.
