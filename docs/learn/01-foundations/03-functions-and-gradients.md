# Functions & gradients — the slope that teaches a model

## In one sentence

A **function** is a machine that turns an input into an output, and a **gradient** is the
**slope** of that machine — the one number that tells a model which way to nudge its knobs to
become less wrong.

## Why it matters

A matrix transforms a vector, but it's full of numbers — **weights** — and at first they're
random and the model is wrong. Learning is the act of **nudging those weights** until the
output is good. To nudge them intelligently you need two things: a way to **measure how wrong
you are** (a function from "knob setting" to "error"), and a way to know **which direction
lowers that error** (the gradient). Every neural network on earth learns by this exact loop:
measure the slope, take a small step opposite it, repeat. Master it here on one knob and the
billion-knob version is the same idea copied a billion times.

## The intuition

### A function is a machine: in → rule → out

A **function** is just a box: put something in, a rule transforms it, something comes out.

```
   input  →  [ rule ]  →  output
     3     →  "double it"  →   6
```

We write it `f(x) = 2x`, read "f of x equals two x":
- **`f`** — the machine's name (just a label).
- **`x`** — the **input**, a stand-in for "whatever you feed in."
- **`f(x)`** — the **output** for that input (`f(3) = 6`).
- **`2x`** — the **rule** (2 times `x`).

"Multiply a vector by matrix `M`" is also a function — vector in, vector out. Same shape
always: **in → rule → out**. The mental image to carry forward: a function turns a **knob**
(the input) and shows a **readout** (the output).

### The function that matters: a "how wrong are we?" meter

Give the model one knob `w` (a *weight*). Build a machine whose input is the knob and whose
output is **how wrong the model is** at that setting — the **error** (a.k.a. **loss**). Small
output = good, big output = bad. Toy example:

```
error(w) = (w − 3)²
```

- **`w`** — the knob setting we choose.
- **`error(w)`** — how wrong we are there.
- **`(w - 3)`** — how far `w` is from the secretly-ideal value `3` (the model does *not* know
  this number — finding it is the whole point).
- **`²`** ("squared") — multiply the thing by itself. It makes the result never negative
  (missing by −2 or +2 is equally bad) and punishes big misses harder.

Turn the knob and read the meter:

```
 w = 0 → 9     w = 1 → 4     w = 2 → 1     w = 3 → 0  ← perfect
 w = 4 → 1     w = 5 → 4
```

Those points trace a **valley** (a U-shape) with its bottom at `w = 3`:

```
 error
   9 |  *                          *
   4 |     *                    *
   1 |        *              *
   0 |____________*_______________  w
     0   1   2   3   4   5
                 ↑ lowest error
```

**Learning = walk to the bottom of the valley.** The model starts somewhere random and must
step downhill. But it's "in fog" — it can only feel the ground right under its feet. How does
it know which way is down?

### Slope: the whole idea of a gradient

**Slope** answers: *if I nudge the input a tiny bit, how much does the output change, and
which way?*

```
slope = (change in output) / (change in input)
```

"Increase the input `w`" = "walk right." Tilts **up** going right → slope **positive**; tilts
**down** → slope **negative**; flat → **zero**. Measure it at `w = 5` with a tiny step right:

```
 w = 5.0 → error = 4.00
 w = 5.1 → error = 4.41
 slope = (4.41 − 4.00) / (5.1 − 5.0) = 0.41 / 0.1 = +4.1   (positive)
```

Positive slope means stepping right makes error go **up**, so downhill is the **opposite**:
step left, toward smaller `w`. The master rule of all learning:

> **To go downhill, step in the direction *opposite* the slope.**
> Positive slope → decrease the knob. Negative slope → increase the knob.

The slope's **size** is a bonus: big (`4.1`) = steep = far from the bottom = take a big step;
near the bottom the valley flattens, slope → 0, steps shrink, and you coast to a stop at the
lowest point. The slope is a **compass needle that always points uphill** — you just walk the
other way.

**Gradient** is the proper name for "slope" once a function has many inputs (a real model has
*billions* of knobs). For one knob, gradient = slope. Same idea, fancier word.

### The update rule: turning a slope into a step

Repeat one tiny rule:

```
w_new = w_old − (step size) × (slope)
```

- **`w_old` / `w_new`** — the knob before / after the step.
- **`slope`** — the gradient (points uphill).
- **minus sign** — bakes in "walk opposite the slope"; subtracting moves you downhill on
  *either* wall automatically.
- **`step size`** — a small number we pick (e.g. `0.1`), the **learning rate**. Too big →
  overshoot the valley; too small → crawl forever.

Walking from `w = 1` with step size `0.1` (the exact slope of this valley is `2(w−3)`):

```
 w=1.00, slope=−4.0 → w = 1.00 − 0.1·(−4.0) = 1.40
 w=1.40, slope=−3.2 → w = 1.72
 w=1.72, slope=−2.56 → w = 1.98 ... 2.18 ... 2.35 ... → creeps toward 3
```

As the knob nears `3` the slope shrinks, steps shrink, and it settles at the bottom. **That
loop — measure slope, take a small opposite step, repeat — is literally how every neural
network learns**, just with billions of knobs at once, each feeling its own slope. This is
called **gradient descent**.

## The gentle math

- **Function notation:** `f(x) = rule`. `f` names the machine, `x` is the input, `f(x)` is the
  output. `error(w) = (w−3)²` maps a knob setting to how wrong we are.
- **Slope (one input):** `slope = (change in output) / (change in input)`, measured by a tiny
  nudge. Sign = direction the function rises; magnitude = steepness.
- **Gradient:** the same notion of slope when there are many inputs — one slope per knob, each
  saying how *that* knob affects the output. For one knob, gradient = slope.
- **Gradient-descent update:** `w_new = w_old − (learning rate) × (slope)`. The minus = go
  downhill; the learning rate = how big a step.

## A picture in words

```
  KNOB w  →  [ error(w) = how wrong ]  →  readout (error)
                       |
              measure the SLOPE here (tiny nudge)
                       |
        slope > 0 ? step knob DOWN : step knob UP
                       |
              w_new = w_old − rate × slope
                       |
                  repeat → slope shrinks → settle at valley bottom
```

## Questions we worked through

- **Q: `f(x) = x + 10` — what's the rule, and what is `f(4)`?**
  A: Add 10 to the input; `f(4) = 14`.
- **Q: At `w = 5` in the valley, which way should the model step, and how can it know from
  only feeling the ground under its feet?**
  A: Step toward smaller `w`. It can know from the local **slope** — probing a tiny step shows
  which way the ground tilts; no need to see the whole valley.
- **Q: Compute the slope at `w = 1` (nudge to `1.1`); positive or negative; which way to step?**
  A: error goes 4 → 3.61, slope = −0.39/0.1 = **−3.9** (negative) → step toward **bigger `w`**.
- **Q: Why are the slopes at `w=1` and `w=5` equal size but opposite sign?**
  A: They're on opposite walls of the same valley, both sloping down toward the bottom at `w=3`.

## Check yourself

1. **What is a function, in the "machine" picture?**
   → A box that takes an input, applies a rule, and returns an output (`in → rule → out`).
2. **What does an error/loss function take in and give out, and is small good or bad?**
   → In: a knob (weight) setting. Out: how wrong the model is. Small = good.
3. **What does the slope/gradient tell you, and how do you use it to improve?**
   → Which way (and how steeply) the error changes as you nudge a knob; step the *opposite*
   way to lower the error.
4. **Write the update rule and say what the minus sign and the learning rate each do.**
   → `w_new = w_old − rate × slope`. Minus = walk downhill (opposite the slope); learning rate
   = how big each step is.

## What's next / depends on

- **Depends on:** [Matrices & matrix multiplication](02-matrices-and-matrix-multiplication.md)
  (the weights we're nudging live in matrices) and
  [Numbers & vectors](01-numbers-and-vectors.md).
- **Next:** *Probability basics* — the model's output isn't one number but a spread of
  *likelihoods* over every possible next word. To measure "how wrong" that spread is (the
  error function we just used), we first need to understand probability.
