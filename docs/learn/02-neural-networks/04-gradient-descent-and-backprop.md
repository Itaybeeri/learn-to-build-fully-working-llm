# Gradient descent & backprop — how the weights actually learn

## In one sentence

**Gradient descent** nudges every weight a little *opposite* its slope so the loss drops a bit,
repeated thousands of times; **backpropagation** is the efficient method that finds each weight's
slope by passing "blame" backward from the loss through the network.

## Why it matters

This is how a model actually *learns*. The forward pass produces a guess, the loss scores how
wrong it is — and gradient descent + backprop are the machinery that turns that one loss number
into a concrete adjustment for every one of the millions of weights. Every trained model, every
LLM, got its weights this way.

## The intuition

### Part 1 — gradient descent over many weights

From Module 1: the loss is a **valley**. For a single weight `w`, the **slope** (gradient) says
which way the loss moves as you nudge `w`, and the update rule walks downhill:

```
   w_new = w_old − (learning rate) × slope
```

Subtract the slope → step *downhill* → loss drops → repeat → reach the bottom.

**What's new at network scale:** there isn't one weight but *millions* (every number in every
layer's matrix). So we need a slope for **every weight**: "if I nudge *this* weight a hair, does
the loss rise or fall, and how steeply?" That whole collection of slopes — one per weight — is
the **gradient**. Picture standing on a foggy hillside: the gradient is the arrow pointing
steepest *uphill* across all weights at once, so to cut the loss each weight steps the opposite
way:

```
   for EVERY weight w:
       w_new = w_old − (learning rate) × (that weight's slope)
```

One sweep = one downhill step for the whole network; thousands of sweeps = training.

```
   loss
    |\                         each weight has its own slope;
    | \        ● you are here  the gradient bundles them all.
    |  \      /                step every weight opposite its
    |   \    /                 slope → loss drops → repeat.
    |    \__/  ← bottom (low loss = good model)
    +------------------ weight space
```

**The sign trick (why subtract — and why `w_new` is NOT always smaller).** What we subtract is
`rate × slope`; `rate` is a small positive, but the **slope can be positive or negative**, and
subtracting a negative *adds*. With `w_old = 5`, `rate = 0.1`:

- slope `+20`: `5 − (0.1×20) = 5 − 2 = 3` → weight **down**.
- slope `−20`: `5 − (0.1×−20) = 5 + 2 = 7` → weight **up**.

So the rule doesn't always shrink the weight — it always moves it **toward lower loss** (down or
up). Valley picture: on the left slope, downhill is to the right (increase); on the right slope,
downhill is to the left (decrease); the slope's sign says which side you're on, the minus sign
sends you to the bottom either way. Like a thermostat: if turning the knob up overheats the room
and you want it cooler, turn it down.

**Slope vs. nudge (two different things).** The **slope (gradient)** is a *measurement* — "how
does the loss react if I wiggle this weight?" The **nudge** is the *action* — the actual change
applied, `nudge = −(rate × slope)`. Example: slope `+20`, rate `0.1` → nudge `−2`. The slope is
`+20` (information); the nudge is `−2` (the move). Backprop *measures* (produces the gradient);
the update *acts* (turns each slope into a nudge and applies it).

### Part 2 — backpropagation

**The problem.** Gradient descent needs every weight's slope, but there are millions of weights.
The naive way — nudge one weight, re-run the *whole* forward pass, see how the loss moved — costs
one full forward pass *per weight*: millions of passes per training step. Hopeless.
**Backpropagation** gets *all* the slopes in **one backward sweep** (~the cost of a single
forward pass). It's what made neural networks practical.

**The idea — blame flowing backward.** After a forward pass you have a loss. Backprop asks "who's
to blame, and how much?" and answers by passing blame from the loss *backward* through the
layers:

```
   forward  →→→  data → L1 → L2 → L3 → ŷ → LOSS   (produces the loss)
   backward ←←←  data ← L1 ← L2 ← L3 ← ŷ ← LOSS   (passes blame back)
```

**Why backward — the reused work.** An early weight's effect on the loss travels *through* every
later layer, so you multiply the effects along the path (the **chain rule**):

```
   L1's blame = (L1→L2) × (L2→L3) × (L3→loss)
   L2's blame =           (L2→L3) × (L3→loss)
   L3's blame =                     (L3→loss)
```

The later pieces are **shared** by everything before them, so compute the end pieces first and
reuse them:

```
   STEP 1 (at the end):  L3's blame = (L3→loss)
   STEP 2 (back one):    L2's blame = (L2→L3) × [L3's blame, already known]
   STEP 3 (back one):    L1's blame = (L1→L2) × [L2's blame, already known]
```

Each step reuses the next step's result and adds only its own local piece — cheap. The reuse is
*only* possible going backward: you can't finish an early layer's blame until you know what the
layers after it did. (Whisper-line analogy: to blame the first person for a garbled message, you
first need to know what everyone after them did to it — so start from the last person and walk
back.) Going *forward* you'd recompute the shared later layers over and over — the waste backprop
avoids.

**The full learning cycle:**

```
   1. FORWARD  — run data through → guess ŷ
   2. LOSS     — score how wrong ŷ is (one number)
   3. BACKWARD — backprop sends blame back → the GRADIENT (every weight's slope)
   4. UPDATE   — turn each slope into a nudge → move every weight (w − rate × slope)
   ↑________________ repeat millions of times ________________|
```

Backprop is **step 3** — it produces the *slopes* (not the nudges); step 4 turns them into
nudges and applies them.

## A picture in words

```
   gradient descent: stand in the loss valley, read the slope of every weight,
   step each one opposite its slope → loss drops → repeat → reach the bottom.

   backprop: the efficient way to READ all those slopes at once —
   send the loss's "blame" backward through the layers, reusing downstream
   results, so one backward pass yields every weight's slope.
```

## Check yourself

- Is `w_new` always smaller than `w_old`? (No — depends on the slope's sign; the rule moves the
  weight toward lower loss, up or down.)
- What's the difference between a **slope** and a **nudge**? (Slope = measurement of how loss
  reacts; nudge = the action `−rate×slope` applied to the weight.)
- Why does backprop work **backward**? (Later layers' effects are shared by earlier weights;
  computing them once at the end and reusing them while walking back avoids recomputation — and
  you can't finish an early layer's blame until the later ones are known.)
- Which step is backprop, and what does it produce? (Step 3; the gradient — every weight's
  slope — which the update step then turns into nudges.)

## Questions we worked through *(so far)*

- **Q: What is the gradient of a network, in terms of weights and slopes?**
  A: The bundle of slopes — one per weight — each saying which way (and how steeply) that weight
  pushes the loss. (Not the nudge itself; the nudge is what the update rule does with the slope.)
- **Q: A weight's slope is positive (nudging it up raises the loss). Which way does the update
  move it, and why?**
  A: Down. We want loss to fall, and raising the weight would raise loss — so lower it. The
  `− slope` in `w_new = w_old − rate × slope` does this automatically (subtracting a positive).
- **Q: Why is `w_new` not always smaller than `w_old`?**
  A: We subtract `rate × slope`, and the slope can be negative; subtracting a negative adds, so
  the weight goes up. The rule moves the weight toward lower loss — down *or* up.
- **Q: Slope vs nudge — what's the difference?**
  A: Slope = the measurement (how the loss reacts to a weight); nudge = the action applied,
  `−(rate × slope)`. Backprop produces slopes; the update turns them into nudges.
- **Q: Why does backprop work backward, not forward?**
  A: Later layers' effects are shared by all earlier weights; computing them once at the end and
  reusing them while walking back avoids recomputing the same chains (the forward way redoes the
  later layers for every weight). You also can't finish an early layer's blame until the later
  ones are known.
- **Q: Which step in the cycle is backprop, and what does it produce?**
  A: Step 3 (BACKWARD). It produces the gradient — every weight's slope — which step 4 (UPDATE)
  turns into nudges and applies.

## What's next / depends on

- **Depends on:** [Loss](03-loss-measuring-wrong.md) (the number we descend),
  [Functions & gradients](../01-foundations/03-functions-and-gradients.md) (slope, the update
  rule, learning rate), [Layers & the forward pass](02-layers-and-the-forward-pass.md).
- **Next:** *The training loop* — ties forward pass → loss → backprop → update into one
  repeating cycle, plus batches and epochs.
