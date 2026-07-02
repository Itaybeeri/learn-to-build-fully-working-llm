# Probability basics — likelihood, and how we score a guess

## In one sentence

A **probability** is a number from 0 to 1 saying how likely something is, and a model's
output is a whole **distribution** of them — one per possible next word, all summing to 1 —
which is exactly what lets us measure "how wrong" a guess was.

## Why it matters

The error function from the last note needed a number for "how wrong is the model?" That
number comes from **probability**. A real LLM never outputs one word; it outputs a confidence
score for *every* possible next word. To understand training at all, you need to read those
scores: what a single probability means, why a spread of them must add to 1, and how the
probability placed on the *true* word becomes the error that gradient descent then shrinks.
This is the last math foundation before we build actual neural networks.

## The intuition

### Part 1 — what a single probability is (the 0–1 scale)

A **probability** answers "how likely?" with one number on a fixed scale from **0 to 1**:

```
 0 ─────────────── 0.5 ─────────────── 1
 impossible      coin-flip          certain
```

- **0** = never happens. **1** = always happens. **0.5** = half the time.
- Anything between = partial likelihood (`0.7` = "7 times out of 10").

We write it with a capital **P** and the event in parentheses:

```
P(heads) = 0.5          "the probability of heads is 0.5"
P(rain)  = 0.3
```

- **P** — the label "probability of."
- **in parentheses** — the event (heads, rain, the next word being *Paris*).
- **the number** — the likelihood, always between 0 and 1, no exceptions. `1.5` ("more than
  certain") and `−0.2` ("less than impossible") are both nonsense.

Two readings, both useful: a **fraction of times** ("0.25 = 1 in 4") and a **degree of
belief** ("I'm 25% sure"). The model uses belief — it's never certain, just more or less
confident. (Percentages are the same number ×100: `0.25 = 25%`.)

### Part 2 — a probability distribution (sums to 1)

A model faces a question with **many** answers ("what's the next word?"), so it outputs not
one probability but a **list** of them, one per option — a **probability distribution**.
Sentence *"The capital of France is ___"* with four pretend candidates:

```
 Paris   0.70
 London  0.10
 Berlin  0.05
 cheese  0.15
```

Picture it as one bar of length 1 cut into slices:

```
 |■■■■■■■■■■■■■■■■■■■■■■■■■■■■|■■■■|■■|■■■■■■|
 |        Paris 0.70        |Lon |Be| cheese|
 └──────────────── total = 1 ───────────────┘
```

**The iron rule: all the probabilities add up to exactly 1.**

```
 0.70 + 0.10 + 0.05 + 0.15 = 1.00  ✓
```

Why 1? Because *something* must come next — one option **will** happen, so the full bar
(length 1 = certainty) is divided among the candidates. They compete for a **fixed budget**:
if one slice grows, the others must shrink to keep the total at 1. That is literally an LLM's
output at every step — a confidence score over its whole vocabulary, summing to 1.

### Part 3 — probability is how we score "how wrong"

Probability is what *makes* the error number from the gradients note. The rule:

> **The model is scored by how much probability it put on the word that actually came next.**

True next word is *Paris*:

```
 Good model:  P(Paris) = 0.70  → lots of belief on the truth → LOW error
 Bad model:   P(Paris) = 0.02  → almost none on the truth    → HIGH error
```

So "how wrong" = "how little probability you gave the correct answer." Training nudges the
weights to push the true word's probability **up** — and because of the fixed budget, the
wrong words automatically go **down**. The full loop, every piece now understood from zero:

```
 weights → distribution over next words → probability on the TRUE word
        → error (low if that prob is high) → gradient → nudge weights → repeat
```

### Why we nudge instead of just setting the weights

Knowing the true word tells us **what we want the output to be** — it does **not** tell us
**what weights produce it**. Three reasons we can't just "set" them:

1. **No dial says "P(4) after 2+2."** The answer comes out of *billions* of weights multiplied
   and added together; it's smeared across all of them, not stored in one place.
2. **The same weights serve every example.** The weights handling *"2 + 2 ="* also handle
   *"capital of France ="* and a trillion others. There's no setting that's correct for one
   without affecting all — training searches for the compromise that's good across all of them.
3. **No inverse map.** We can't run "desired output → weights" directly. We *can* compute the
   **slope** ("does nudging this weight up raise the true word's probability?"), which is the
   only handle we have. So we step, re-measure, step again.

The split to remember: **the true word gives us the error** (how wrong we are); **the gradient
gives us how to change the weights** — direction and a small step, repeated, not a teleport.

## The gentle math

- **Probability:** `P(event)` = a number in `[0, 1]`. `0` = impossible, `1` = certain.
- **Distribution:** a probability for each possible outcome, with `sum of all = 1`. For next
  outcomes `o1…oN`: `P(o1) + P(o2) + ... + P(oN) = 1`.
- **Fixed budget:** raising one outcome's probability forces the others down (they must still
  total 1).
- **Scoring a guess:** error is low when `P(true word)` is high, and high when it's low. (The
  exact formula — "cross-entropy loss" — comes in Module 5; the idea is all we need now.)

## A picture in words

```
        the question: "what comes next?"
                       |
   model → P(word_1), P(word_2), ... P(word_N)   ← a distribution, sums to 1
                       |
        look at P(the word that TRULY came next)
                       |
        high  → low error   |   low → high error
                       |
            gradient nudges weights to raise P(true word)
                       |
        wrong words shrink (fixed budget = 1) → repeat
```

## Questions we worked through

- **Q: `P(rain) = 0.8` — how likely, and as a percent?**
  A: Very likely; 80% (8 times in 10).
- **Q: Can a probability be `1.5` or `−0.2`?**
  A: No — always between 0 and 1; outside that is meaningless.
- **Q: `coffee 0.6, tea 0.3, rock ?` — what must P(rock) be?**
  A: `0.1`, because the slices must sum to exactly 1.
- **Q: If P(coffee) jumps to 0.9, what happens to the others?**
  A: They shrink — the fixed budget of 1 forces it.
- **Q: If we know the true word, why not instantly set the weights right?**
  A: Knowing the true word gives the *error*, not the weights. There's no per-fact dial, the
  same weights serve every example, and we can't invert "output → weights." The gradient only
  gives a small step's direction; we nudge and repeat.
- **Q: "2+2=4", Model X P(4)=0.95 vs Model Y P(4)=0.05 — which has lower error, and what
  happens to wrong tokens when training raises P(4)?**
  A: Model X (more belief on the truth). The wrong tokens must shrink to keep the total at 1.

## Check yourself

1. **What is a probability, and what range does it live in?**
   → A number saying how likely something is, always between 0 and 1.
2. **What is a probability distribution, and what must its numbers add up to?**
   → A probability for every possible outcome; they sum to exactly 1.
3. **How does probability give us "how wrong" the model is?**
   → Look at the probability it placed on the word that actually came next: high → low error,
   low → high error.
4. **Why must raising the true word's probability lower the others?**
   → Total probability is a fixed budget of 1; one slice can only grow by shrinking the rest.
5. **Knowing the true word gives us ___; the gradient gives us ___.**
   → the error (how wrong we are); how to change the weights (direction + a small step).

## What's next / depends on

- **Depends on:** [Functions & gradients](03-functions-and-gradients.md) (probability supplies
  the error that the gradient shrinks) and
  [What is an LLM?](../00-orientation/01-what-is-an-llm.md) (the model outputs probabilities).
- **Next:** Module 2 — **Neural networks**, starting with *What a neuron is*. We've now got
  every math ingredient (vectors, matrices, gradients, probability); next we assemble them
  into the actual unit a network is built from — and write our first runnable demo.
