# Loss — measuring "wrong"

## In one sentence

The **loss** is a single number that compares the model's guess to the true answer and says how
wrong it was — `0` is perfect, bigger is worse — and training is the job of pushing it down.

## Why it matters

A forward pass produces a guess `ŷ`, but a guess alone says nothing about *quality*. The loss
turns "the model ran" into "the model was *this* wrong" — one score we can then minimize. It's
the bridge from running the model to **training** it: you met it in Module 1 as the
[error/loss meter — the valley](../01-foundations/03-functions-and-gradients.md); loss is the
height of the valley, and we want the bottom.

## The intuition

### Part 1 — loss is one number: how wrong a single guess is

A loss function obeys three rules:

- It compares the model's output `ŷ` to the **true answer** `y` (the correct label, known for
  training examples).
- It returns **one number**.
- **Lower = better.** `0` = perfect; the more wrong, the bigger.

```
   true answer  y   ─┐
                     ├──► [ loss function ] ──► one number: how wrong
   model guess   ŷ  ─┘                          (0 = perfect, bigger = worse)
```

**Simplest example — squared error.** Say the model predicts a house price `ŷ = 300` (thousand)
but the truth is `y = 350`. A natural "how wrong" is the gap `ŷ − y = −50`. But a raw gap can be
negative or positive and cancel out, so we **square it**:

```
   loss = (ŷ − y)²  =  (300 − 350)²  =  (−50)²  =  2500
```

Squaring does two things: (1) kills the sign (always ≥ 0), and (2) **punishes big misses
disproportionately**. Example: a miss of 2 → loss 4; a miss of 6 (3× bigger) → loss 36 (**9×**
bigger). So squared error "hates" big mistakes, nudging training to fix the worst errors first.
This is **squared error**; averaged over many examples it's **Mean Squared Error (MSE)**.

### Part 2 — loss for a probability prediction (cross-entropy)

An LLM doesn't predict one number — it outputs a **probability distribution** over the whole
vocabulary (one probability per word, summing to 1), and the truth is *"the actual next word was
`mat`."* So the question is: given that whole table of probabilities, how wrong was it?

**The one idea:** look *only* at the probability the model gave to the **correct** word. High →
low loss; low → high loss. A perfect model would have put all its probability (1.0) on the word
that actually came next.

```
   true next word = "mat"
   model A:  P(mat) = 0.9   →  confident & correct  →  LOW loss
   model B:  P(mat) = 0.1   →  barely believed it   →  HIGH loss
   model C:  P(mat) = 0.01  →  almost ruled it out  →  HUGE loss
```

**The formula (cross-entropy / negative log-likelihood):**

$$\text{loss} = -\log(p_{\text{correct}})$$

- **p_correct** — the single probability the model assigned to the word that was actually
  correct. One number in (0, 1).
- **log** — the logarithm. What it does to numbers in (0,1): turns them negative and dives toward
  −∞ as the input → 0. `log(1)=0`, `log(0.5)≈−0.7`, `log(0.1)≈−2.3`, `log(0.01)≈−4.6`.
- **the minus sign** — flips those negatives to positive, so loss is positive, small near
  p=1, large near p=0.

```
   p_correct = 0.9  → −log(0.9)  = 0.10   (good)
   p_correct = 0.5  → −log(0.5)  = 0.69
   p_correct = 0.1  → −log(0.1)  = 2.30   (bad)
   p_correct = 0.01 → −log(0.01) = 4.61   (confidently wrong — huge)
```

Reward being confident and right; punish being confident and wrong, harshly.

**Why the *other* words still matter (the shared budget).** The loss reads only `p_correct`
directly — a wrong word's probability (e.g. `P(dog)`) never appears in the formula. But all the
probabilities share a fixed budget of 1.0, so pouring probability onto wrong words **steals** it
from the correct word, lowering `p_correct` and *raising* the loss indirectly. That is exactly
the mechanism training exploits: to cut the loss, the model must move probability **off** wrong
words and **onto** the true one.

### Part 3 — one loss for the whole batch (averaging)

Each `−log(p_correct)` scores **one** prediction. Training runs over millions, so we **average**
the per-word losses into a single number:

```
   total loss = average of ( −log(p_correct) ) over all predictions
```

That average is the one number training minimizes. Low average loss = "across all these
examples, the model usually put high probability on the actual next word" = a good language
model. The full chain:

```
   forward pass → probabilities → −log(p on the true word) → average over all words
      (running)     (the guess)        (per-word loss)        (THE number to minimize)
```

## A picture in words

```
   true answer  y   ─┐
                     ├──► [ loss ] ──► one number (0 = perfect, bigger = worse)
   model guess   ŷ  ─┘

   regression (one number):   loss = (ŷ − y)²            ← squared error / MSE
   classification / LLM:       loss = −log(p_correct)     ← cross-entropy
   over a whole batch:         loss = average of the above
```

## Check yourself

- What three rules does any loss obey? (Compares ŷ to y; returns one number; lower = better.)
- Why square the error instead of using the raw gap? (Kills the sign; punishes big misses
  disproportionately.)
- For an LLM, what single thing does cross-entropy look at, and what's the loss when the model
  put probability 1.0 on the correct word? (`p_correct`; loss = −log(1) = 0.)
- A wrong word's probability isn't in the formula — so why does it still hurt to put probability
  on it? (Shared budget of 1.0: it steals from `p_correct`.)
- How do we get one loss number from millions of predictions? (Average the per-word losses.)

## Questions we worked through *(so far)*

- **Q: Model guesses `ŷ = 8`, true `y = 10`. Squared-error loss?**
  A: `(8 − 10)² = 4`.
- **Q: Second guess `ŷ = 4`, true still 10 — bigger or smaller loss, and what does it show?**
  A: Bigger: `(4 − 10)² = 36`. The miss tripled (2→6) but the loss grew 9× — squaring punishes
  big misses far more than small ones.
- **Q: True word `cat`. Model A: P(cat)=0.8, Model B: P(cat)=0.2 — lower loss and why?**
  A: A. Cross-entropy reads only `p_correct`; 0.8 is closer to 1, so `−log(0.8)≈0.22` beats
  `−log(0.2)≈1.6`. (Sign care: log(0.2)≈−1.6, the minus makes loss +1.6 — loss is always
  positive.)
- **Q: Does `P(dog)=0.05` affect this loss? Why/why not?**
  A: Not directly — the formula only reads the probability on the true word (cat). But because
  all probabilities sum to 1, probability on dog steals from cat and raises the loss indirectly.
- **Q: If P(cat) rises 0.2 → 0.6, what happens to the loss and to the leftover budget?**
  A: Loss decreases (closer to 1); leftover for all other words combined = 0.4.

## What's next / depends on

- **Depends on:** [Layers & the forward pass](02-layers-and-the-forward-pass.md) (produces the
  guess `ŷ`), [Functions & gradients](../01-foundations/03-functions-and-gradients.md) (loss is
  the valley to descend), [Probability basics](../01-foundations/04-probability-basics.md).
- **Next:** *Gradient descent & backprop* — now that we have a single loss number, how do we
  find which way to nudge every weight to make it smaller?
