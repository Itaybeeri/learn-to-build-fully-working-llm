# The training loop — spinning the four steps until the model learns

## In one sentence

**Training** is just one cycle — forward pass → loss → backprop → update — repeated millions of
times, each turn nudging the weights a hair downhill until the loss stops dropping.

## Why it matters

This is the concept that *assembles* Module 2: every piece you learned (layers, loss, backprop,
gradient descent) is one step of this loop. "Training an LLM" — no matter how huge — literally
means spinning this loop on enormous data. Understand the loop and you understand how every model
on earth got its weights.

## The intuition

### Part 1 — the loop itself

The training loop is the four steps you already met, run over and over:

```
   repeat many times:
     1. FORWARD   — feed a chunk of data through → predictions ŷ
     2. LOSS      — compare ŷ to the true answers → one loss number
     3. BACKWARD  — backprop → the gradient (every weight's slope)
     4. UPDATE    — nudge every weight a little (w − rate × slope)
```

Each step uses one piece of the module: Forward = the **forward pass**, Loss = **cross-entropy**,
Backward = **backprop**, Update = **gradient descent**.

**The key mindset: one turn barely changes anything.** The nudge is `−(rate × slope)`, and the
learning rate is deliberately *small*, so each weight moves by a hair and the loss drops only a
tiny bit per turn. The power is in *repetition* — millions of tiny, stable downhill steps
accumulate into a trained model. (Why not take big steps to go faster? Too large a learning rate
overshoots the valley and the loss bounces or blows up — small and many is stable.)

```
   loss over loop turns:
   high │●
        │ ●●●
        │    ●●●●
        │        ●●●●●●●●
    low │                ●●●●●●●●●●●●●  ← trained: loss has flattened
        └──────────────────────────────── loop turns
```

**When to stop:** when the **loss stops dropping** (it flattens). Near the bottom of the valley
the slope is small, so `nudge = rate × slope` is small, so the loss barely changes turn to turn —
that plateau is the signal the model has learned what it can from the data.

### Part 2 — batches & epochs

Step 1 says "feed a chunk of data through" — but how big a chunk? With, say, a million examples:

- **All million at once, every turn** — most accurate slope, but each turn is enormously slow and
  memory-hungry (very few turns possible).
- **One example at a time** — fast per turn, but each slope rests on a single example: noisy and
  jumpy; one weird example yanks the weights.
- **A small handful (e.g. 32 or 64)** — the sweet spot everyone uses. That handful is a **batch**
  (mini-batch): steady enough (averages out single-example noise) yet small enough to be fast and
  fit in memory.

**Batch = one chunk = one loop turn.** Each turn forwards the batch, averages its loss into one
number, backprops, and updates — then the next turn takes the next batch. (This is why the real
name is **stochastic gradient descent / SGD** — "stochastic" = using a random sample, the batch,
rather than the whole dataset each step.)

**Epoch = one full pass over all the data.** Chop the dataset into batches and feed them through,
one batch per turn; when every example has been seen once, that's one **epoch**. One pass is
rarely enough (tiny nudges), so we run **many epochs** — shuffle, sweep all batches again, and
again — until the loss flattens.

```
   BATCH = how many examples per loop turn   (e.g. 32)   — a small chunk
   EPOCH = one full sweep through the dataset            — many batches

   training =  for each epoch:
                  for each batch:
                      forward → loss → backward → update
```

A full run is **two nested loops**: outer counts epochs (full passes), inner walks batch by batch
(each batch = one update). Both `batch size` and `number of epochs` are **hyperparameters** — you
choose them, training doesn't. (Quick arithmetic: 1,000 examples with batch size 100 → 10 batches
→ **10 updates per epoch**.)

## A picture in words

```
   the whole module, assembled:

   for each epoch (full pass over data):
     for each batch (e.g. 32 examples):
        FORWARD   layers re-express the batch → predictions ŷ
        LOSS      cross-entropy: how wrong, averaged over the batch
        BACKWARD  backprop → the gradient (every weight's slope)
        UPDATE    w ← w − rate × slope   (a hair toward lower loss)
   stop when the loss flattens.
```

## Check yourself

- Name the four steps of one loop turn, and which Module-2 concept each one is.
- Why do we repeat the loop millions of times instead of taking one big step? (Small learning
  rate → each turn moves weights a hair; big steps overshoot the valley.)
- Batch vs. epoch? (Batch = examples per turn; epoch = one full pass over all data.)
- 60,000 examples, batch size 64 — roughly how many updates per epoch? (≈938: 60000 ÷ 64.)
- When do we stop training? (When the loss stops dropping / flattens.)

## Questions we worked through *(so far)*

- **Q: Why isn't one turn of the loop enough to train a model?**
  A: The learning rate is small, so each weight moves only a hair and the loss drops just a tiny
  bit per turn. Training needs millions of these tiny steps to accumulate — repetition is the
  point.
- **Q: How do we know when to stop?**
  A: When the loss stops dropping (flattens). Near the valley bottom the slope is small, so the
  nudges and the loss barely move.
- **Q: Batch vs epoch?**
  A: Batch = how many examples per loop turn (e.g. 32); epoch = one full pass through all the
  data (many batches), after which we shuffle and go again.
- **Q: 1,000 examples, batch size 100 — how many updates per epoch?**
  A: 10 — the data splits into 10 batches, and each batch is one update.

## What's next / depends on

- **Depends on:** [Gradient descent & backprop](04-gradient-descent-and-backprop.md),
  [Loss](03-loss-measuring-wrong.md), [Layers & the forward pass](02-layers-and-the-forward-pass.md).
- **Next:** Module 2's **runnable demo** (a tiny network trained with this exact loop), then
  Module 3 (Language as data).
