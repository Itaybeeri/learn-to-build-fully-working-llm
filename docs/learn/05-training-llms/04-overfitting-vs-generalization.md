# Overfitting vs. generalization

> **Status: complete.** ① the core distinction — training only lowers loss on the *text we show* the
> model, but a high-capacity model can lower that loss two ways: **memorize** the exact examples (brittle) or
> **learn the underlying patterns** (transfers). Low training loss is only a *proxy*; what we actually want is
> performance on **unseen** text. *Overfitting* = great on training data, weak on new data. *Generalization* =
> learning patterns that work on inputs never seen. ② **detection** — split data into a **training set** and a
> held-out **validation set** (never trained on, so it stays a fair stand-in for unseen text); watch both
> losses — overfitting begins when they **split apart** (training loss keeps falling while validation loss
> bottoms out and **rises**). ③ **causes & remedies** — three causes (model capacity too big for the data ·
> too little/too repetitive data · training too long) each met by a remedy (right-size the model ·
> more/cleaner/deduplicated data · **early stopping**), plus **regularization** (dropout, weight decay) that
> actively makes memorizing harder. **Module 5 COMPLETE.**

## In one sentence

**Overfitting** is when a model lowers its training loss by *memorizing* the specific training text instead of
learning patterns that transfer, so it looks great on data it has seen and does poorly on new data;
**generalization** is the opposite — learning patterns that still work on text the model never saw.

## Why it matters

Training does exactly one thing: it pushes the loss down **on the text we show the model**. But that is not
what we actually want. We don't care how well the model predicts text it has already seen — we care how well
it does on **new** input (your next question, a sentence nobody has ever written). So "low training loss" is
only a **proxy** for the real goal, and a proxy that a big model can cheat: with billions of weights it has
enough capacity to literally memorize huge chunks of its training data. Knowing this is the whole reason we
later measure success on held-out text, not on the training set.

## The intuition

### Part 1 — memorizing vs. learning (two ways to lower the same loss)

There are two very different ways a model can drive its training loss down:

- **Memorize.** Build, in effect, a giant lookup table: "I've seen this exact sentence; the next word was
  *Paris*, so I'll say *Paris*." This can drive training loss to nearly zero — but it's brittle. Change one
  word or ask something slightly new and the lookup fails. The model learned the *answers*, not the
  *reasoning*.
- **Generalize.** Learn the underlying **pattern**: "'the capital of X is ___' is answered by X's capital."
  That same pattern works on countries seen rarely, or sentences phrased differently. The model learned a
  *rule that transfers* to new inputs.

Both lower the training loss. **Only the second is what we actually want.**

**The exam-student analogy.** Two students study 50 practice problems:
- *Student A* memorizes the worked answers. Perfect on the practice set; lost on the real exam (new numbers).
- *Student B* learns *how* to solve each type. Maybe a small slip on practice; fine on the exam — the method
  transfers.

Student A is **overfitting** (clinging to specific examples instead of the general method). Student B is
**generalizing**. With billions of weights, an LLM has enough capacity to be Student A if memorizing is the
easy path to low loss — which is exactly the danger.

```
   low TRAINING loss  ──►  could mean MEMORIZED (brittle)   or   GENERALIZED (transfers)
   what we actually want:  low loss on UNSEEN text  ←─ training never measures this directly
```

### Part 2 — detecting it: the train/validation split and the two loss curves

We can't show the model the real world's *future* text, so we fake "unseen" text from data we already have.
**Don't invent — withhold.** Before training, **split** the pile:

- **Training set** — the bulk; the model trains on this (sees it, lowers loss on it, updates weights from it).
- **Validation set** (held-out / dev set) — a chunk we **lock away and never train on**. Because no weight is
  ever updated from it, the model has effectively never seen it — a fair stand-in for unseen, real-world text.

Then during training we watch **two** loss numbers: **training loss** (on the seen text) and **validation
loss** (on the held-out text). The relationship between them is the whole diagnostic:

- **Phase 1 — both fall together.** The model is learning genuine patterns that help on seen *and* unseen text.
  Healthy generalization.
- **The bottom of the validation curve** = the **best** version of the model (as good as it gets on unseen
  text). This is where you'd want to stop.
- **Phase 2 — the curves split.** Training loss keeps dropping (it's now memorizing the exact training text),
  but validation loss bottoms out and **turns back up** — the model is getting *worse* on unseen text. That
  widening **gap** is the signature of overfitting.

```
  HIGH loss ▲
            │ V\                              the GAP = how much it overfit
            │   \  T\                        ╭───┴───╮
            │    \    \         V V V V V V V        ← validation loss CLIMBS (worse on unseen)
            │     V     T\  V V
            │      \   V   ╳ T\                       ╳ = lowest validation loss
            │       T     V    T\                         = BEST model (stop here!)
            │              V      T  T  T  T  T  T  T ← training loss keeps sinking (memorizing)
   LOW loss └────────────────────────────────────────►  training time
              PHASE 1: fall together          PHASE 2: split apart
              (real patterns — helps           (T keeps dropping, V turns
               on seen AND unseen)              back UP = overfitting)
```

**The rule:** watch the **validation** line, not the training line. Training loss falling always *looks* like
progress, but once validation loss starts rising, "progress" has become memorization.

### Part 3 — causes, and the toolbox against overfitting

Overfitting is a **mismatch** between three things: how much the model *can* memorize, how much data it has,
and how hard we push it. Three causes, each with a matched remedy:

| Cause | Why it overfits | Remedy |
|---|---|---|
| **Model capacity too big for the data** | Billions of weights → spare room to memorize specifics instead of finding compact general rules | **Right-size / shrink the model** (less spare capacity) |
| **Too little or too repetitive data** | Not enough varied examples to *force* a general rule; memorizing the few you have is easiest | **More / cleaner / deduplicated data** (ties back to Module 5 ②) |
| **Training too long** | Too many passes (epochs) over the same text → eventually it memorizes it | **Early stopping** — halt at the lowest validation loss, before it climbs |

Plus one active defense that attacks capacity directly:

- **Regularization** — an umbrella term for tricks that make memorizing *harder*, so the model is pushed
  toward general patterns ("putting hurdles in the model so it has to work harder"). Two common ones:
  - **Dropout** — during training, randomly switch off a fraction of neurons each step, so the model can't
    lean on any single memorized pathway and must spread knowledge into robust, general patterns (connects to
    the *distributed representation* idea from ③).
  - **Weight decay** — gently pull every weight toward zero, so a weight only grows when it truly earns it;
    smaller weights → a simpler, smoother function → less memorizing.

**Headline:** overfitting is a *balance* between model capacity, data, and training time — kept in check by
right-sizing the model, more/cleaner data, early stopping, and regularization (dropout, weight decay).

```
   capacity ↔ data ↔ training-time   (keep them in balance)
   too much capacity / too little data / too long → memorize → validation loss rises
```

## Check yourself

- Why isn't "low training loss" the real goal? (It's only a **proxy** — a big model can reach it by
  *memorizing* the exact training text, which doesn't transfer. We actually want good predictions on
  **unseen** text.)
- Define overfitting in one line. (Lowering training loss by memorizing specifics → great on training data,
  weak on new data.)
- Define generalization. (Learning patterns that still work on inputs the model never saw.)
- How do we *detect* overfitting with only one pile of data? (Hold out a **validation set** we never train on;
  watch training vs. validation loss — overfitting begins when they **split** (training keeps falling,
  validation **rises**).)
- Why must the model never train on the validation set? (So it stays genuinely **unseen** — a fair test the
  model couldn't have memorized.)
- Name the three causes of overfitting and a remedy for each. (Capacity too big for the data → right-size the
  model; too little/repetitive data → more/cleaner/deduplicated data; training too long → early stopping.)
- What is regularization? (Tricks that make memorizing harder so the model generalizes — e.g. **dropout**
  (randomly switch off neurons) and **weight decay** (pull weights toward zero).)

## Questions we worked through

- **Q (top rung, ①): If lowering training loss is the only thing training does, why is "low training loss" not
  the goal — what failure are we worried about, and what do we want instead?** A: We want the model to
  understand/reason, not memorize; if it memorizes it aces the exact questions but a slight change makes it
  fail. *(Answered fully; added the vocabulary — training loss is a proxy, the true target is unseen-text
  performance, and the gap "great on training, weak on new" is the definition of overfitting.)*
- **Q (②): With only one pile of text, how can we measure generalizing vs. memorizing?** A: invent/mutate new
  text and test on it. *(Right instinct — test on unseen text — but steered to the standard method: don't
  invent, **withhold** a real validation set. Then nailed the full restatement: split into training +
  validation, never train on validation so it stays unseen, overfitting begins when the two loss curves split
  apart.)*
- **Q (top rung, ③): Name the main causes of overfitting and a remedy for each, and define regularization.**
  A: repetitive/non-unique data → more unique data; too little data for a big model → less spare capacity
  (smaller model); training too long → early stopping; regularization = "putting hurdles in the model so it
  works harder," e.g. lower weights (weight decay) and hiding neurons (dropout). *(All three cause/remedy
  pairs and the regularization definition correct; only tightened "too much data" → "too *repetitive* data.")*

## What's next / depends on

- **Depends on:** [Loss — measuring "wrong"](../02-neural-networks/03-loss-measuring-wrong.md),
  [The training loop](../02-neural-networks/05-the-training-loop.md) (epochs),
  [Width / Depth & Hyperparameters](../02-neural-networks/02-layers-and-the-forward-pass.md) (model capacity),
  [Datasets & scale](02-datasets-and-scale.md) (cleaning/dedup),
  [What "learning" actually adjusts](03-what-learning-adjusts.md) (billions of weights = capacity to memorize;
  distributed representation).
- **This concept is complete — and it finishes Module 5.** Next: **Module 6 — Using & aligning LLMs**
  (inference & sampling, context windows, fine-tuning, instruction tuning & RLHF, limitations). Recommended
  first: a short bridge recap of the pieces Module 6 leans on.
