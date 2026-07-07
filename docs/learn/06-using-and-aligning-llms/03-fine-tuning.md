# Fine-tuning — specializing a pre-trained model

## In one sentence

**Fine-tuning** takes a fully pre-trained, generalist model and trains it a little further on a small, targeted
dataset, nudging its weights to specialize it for a specific job — far cheaper than training from scratch.

## Why it matters

Pre-training a large model costs months of compute and enormous money, and most teams will never do it. What
they *can* do — cheaply — is take an open pre-trained model and **fine-tune** it for their task: a support bot
in their brand voice, a model fluent in medical or legal text, a coding assistant for their codebase.
Fine-tuning is how the broad, expensive generalist becomes a useful, affordable specialist.

## The intuition

### Part 1 — the two-stage idea (pre-train, then specialize)

Everything in Module 5 — trillions of tokens of web text, the n² attention, every weight nudged — is
**pre-training**. It's enormously expensive (months of compute, huge cost) and produces a **broad generalist**:
the model knows grammar, facts, reasoning, a bit of everything.

But "a bit of everything" often isn't what you need. Suppose you want a model excellent at **medical Q&A**, or
that always writes in **your company's support-email style**, or is fluent in **legal contracts**. The
generalist does these passably, not expertly.

Do you train a brand-new model from scratch on medical text? **No** — that costs a fortune *and* wastes effort
relearning grammar and basic reasoning the generalist already has.

**Fine-tuning** instead: take the already-pre-trained model and **train it a little further on a small,
targeted dataset**. You don't start from random weights — you start from a model that already understands
language and just **nudge** it toward your specialty.

```
   STAGE 1: PRE-TRAINING                    STAGE 2: FINE-TUNING
   ┌────────────────────────┐               ┌────────────────────────┐
   │ trillions of tokens    │               │ small targeted dataset  │
   │ (the whole web)        │   ──────►      │ (e.g. medical Q&A)      │
   │ months, $$$ huge       │   start from   │ hours/days, $ small     │
   │ random weights → broad │   this model   │ broad weights → expert  │
   │       GENERALIST       │               │     SPECIALIST          │
   └────────────────────────┘               └────────────────────────┘
```

Fine-tuning is **dramatically cheaper** than pre-training because it **stands on the shoulders** of all the
general knowledge already in the weights. That's why most teams never pre-train — they download an open
pre-trained model and fine-tune it.

The deep reason it works: the pre-trained model has **already learned the hard, general stuff** (language
itself). Specializing is a *small adjustment on top*, not a fresh start.

> **A note on "which weights change."** By default, fine-tuning updates **all** the weights — but only
> **slightly**, over a **small dataset** for a **few steps**. The cheapness comes from (1) starting from an
> already-good point so only a small adjustment is needed, and (2) the tiny dataset + short run — *not* from
> touching fewer weights. (A variant that updates only a small subset of weights — parameter-efficient
> fine-tuning — comes in Part 3.)

### Part 2 — how fine-tuning works mechanically

There's almost **nothing new** here: fine-tuning *is* the Module 2 training loop — **forward pass → measure
loss → backprop → nudge weights**, repeated — run with different settings. Three knobs change:

**1. The data is small and targeted.** Instead of trillions of web tokens, you feed your specialty dataset
(thousands to millions of examples of medical Q&A, support emails, etc.). The task is still next-token
prediction (or a labeled variant); only the *content* is focused on your domain.

**2. The learning rate is small.** The learning rate (Module 1/2) is the **size of each step** when nudging
weights. In fine-tuning you deliberately use a **much smaller** one than pre-training, because the pre-trained
weights are **already good** — you want to *gently* tilt them toward your task, not bulldoze the general
knowledge they hold. Big steps would **overwrite** those carefully-tuned values and wreck what the model
already knew.

```
   pre-training:  random weights, BIG steps   →  learn everything from scratch
   fine-tuning:   good weights,   tiny steps  →  gently lean toward the specialty
```

**3. Far fewer steps.** Pre-training is months; fine-tuning might be hours or a day — you only need a small
adjustment from a strong starting point.

Put together: **same loop, small targeted dataset, small learning rate, few steps.** The weights drift a little
(embeddings, attention matrices, FFN all shift slightly), so outputs lean toward your domain while the language
ability underneath stays intact. The small learning rate + few steps aren't only for speed — they're a **safety
setting**: push too hard and the model loses its general ability (Part 3).

### Part 3 — the trade-offs (catastrophic forgetting & the efficient fix)

Two real problems come with fine-tuning, and one elegant solution.

**Problem 1 — catastrophic forgetting.** Train hard on a narrow dataset and the weights drift so far toward the
specialty that the model **loses general abilities it used to have** — great at legalese, but suddenly poor at
everyday conversation or basic arithmetic. The *cause*: the general knowledge was stored **in the weight values
themselves** (Module 5 — there's no separate backup). When aggressive fine-tuning takes big steps and changes
those values to fit the new data, it **overwrites** them — the old knowledge is written over and lost.
"Catastrophic" because it can be sudden and severe. Defenses (all already met): a **small learning rate** and
**few steps** (gentle nudges overwrite little), a **mixed dataset** (blend general data with the specialty so
the model keeps practicing both), and **early stopping** (stop before it drifts too far).

**Problem 2 — full fine-tuning is still heavy.** Even far cheaper than pre-training, updating **all** the
weights of a large model means storing and adjusting *billions* of numbers — and ten specialties would need
ten *full copies* of the giant model. Expensive in compute and storage.

**The fix — parameter-efficient fine-tuning (PEFT), e.g. LoRA.** Instead of nudging all billions of weights,
**freeze the original model entirely** and train only a **tiny set of new add-on weights** alongside it. The
big model stays untouched; the small add-on learns the specialty.

```
   FULL fine-tuning:   update ALL billions of weights        → 1 full model copy per specialty
   PEFT / LoRA:        FREEZE the big model, train a small
                       add-on (a few million weights)         → tiny file per specialty, swap in/out
```

Two wins:
- **Cheap & small.** Train and store only a few million new weights, not billions — each specialty is a tiny
  file you can swap into the same base model.
- **No catastrophic forgetting.** Because the original weights are **frozen**, they **can't be overwritten** —
  the general knowledge is physically safe. You *add* a specialty on top rather than rewriting the base.

## Check yourself

- What is pre-training vs. fine-tuning? (Pre-training = the giant, expensive run on trillions of tokens →
  broad generalist; fine-tuning = a short, cheap run on a small targeted dataset that specializes it.)
- Why is fine-tuning so much cheaper than training from scratch? (It starts from a model that **already knows
  language, grammar, facts, reasoning** — only a small adjustment on little data is needed, instead of
  relearning everything.)
- By default, does fine-tuning change a few weights or all of them? (All of them — but only **gently**, on a
  small dataset for few steps. The cheapness is the good starting point + small data, not fewer weights.)
- Which settings change versus pre-training? (Same loop, but: **small targeted dataset**, **small learning
  rate**, **far fewer steps**.)
- Why keep the learning rate small? (The weights are **already good**; big steps would **overwrite** the
  general language/knowledge they hold — small steps gently tilt toward the specialty without trashing it.)
- What is catastrophic forgetting and what causes it? (Aggressive fine-tuning **overwrites** the weight values
  that encoded the model's general knowledge — stored *in* those weights, with no backup — so it loses everyday
  abilities it used to have.)
- How does LoRA / PEFT both avoid forgetting and save cost? (It **freezes** the original weights and trains
  only a small add-on of a few million new weights → cheap & storage-light *and* the frozen base **can't be
  overwritten**, so general knowledge stays safe.)

## Questions we worked through

- **Q: Why is fine-tuning so much cheaper than pre-training from scratch, and what does the pre-trained model
  already give you?** A: it's cheaper because it doesn't train a full model with language and grammar — it only
  nudges weights on a small dataset for a small training. *(Correct on the core: the model already knows
  language/grammar/reasoning, so only a small specialization is needed. Sharpened: by default it nudges **all**
  weights, just gently on little data — cheapness = good starting point + small data/short run, not "fewer
  weights.")*
- **Q: Fine-tuning reuses the same training loop — name the settings that change vs. pre-training, and explain
  why the learning rate is kept small.** A: the learning rate stays small so we don't push the model too far
  and make it lose its balance; the settings are small/targeted data, small learning rate, and far fewer steps.
  *(All three settings correct, and the right reason for the small LR. Sharpened "lose its balance" → the big
  steps would **overwrite the general knowledge** already in the weights.)*
- **Q (top rung): (1) What is catastrophic forgetting and what causes it? (2) How does LoRA avoid it and save
  cost vs. full fine-tuning?** A: first passes had the LoRA half solid (freeze most weights, train only a few
  million not billions) and the *symptom* of forgetting ("model loses balance, can't do general predictions"),
  but kept describing the symptom rather than the cause. Stepped down to isolate it: *the general knowledge is
  stored in the weight values — if a weight changes from 0.7 to 0.2, is the old value still anywhere?* → "Gone,
  it was overwritten." Then climbed back: catastrophic forgetting = heavy fine-tuning **overwrites** the
  weights that encoded general knowledge → it fails general tasks; LoRA **freezes** the base (can't be
  overwritten → no forgetting) and trains a tiny add-on (cheap & storage-light). *(Cleared the top unaided
  once the "overwrite" mechanism landed.)*

## What's next / depends on

- **Depends on:** [The training loop](../02-neural-networks/05-the-training-loop.md) (fine-tuning *is* the same
  loop, run briefly on new data), [Datasets & scale](../05-training-llms/02-datasets-and-scale.md) (pre-training
  is the giant run this builds on), [What "learning" adjusts](../05-training-llms/03-what-learning-adjusts.md)
  (the weight groups being nudged).
- **Also leans on:** [Overfitting vs. generalization](../05-training-llms/04-overfitting-vs-generalization.md)
  (early stopping as a forgetting defense).
- **This concept is complete.** Next in Module 6: **Instruction tuning & RLHF (alignment)** — a special,
  important kind of fine-tuning that turns a raw text-predictor into a helpful assistant that follows
  instructions and matches human preferences.
