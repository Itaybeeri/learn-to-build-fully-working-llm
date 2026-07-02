# Multi-head attention — several attentions in parallel, each catching a different relationship

> **Status: complete.** All three parts: ① why one head isn't enough; ② the mechanism (h heads in
> parallel → concatenate → ×W_O); ③ heads specialize via training, capturing many relationships at
> once.

## In one sentence

**Multi-head attention** runs several self-attention computations ("heads") side by side, each with
its own learned Q/K/V, so each head can focus on a *different* kind of relationship between words —
then their results are combined into one vector per word.

## Why it matters

A single self-attention head can only learn *one* notion of "what's relevant." But real sentences
have many relationships at once (a verb to its subject, the same verb to its object, an adjective
to its noun, a pronoun to what it refers to). Multi-head attention is how a Transformer captures
all of those simultaneously — and it's what the "multi-head" in every real model refers to.

## The intuition

### Part 1 — why one head isn't enough

A name for what you already know: **one complete self-attention computation — one set of `W_Q`,
`W_K`, `W_V` — is called one *head*.** Score q·k → softmax → blend values *is a single head*.

A single head learns **one** notion of "what's relevant." Our earlier head learned *"a verb attends
to its subject"* (eat → lion). But even a short sentence has many relationships at once:

```
   "the  big  lion  will  eat  the  meat"
         adjective→noun       eat→lion (verb→subject: who eats)
                              eat→meat (verb→object: what's eaten)
                              big→lion (adjective→noun: describes which)
```

For `eat` to be understood, it wants to look at **both** its subject (`lion`) *and* its object
(`meat`) — two different relationships. And `lion` wants to absorb the adjective `big`. But **one
head has only one set of Q/K/V**, so it can specialize in only **one** "what's relevant" rule at a
time. If it does verb→subject, it isn't also doing verb→object or adjective→noun.

So the fix: **run several attention heads in parallel**, each with its *own* `W_Q/W_K/W_V`, so each
can specialize in a different relationship — then combine what they all found.

### Part 2 — the mechanism: heads in parallel, then combine

A real model has several heads (say 8 or 12). Three moves:

**Move 1 — run the heads in parallel.** Each head is its own full self-attention with its own
`W_Q/W_K/W_V`. For a word, every head independently does score q·k → softmax → blend values, so
**each head produces its own new vector** for that word:
```
   head 1 (learned verb→subject):  eat → absorbs lion  →  vector₁
   head 2 (learned verb→object):   eat → absorbs meat  →  vector₂
   head 3 (...):                   eat → absorbs ...    →  vector₃
```

**Move 2 — concatenate.** Stick those head-vectors together side by side into one long vector:
```
   [ vector₁ | vector₂ | vector₃ | … ]
```

**Move 3 — combine with an output matrix.** Multiply that long vector by one more learned matrix
`W_O` (the *output* matrix), which mixes all heads' findings into one final vector of the normal
size:
```
   [ vector₁ | vector₂ | … ] × W_O  →  final vector  (knows subject AND object AND …)
```

So a word's final vector carries information from **all** the relationships the heads found, not
just one.

```
   one head:    a word learns ONE relationship
   multi-head:  many at once → concatenate → ×W_O → one rich vector
```

(Efficiency note: each head usually works on a *smaller* slice of the vector, so gluing `h` heads
back together lands at the original size — multi-head costs about the same as one big head but is
far more expressive.)

### Part 3 — what the heads specialize in, and why it helps

**Heads are not assigned roles — they *learn* them.** Each head's `W_Q/W_K/W_V` start random and
**training** drifts them toward whatever relationships help predict the next word. Inspecting
trained models, researchers *find* heads that track subjects, the previous word, matching brackets,
pronoun references — all **emergent**, never designed.

**Why it helps:** understanding language *is* juggling many relationships at once. Multi-head lets
the model attend to several in parallel and fuse them, so one word's vector can simultaneously
reflect who did the action, what it acted on, which one (adjectives), and more.

```
   a sentence has MANY relationships at once;
   one head captures ONE; several heads run in PARALLEL and capture many → fused by W_O.
```

## A picture in words

```
   word "eat", embedding x
     ├─ head 1 (W_Q¹/W_K¹/W_V¹):  score q·k → softmax → blend values → vector₁  (subject: lion)
     ├─ head 2 (W_Q²/W_K²/W_V²):  …                                   → vector₂  (object: meat)
     ├─ head 3 …                                                       → vector₃
     ▼
   concatenate [ vector₁ | vector₂ | vector₃ | … ]  ×  W_O  →  one rich final vector for "eat"
```

## Check yourself

- What is a "head"? (One complete self-attention computation — one set of Q/K/V.)
- Why use several heads? (A sentence has many relationships at once; one head captures one, several
  run in parallel and capture many.)
- How do the heads' outputs become one vector? (Concatenate them, then multiply by the learned
  output matrix W_O.)
- Who decides what each head specializes in? (Nobody — it's learned during training; specialties
  emerge.)

## Questions we worked through

- **Q: Why isn't one head (one set of Q/K/V) enough?**
  A: A single set of Q/K/V is tuned for one relationship type (e.g. verb→subject), so it can't also
  capture other relationships (verb→object, adjective→noun) in the same sentence at the same time.
- **Q: For 8 heads, what does each produce for a word, and how do we get one vector out?**
  A: Each head has its own (non-identical) Q/K/V, so each produces a different context-aware vector;
  concatenate all of them into one long vector, then multiply by the learned matrix W_O → one final
  vector.
- **Q: Who decides each head's specialty, and why is multi-head better than one head?**
  A: Training decides (not assigned). Better because a sentence has many relationships at once — one
  head captures one, several heads run in parallel and capture many.

## What's next / depends on

- **Depends on:** [Self-attention (Q/K/V)](02-self-attention-qkv.md) (one head = one set of
  Q/K/V; score q·k → softmax → blend values).
- **Next:** *Positional information* — attention as built so far has no sense of word *order*;
  positional encoding adds it.
