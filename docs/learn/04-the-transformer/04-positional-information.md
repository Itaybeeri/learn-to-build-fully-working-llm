# Positional information — giving the model a sense of word order

> **Status: complete.** All three parts: ① the gap — attention is order-blind (a bag of words,
> not a sequence); ② the fix — add a position vector to each token's embedding so the input
> carries *what* + *where*; ③ how the position vectors are made — learned table vs. sinusoidal
> wave formula.

## In one sentence

**Positional information** is an extra "where am I?" signal added to each token's vector, because
self-attention on its own treats a sentence as an unordered *bag of words* and can't tell
"dog bites man" from "man bites dog."

## Why it matters

Word order carries meaning — who did what to whom. But attention as built (score q·k → softmax →
blend) is **order-blind**: it would give the exact same answer if you shuffled the words. Every
real Transformer therefore injects position into the input. Without it, an LLM couldn't tell a
question from its reverse, a subject from an object, or "not good" from "good… not."

## The intuition

### Part 1 — attention is blind to word order

Everything in attention is, at its core, a **weighted sum** of the other words' value vectors. And
a sum doesn't care about order: `lion + meat` equals `meat + lion`. So when a word blends in its
neighbors, it gathers the **same bag** of neighbors no matter how they were arranged.

That means these look **identical** to attention:

```
   "dog bites man"
   "man bites dog"
```

Same three words → same embeddings → same Q/K/V → same scores → same blend. Attention cannot tell
them apart — yet they mean opposite things.

**Why:** when `bites` attends, it looks at the set `{dog, man}` and blends by relevance — but
"the set of words around me" is the same set in both sentences. Nothing in the machinery records
**where** each word sat (position 1, 2, 3…). The model sees a **bag of words, not a sequence.**
(Picture word-magnets handed to you in a pile: you know *which* words, not their *order*.)

**So what's missing:** each token's vector currently encodes only **what word it is** (its
embedding) and **nothing about where it sits**. The fix will be to stamp each word with its
**position**, so "dog" at position 1 is distinguishable from "dog" at position 3.

### Part 2 — the fix: add a position vector to each token's embedding

Invent a **position vector** for each slot in the sentence: a vector meaning "position 1," another
meaning "position 2," and so on. Each is the **same size** as a word embedding (e.g. 256 numbers).
Then, before a token enters attention, **add** its position vector onto its embedding, slot by slot:

```
   input vector  =  token embedding  +  position vector
                     (what word)        (where it sits)
```

That summed vector is what flows into Q/K/V — so every token now carries **both** *what I am* and
*where I am*, fused into one vector of the normal size.

**Why adding fixes order-blindness.** Tiny made-up numbers:

```
   embeddings:  dog=[2,0]   bites=[0,2]   man=[1,1]
   positions:   pos1=[0.1,0]   pos2=[0,0.1]   pos3=[0.2,0.2]

   "dog bites man":  dog@1=[2.1,0]   bites@2=[0,2.1]   man@3=[1.2,1.2]
   "man bites dog":  man@1=[1.1,1]   bites@2=[0,2.1]   dog@3=[2.2,0.2]
```

`dog` is `[2.1,0]` in the first sentence but `[2.2,0.2]` in the second — **the same word is now a
different vector depending on where it sits.** So when attention does its (still order-blind) sum,
the ingredients differ → the two sentences finally produce different results.

The key move: **we didn't change attention at all.** The blend is still a sum. We just made the
things going *into* the sum already carry position, so order survives.

**Why add, not concatenate?** Adding keeps the vector the **same length** (no extra size or cost),
and the network learns to read both the "what" and the "where" out of the summed vector. Real
Transformers add.

### Part 3 — where the position vectors come from (learned vs. sinusoidal)

Part 2 said "add a position vector for each slot" — but who *makes* those vectors? Two methods;
real models use one or the other. Both do the same job: hand each slot a distinct vector to add on.

**Method A — learned position vectors (simple).** Just a second lookup table, exactly like the word
embedding table (Module 3) but indexed by *slot number* instead of word ID:

```
   position table:   slot 0 → [vector]   slot 1 → [vector]   slot 2 → [vector]   …
```

Rows start random and are shaped by **gradient descent** during training; the model learns whatever
position vectors help it predict. To position a token, look up its slot and add that row.
- **Pro:** dead simple; the model learns what's most useful.
- **Con:** the table has a **fixed number of rows** (e.g. 2048). Feed text longer than that and there
  is **no row** for the new position — it can't handle lengths it never trained on. (GPT, BERT use
  this learned approach.)

**Method B — sinusoidal position vectors (a fixed wave formula).** The original Transformer didn't
learn positions — it **computed** them with sine and cosine waves. No table, no learning. Intuition:
an **odometer** / a row of **clock hands turning at different speeds**:

```
   wheel 1 (fast):   moves every position      → distinguishes neighbors
   wheel 2 (slower): moves every few positions
   wheel 3 (slow):   barely moves              → distinguishes far-apart regions
```

Each position is a unique **snapshot of all the wheels at once** (like an odometer reading `00427`
is a unique combination of wheel angles). Two nice properties: every position gets a **unique**
pattern, yet **nearby positions get similar** patterns (so the model can sense close vs. far); and
because it's a **formula**, it works for **any** position — even one longer than anything trained on
(no fixed maximum length — the main edge over Method A).

The formula's shape, just to have seen it (don't memorize): for position `p`, dimension `i`,
alternate `sin( p / big_number^(i/d) )` and `cos( … )` — small `i` = fast wave, large `i` = slow
wave (the "wheels at different speeds"). The takeaway is the **odometer idea**, not the algebra.

**Bottom line:** learned = trainable table (simple, but capped length); sinusoidal = fixed wave
formula (any length). Either way Part 2's mechanism is unchanged — you just *add it on*.

## A picture in words

```
   without position:   {dog, bites, man}  ≡  {man, bites, dog}     ← attention can't distinguish
   with position:      dog@1 bites@2 man@3  ≠  man@1 bites@2 dog@3  ← now order is visible

   per token:   [ token embedding ]  +  [ position vector ]  =  [ input to attention ]
                    what word              where it sits           carries both

   making the position vector:
     Method A (learned):    slot number → lookup row in a trained table
     Method B (sinusoidal): slot number → plug into a sine/cosine wave formula (odometer wheels)
```

## Check yourself

- Why is plain self-attention blind to word order? (Its core is a weighted *sum* of neighbors'
  values; a sum is order-independent, so it sees a bag of words, not a sequence.)
- Give two sentences attention can't tell apart, and say what's missing. ("dog bites man" /
  "man bites dog" — same words, same blend; missing = each token's *position*.)
- How do we fix it without touching attention? (Add a *position vector* to each token's embedding
  before it enters attention, so the same word at a different slot becomes a different vector.)
- Why does adding work even though the blend is still a sum? (The ingredients going into the sum now
  carry position, so different orders give different vectors → different results.)
- Name the two ways position vectors are made, with one trade-off each. (Learned = a trainable
  lookup table, simple but capped to a fixed max length; sinusoidal = a fixed sine/cosine wave
  formula — "odometer wheels" — works for any length, no learning.)

## Questions we worked through *(so far)*

- **Q (top rung): Why can't self-attention tell "dog bites man" from "man bites dog" — what about
  its core mechanic ignores arrangement — and what's the missing ingredient we'll add?**
  A: The blend step is a weighted **sum**, and a sum is order-independent (`3+5 = 5+3`), so
  attention sees the same bag of words either way — what's missing from each token is its
  **position**.
- *Ladder we climbed:* first answer had the right shape (attention ignores order, just blends,
  order matters) but missed the precise *why* and the exact missing piece. Stepped down to isolate:
  (1) "is `3+5` different from `5+3`?" → he saw a sum can't depend on order (and intuited the fix:
  do something extra per position — previewing Part 2); (2) "name the one missing thing" → **position**.
  Then climbed back up: the *why* = the blend is a **sum**; the missing ingredient = **position** —
  combined into one clean sentence.
- **Q (Part 2 top rung): What do we attach to each token, how, and why does it make "dog bites man"
  ≠ "man bites dog" even though the blend is still a sum?**
  A: Add a small **position vector** to each token's embedding (vector addition), after the embedding
  lookup and before attention; the whole model is trained with positions present, so it never
  expects a position-free embedding. It works because `dog` gets `pos1` added in one sentence and
  `pos3` in the other → the same word becomes a **different vector**, so the sum's *ingredients*
  differ → different result. We didn't make the sum order-aware; we position-stamped its inputs.
- *Two probing questions worth working through (they come up before the top rung):* "Where exactly is it
  added?" → once, at the input, right after embedding lookup and before Q/K/V. "Doesn't adding a
  vector corrupt the meaning / make predictions wrong?" → no: everything is trained together so the
  network expects position to be present (nothing to corrupt), and embeddings have hundreds of
  dimensions so "what" and "where" coexist and stay recoverable; the same position always gets the
  same stamp (consistent signal, not noise).
- **Q (Part 3 top rung): Name the two ways to make position vectors, how each produces a vector for
  a slot, and the key trade-off.**
  A (the full answer): **Learned** — a trainable table nudged by gradient
  descent like word embeddings; simple/straightforward but **capped** at the trained length (~2048).
  **Sinusoidal** — built from waves ("wheels") at different speeds; **not capped**, so it gives a
  vector for positions never seen, at the cost of being a less intuitive formula-based mechanism.
  Trade-off: learned can't exceed its table; sinusoidal handles any length.

## What's next / depends on

- **Depends on:** [Self-attention (Q/K/V)](02-self-attention-qkv.md) (blend = weighted sum of
  values — the sum is what's order-blind), [Embeddings — words as vectors](../03-language-as-data/04-embeddings-words-as-vectors.md)
  (the position table is just a second embedding table; adding vectors slot-by-slot).
- **Next concept:** *A transformer block (feed-forward, residuals, layer norm)* — assembles
  attention + position into the repeatable building block that stacks into a full Transformer,
  finishing Module 4.
