# The attention idea — letting words look at each other

## In one sentence

**Attention** is the step that lets every word in a sentence look at all the other words,
decide which ones are relevant to it, and rewrite its own vector as a blend of them — turning
each word's fixed, context-free embedding into a context-aware representation.

## Why it matters

This is the core idea of the Transformer — the architecture behind every modern LLM. Modules
1–3 got us to the door: each word is now a vector (an embedding), and we know the dot product
scores how similar two vectors are. Attention is what those pieces were *for*. It is the
mechanism that lets a model understand that "bank" means something different by a river than in
a sentence about savings, and that "it" refers to the milk — context, captured in numbers.

## The intuition

### Part 1 — why a fixed embedding isn't enough

After Module 3, every word becomes an **embedding vector** that captures its general meaning.
But that vector is **fixed**: the word `"bank"` looks up the *same* row of the embedding table
no matter the sentence.

```
"river bank"     →  bank = [ … same vector … ]
"savings bank"   →  bank = [ … same vector … ]
```

Yet those are different things — mud and water vs. money. The embedding table can't tell them
apart, because it hands out the word's vector **before it has seen any of the neighbors**. The
embedding is **context-free**.

Language is full of words whose meaning depends on their neighbors:

- **`"it"`** — *"The cat drank the milk because **it** was warm."* "It" = the milk, but the
  bare embedding of "it" has no idea.
- **`"play"`** — *play* a song vs. *play* a game vs. a Broadway *play*.
- Even a plain sentence: in *"the cat sat,"* to really represent `"sat"` you'd want to know
  **who** sat — and that lives in a *different* word, `"cat"`.

So the embedding is a good **starting point** — a word's general, out-of-context meaning — but
it isn't the finished representation. We need a step that **updates each word's vector using
the other words in this particular sentence**, turning a context-free vector into a
**context-aware** one.

> That updating step is **attention**: every word looks around at its neighbors and rewrites
> itself in light of them.

### Part 2 — score by similarity, blend by relevance

If a word just absorbed **all** its neighbors equally it'd be a mess — it would soak up "the",
"a", punctuation, every irrelevant word. So attention is selective, in two steps:

**Step 1 — score every other word (relevance).** The current word takes its vector and computes
a **dot product** with each other word's vector. A dot product is a **similarity score** (from
Module 1), so a high score means "you're relevant to me," low means "ignore you."

```
   word "it" in "the cat drank the milk because it was warm" scores each word:
       it · the     →  0.1   (irrelevant)
       it · milk    →  4.8   (very relevant!)
       it · cat     →  1.2
       it · drank   →  0.4
```

**Step 2 — blend, weighted by those scores.** Build the word's new vector as a **weighted blend**
of all the words' vectors — each weighted by its relevance score. High scorers contribute a lot,
low scorers almost nothing.

```
   new "it" vector ≈ (mostly milk's vector) + (a little cat) + (tiny bits of the rest)
```

The result: "it"'s vector now *contains* "milk" — context-aware, and it picked up the **right**
neighbor because **similarity decided the weights**. The whole attention idea in one line:
**score everyone by similarity (dot product), then blend them in proportional to that score.**

Worked example — *"the lion will eat ___"*, focusing on `"eat"`: it scores `lion` high and `the`
low, so its new vector mostly contains `lion`. That "lion-flavored" `eat` then predicts **meat**,
whereas in *"the rabbit will eat ___"* a "rabbit-flavored" `eat` predicts **grass** — the *same*
context-free word now produces *different*, correct predictions because attention gave it
different context.

### Part 3 — the weighted-average mechanic

"Blend the vectors, weighted by relevance" has a precise meaning: a **weighted average**, in two
stages.

**Stage A — turn the raw scores into weights that add up to 1.** The dot-product scores can be
any numbers (big, small, even negative). We convert them into **fractions that sum to 100%** —
those become the blend weights. The standard tool for this is **softmax** (it squashes any list
of scores into positive fractions that total 1; opened up properly in a later concept).

```
   raw scores:  lion 4.0,  will 0.5,  eat 1.0   →   weights: lion 0.7, will 0.1, eat 0.2  (sum 1)
```

**Stage B — multiply each word's vector by its weight, then add them up (slot by slot).** Using
tiny 2-D vectors where slot 1 = "predator-ness", slot 2 = "plant-ness" (just made-up, readable
numbers — real embeddings have hundreds of unnamed slots):

```
   lion = [10, 0]   will = [0,0]   eat = [1,1]      weights 0.7 / 0.1 / 0.2

   new eat = 0.7·[10,0] + 0.1·[0,0] + 0.2·[1,1] = [7.2, 0.2]   → predator-flavored → predicts meat
```

Swap `lion` for `rabbit = [0,10]` (weights `rabbit 0.6, the/will 0.1, eat 0.2`):

```
   new eat = 0.6·[0,10] + 0.2·[1,1] = [0.2, 6.2]   → plant-flavored → predicts grass
```

The same plain `eat = [1,1]` became `[7.2,0.2]` or `[0.2,6.2]` depending on its neighbors — that
is "making the vector more lion-ness / rabbit-ness": multiply the relevant neighbor's vector by a
fat weight and add it in.

**Why the weights sum to 1:** it makes the result a true *average*, so the new vector stays on
the same scale as the originals — it can't blow up or vanish. A genuine blend, not a pile-up.

## A picture in words

```
   context-free embeddings            after attention (context-aware)
   "river"  bank  "of"          →      "river"  bank′  "of"
            ▲ same vector                       ▲ bank′ now leans "geography"
   "savings" bank "account"     →      "savings" bank″ "account"
            ▲ same vector                       ▲ bank″ now leans "money"
   one word looked at its neighbors and rewrote itself accordingly
```

## Check yourself

- Why isn't a word's embedding, on its own, enough to predict the next word? (It's
  context-free — assigned before seeing neighbors — so it can't distinguish "river bank" from
  "savings bank.")
- Give a word whose meaning depends on its neighbors, and say what attention should do to its
  vector. (e.g. "it" — attention should pull in the vector of whatever "it" refers to, so "it"
  becomes context-aware.)

## Questions we worked through *(so far)*

- **Q: Why isn't a word's embedding, on its own, enough to predict the next word? Give a word
  whose meaning needs its neighbors, and say what should happen to its vector.**
  A: The embedding is context-free — assigned before seeing neighbors — so it can't tell
  "river bank" from "savings bank," or know what "it" refers to. Once a word can look around,
  its vector should be **updated to contain information from the relevant neighbors** (e.g.
  "eat" in "lion will eat" should absorb "lion").
- **Q: In "the lion will eat ___", which words does "eat" score high/low, and what does its new
  vector mostly contain?**
  A: `lion` high, `the` low. After the blend, `eat`'s new vector mostly contains `lion`.
- **Q: Why does a "lion-flavored" eat predict differently than a "rabbit-flavored" eat?**
  A: Plain context-free `eat` is identical in both sentences, so it would force the *same*
  guess. Attention makes the vector different (predator-leaning vs plant-leaning), so the
  prediction step leans toward meat vs grass.
- **Q: How do you actually "make the vector more rabbit-ness"?**
  A: A weighted average — multiply `rabbit`'s vector by a large weight and add it into the
  blend. Worked by hand: `eat=[1,1]`, weights `rabbit 0.6 / the 0.1 / will 0.1 / eat 0.2`,
  `rabbit=[0,10]` → new eat = `[0.2, 6.2]` (plant-flavored). (Initial slip: answered `[1,1]`,
  i.e. eat keeping 100% of itself — corrected by seeing eat's own weight is only 0.2.)
- **Q: Why do the attention weights add up to 1?**
  A: So the blend is a true *average* — the new vector stays on the same scale as the originals
  and can't blow up or vanish ("nothing breaks the fences").

## What's next / depends on

- **Depends on:** [Embeddings — words as vectors](../03-language-as-data/04-embeddings-words-as-vectors.md)
  (each word is a vector), [Matrices & matrix multiplication](../01-foundations/02-matrices-and-matrix-multiplication.md)
  (the dot product = a similarity score), [Layers & the forward pass](../02-neural-networks/02-layers-and-the-forward-pass.md).
- **Demo:** portal → **Module 4 tab** ([`portal/modules/module4.js`](../portal/modules/module4.js))
  — Attention blend: pick a subject, watch `eat` score every word (dot product), soften to weights
  (softmax), blend the vectors, and see the next-word guess flip between meat & grass.
- **Next (within this concept):** Part 2 — the idea: each word scores every other word by
  similarity (dot product) and blends in the relevant ones. Then Part 3 — the weighted-average
  mechanic. After this concept: *Self-attention (Q/K/V)*.
