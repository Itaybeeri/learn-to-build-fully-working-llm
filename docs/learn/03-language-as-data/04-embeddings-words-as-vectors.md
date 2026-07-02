# Embeddings — words as vectors

## In one sentence

An **embedding** is the learned vector a token's ID looks up in a big table — the real,
meaning-carrying representation the network reads — with similar words ending up at nearby vectors
because training arranged them that way.

## Why it matters

This is the destination of the whole module: the final step that turns a token ID (a meaningless
integer) into numbers that *carry meaning*, so the network's multiply-add-bend machinery has
something real to work with. Embeddings are the bottom layer of every LLM — the very first thing
the model does with your text after tokenizing it.

## The intuition

### Part 1 — the embedding is a lookup table

We keep one big table of vectors — the **embedding table** (or embedding matrix):

```
   embedding table  (vocab_size rows × d columns, e.g. 50,000 × 256)

   id 0   →  [ 0.01, -0.20,  0.11, … ]   ← d numbers
   id 1   →  [-0.33,  0.05,  0.91, … ]
   id 2   →  [ 0.20, -0.70,  0.10, … ]   ← "cat"'s vector
   …
```

- **One row per token** in the vocabulary.
- Each row is a **vector of `d` numbers** (`d` = the *embedding dimension*, a design choice — e.g.
  256 or 768).
- To embed a token: take its ID and **look up that row**. That row *is* the token's **embedding** —
  the vector that flows into the network.

So the ID's only job is to be an **address** ("go to row 2"); the real representation is the
*vector* at that row. The full pipeline is now complete:

```
   "cat"  →  [cat]  →  ID 2  →  look up row 2  →  [0.20, -0.70, 0.10, …]  →  network
```

**Why this isn't the rejected "numbering words" scheme.** In Concept 1, numbering failed because the
*number itself* was used as the representation (and the network did arithmetic on it, inventing fake
order). Here the ID is **only a lookup key** — the network never multiplies the ID; it multiplies
the *vector* the ID fetches. All meaning lives in that learned vector, so the fake-order problem
can't arise.

### Part 2 — positioned by meaning, learned by training

The magic is in *what* those vectors end up being.

**They are positioned by meaning.** In a trained table, "cat" and "kitten" sit at nearby vectors,
"king" near "queen," "car" off in another region — **distance = relatedness.** Because the vectors
carry real structure, the network's multiply-add machinery can do meaningful work and even
*generalize* (what it learns near "cat" partly transfers to nearby "kitten"). Even **directions**
carry meaning, giving the famous `king − man + woman ≈ queen` — consistent "royalty" and "gender"
directions you can do arithmetic along.

**They are learned — they start random.** Nobody hand-places the words. The embedding table begins
as **random numbers**; the embedding table is **just more weights** (the same tunable dials as the
rest of the network). **Training** arranges them: **backprop** (step 3) computes how each embedding
number should change to lower the loss, and **gradient descent** (the step-4 update) nudges them a
hair, millions of times, until the random cloud organizes into a meaningful map. (ReLU/sigmoid are
forward-pass shaping only — they never change weights.)

### Part 3 — *why* training arranges them by meaning

The model's only job is **predicting the next token**. Words that appear in the **same kinds of
contexts** ("cat" and "dog" both near *pet, vet, fur, sat…*) are easiest to predict around if they
have **similar vectors** — so training is *pushed* to pull their embeddings together. The causal
arrow runs **similar usage → training pulls vectors close → similar property-values emerge**. Nobody
labels "cat" and "dog" as similar; the model **discovers** it purely from the goal of predicting
text, and the king/queen geometry falls out as a side effect. (The classic phrase: *"you shall know
a word by the company it keeps."*)

## A picture in words

```
   text "cat"  →  token [cat]  →  ID 2  →  look up row 2 of the embedding table
                                            → [0.20, -0.70, 0.10, …]  →  network

   the table (learned weights), after training:
        cat • kitten            king • queen          (similar words cluster)
                        car • truck • bus
   distance = relatedness;  directions carry meaning (king − man + woman ≈ queen)
```

See it live in the **portal → Module 3 tab** (Word map + Tokenizer demos).

## Check yourself

- What is an embedding, and what's the embedding table? (A token's learned meaning-vector; the
  table is one row per token in the vocabulary.)
- The ID is an integer — why isn't that the rejected "numbering words"? (The ID is only a lookup
  address; the network computes on the *vector* it fetches, never the ID's value.)
- What kind of thing are the embedding numbers, and what arranges them? (Weights; arranged by
  training — backprop + gradient descent — starting from random.)
- Why do "cat" and "dog" end up with similar vectors? (They appear in similar contexts, so similar
  vectors make the surrounding words easier to predict — training pulls them together.)

## Questions we worked through *(so far)*

- **Q: We rejected "numbering words" in Concept 1, yet now we give each token an integer ID. Why is
  this not the same failure?**
  A: The ID isn't the representation — it's just a key into a lookup table. The network never does
  arithmetic on the ID; it fetches the token's *vector*, and that learned vector holds the meaning.
  So no fake order/size is ever used in a calculation.
- **Q: Embeddings start random and end up meaningful — what process arranges them, and what kind of
  thing are they?**
  A: They're **weights** (a matrix of learned dials), arranged by **training** — backprop computes
  the slopes, gradient descent (the update step) nudges them, repeated millions of times. (Not
  ReLU/sigmoid — those are forward-pass shaping and never change weights.)
- **Q: Why do "cat" and "dog" get similar vectors?**
  A: They appear in similar contexts, so giving them similar vectors makes the surrounding words
  easier to predict — training pulls them together. Similar usage → similar vectors.

## What's next / depends on

- **Depends on:** [Vocabularies](03-vocabularies.md) (token IDs), [Why representing language is
  hard](01-why-representing-language-is-hard.md) (need vectors positioned by meaning), [Numbers &
  vectors](../01-foundations/01-numbers-and-vectors.md), [The training loop](../02-neural-networks/05-the-training-loop.md).
- **Demo:** portal → **Module 3 tab** ([`portal/modules/module3.js`](../portal/modules/module3.js))
  — Word map (embeddings + `king − man + woman ≈ queen`) and Tokenizer (BPE merge-replay).
- **Next:** Module 3 is complete — give the **Module 4 bridge recap**, then start *Module 4 — The
  Transformer* at *The attention idea*.
