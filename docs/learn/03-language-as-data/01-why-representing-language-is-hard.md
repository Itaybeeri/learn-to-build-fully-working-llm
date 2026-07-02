# Why representing language is hard

## In one sentence

A neural network can only do arithmetic on **numbers**, but language is made of **words** —
symbols that carry meaning, depend on context, and come in near-endless variety — so before any
LLM can run, we must turn text into numbers that *capture meaning*, and doing that well is
genuinely hard.

## Why it matters

Everything in Modules 1–2 quietly assumed the input was already a vector of numbers (the "should
I go outside?" neuron got `temperature`, `raining`, `weekend` as numbers). But the real input to
an LLM is **text**: "The cat sat on the…". A network has no idea what a letter or a word *is* — it
only multiplies, adds, and bends numbers. So the very first job in building an LLM is the
translation: **text → numbers**. Module 3 is that bridge, and this note explains why it's not as
simple as it sounds.

## The intuition

### Part 1 — the mismatch: networks eat numbers, language is meaning

Recall what a network actually does (Module 2): every neuron multiplies its inputs by weights,
adds them up, and bends the result. **Multiply, add, bend.** That is *all* it can do — pure
arithmetic on numbers.

Now look at language. A word like **"cat"** is not a number. It's a **symbol** standing for a
whole bundle of meaning: a small furry animal, related to "kitten" and "dog," the kind of thing
that "sat on a mat." None of that is arithmetic. You can't multiply "cat" by a weight.

So there's a fundamental **mismatch**:

```
   what a network can read        what language actually is
   ───────────────────────        ─────────────────────────
   numbers (1.7, -0.3, …)         symbols that carry MEANING
   you can multiply/add them      "cat", "queen", "running"
   order & size mean something    a word's spelling ≠ its meaning
```

Before the network can process a single sentence, we must **convert the words into numbers** —
and not just any numbers, but numbers that somehow preserve the *meaning*, because meaning is the
whole point. That conversion is the central problem of this module.

The hard part isn't "assign a number to each word" — that's easy and, as we'll see next, mostly
useless. The hard part is making the numbers **carry the meaning**, so that the network's
multiply-add-bend machinery can actually do something useful with them.

### Part 2 — why the obvious fix (just number the words) fails

The first idea everyone has: build a list and give each word a number. `cat = 1`, `dog = 2`,
`queen = 3`, `king = 4`, … Feed *that* number in. It seems to solve the mismatch — but it quietly
breaks in two fatal ways.

**Problem 1 — it invents fake order and size.** Numbers carry **order** and **magnitude**, and the
network's arithmetic *will* use them. With `cat = 1`, `dog = 2`, `queen = 3`, the numbering
secretly claims that `queen > dog > cat`, that `dog` sits *halfway between* `cat` and `queen`, and
that `queen` is "three times" `cat`. All nonsense — the numbering was arbitrary (alphabetical, or
random). But the network can't tell; it will happily compute things like "average of cat(1) and
queen(3) = dog(2)" and read meaning that was never there.

**Problem 2 — it captures no similarity.** A word's number says nothing about which words *mean*
similar things. `cat = 1` and `kitten = 5000` could be miles apart, while `cat = 1` and `car = 2`
sit right next to each other. So the network has no way to know "cat" and "kitten" are related and
"cat" and "car" are not — and therefore can't generalize (if *cat* sat on a mat, a *kitten*
probably could too).

The deeper reason both problems exist: a single number is **one dimension** — a point on a line.
But meaning has **many facets** at once (furriness, royalty, is-it-an-action, …). You cannot place
every word meaningfully on a single line. This is the exact lesson from
[Numbers & vectors](../01-foundations/01-numbers-and-vectors.md): one number isn't enough — you
need a **vector**. That realization is what points us toward **embeddings** (Part 3 and the
Embeddings note).

### Part 3 — what a good representation must do

The two failures of simple numbering point straight at the fix. A good representation must:

1. **Give each word many numbers — a vector, not one.** A single number is one dimension (a point
   on a line), too cramped to hold meaning's many facets. Meaning needs room: is-it-furry,
   is-it-royal, is-it-an-action, … Each becomes a slot in a vector. (Exactly
   [Numbers & vectors](../01-foundations/01-numbers-and-vectors.md): one number isn't enough.)
2. **Place words by meaning — similar words close, unrelated words far.** In this space, *closeness
   means relatedness*: "cat" near "kitten," "king" near "queen," "car" off in a different region.
   Now the network's multiply-add machinery has real geometry to work with, and it can generalize
   (what's true near "cat" likely holds near "kitten").
3. **Be learned, not hand-assigned.** Nobody places the words by hand — **training arranges the
   vectors**, nudging related words together (precisely the "learning arranges the numbers"
   picture from the recap).

A learned vector-per-word, positioned by meaning, is called an **embedding** — the destination of
this module. There's also a second, practical hurdle: the sheer **variety** of language (enormous
vocabularies, new words, typos, word-forms like run/runs/running). We can't pre-list every
possible word, so we first break text into reusable pieces called **tokens**. That sets the path
for the rest of Module 3: **Tokens & tokenization → Vocabularies → Embeddings**.

## A picture in words

```
   text  "The cat sat"
     │   ① chop into reusable units ............  TOKENS
     ▼
   tokens  [The] [cat] [sat]
     │   ② each unit looked up as a vector ......  EMBEDDINGS
     ▼                                              (learned; similar meanings sit close)
   numbers  cat → [0.2, -0.7, 0.1, …]   ← now the network can multiply/add/bend
```

Simple numbering (`cat=1, dog=2, …`) skips straight to one number and fails: fake order/size, and
no similarity. The real pipeline earns its numbers.

## Check yourself

- Why can't raw text go straight into a network? (It only does arithmetic on numbers.)
- Name the two reasons simple word-numbering fails. (Fake order/magnitude; captures no
  similarity.)
- What three things must a *good* word representation do? (Many numbers per word; positioned by
  meaning so similar words are close; learned by training.)
- What's the name for a learned, meaning-positioned vector per word? (An embedding.)

## Questions we worked through

- **Q: Why can't we hand raw text "The cat sat" straight to a neural network?**
  A: A network only does arithmetic on numbers — you can't multiply the word "cat" by a weight.
  So we must first transform words into numbers, and the hard part is making those numbers carry
  the *meaning*.
- **Q: If we number words `cat=1, dog=2, queen=3, …`, what goes wrong (find two problems)?**
  A: (1) Fake order/magnitude — the numbers imply "queen is 3× cat" and "dog is halfway between,"
  which is meaningless, yet the network's arithmetic uses it. (2) No similarity — arbitrary
  numbering puts unrelated words adjacent (cat=1, car=2) and related words far (cat=1,
  kitten=5000), so the network can't tell what's related.
- **Q: How does a "vector per word, positioned by meaning" fix each failure?**
  A: It replaces one arbitrary ID with a vector of learned, meaningful *measurements*. (Fixes
  similarity: related words share facet values — king & queen both score high on "royal," differ
  on "male" — so they sit close.) (Fixes fake magnitude: each number now measures a real property,
  so its size means something real, and there's no single arbitrary ID forcing "queen = 3× cat.")

## What's next / depends on

- **Depends on:** [Numbers & vectors](../01-foundations/01-numbers-and-vectors.md) (a vector
  represents a thing), [What a neuron is](../02-neural-networks/01-what-a-neuron-is.md) (networks
  only multiply/add/bend numbers), [How an LLM learns](../00-orientation/02-how-llms-learn-and-where-knowledge-lives.md)
  (knowledge lives in numbers/weights).
- **Next:** Part 2 (why numbering words fails) → Part 3 (what a good representation must do), then
  *Tokens & tokenization*.
