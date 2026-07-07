# Next-token prediction — how the text teaches itself

## In one sentence

An LLM learns by **self-supervised next-token prediction**: it hides the next word of real text, guesses
it, and corrects itself — and because the next word is *already in the text*, every sentence is a pile
of free training examples needing no human labels at all.

## Why it matters

This is the single idea that makes modern LLMs possible. Older machine learning needed humans to
hand-label every example, which caps how much you can train on. Next-token prediction removes that cap:
**all text ever written is instantly training data.** The scale that makes LLMs powerful comes directly
from this trick — no labeling, no bottleneck.

## The intuition

### Part 1 — the text labels itself (no humans needed)

To train a machine the usual way ("supervised learning") you need **labeled examples**: an input paired
with the correct answer. To teach "is this a cat?", a person must tag thousands of photos `cat` /
`not cat` by hand. That labeling is slow and expensive — humans are the bottleneck.

Next-token prediction sidesteps it entirely. Take any ordinary sentence:

```
   "the cat sat on the mat"
```

A training example needs an **input** and a **correct answer (label)**. Just cut the sentence:

```
   input:  "the cat sat on the"   →   correct answer:  "mat"
```

Where did the label `mat` come from? **From the text itself** — the next word *is* the answer, already
sitting there. Nobody labeled anything. This is **self-supervised learning**: the data provides its own
supervision. Every book, web page, article, and code file is *already* a labeled dataset, because every
word is the label for the words before it.

That is the unlock — it is *why* an LLM can train on essentially the entire internet. There is no
human-labeling step to slow it down; the raw text **is** the labeled data.

```
   a plain sentence  ─────►  hides free (input → next-word) examples inside it
   "the cat sat..."          the next word is always the answer — no human tags a thing
```

### Part 2 — the sliding window (one sentence, many examples)

One *cut* of a sentence makes one example — but you don't cut just once. Make an example at **every
position** by sliding along the text:

```
   "the cat sat on the mat"

   input: "the"                    → answer: "cat"
   input: "the cat"                → answer: "sat"
   input: "the cat sat"            → answer: "on"
   input: "the cat sat on"         → answer: "the"
   input: "the cat sat on the"     → answer: "mat"
```

One 6-word sentence just produced **5 training examples**. At each position the **input is everything
before** and the **label is the very next word**. This is the **sliding window**: walk through the text
one token at a time, and at each step the model practices "predict the next word from what I've seen so
far." Two payoffs:

1. **Data multiplies.** A 1,000-word document is ~1,000 examples (one per position), not one. Combined
   with Part 1 ("all text is free"), the amount of practice becomes astronomical.
2. **It learns to predict from *any* context length** — sometimes 1 word of history, sometimes 500 —
   because it's drilled on every prefix. (In practice the model predicts at *all* positions at once for
   efficiency, but each position is still its own labeled example.)

```
   the   cat   sat   on   the   mat
    └────►              each arrow = one training example:
         └────►         "predict the next token from everything to the left"
               └────►
                    └────►
                          └────►
```

Every arrow is a free (input → next-word) example — a whole book is millions of these arrows.

### Part 3 — why "just predict the next word" teaches *everything*

Predicting the next word sounds **trivial** — almost dumb. So how does a model that only plays "guess the
next word" end up with grammar, facts, translation, even reasoning? The answer is the deepest idea in the
whole curriculum: **to predict the next word *well*, you are forced to understand everything before it.**
The simple task is a doorway — the better you want to be at it, the more you're *forced* to learn:

- **Grammar** — to finish "The keys to the cabinet ___" it must agree with "keys" (are), not "cabinet".
- **Facts** — "The capital of France is ___": the only reliable way to predict "Paris" is to have stored
  the fact.
- **Meaning / context** — "I poured water from the bottle until it was ___" (empty? full?) depends on
  tracking what "it" is.
- **Reasoning** — "All cats are mammals; Tom is a cat, so Tom is a ___": predicting "mammal" means
  following the logic.
- **Translation, style, code…** — each is just "the next word," each demands real competence.

The pattern: **next-word prediction is not the goal — it's the *pressure*.** And the reason that pressure
can't be faked by memorizing is **endless novelty**: across billions of sentences the model never sees the
same one twice, so there's nothing to look up. The only thing that keeps the loss dropping on
forever-new sentences is to learn the **general machinery** — grammar, facts, logic — that works on
sentences it has never encountered. Pressure + novelty ⇒ it *must generalize*, i.e. understand.

```
   "just predict the next word"
            │  (over billions of ever-new sentences, pushed to predict well)
            ▼
   forced to learn grammar + facts + meaning + reasoning  =  a model that "understands"
```

The task is simple. Being *good* at it is not — and that gap is exactly where the intelligence gets packed
in. ("Autocomplete" is the surface; understanding is what doing it well actually requires.)

## Check yourself

- Where does the "correct answer" for each training example come from? (From the text itself — the
  actual next word *is* the label; no human labels it.)
- What is this style of learning called, and why does it let LLMs train on so much data? (Self-supervised
  learning — because every sentence already contains its own answers, all text in the world is instantly
  training data, with no labeling bottleneck.)
- How does one sentence give many training examples? (The sliding window — at every position, input =
  everything before, label = the next word; an N-word text gives ~N examples.)
- Why isn't "predict the next word" just glorified autocomplete? (Because being *good* at it across
  billions of ever-new sentences has no memorize-shortcut — the only way to keep the loss low is to
  genuinely learn grammar, facts, meaning, and reasoning. The task is the pressure; understanding is what
  it forces.)

## Questions we worked through

- **Q: Training normally needs humans to label every example, but an LLM trains on raw text with no human
  labels. Where does each example's correct answer come from, and why does that mean an LLM can learn from
  essentially unlimited text?** A: the correct answer is the next word, which is already in the sentence
  being trained on; since all text already contains its own labels, the whole internet becomes training
  data. *(Answered fully on the first try.)*
- **Q: A single sentence gives several training examples — how? What are we sliding, and what is the
  input vs. the answer at each position?** A: every word in turn is the next-word target for the words
  before it — slide along the sentence; at each position input = everything before, answer = the next
  word. *(Answered fully on the first try.)*
- **Q (top rung): Someone says "an LLM only predicts the next word — it's glorified autocomplete, it
  can't really understand." Answer them, including *why* such a simple task requires real understanding.**
  A: the task is simple but being *good* at it isn't — across billions of varied sentences you must know
  grammar, facts, meaning, and reasoning to keep getting the next word right; next-word prediction is the
  *pressure*, not the point.
- *Ladder we climbed:* a common first answer gives the *what* (it needs grammar/facts/reasoning) but
  not the *why*. Isolate the missing lever: why can't a dumb autocomplete still score well? Nudge
  toward "to keep getting it right you must ___" → "learn." Push once more for the linchpin — *why no
  shortcut* — and land the **endless-novelty** point: billions of ever-new sentences mean nothing to
  memorize, so the model must generalize (understand). That affirms the pressure framing; the close is
  the full one-paragraph comeback tying it together.

## What's next / depends on

- **Depends on:** [What is an LLM?](../00-orientation/01-what-is-an-llm.md) (next-token prediction as the
  model's job), [How LLMs learn](../00-orientation/02-how-llms-learn-and-where-knowledge-lives.md) (the
  guessing game), [The training loop](../02-neural-networks/05-the-training-loop.md) and
  [Cross-entropy](../02-neural-networks/03-loss-measuring-wrong.md) (one training step on a next-word guess).
- **This concept is complete.** Next concept in Module 5: **Datasets & scale** — where the training text
  comes from, how much of it there is, cleaning/dedup, and why scale (data + parameters + compute) matters
  so much for what the model can do.
