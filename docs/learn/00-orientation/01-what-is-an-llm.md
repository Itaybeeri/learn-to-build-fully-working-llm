# What is an LLM? (the big picture)

## In one sentence

An LLM (Large Language Model) is a computer system that, given some text, **predicts what
text most likely comes next** — and by doing that over and over, it can write, answer, and
converse.

## Why it matters

This is the single most important idea in the whole curriculum. *Everything* else we'll
learn — tokens, embeddings, attention, training — exists to make this one prediction good.
If you hold onto "it's a next-piece-of-text predictor," nothing later will feel like magic.

## The intuition

You already use a tiny version of this every day: the **autocomplete** on your phone
keyboard. You type "I'll be there in five —" and it suggests "minutes." It's guessing the
next word from the words so far.

An LLM is the same *kind* of thing, scaled up enormously and made far smarter. Give it:

> "The capital of France is"

and it predicts the next word is very likely "Paris."

The clever trick is what happens next: it **takes its own prediction, adds it to the text,
and predicts again.** That loop is how a next-word guesser produces whole sentences,
paragraphs, and answers:

```
"The capital of France is"            → predicts "Paris"
"The capital of France is Paris"      → predicts "."
"The capital of France is Paris."     → predicts (maybe stop, or keep going)
```

So the model never "writes a paragraph" in one shot. It writes **one chunk, then the next,
then the next** — each time asking the same question: *given everything so far, what's the
most likely next chunk?*

## A picture in words

Think of a loop:

```
   ┌─────────────────────────────────────────┐
   │  text so far  →  [ LLM ]  →  next chunk  │
   │        ▲                          │      │
   │        └──────  append  ◀─────────┘      │
   └─────────────────────────────────────────┘
```

Feed text in, get one chunk out, glue it on, repeat. That's generation.

## The gentle math (just a taste — we'll build this properly later)

We'll make this precise in later modules, but here's the shape of it. The model doesn't
output *one* word with certainty. It outputs a **probability** for every possible next word —
a number between 0 and 1 saying "how likely is this one." For our example it might be:

```
P("Paris")  = 0.91     ← very likely
P("France") = 0.02
P("a")      = 0.01
P("banana") = 0.000001 ← basically never
```

(`P(...)` just means "the probability of...". All these probabilities across every possible
word add up to 1.) Generating text is then: look at these probabilities and **pick** a next
word — usually a likely one. We'll explore exactly *how* it picks in Module 6 (that's
"sampling").

## Questions we worked through

The questions we actually talked through to reach this idea — try to answer before reading
the resolution.

- **Q: If an LLM can only ever predict the *next* word, how can it possibly write a whole
  paragraph, or answer a full question?**
  A: It doesn't produce the answer in one shot. It generates **one word, appends that word to
  the text, and predicts again** — looping. Each new word becomes part of the input for the
  next prediction, so it builds the whole answer one word at a time, reading its own output
  as it goes.

## Check yourself

1. **In your own words, what is the one job an LLM does?**
   → It predicts the next chunk of text, given the text so far.
2. **An LLM only predicts one chunk at a time. So how does it produce a whole answer?**
   → It loops: predict a chunk, append it to the text, predict the next, and so on.
3. **Does the model output a single sure word, or something else?**
   → Something else: a probability for *every* possible next word; then one is picked.

## What's next / depends on

- **Depends on:** nothing — this is the starting point.
- **Next:** *How to use these notes* (then we begin the foundations in Module 1).
