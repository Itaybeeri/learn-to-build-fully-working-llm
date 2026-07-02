# How an LLM learns, and where its "knowledge" lives

## In one sentence

An LLM gets good by **practicing next-word prediction** on huge amounts of text, and what it
ends up *being* is a giant pile of tuned numbers called **weights** — not stored text.

## Why it matters

People imagine an LLM as a search engine with the internet saved inside it. It isn't. Getting
this right now prevents a hundred confusions later: it explains why models can be "wrong,"
why they don't "look things up," and what we actually mean by "training" in Module 5.

## The intuition

**How it learns — the guessing game.** During training, the model is shown mountains of text
and forced to play one game over and over, billions of times: *predict the next word.* Each
round it guesses, checks the real next word, and **nudges its internal settings** to be a
little less wrong. Repeat across the whole internet's worth of text and it gradually becomes
excellent at the game. It has seen "the capital of France is Paris" in so many forms that
predicting "Paris" becomes automatic.

**What it stores — dials, not text.** After training, the text is thrown away. What survives
is the *adjustments* — the final settings. Think of the model as **billions of tiny dials**,
each set to a number like `0.3`, `-1.7`, or `0.0008`. Training is the slow turning of those
dials; the trained model is just their final positions.

An analogy: you can recognize a friend's face instantly, but you don't keep a *photo* filed
in your head. Your brain got *tuned* — connections adjusted — so their face triggers
"that's them!" No image stored, just a tuned system. An LLM is the same: no text stored,
just tuned numbers.

## A picture in words

```
   TRAINING (done once, slowly):
     lots of text → play "guess next word" → wrong? nudge the dials → repeat ×billions

   WHAT'S LEFT (the model):
     [ 0.31 ][ -1.74 ][ 0.0008 ][ 0.62 ] ... billions of dials, frozen at their final values
```

## The gentle math (just the vocabulary for now)

- Those numbers are called **weights** or **parameters** (same thing).
- When you hear *"a 7-billion-parameter model,"* that number is literally **how many dials it
  has.** More dials → more capacity to capture patterns (up to a point).
- We are *not* doing the math of how a dial affects a prediction yet — that's Module 2
  (neurons) and Module 5 (training). For now, just hold: **knowledge = a pile of numbers.**

## Questions we worked through

The chain of questions we talked through to get from "it knows things" to "it's a pile of
numbers." Try each before reading the resolution.

- **Q: Where does the model's "knowledge" come from — how does it know "Paris" is likely
  after "The capital of France is," and "banana" isn't?**
  A: From learning on huge amounts of text — practicing next-word prediction and nudging its
  internal numbers to be less wrong, over and over.
- **Q: Did it *store* all that text, like a library or search engine it looks the answer up
  in?**
  A: No — something else. It does not keep the text at all.
- **Q: If it throws the text away and only keeps the "internal settings" it nudged during
  training, what are those settings actually made of?**
  A: Numbers. Billions of adjustable numbers ("dials"), called **weights / parameters**.
  Those numbers — not stored text — are what the model *is*.

## Check yourself

1. **How did the model get good at predicting the next word?**
   → By practicing the guess on huge amounts of text and nudging its settings to be less
   wrong, billions of times.
2. **Does the model store the text it read?**
   → No. It throws the text away; only the tuned numbers (weights) remain.
3. **What is a "parameter," and what does "7-billion-parameter model" mean?**
   → A parameter (weight) is one of the model's adjustable numbers/dials; 7 billion of them
   means it has 7 billion such dials.

## What's next / depends on

- **Depends on:** [What is an LLM? (the big picture)](01-what-is-an-llm.md).
- **Next:** *How to use these notes*, then Module 1 — Foundations, where we start meeting the
  gentle math (numbers and vectors) that those "dials" are made of.
