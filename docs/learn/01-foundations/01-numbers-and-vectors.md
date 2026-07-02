# Numbers & vectors — what a vector represents

## In one sentence

A **vector** is an ordered list of numbers, where each position ("slot") carries a fixed,
agreed meaning.

## Why it matters

Vectors are the language of everything ahead. A word's *meaning*, a model's *dials*
(weights), the *input* and *output* of every layer — all are vectors. If "an LLM is a pile
of numbers," vectors are how that pile is **organized** so it can mean something.

## The intuition

Start with a **location on a flat map**. One number can't pin a spot. Say *"3 east"* — you
know how far east, but not how far north, so the treasure could sit anywhere on a vertical
line. Add a second number, *"5 north,"* and now there's exactly **one** spot: `[3, 5]`.

```
   one number  → narrows you to a whole LINE of possible spots
   two numbers → picks the exact POINT on that line
```

Two independent facts (how far east, how far north) need two numbers. That ordered pair
`[3, 5]` is a vector.

The same trick describes *things*, not just places. A person:

```
[180, 75, 30]   →   slot1 = height(cm), slot2 = weight(kg), slot3 = age(yrs)
```

Each slot means something fixed; the list captures the person across several attributes at
once. **That's a vector: an ordered list of numbers, each position with a meaning.**

## The gentle math

- **Notation:** a vector is written as a list in brackets, e.g. `v = [3, 5]`. The numbers
  inside are its **components** (or entries).
- **Dimension = how many numbers are in the list.** `[3, 5]` is 2-dimensional (2-D);
  `[3, 5, 9]` is 3-D; a list of 300 numbers is 300-D. Each dimension is one independent
  direction / attribute.
  ```
  [3, 5]              2-D   (a point on a flat map)
  [3, 5, 9]           3-D   (a point in space — the 9 is altitude)
  [3, 5, 9, …, 2]     300-D (300 independent pieces of info)
  ```
- **High dimensions are fine.** You can't *picture* 300-D space, but every rule that holds
  for `[3, 5]` holds identically for 300 numbers. So we build intuition in 2-D/3-D (where we
  can see it) and trust the same math up high. No new magic appears — just more slots.

## A picture in words

```
        north
          ↑
   . . . [3,3]
   . . . [3,2]
   . . . [3,1]
   . . . [3,0]  ← "3 east" alone = this whole vertical line of possibilities
          |
          +--------→ east
   The second number ("north") picks ONE point off that line.
```

A vector can be seen as a **point** (where you land) or an **arrow** from the origin to that
point — same thing, two views we'll both use later.

## How this connects to LLMs

An LLM represents the **meaning of a word** as a long vector — often hundreds or thousands of
numbers — where each slot captures some aspect of meaning. The model's weights are organized
into vectors too. We'll build this up in Module 3 ("embeddings"); for now, just hold:
*meaning and parameters live as lists of numbers.*

## Questions we worked through

- **Q: To locate a spot on a flat map, why isn't one number enough?**
  A: One number (e.g. "3 east") only narrows you to a **line** — it leaves the other
  direction unspecified, so infinitely many spots qualify. A second number ("north") picks
  the exact point.
- **Q: But if the treasure is "only 3 east," isn't one number enough?**
  A: No — "only 3 east" secretly means *3 east and **0** north*. You still used a second
  number (`0`); you just chose zero. Without stating it, `[3,0]` can't be told apart from
  `[3,5]`.
- **Q: In 3-D (e.g. a drone in the air), how many numbers, and what's the third?**
  A: Three. The first two place it over a spot on the ground; the third is its **altitude**
  (how high up). Each extra number locks one more independent direction.

## Check yourself

1. **What is a vector, in one sentence?**
   → An ordered list of numbers where each position has a fixed meaning.
2. **What does a vector's "dimension" mean?**
   → How many numbers are in the list (2-D = 2 numbers, 300-D = 300 numbers).
3. **Can you picture 300-dimensional space? Does it matter?**
   → No, nobody can — and it doesn't matter: the same math that works in 2-D works in 300-D.
4. **How does an LLM use vectors?**
   → It represents word meanings (and its weights) as vectors — long lists of numbers.

## What's next / depends on

- **Depends on:** [How an LLM learns, and where its "knowledge" lives](../00-orientation/02-how-llms-learn-and-where-knowledge-lives.md)
  (the model is a pile of numbers — vectors are how we organize them).
- **Next:** *Matrices & matrix multiplication* — what happens when we line up many vectors
  and let them interact (the core operation inside every layer).
