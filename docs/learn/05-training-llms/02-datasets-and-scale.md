# Datasets & scale — the fuel, and why bigger won

> **Status: complete.** ① where the text comes from (web is the bulk + books, Wikipedia, code) and how
> much (**tokens**, in the **trillions**); self-supervision makes it assemblable, compute makes it
> trainable; ② **quality** — raw web is mostly junk, *garbage in → garbage out* (the model becomes what it
> reads), so the corpus is heavily cleaned: quality-filtering, **deduplication**, safety filtering, and
> **quality often beats raw quantity**; ③ **scale** — three dials (data + parameters + compute) grown
> *together*; **scaling laws** make "bigger = lower loss" smooth and *predictable*; and **emergent
> abilities** switch on past a size threshold. Recipe: simple task + clean data + scale → predictably
> better, plus new abilities for free.

## In one sentence

An LLM is trained on a colossal pile of raw text — mostly the open web, plus books, Wikipedia, and code,
measured in **trillions of tokens** — which is only assemblable because **self-supervision** turns every
piece of text into free labeled examples, and only *trainable* because of massive compute.

## Why it matters

The model is only as good as what it reads. Where the text comes from, how clean it is, and *how much* of
it there is largely determine what the model can do. And the single biggest lesson of the last years —
"scale" — lives here: making the data, the model, and the compute all bigger turned a neat idea into
systems that can write, reason, and code.

## The intuition

### Part 1 — where the training text comes from (and how much)

Concept ① gave the unlock: *all text is free training data* because it labels itself. So the move is to
gather as much text as possible. The main sources, biggest first:

- **The open web** — the giant one. Projects "crawl" the internet (download billions of public pages) into
  huge archives; the best-known is **Common Crawl**, a free public snapshot of much of the web. This is the
  bulk of most training sets.
- **Books** — large digitized collections; long, well-edited, coherent over many pages → good for
  long-range structure and good writing.
- **Wikipedia** — encyclopedic, factual, clean, multilingual. Small but high quality.
- **Code** — enormous amounts of public source code (e.g. GitHub). A big reason models can program.
- **Curated extras** — news, academic papers, Q&A sites, forums, etc.

**How much?** We don't count sentences or files — we count **tokens** (Module 3: the sub-word chunks the
model actually reads). The numbers are staggering:

```
   GPT-3 (2020):        ~300 billion tokens
   modern big models:   trillions of tokens  (often 1–15+ trillion)
```

A trillion tokens is far more than any human could read in thousands of lifetimes — the equivalent of many
millions of books. The model effectively reads the public written output of humanity.

**Two distinct ingredients make this possible** (don't confuse them):

1. **Self-supervision (why it's *possible* to assemble)** — normally each training example needs a human to
   label the correct answer; labeling trillions of next-word answers by hand would take armies of people
   millions of years. But the next word *is* the label, already in the text — so any text is instantly a
   pile of labeled examples with **zero labeling effort**. That's the only reason a trillion-example dataset
   can exist.
2. **Compute / GPUs (why it's *feasible* to train)** — having the data isn't enough; you need vast amounts of
   parallel computation (many GPUs) to actually run the forward/backward passes over that mountain of text.

```
   web + books + Wikipedia + code + ...
            │   (crawled, collected, merged)
            ▼
   one gigantic pile of raw text  ──tokenize──►  trillions of tokens to predict
```

### Part 2 — quality: garbage in, garbage out

"Gather as much text as possible" has a catch: **most of the open web is junk** — spam, broken HTML,
auto-generated gibberish, keyword-stuffed ad farms, endless duplicates, toxic content, pages that are just
menus. And the iron rule of machine learning is:

> **Garbage in, garbage out.** The model *becomes what it reads.* It only ever imitates the next-token
> patterns in its data — it has no taste of its own — so training on spam and gibberish teaches it to
> *produce* spam and gibberish. Clean text in = clean model out.

So raw crawled text is **not** trained on directly; it's heavily **cleaned** first. The main jobs:

1. **Quality filtering** — keep text that looks like real, fluent human writing; drop gibberish,
   boilerplate, spam, machine-generated junk (heuristics + a small "is this good text?" classifier).
2. **Deduplication** — remove repeated text (syndicated articles, mirrored pages, pasted quotes). Two
   reasons: (a) dupes **waste effort and skew weighting** — a paragraph seen 10,000× gets over-practiced as
   if it mattered 10,000× more; (b) seeing the *exact same* string many times pushes the model toward
   **memorizing** it (and regurgitating training text verbatim) instead of learning general patterns.
3. **Safety / unwanted-content filtering** — remove or down-weight toxic, harmful, or private content you
   don't want the model absorbing and reproducing.

The result is a smaller but **much cleaner** corpus — and that's a real tradeoff with Part 1's "more is
better":

```
   raw web (huge, filthy)  ──filter + dedup + safety──►  smaller, clean corpus  ──►  better model
            more                                                 better
```

The field's hard-won lesson: **quality often beats raw quantity.** Less but cleaner text frequently beats
more but dirtier text. So "more is better" carries a big asterisk — more *good* text helps; more garbage
actively hurts.

### Part 3 — scale: why "just make it bigger" worked

The big surprise of the modern era. Usually "make it bigger" hits diminishing returns. With LLMs,
researchers found that **making everything bigger keeps making the model better — predictably — over many
orders of magnitude.** That discovery is *why* LLMs exploded.

**Three "bigger" dials, which must grow *together*:**

1. **Data** — more (clean) training tokens.
2. **Parameters** — more weights (the Module 0 dials) = more capacity to store patterns. (GPT-2: ~1.5B;
   later models: hundreds of billions.)
3. **Compute** — more processing (GPUs × time) to actually do the training.

Analogy: **parameters = the size of the brain** (how much it *can* learn); **data = how much it gets to
read** (how much there *is* to learn); **compute = study time** (to actually absorb it). Grow only one and
you stall — a huge brain with nothing to read learns nothing; a tiny brain with the whole library can't
hold it; either with no study time never finishes. The weak link caps the rest, so they scale together.

**Scaling laws.** The relationship turned out to be *smooth and predictable*: plot loss against scale and it
follows a clean downward trend. So you can **predict** how much better a bigger model will be *before*
building it — which is exactly why labs confidently spend fortunes on ever-larger models. Bigger reliably =
lower loss = better.

**Emergent abilities.** The wild part: some abilities **don't improve gradually — they appear suddenly**
once the model is big enough. A small model simply *can't* do multi-step arithmetic / follow a tricky
instruction / do some reasoning (scores ~0), then past a size threshold the ability **switches on**, like a
phase change (water → ice). Capabilities nobody explicitly trained for *emerge* purely from scaling up
next-word prediction — a big reason LLMs surprised even their builders.

```
   bigger DATA + bigger MODEL + bigger COMPUTE  (grown together)
            │
            ▼
   predictably lower loss  ──and──►  new abilities switch on (emergence)
```

**Whole-concept takeaway:** the recipe behind modern LLMs is almost embarrassingly simple to state — *a
simple task (next-word prediction) + a clean dataset + scale (data + parameters + compute, grown together).*
No new magic per ability; the abilities fall out of doing that one simple thing, bigger.

## Check yourself

- What's the biggest single source of LLM training text, and what unit/ballpark describes the amount? (The
  open web — e.g. Common Crawl; measured in **tokens**, in the **trillions**.)
- Why is it even possible to assemble a training set of trillions of examples? (Self-supervision — every
  text labels itself with its own next words, so no human labeling is needed.)
- What's the difference between self-supervision and compute here? (Self-supervision makes such a huge
  *labeled dataset possible*; compute/GPUs make it *feasible to actually train* on it. Different questions.)
- Why isn't raw web text trained on directly? (Garbage in, garbage out — the model becomes what it reads,
  so junk text → junk model. The corpus is cleaned: quality-filter, deduplicate, safety-filter.)
- Why specifically remove duplicate text? (Dupes waste effort and over-weight whatever's repeated, and push
  the model toward memorizing/regurgitating exact strings instead of learning general patterns.)
- Is more data always better? (No — *more good text* is better; more garbage hurts. Quality often beats
  raw quantity.)
- What three things do you scale, and why together? (Data, parameters, compute — the weak link caps the
  rest: big brain + nothing to read, or tiny brain + whole library, both stall.)
- What are "scaling laws" and why are they useful? (Loss-vs-scale is a smooth, predictable downward trend,
  so you can predict a bigger model's gain *before* building it — worth the investment.)
- What's an emergent ability? (A capability that appears *suddenly* past a size threshold rather than
  ramping up gradually — abilities nobody trained for, falling out of scaling next-word prediction.)

## Questions we worked through

- **Q: (a) Biggest single source of training text + how we measure the amount (unit, ballpark); (b) why is
  it even possible to assemble a set this enormous?** A: (a) the **web**, measured in **tokens** (trillions);
  (b) — a common first answer is *compute/GPUs*, which is the wrong axis (that's how we *process* it, not why
  the dataset can exist). Nudge back to Concept ①: the reason there can be *trillions of labeled examples
  with no labeling effort* is that **every text is already a "guess the next word" dataset** — it labels
  itself. Keep the two distinct ingredients separate (self-supervision = possible; compute = feasible).
- **Q: We gather as much text as possible, yet we don't train on raw web text — why? Name a cleaning step
  and why it matters (and why dedup specifically helps).** A: **garbage in, garbage out** — the model
  becomes what it reads, so junk in = junk out; steps include dropping gibberish/low-quality pages and
  **deduplication**, which avoids over-training/memorizing and verbatim regurgitation. *(Dedup + its
  memorize/regurgitate reason and the gibberish filter are the pieces to land; reinforce the headline
  "garbage in, garbage out / the model becomes what it reads" — trained on spam → it learns to produce
  spam.)*
- **Q (top rung): Explain the "scale" story — (1) the three things you scale and why together, (2) what
  scaling laws mean and why they're useful, (3) what an emergent ability is and why it's surprising.**
  A: (1) **data, parameters, compute** — must grow together because the weakest link caps the rest; (2)
  **scaling laws** = you can measure/predict the bigger model's gain before building it, so you don't spend
  the money blind; (3) **emergent ability** = a capability that suddenly appears in a big model that a small
  one can't do, that nobody trained for. *(Answered all three fully on the first try; added nuance — the
  smooth predictable loss trend, the brain/read/study-time analogy, and "switches on like a phase change.")*

## What's next / depends on

- **Depends on:** [Next-token prediction](01-next-token-prediction.md) (self-supervision = why the dataset is
  assemblable), [Tokens & tokenization](../03-language-as-data/02-tokens-and-tokenization.md) (we count data
  in tokens), [Weights/parameters](../00-orientation/02-how-llms-learn-and-where-knowledge-lives.md) (the
  "parameters" scaling dial).
- **This concept is complete.** Next concept in Module 5: **What "learning" actually adjusts** — a closer
  look at *which* numbers change during training (embeddings, attention Q/K/V, FFN, etc.) and what it means
  for knowledge to live in those weights.
