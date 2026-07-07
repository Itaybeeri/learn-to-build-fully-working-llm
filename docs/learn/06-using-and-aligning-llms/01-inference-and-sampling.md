# Inference & sampling — turning the distribution into actual words

## In one sentence

**Inference** is running the finished model to produce text, and **sampling** is the step that turns the
model's final **probability distribution** over the vocabulary into one concrete next token — where
**temperature** (and top-k/top-p) control *how* adventurously that token is chosen.

## Why it matters

Training is over; now we actually *use* the model. But the model never outputs a word — its last step outputs
a **probability for every token in the vocabulary**. To put text on the screen, something has to *pick one*.
*How* we pick is a real choice with real consequences: pick too rigidly and the model is repetitive and
lifeless; pick too wildly and it's incoherent. Sampling settings (temperature, top-k, top-p) are the knobs
that tune this, and they're why the *same* model can sound like a careful encyclopedia or a freewheeling poet.

## The intuition

### Part 1 — inference, and the sampling problem (greedy vs. sampling)

🔁 *Recap (Module 4):* the model's final step produces **logits** — one raw score per vocabulary token — and
**softmax** turns them into a **probability distribution**: one probability per possible next token, all
positive, summing to 1.

So at generation time the model hands us something like:

```
   "The capital of France is ___"
        ▼   (model's final output — a distribution)
   Paris  0.70   Lyon 0.08   France 0.05   Nice 0.02   ... (49,996 more tiny probabilities)
```

The model's job is **done** — it gave the distribution. But we wanted **one word**. The step that turns the
distribution into a single chosen token is **sampling**; doing this over and over (pick a token, append it,
feed it back, repeat) is how the model **generates** text. The whole act of running the trained model this way
is **inference** (as opposed to *training*).

**The simplest rule: greedy / argmax.** Just take the single highest-probability token every time (`Paris`).
"Argmax" = *the argument that maximizes* — the token with the max probability. Its fatal flaw: it's
**deterministic**. Same prompt → same distribution → same top token → **identical output every time**. Two
consequences:

- **No variety** — ask for a poem twice and get the exact same poem, word for word.
- **Bland & loop-prone** — always grabbing the locally "safest" word doesn't make the best overall sentence;
  greedy text tends to fall into repetition loops ("I think that I think that…").

Greedy is fine when you *want* the top answer (facts). For anything creative or long, it's too rigid.

**The fix: sampling.** Instead of always taking the top token, roll a **weighted die** where each token's
chance of being picked equals its probability. Over many runs `Paris` (0.70) is chosen ~70% of the time,
`Lyon` (0.08) ~8%, and so on — likely words still dominate, but the output is no longer frozen to one path.
This buys **variety without chaos**: fewer loops, different results each run.

### Part 2 — temperature (the randomness dial)

Pure proportional sampling gives *one fixed* amount of randomness (always exactly the model's probabilities).
But different tasks want different amounts — a factual answer should be near-deterministic; a story should be
adventurous. **Temperature** is the dial that controls *how* random the sampling is, by **reshaping the
distribution before the die roll**:

```
   raw:                 Paris 0.70   Lyon 0.08   France 0.05   ...
   LOW  temp (0.3) →  sharpen:  Paris 0.95   Lyon 0.02   ...   (top word dominates → near-greedy)
   temp = 1.0      →  unchanged: Paris 0.70  Lyon 0.08   ...   (the model's own probabilities)
   HIGH temp (1.5) →  flatten:  Paris 0.45   Lyon 0.15   ...   (long shots get a real chance → creative)
```

The name is an analogy to heat: **low temperature = calm, ordered** (sticks to favorites); **high temperature
= hot, jittery** (takes risks). Mechanically it **exaggerates or shrinks the gaps** between probabilities —
low temp makes the rich richer (→ greedy in the limit, temp→0), high temp evens everyone out (→ more random).

- **Low temp →** less creative, reliably the top word → use for **factual / precise** answers.
- **High temp →** more creative, lets lower-ranked words step in → use for **imaginative / varied** writing.

### Part 3 — top-k and top-p (trimming the tail)

There's a problem hiding inside high temperature. The distribution has **~50,000 tokens**: a top handful
(`Paris` 0.70, `Lyon` 0.08, …) and then a giant **tail** of tens of thousands of tokens each with a *tiny*
probability (~0.00001) — odd punctuation, foreign-script characters, broken fragments. Individually each tail
token is almost never picked, but there are **so many** of them that **their probabilities add up to a
substantial total**. And turning temperature **up** *flattens* the distribution, which **lifts that whole junk
tail** — so the weighted die now has a real chance of landing on nonsense. That's exactly why pure
high-temperature text can suddenly go **incoherent**.

The fix: **before** rolling the die, **cut off the tail** so only sensible tokens remain. Two ways to cut:

- **Top-k — keep a fixed number.** Keep only the **k most-likely** tokens (e.g. k = 40), drop the rest,
  re-normalize, then sample from those. The junk tail is gone.
  ```
  top-k = 3:  keep [Paris 0.70, Lyon 0.08, France 0.05]  →  drop the other 49,997  →  sample from these 3
  ```
- **Top-p (nucleus) — keep a fixed *amount of probability*.** Keep the **smallest set of top tokens whose
  probabilities add up to p** (e.g. p = 0.9), drop the rest.

**The key difference:** top-k is a **fixed headcount** (always k tokens, no matter what); top-p **adapts to the
model's confidence** — when the model is sure (`Paris` 0.95) it keeps just a couple of tokens; when it's unsure
(probability spread thin) it keeps more, because it takes more of them to reach p. That adaptiveness is why
top-p is usually preferred. They combine with temperature: temperature controls *how bold* the pick is,
top-k/top-p control *how wide the candidate pool* is.

```
   full distribution ─► [top-k / top-p: cut the junk tail] ─► [temperature: sharpen/flatten] ─► roll weighted die ─► next token
```

## Check yourself

- What is inference vs. training? (Inference = running the finished model to generate text; training = the
  earlier process that set its weights.)
- What does the model's final layer actually output, and what must happen next? (A **probability distribution**
  over the whole vocabulary; **sampling** must then pick one concrete token from it.)
- What is greedy/argmax and its flaw? (Always take the highest-probability token → **deterministic**: identical
  output every time, and prone to bland repetition loops.)
- What does plain sampling do? (Pick a token at random **in proportion to its probability** — a weighted die;
  `Paris` 0.70 → chosen ~70% of the time. Buys variety without chaos.)
- What does temperature do? (Reshapes the distribution before sampling: **low** sharpens it (near-greedy,
  reliable — facts); **high** flattens it (long shots get a chance — creative).)
- Why does high temperature risk incoherence? (Flattening **lifts the junk tail** — tens of thousands of
  tiny-probability tokens whose total is substantial — so the die can land on gibberish.)
- What do top-k and top-p do, and how do they differ? (Both **trim the tail before sampling**. Top-k keeps a
  **fixed number** of top tokens; top-p keeps the smallest set whose probabilities **sum to p** — top-p
  **adapts** to the model's confidence, top-k doesn't.)

## Questions we worked through

- **Q: What's the simplest rule to pick one word, and what happens if you use it every time?** A: the output
  is always the same, because you always choose by the same rule. *(Correct — named the rule **greedy/argmax**
  and added the second flaw: bland, repetition-prone.)*
- **Q: With a weighted die, roughly how often is each token chosen, and what has it bought us?** A: it bought
  randomness "with percentage." *(Correct — weighted randomness; pinned the numbers: `Paris` 0.70 → ~70%,
  `Lyon` 0.08 → ~8%; bought variety without chaos.)*
- **Q: What does turning temperature down vs. up do, and which fits a factual question vs. a story?** A: down
  = less creative, picks the top-rated word almost always; up = more creative, lets other results step in.
  *(Correct; made explicit: factual → low temp, imaginative story → high temp.)*
- **Q (top rung): Why does high temperature risk incoherent text, and how do top-k/top-p each fix it (and
  differ)?** A: first pass was partial — got "high temp gives other words a fair chance" and the top-k/top-p
  mechanics, but mis-attributed *loops* to high temp and hadn't named the tail mechanism. Stepped down to
  isolate it: *do the tens of thousands of tiny tail probabilities add up to something substantial?* (yes) and
  *does flattening lift or squash that tail?* (lifts it). Then climbed back and gave the full answer cleanly:
  high temp lifts the substantial **junk tail** → die can pick gibberish; **top-k** cuts to a fixed number of
  top tokens, **top-p** keeps tokens that sum to a target probability — top-p **adapts** to confidence.

## What's next / depends on

- **Depends on:** [Probability basics](../01-foundations/04-probability-basics.md) (a distribution sums to 1),
  [A transformer block](../04-the-transformer/05-a-transformer-block.md) (logits → softmax → distribution),
  [Softmax](../04-the-transformer/02-self-attention-qkv.md).
- **This concept is complete.** Next in Module 6: **Context windows** — the maximum span of tokens the model
  can attend over at once, why there's a limit, and what happens at the edges.
