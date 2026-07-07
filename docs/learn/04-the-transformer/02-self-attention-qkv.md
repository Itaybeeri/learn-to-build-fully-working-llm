# Self-attention (Q/K/V) — the attention idea, made precise and learnable

## In one sentence

**Self-attention** is the attention idea built for real: instead of a word using its one raw
embedding to do everything, each word produces three separate vectors — a **query** (what I'm
looking for), a **key** (how I'm found), and a **value** (what I contribute) — and attention
scores query·key, then blends the values.

## Why it matters

This is the actual engine inside every Transformer. The "attention idea" gave us the shape
(score by similarity, blend by relevance); Q/K/V is how that shape is made precise and, crucially,
**learnable** — the model gets to *learn* how each word should ask, advertise, and contribute,
rather than being stuck with one fixed embedding for all three jobs.

## The intuition

### Part 1 — one vector can't do three jobs

In the plain "attention idea," every word used its **single raw embedding** for everything. In
*"the lion will eat ___"*, focusing on `eat`, that one vector had to play three different roles:

1. **The query (shopping).** `eat` computed `eat · lion`, `eat · the`, … — using its vector to
   *ask* "who's relevant to me?" Here the embedding is a **search query**.
2. **The key (being found).** `lion` was *scored by* its vector when `eat` came shopping — the
   embedding acting as a **label/tag** others match against. A different role: being found, not
   searching.
3. **The value (content delivered).** Once `lion` won a high weight, the thing **blended in** was
   again `lion`'s vector — the actual **information** poured into the new `eat`.

So one vector is forced to be the **question you ask**, the **label others match against**, and
the **content you deliver**, all at once.

**Why that's too much.** Those three jobs genuinely want to differ. Dating-app analogy:

- **Query** = "what am I looking for?" (tall, likes hiking)
- **Key** = "how do I advertise myself?" (my profile headline)
- **Value** = "what you actually get once matched" (the real me)

"What I'm looking for" and "how I describe myself" are not the same — you can be a homebody
*seeking* an adventurer. Forced to use one list of traits for both, you can't express that. The
single embedding has exactly this handcuff.

**The fix (headline).** Self-attention gives each word **three separate vectors**, one per role:

| Vector | Role | Answers… |
|--------|------|----------|
| **Query** (Q) | what I'm looking for | "who is relevant *to me*?" |
| **Key** (K) | how I'm found | "what do I match *against*?" |
| **Value** (V) | what I contribute | "what do I pour into the blend?" |

Scoring becomes **query · key** (my question vs. your label); the blend is a **weighted sum of
values** (everyone's content, weighted). Three roles, three vectors — no vector doing all three.

*(How the three vectors are made from the embedding, and why that makes attention learnable, is
Part 2.)*

### Part 2 — three learned weight matrices make Q, K, V

A word starts with just **one** thing: its embedding `x` (Module 3). We get three vectors out of
it by **transforming** it three ways. For each word's embedding `x`:

```
   query  q = W_Q · x
   key    k = W_K · x
   value  v = W_V · x
```

- **`x`** — the word's embedding (its context-free meaning vector).
- **`W_Q`, `W_K`, `W_V`** — three **weight matrices**; each, times `x`, produces a new vector.
- **`q`, `k`, `v`** — that word's query, key, value.

Why does a matrix-times-vector help? From Module 1, **matrix × vector re-expresses the vector** —
a learned "lens" that emphasizes some features, downplays others. So `W_Q` is a lens that turns the
embedding into a good *"what am I looking for"* vector; `W_K` into a good *"how I advertise"*
vector; `W_V` into a good *"what I contribute"* vector.

Two crucial facts:

1. **The matrices are learned weights** (Module-2 dials): random at first, shaped by **training**
   (backprop + gradient descent). *This is what makes attention learnable* — the model isn't stuck
   using the raw embedding for all three jobs; it discovers three transformations that produce a
   genuinely good query, key, and value.
2. **The same three matrices are shared across every word** — one `W_Q` for the whole sentence,
   applied to each word's embedding. The model learns a *general rule* for asking / advertising /
   contributing, then applies it everywhere.

```
   "lion":  x_lion → (×W_Q) q_lion   (×W_K) k_lion   (×W_V) v_lion
   "eat":   x_eat  → (×W_Q) q_eat    (×W_K) k_eat    (×W_V) v_eat
   …every word, same three matrices, three vectors each.
```

### The key clarification — Q/K/V are *hats every word wears*, not different words

Query/Key/Value are **not three kinds of word**. They're **three hats every single word wears at
once.** `eat` has a query, a key, and a value; so does `lion`; so does every word. The question is
never "is this word a query or a key?" — it's "in *this* comparison, which hat is each word using?"

**Google-search picture (no math):**

- **Query** = the text you type in the search box — your *question*, what you're looking for.
- **Key** = the keywords a page is *listed under* (`pizza, italian, delivery`) — how it gets
  **found**. Matched against your query; you never really read it.
- **Value** = the page's actual content (the menu, prices) — what you **read after clicking**.

A search: your **query** is matched against each page's **key**; the high-matching pages win; you
then read their **value**. *Match on the key, consume the value.*

Mapped to *"the lion will eat ___"* with `eat` as the searcher: `eat` searches with its **query**
("who's doing the eating?"); `lion` advertises with its **key** ("animal, predator, subject") which
matches strongly; then `lion`'s **value** (its predator-meaning) flows into the new `eat`. And it's
*self*-attention because every word searches at the same time — everyone is both a searcher and a
page in the results.

### Part 3 — the full mechanic (score → softmax → blend)

Each word now has a query `q`, a key `k`, and a value `v`. To compute **one** word's new
context-aware vector — say `eat` (the searcher) — three steps:

**Step 1 — Score.** The searcher's **query** · **every** word's **key**:
```
   q_eat · k_lion = 4.0     q_eat · k_will = 0.5     q_eat · k_eat = 1.0
```
(The one change from the plain attention idea: score **query · key**, not raw embedding · embedding.)

**Step 2 — Weights (softmax).** Squash the scores into weights that sum to 1:
```
   4.0 / 0.5 / 1.0   →  softmax  →  0.7 / 0.1 / 0.2   (sum 1)
```

**Step 3 — Blend the values.** New vector = weighted sum of every word's **value**:
```
   new eat = 0.7·v_lion + 0.1·v_will + 0.2·v_eat
```
(The other change: blend the **values** `v`, not raw embeddings.)

The **same three steps run for every word at once** — each is a searcher with its own query, so
every word gets a fresh context-aware vector. Self-attention is the attention idea's exact shape —
**score → softmax → blend** — with two upgrades: score with **q·k**, blend the **values**. Those
upgrades let the learned matrices pick the *right* neighbors.

## A picture in words

```
   for each word as the searcher:
     1. SCORE   its query · every word's key      → relevance scores
     2. SOFTMAX those scores                       → weights that sum to 1
     3. BLEND   weights × every word's value, summed → its new context-aware vector

   attention idea:   score x·x,   blend the raw embeddings
   self-attention:   score q·k,   blend the values v        (q,k,v = learned re-expressions of x)
```

## Check yourself

- Q/K/V are not three kinds of word — what are they? (Three hats *every* word wears at once.)
- In one comparison between two words: the word doing the looking uses its **____**; the word being
  looked at is matched on its **____**, and if it wins, contributes its **____**.
  (query; key; value.)
- Why split one embedding into three? (One vector can't be the question, the label, and the content
  at once — those are genuinely different jobs.)

## Questions we worked through *(so far)*

- **Q (top rung): A word's single raw embedding was forced to play three roles in attention. Name
  them, and say which role belongs to the word doing the looking vs. the word being looked at.**
  A (reached after climbing the ladder): Query, Key, Value. The looker uses its **query**; the
  word being looked at is matched on its **key**, and once it wins, contributes its **value**.
- *Ladder we descended/climbed to get there:*
  - First answer named Q/K/V correctly but missed the looker-vs-looked-at split, and the roles
    felt like "sentence understanding, not vectors."
  - **Down a rung (Google, no math):** "best pizza" — which is the text you typed (query), which
    is the menu you read after clicking (value)? → got query; confused key/value.
  - **Down again (isolate key vs value):** keywords `pizza, italian, delivery` vs. the full menu —
    which is key, which is value? → **cleared:** key = how you're found, value = what you read once
    found.
  - **Up a rung (back to the sentence):** in "the lion will eat ___", `eat` searches with its
    query, `lion` advertises with its key, `lion`'s value flows in once it scores high. → cleared.
  - **Top rung re-asked:** answered fully and correctly on his own.

### Part 2 — questions we worked through

- **Q (top rung): How are Q/K/V made from one embedding, and why does that make attention
  learnable (vs. the raw-embedding "attention idea")?**
  A: Each is `W_Q·x`, `W_K·x`, `W_V·x` — three learned weight matrices that re-express the
  embedding for each role. Learning them lets the model discover a *general recipe* for the right
  connections between words (verb→subject, etc.), tuned to its task — which raw-embedding
  *similarity* can't capture (`eat` and `lion` aren't "similar," yet `lion` is the relevant one).
- *Side questions that built the answer (important confusions cleared):*
  - **"Where do W_Q/W_K/W_V come from?"** → They're **weights** (like neuron weights / the
    embedding table): random at first, given their values by **training** — the same one process.
  - **"Is it all just training? What do they do when I ask a question?"** → Two phases (like BPE
    build-vs-use): **training** twitches all weights once; **using** freezes them and just runs the
    forward pass. At use-time attention *computes* context fresh for your specific sentence with the
    fixed matrices — it doesn't memorize "eat↔lion."
  - **"What if it's a totally new, unrelated sentence?"** → The matrices are trained over *millions*
    of sentences, so they can't memorize one; they learn **general** relationship rules that apply
    to any new sentence (e.g. "the chef will cook ___").
  - **"Single set of weights? When do you make new sets?"** → One head = one shared set
    (W_Q/W_K/W_V) for all words; new sets come from **multiple heads** (different relationship
    types — next concept) and **stacked layers** (depth).
  - **"Q/K/V is a much smaller pile than the embeddings?"** → Yes: embeddings store something per
    word (scales with vocab, ~25M); Q/K/V is one shared *recipe* (size depends only on `d`, ~0.8M
    per head). A general transformation is cheap; per-word storage is expensive.

### Part 3 — questions we worked through

- **Q (top rung): Walk through how self-attention computes one word's new vector end to end (the
  three steps), and name the two things different from the plain attention idea.**
  A (climbed to on his own): **1.** score `q_eat · k_word` for every word; **2.** softmax the scores
  into weights that sum to 1; **3.** weighted blend — each word's **value** × its weight, summed →
  the new `eat` vector. Differences: score with **q·k** (not raw embedding · embedding), and blend
  the **values** (not raw embeddings). *(Ladder: first answer had the skeleton but summed values
  equally and wrote `W_Q*W_K`; isolated "do all words count equally?" → got weighted sum; supplied
  the score→softmax→weight link; then he listed the three steps correctly himself.)*

## What's next / depends on

- **Depends on:** [The attention idea](01-the-attention-idea.md) (score by similarity, blend by
  relevance), [Embeddings — words as vectors](../03-language-as-data/04-embeddings-words-as-vectors.md),
  [Matrices & matrix multiplication](../01-foundations/02-matrices-and-matrix-multiplication.md)
  (dot product = similarity).
- **Next (within this concept):** Part 2 — three learned weight matrices turn each embedding into
  its Q, K, V. Then Part 3 — the full self-attention mechanic, end to end.
