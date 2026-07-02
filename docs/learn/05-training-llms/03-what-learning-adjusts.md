# What "learning" actually adjusts — the model's weight groups

> **Status: complete.** ① the **inventory** — a Transformer's learnable weights aren't one blob but distinct
> groups (embedding table, Q/K/V matrices, output W_O, the two feed-forward matrices, LayerNorm's γ/β, the
> positional table if learned, the unembedding), all adjusted at once by gradient descent; ② **what each
> group learns** — embeddings learn *meaning*, Q/K/V learn *relationships*, W_O *combines* heads, the FFN
> *processes* and stores much of the *patterns/facts*; ③ **knowledge is distributed** (no single "Paris"
> weight — facts live spread across thousands of weights) and **training only changes the weight values,
> never the architecture** (the container's shape is fixed before training; learning just fills it).

## In one sentence

"Learning" means **gradient descent simultaneously nudging every one of the model's weight groups** — the
embeddings, the attention Q/K/V/W_O matrices, the feed-forward matrices, the layer-norm dials, and the
unembedding — until those millions-to-billions of numbers settle into values that make the next word
predictable.

## Why it matters

"Knowledge lives in the weights" (Module 0) is true but vague — *which* weights? A Transformer has several
**distinct kinds** of weights, each with a different job. Seeing them as one concrete list turns the slogan
into something you can point at, and sets up the deeper idea that knowledge is spread *across* all of them,
not stored in any one place.

## The intuition

### Part 1 — the inventory: which numbers are the learnable weights?

Trace the pipeline from a token ID to the output probabilities and you pass through every weight group. They
are all "just numbers," and training adjusts **all of them at the same time**:

| # | Weight group | What it does |
|---|---|---|
| 1 | **Embedding table** | token ID → its meaning vector (Module 3) |
| 2 | **Q / K / V matrices** (`W_Q`, `W_K`, `W_V`) | each word → its query, key, value (per head) |
| 3 | **Output matrix `W_O`** | recombine the multi-head outputs into one vector |
| 4 | **Feed-forward matrices** (two, **shared** across positions) | the "think" sub-layer: matrix → ReLU → matrix |
| 5 | **LayerNorm's γ (scale) and β (shift)** | the small learned dials in each layer norm |
| 6 | **Positional table** *(only if learned, not sinusoidal)* | the "where" vectors added to embeddings |
| 7 | **Unembedding matrix** | final vector → logits over the vocab (often **tied** to the embedding table) |

Two things to hold onto:

- **The feed-forward weights are shared** — one little network reused at every word position (like the
  attention matrices). Not a separate FFN per position.
- **There is no special treatment.** Backprop computes a slope for *every* one of these numbers in one
  backward pass, and the update nudges each by `w ← w − rate × slope`. So "the model learns" literally means
  *all* these groups move together, every training step, toward lower loss.

```
   token ID ─► [embedding] ─► [+positional] ─► [ Q/K/V → W_O | FFN | γ,β ] ×N blocks ─► [unembedding] ─► logits
                  ▲                ▲                ▲      ▲      ▲    ▲                      ▲
                  └──────────── all of these are learnable weights, all nudged by gradient descent ───────────┘
```

### Part 2 — what each weight group learns to represent

All these numbers get nudged toward lower loss — but because each group sits at a *different spot* in the
pipeline, each ends up specializing in a **different job**. Nobody assigns the jobs; they *emerge* because
that division of labor is what makes the next word predictable:

- **Embedding table → *meaning*.** The vectors drift until words that behave alike land near each other
  (`king − man + woman ≈ queen`). They encode **what each token means**, in isolation.
- **Q / K / V matrices → *what to look at* (relationships).** They learn which words should attend to which
  — one head may learn "verb → its subject," another "adjective → its noun." They learn the **routing**:
  who is relevant to whom.
- **W_O → *how to combine*** the heads' findings into one useful vector.
- **Feed-forward matrices → *processing + stored patterns/facts*.** The big one: after attention *gathers*
  context, the FFN *transforms* it, and much of the model's **factual knowledge and learned patterns** is
  thought to live here ("after 'the capital of France is' → push toward 'Paris'"). FFNs are the bulk of the
  parameters.
- **LayerNorm γ/β → keep each layer's numbers well-scaled** (a supporting role, not meaning).

The key idea: **the architecture provides empty slots with different *roles*; training fills them so that,
together, they predict the next word.** Meaning, relationships, and patterns/facts are a division of labor
that emerged purely from minimizing loss.

```
   embeddings: WHAT words mean   ·   Q/K/V: WHICH words matter to which   ·   FFN: PROCESS it + store patterns/facts
                          all shaped by one pressure: predict the next token
```

### Part 3 — knowledge is distributed (and the architecture never changes)

Two final ideas that clear up a common misconception.

**① No single weight "is" a fact.** It's tempting to imagine one neuron that *is* "Paris." It doesn't work
that way. A fact like *capital of France → Paris* is spread across **thousands of weights acting together** —
a pattern in the combination, not a slot you could point to. And each individual weight takes part in **many**
facts/patterns at once. This is a **distributed representation**: meaning lives in the *combination* of many
numbers, not in any one. Consequences: you can't open the model and edit "the Paris weight" (knowledge is
smeared across the network), and this overlap is part of why models generalize to inputs they've never seen
exactly.

**② Training changes the *weights*, never the *architecture*.** Before training, the model already has its
full **shape** — number of layers, heads, embedding size, the wiring — all **fixed by the designer** (the
*hyperparameters* from Module 2, chosen *before* training). Training does not add layers or rewire anything;
it only sets the numbers.

> The architecture is the **empty container**, decided up front. Training only **fills it** — pouring values
> into the millions of weight-slots. Same wiring throughout; only the numbers inside change.

So "the model learned French / learned to code" means: the *exact same network* now holds different numbers.
**Learning = finding good values for a fixed set of slots.**

```
   FIXED (chosen before training):  #layers, #heads, embedding size, wiring   ← the empty container
   LEARNED (set by training):       the values in every weight slot           ← what gets poured in
```

## Check yourself

- Name the model's main learnable weight groups. (Embedding table; Q/K/V matrices; output `W_O`; the two
  feed-forward matrices; LayerNorm's γ/β; positional table if learned; the unembedding matrix.)
- Does each word position have its own feed-forward weights? (No — the **same shared** FFN weights are reused
  at every position.)
- What does "the model is learning" mean concretely? (Gradient descent nudging **all** these weight groups at
  once — each gets a slope from backprop and a nudge `w ← w − rate × slope` every step.)
- What does the embedding table learn vs. what the Q/K/V matrices learn? (Embeddings → **meaning** of
  individual words; Q/K/V → **relationships**, which words attend to which.)
- Where does much of the model's factual knowledge live? (In the **feed-forward** matrices — after attention
  gathers context, the FFN processes it and stores patterns/facts; FFNs are most of the parameters.)
- Is there a single weight that stores "Paris is the capital of France"? (No — it's a **distributed
  representation**: the fact is spread across thousands of weights together, and each weight serves many
  facts. No "Paris weight" to point at.)
- Does training change the model's architecture? (No — the shape/wiring is fixed before training
  (hyperparameters); training only sets the **values** in the fixed weight slots.)

## Questions we worked through

- **Q: Trace the pipeline and name as many learnable weight groups as you can.** A: a strong first pass
  names the embedding table (token ID → vector), the attention **Q/K/V** matrices, and the output **W_O**.
  The feed-forward sub-layer is easy to miss; the hint "what's the 'think' sub-layer made of?" reconnects
  it to the Module 2 layers (matrix → ReLU → matrix = two weight matrices). Complete the list with the
  smaller groups (LayerNorm γ/β, positional table, unembedding) and the headline: all are nudged together
  by the same gradient descent.
- **Q: Does each position use its own FFN weights or shared ones?** A: **shared** — one FFN reused at every
  position. *(Answered correctly.)*
- **Q: What does the embedding table learn to capture vs. the Q/K/V matrices?** A: embeddings learn
  **meaning**; Q/K/V learn **relationships**. *(Answered correctly; added that the FFN holds much of the
  stored patterns/facts.)*
- **Q (top rung): When an LLM "learns," what literally changes — and where is a fact like "Paris is the
  capital of France" stored (is there a single Paris weight)?** A: training just **adjusts the numbers
  (weights)** so the model predicts the next token better; the values are stored **across many parameters**,
  not in a single answer/weight. *(Answered both halves fully; reinforced the explicit nuance that training
  changes only the weight values, never the fixed architecture.)*

## What's next / depends on

- **Depends on:** [Embeddings](../03-language-as-data/04-embeddings-words-as-vectors.md),
  [Self-attention (Q/K/V)](../04-the-transformer/02-self-attention-qkv.md),
  [Multi-head attention](../04-the-transformer/03-multi-head-attention.md),
  [A transformer block](../04-the-transformer/05-a-transformer-block.md) (FFN, layer norm),
  [Gradient descent & backprop](../02-neural-networks/04-gradient-descent-and-backprop.md) (what does the
  nudging).
- **This concept is complete.** Next concept in Module 5: **Overfitting vs. generalization** — the difference
  between *memorizing* the training text and *learning patterns that work on unseen text*, and how we tell
  them apart (train vs. validation loss).
