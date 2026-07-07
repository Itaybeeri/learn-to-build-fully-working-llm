# A transformer block — mix, then think (and how to stack it deep)

## In one sentence

A **transformer block** is the repeatable building unit of a Transformer: first an attention
sub-layer lets words **mix** information between each other, then a small feed-forward network lets
each word **think** about what it gathered — wrapped with two helpers (residual connections and
layer norm) that make the block safe to stack dozens deep.

## Why it matters

Attention by itself isn't a whole model. The transformer block is the actual unit that gets
**stacked** — GPT-style models are just many identical blocks in a row. Understanding one block is
understanding the entire body of an LLM; everything else (more layers, more heads) is just more of
this same thing.

## The intuition

### The big picture first — a transformer is a tower of "refine a little" blocks

Before the parts, the *why*. The model has one goal: **predict the next word**. To guess well it
needs each word understood **in context** (a raw embedding is context-free — `eat` is the same
vector everywhere — so on its own it can't know meat vs. grass). So the transformer's real job is to
**turn each word's bare embedding into a rich, context-aware understanding**, good enough that the
last word's vector practically points at the answer.

Understanding a sentence isn't one-shot — it takes several passes (like re-reading a tricky
sentence: each read starts from the understanding the last one gave you, and goes deeper). So a
transformer defines **one unit of "improve the understanding a bit" — the block — and stacks it many
times.** Each block takes the words as currently understood and hands the next block a slightly
better version:

```
   word+position vectors → [block 1] → [block 2] → ⋯ → [block N] → predict next word
                            (each block refines the understanding a little more)
```

That tower of identical blocks *is* the transformer. Everything below is what's inside one block.

### Part 1 — two sub-steps: mix, then think

A block runs two sub-layers one after the other.

**Step 1 — attention sub-layer (mixing / communication).** This is multi-head self-attention,
which you already know: each word looks at all the others and rewrites itself as a weighted blend of
them, so it now carries context (`eat` picks up `lion`). Its job is to **move information between
words**. But at heart attention is a weighted **sum** — it gathers and averages; it doesn't deeply
*process* what it gathered. The word has now *collected* the right info but hasn't *thought about*
it yet.

**Step 2 — feed-forward sub-layer (thinking / computation).** Each word's vector is then passed
through a small **feed-forward network (FFN)** — nothing new: it's the **Module 2 layers**, a matrix
multiply → activation (a bend, e.g. ReLU) → another matrix multiply. Two things about how it's
applied:

1. **Per word, independently** — each word's vector goes through the FFN on its own, no looking at
   neighbors (attention already did the looking; now each word digests privately).
2. **Same FFN weights for every word** — one shared little network, reused at every position.

Its job is to **transform** each word's gathered information into something more useful — the model's
real "thinking" about what each in-context word means for the prediction.

**The rhythm — mix, then think:**

```
   words → [ attention: mix info BETWEEN words ] → [ feed-forward: think, EACH word alone ] → richer words
                     communication                          computation
```

- **Attention** = talk to each other (move information around).
- **Feed-forward** = go off and process it (transform it, each word by itself).

That pairing — gather context, then process it — is the heartbeat of a Transformer.

### Part 2 — residual connections (add, don't replace)

**The problem.** We stack dozens of blocks. If each block *completely replaces* the vector with its
output, two things break: (1) the word's original meaning gets washed away after many rewrites, and
(2) training breaks — the backprop "blame" signal (Module 2) has to travel back through every block
and **shrinks to nearly nothing**, so early layers barely learn. Deep stacks become untrainable.

**The fix.** A **residual connection** (skip connection): instead of a sub-layer *replacing* the
vector, it **adds its result on top of the original**:

```
   output = input + sublayer(input)
              ▲          ▲
         the original   the change the sublayer computed
         (skips around, untouched)
```

Each sub-layer gets its own residual:

```
   a = x + Attention(x)        ← attention adds its mixing on top of x
   y = a + FeedForward(a)      ← feed-forward adds its thinking on top of a   (y = block output)
```

**Why it's better:**

1. **Each sub-layer only learns the *adjustment*** — a small correction to add, not the whole vector
   rebuilt from scratch. If a block has nothing useful to add, it can output ≈0 and the vector passes
   through unchanged.
2. **The original always survives** — a clean straight path (`x +`) carries the word's identity all
   the way up the tower without being mangled.
3. **Training flows** — that straight path is a **highway for the backprop signal**, so it reaches
   even the earliest blocks without vanishing. This single trick is what made very deep networks
   practical.

**Image — editing a document.** Bad way: every editor rewrites the whole page from memory → after
30 editors it's unrecognizable. Residual way: each editor adds small **edits/margin notes on top** of
the existing text → the original survives and improvements accumulate gently.

### Part 3 — layer normalization (keep the numbers stable)

**The problem.** A word's vector flows up the tower, and at every step a residual **adds** something
on top of it (`x + sublayer(x)`). Add, add, add, dozens of times — and the numbers **drift**: they
can balloon huge, collapse tiny, or get lopsided (some slots enormous, others near zero). That breaks
training: recall from Module 2 that activations and gradients only behave in a **moderate range**.
Numbers too large → activations saturate (gradient ≈ 0, learning stalls) or gradients **explode**
(loss bounces); too tiny → the signal **vanishes**. Drift isn't just untidy — it directly destabilizes
the whole network.

**The fix — layer normalization (LayerNorm).** Before each sub-layer, reset the word's vector back to
a clean, standard "volume." Think of a **sound engineer auto-leveling speakers** who talk at wildly
different loudness: bring everyone to a consistent level so the *content* comes through, not the raw
volume. LayerNorm does that to a vector — it evens out the scale so the *pattern* in the numbers is
what matters, not their size. Crucially it **keeps the relative pattern** (the biggest slot stays the
biggest); it only rescales.

**The two moves** (computed across the `d` slots of one word's vector `x = [x₁ … x_d]`):

1. **Subtract the mean μ** → *recenter*. `μ = (x₁ + … + x_d) / d` is the vector's overall center. Take
   it away from every slot → the vector now sits centered on 0 (fixes "too high / too low overall").
2. **Divide by the spread σ** → *re-scale*. σ = the typical distance of a slot from μ (how big the
   swings are). Dividing by it is an **auto-leveling volume knob**: huge swingy numbers (big σ) shrink,
   tiny cramped numbers (small σ) grow — both land at the *same* standard swing-size of ~1 (fixes
   "swings too big / too small").

```
        xᵢ − μ
  x̂ᵢ = ─────────      → x̂ always comes out centered at 0, spread ≈ 1, pattern intact
           σ
```

*Tiny example:* `[100, 300, 200]` → subtract μ=200 → `[-100, +100, 0]` (centered, but swings ±100) →
divide by σ≈100 → `[-1, +1, 0]`. Same shape, gentle swings. A tiny vector `[0.1, 0.3, 0.2]` runs the
same two steps and lands at `[-1, +1, 0]` too — *whatever volume it arrives at, it leaves at volume 1.*

**Two learned dials — γ and β.** Forcing exactly "center 0, spread 1" is rigid; maybe a different
scale helps the next sub-layer. So LayerNorm finishes with `out = γ · x̂ + β`, where **γ (gamma)**
re-scales and **β (beta)** re-shifts — both *learned* like any weight. It forces a clean range, then
lets training pick the final scale/shift. Stable **and** flexible.

**Where it sits / how often.** LayerNorm is **not** a one-time cleanup. After every residual add the
drift creeps back, so a LayerNorm guards **each** sub-layer — one before attention, one before
feed-forward → **two LayerNorms per block**, repeated up the whole tower.

```
   x → [LayerNorm] → [Attention] →(+x)→ [LayerNorm] → [FeedForward] →(+)→ block out
        reset #1                          reset #2
```

### Part 4 — stacking blocks = a full Transformer

You now know the **engine** (one block). The last step is what wraps around a *stack* of them: how a
sentence gets **into** the first block, and how the last block's output becomes an actual next-word
guess. The middle is just "the same block, N times."

**① Input — words → vectors (Module 3 + positional).**

```
   "the lion will" → [tokenize] → IDs → [embedding lookup] → vectors → [+ position] → input vectors
```

Each word becomes a vector carrying **what** it is (embedding) **+ where** it sits (positional). That's
the input to block 1.

**② Body — a stack of N identical blocks.**

```
   input vectors → [block 1] → [block 2] → ⋯ → [block N] → refined vectors
```

Every block does the same mix→think (with residuals + layer norm), each handing the next a richer,
more context-aware version. **N is a design choice** (GPT-2 small = 12; big models 96+). Crucially the
**shape never changes** — N vectors in, N vectors out, same size all the way up. The *container* stays
fixed; the *contents* get better. This stacking is exactly why these are called **deep** networks.

*Why many small blocks instead of one big one — staged refinement.* Each block can only build on what
the previous block already worked out: block 1 catches simple/local relationships, block 5 builds on
that for deeper ones, block 12 deeper still — like re-reading, where each pass goes further only
because the last pass laid a base. One single block, however large, is just **one pass** and can't
build on its own output. Depth = layers of understanding stacked on each other.

**③ Output — final vectors → next-word probability.**

We take the vector at the **last position** (the word we're predicting *after*) and turn it into a
guess over the whole vocabulary:

```
   final vector → [final layer norm] → [unembedding: × big matrix] → one score per vocab word → [softmax] → probabilities
                                                                       └─ these scores = "logits" ─┘
```

- a final **layer norm** tidies the vector;
- a big matrix (vocab_size rows) produces **one raw score per vocabulary word** — these are the
  **logits**;
- **softmax** squashes the logits into a probability distribution summing to 1.

That distribution is the **next-token prediction** from Module 0 — now you know every piece that makes
it. The whole pipeline:

```
   WORDS → embed + position → [ block ×N ] → final norm → logits → softmax → next-token probabilities
            └─ Module 3 ──┘     └─ Mod 4 ─┘                 └──── back to Module 0 ────┘
```

**That diagram is a complete GPT.** Everything left (Modules 5–7) is *training* it and *running* it —
the architecture itself is now fully assembled.

## Check yourself

- A block has two sub-layers — name each and its job. (Attention = mix information *between* words;
  feed-forward = each word *processes* what it gathered, on its own.)
- The feed-forward part isn't a new idea — what is it? (The Module 2 layers: matrix → activation →
  matrix, applied per word with shared weights.)
- Why isn't attention alone enough? (It's a weighted *sum* — it gathers/averages but doesn't deeply
  process; the FFN does the transforming.)
- What does a residual connection do, and why? (`output = input + sublayer(input)` — adds the
  sub-layer's change on top of the original instead of replacing it; keeps the original intact and
  lets the backprop signal flow, so deep stacks are trainable and each sub-layer only learns a
  correction.)
- What problem does layer normalization solve, what two moves does it make, and where does it sit?
  (Drift — residual adds make numbers grow huge/tiny across the deep stack, destabilizing training. It
  **subtracts the mean** (recenter to 0) and **divides by the spread σ** (re-scale to standard volume),
  keeping the relative pattern. Sits **before each sub-layer** → two per block; γ and β are learned
  dials that let training pick the final scale/shift.)
- Trace a full Transformer end to end. (**Input:** tokenize → embedding lookup → add positional →
  input vectors. **Body:** a stack of N identical blocks, each refining the vectors — same shape in and
  out, contents get richer; depth gives *staged* refinement (each block builds on the previous one's
  output). **Output:** take the last position's vector → final layer norm → multiply by the unembedding
  matrix to get **logits** (one score per vocab word) → softmax → next-token probability distribution.)
- Why stack many blocks instead of one big one? (Staged refinement — each block can only build on what
  the previous block worked out; one block is a single pass and can't build on its own output.)

## Questions we worked through

- **Q (top rung): A block has two sub-layers run one after the other — name both, say which moves
  information between words vs. processes each word on its own, and what the second is built from.**
  A: First = **attention** (moves info *between* words); second = **feed-forward** (each word
  processes its info *on its own*), built from the **Module 2 layers** — matrix → activation → matrix,
  per word with shared weights.
- *Ladder we climbed:* a common sticking point is not seeing the *core* of the transformer / why it
  exists. Step back to the big picture: the transformer's one goal is next-word prediction, which
  needs each word understood *in context*; a "block" is one unit of "refine the understanding a
  little," and stacking many blocks (each building on the previous one's output, like re-reading)
  deepens understanding of the whole sentence. A gentler rung — "why stack the same block many
  times?" — leads to "to understand the whole sentence, not just one attention step." Climbing back
  to the two-sub-layer question then closes it.
- **Q (top rung): Walk through layer normalization end to end — what problem it solves, the two moves
  it makes (and what each fixes), and where it sits / how often.** A: solves number **drift** (residual
  adds make vectors huge/tiny across the stack); **subtracts the mean** to fix "too high/low" and
  **divides by the spread** to fix "swings too big/small"; sits **before attention and before
  feed-forward** (twice per block).
- *Ladder we climbed:* the *problem* (drift → huge/tiny, wanting consistent numbers) usually lands on
  the first try, but the divide-by-σ math is easy to lose. Step down: drop the formula and use a
  **volume-knob** analogy + a concrete `[100,300,200]→[-1,+1,0]` example (and the same for a tiny
  vector) to show the divide auto-levels to volume 1 — "both end up the same standard volume, no
  drift." A sharp question worth raising — *can a vector come out big again after normalizing?* —
  opens the **placement** point (residual re-adds drift, so LayerNorm guards every sub-layer). On the
  two-moves question the *where* (before attention/FFN) comes easily; a common slip is calling both
  moves "dividing" — isolate move #1 to see it's **subtracting** the mean. Re-asking the full top rung
  then brings the whole thing together in one connected answer.
- **Q (top rung): As N blocks are stacked the vectors keep the same shape in and out — so what is each
  block actually doing, and why stack many instead of one bigger block?** A: each block **refines** the
  vectors a little more toward the goal (next prediction), folding in the other words (attention) and
  processing them (FFN), "building better and better values"; you stack many because of **staged
  refinement** — each block builds on the previous one's output, so depth = layers of understanding,
  which one single (even huge) pass can't give.
- *Ladder we climbed:* the refinement idea usually comes easily ("more relation to the goal… building
  better and better values"). Sharpen the *why many vs. one big* into the explicit **staged-refinement**
  point (a block can't build on its own output; depth lets block 5 build on block 1). Then frame the
  full input→body→output pipeline (embed+position → N blocks → final norm → logits → softmax) as the
  close of the part and of Module 4.

## What's next / depends on

- **Depends on:** [Multi-head attention](03-multi-head-attention.md) and
  [Self-attention (Q/K/V)](02-self-attention-qkv.md) (the "mix" sub-layer),
  [Layers & the forward pass](../02-neural-networks/02-layers-and-the-forward-pass.md) (the FFN is
  just these layers), [Positional information](04-positional-information.md),
  [Embeddings](../03-language-as-data/04-embeddings-words-as-vectors.md) and
  [Softmax / probability distribution](../01-foundations/04-probability-basics.md) (the input and
  output stages of the full Transformer).
- **This concept is complete, and it finishes Module 4 — the Transformer architecture is fully
  assembled.** Next up is **Module 5 — Training LLMs** (next-token prediction at scale, datasets,
  what learning adjusts, overfitting vs. generalization). Recommended first: a short recap of the
  pieces Module 5 leans on — the training loop, loss/cross-entropy, gradient descent & backprop
  (Module 2), and how this whole Transformer produces the next-token distribution it will be trained
  on.
