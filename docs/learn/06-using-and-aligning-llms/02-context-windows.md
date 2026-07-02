# Context windows — the model's field of vision

> **Status: complete.** ① **what a context window is** — the **maximum number of tokens** the model can
> attend over in a single forward pass (a.k.a. *context length*). It's a **hard ceiling baked into the model**
> (GPT-2 ≈ 1,024; modern models 8K–1M+), measured in **tokens** (prompt + generated output share the same
> budget). It is *not* a stored memory carried between requests — it's **how much text gets re-fed into the
> model each step**. When the conversation grows past the window, the **earliest tokens are dropped entirely**:
> not "remembered a little less" but simply **not fed in** — a hard cliff, not a fade. ② **why the limit
> exists** — two costs. (1) **Attention is n²**: each of the n tokens in the window is scored against every
> token (the Q·K step), so the work and memory grow with the **square** of the length — double the window and
> the cost *quadruples*. This is the practical ceiling. (2) The **positional setup is fixed-size**: a learned
> position table has a fixed number of rows set at training time, so a position past the last row has **no
> vector** — the model can't even represent, let alone accept, a longer sequence. A structural ceiling. ③
> **what happens at the edges** — when a conversation outgrows the window, the default is **truncation**: keep
> the most recent tokens that fit and **drop the oldest**, so the window *slides* forward like a frame on a
> filmstrip. This is lossy — early instructions/facts simply vanish (why long chats "forget the beginning";
> why key instructions go at the *end* of the prompt). A smarter variant **summarizes** the old part into a few
> tokens instead of dropping it. Separately, builders **enlarge** the window by attacking the same two Part-2
> costs: **efficient attention** (each token attends only to nearby / a few summary tokens → n² shrinks toward
> ~n) and **formula-based positions** (sinusoidal / RoPE → a vector for *any* length, beating the fixed table).

## In one sentence

A **context window** is the maximum span of tokens the model can look at — prompt plus everything generated so
far — when predicting the next token; anything past that limit is dropped from view entirely.

## Why it matters

The context window is the single most felt limit when *using* an LLM. It's why a long document has to be
chunked, why a chatbot eventually "forgets" the start of a long conversation, and why prompt + answer compete
for the same budget. Knowing it's a **hard token ceiling** (not a soft, fading memory) tells you exactly what
the model can and cannot take into account on any given step.

## The intuition

### Part 1 — what a context window is

Recall the **sliding window** from Module 5: to generate the next token, the model looks back at the tokens
before it — the prompt plus everything it has generated so far. That "everything before" is the model's
**context**: the tokens it's allowed to look at when deciding what comes next.

The catch: that context is **not unlimited**. Every model has a fixed maximum number of tokens it can hold in
view at once. That maximum is the **context window** (also *context length*) — the width of the model's "field
of vision," measured in tokens.

```
   context window = 8 tokens (toy example)
   ┌─────────────────────────────────────────┐
   │  The  cat  sat  on  the  mat  and  it    │  ← model can "see" all 8
   └─────────────────────────────────────────┘
                                          ▼
                                    predict next token

   text grows past 8 tokens:

      The  cat  sat  on  the  mat  and  it   purred   loudly  ...
     └────────── falls outside ──────────┘  └──── inside window ────┘
       (no longer visible to the model)
```

Pin down:

- It counts **tokens, not words** — and the prompt *and* the model's own output both spend from the same
  budget.
- It's a **hard ceiling baked into the model.** GPT-2 was ~1,024 tokens; modern models reach 8K, 128K, even a
  million+. Whatever the number, there's always *a* limit.
- "Inside the window" = the model attends to it. "Outside the window" = on this step, **it doesn't exist** — no
  fading memory, it's simply not fed in.

A subtle but important framing: the window is **not a memory the model stores between requests.** It's **how
much text gets re-fed into the model on each step.** Every prediction, the model is handed a fresh slab of
tokens (up to the window size) and reads it from scratch. "What it remembers" = "what fits in the slab this
time." Past that — a hard cliff, gone.

### Part 2 — why the limit exists

If a bigger window is so useful, why not make it infinite? Because it isn't free — there are two real costs.

**Reason 1 — attention's cost grows with the *square* of the length (n²).** Recall self-attention (Module 4):
to build each token's new vector, the model scores it against **every other token** in the window (the Q·K dot
products), softmaxes, then blends. So *every token looks at every token.* Count the comparisons: if the window
holds **n** tokens, each of the n tokens is scored against ~n tokens → roughly **n × n = n²** dot products.

```
   n = 100 tokens   →  100 × 100   =     10,000 comparisons
   n = 1,000        →  1,000²      =  1,000,000
   n = 10,000       →  10,000²     = 100,000,000
```

Double the window and the work doesn't double — it **quadruples**. This "n-squared" growth eats both compute
and memory (the model holds the whole n×n grid of scores). That's the **practical** ceiling.

**Reason 2 — position information is set up for a fixed maximum length.** Recall positional info (Module 4):
the model adds a **position vector** to each token so attention can tell word order apart. If those come from a
**learned table**, the table has a *fixed number of rows* — one per slot, up to the max length trained on.
Position 5,000 has no row if the model was built for 1,024 positions; the model has **no representation for
that slot**, so it can't accept a sequence that long. That's the **structural** ceiling.

### Part 3 — what happens at the edges (and how the window gets stretched)

The window is a hard ceiling, so: your conversation has 5,000 tokens but the window is 4,000. The model
**cannot** take all 5,000. What does the system around the model do?

**The default: truncation (a sliding cutoff).** Keep only the most recent tokens that fit and **drop the
oldest** to make room. As the chat grows, the window slides forward like a frame along a filmstrip:

```
   window = 4 tokens

   step A:   [ T1 T2 T3 T4 ]                      ← full
   step B:      T1  [ T2 T3 T4 T5 ]               ← T1 dropped to fit T5
   step C:         T1 T2 [ T3 T4 T5 T6 ]          ← T2 dropped to fit T6
```

This is exactly why a long chatbot conversation "forgets the beginning" — those early tokens slid out of the
frame. Two consequences:

- **It's lossy.** Whatever was in the dropped tokens (an early instruction, a fact established at the start) is
  **gone** on this step — the model truly can't see it.
- **It's why prompt ordering matters.** Critical instructions often go at the *end* of the prompt (closest to
  the prediction) so they survive longest and aren't the first to fall off.

A common smarter variant: instead of blindly dropping the oldest, **summarize** the old part into a few tokens
and keep that summary in-window — trading exact detail for a compact gist that still fits.

**Stretching the window itself.** Builders also make the window *bigger* so truncation bites later — and the
fixes target exactly the two costs from Part 2:

- **For the n² cost:** *efficient attention* — schemes where not every token attends to every token (each
  token looks only at nearby tokens, or a small set of "summary" tokens), cutting n² down toward ~n. This is
  why million-token windows are now feasible.
- **For the fixed positional table:** position methods that **extend to lengths never trained on** — the
  *sinusoidal* formula from Module 4 (a formula, not a table, so it yields a vector for *any* position), and
  modern relative schemes like **RoPE** that can be stretched past their training length.

```
   conversation longer than window
        │
        ├─ default ─► TRUNCATE: drop oldest tokens (window slides) ─► early context lost
        │              └─ smarter: SUMMARIZE old part into a few tokens
        └─ or enlarge the window ─► efficient attention (beat n²) + formula positions (beat fixed table)
```

## Check yourself

- What is a context window? (The maximum number of **tokens** the model can attend over in one forward pass —
  prompt + generated output combined.)
- Why not make the window infinite — what's the main cost? (Attention is **n²**: every token is scored against
  every token, so cost and memory grow with the **square** of the length; double the window → 4× the work.)
- What's the second, structural limit? (The **positional table is fixed-size** — rows set at training time — so
  positions past its last row have no vector and can't be represented.)
- A conversation outgrows the window — what does the system do by default, and what's the visible effect? (It
  **truncates**: drops the **oldest** tokens to fit the newest, so the window slides — early instructions/facts
  vanish, i.e. it "forgets the beginning." Smarter variant: **summarize** the old part instead.)
- How do builders make the window *bigger*, and which Part-2 cost does each fix attack? (**Efficient
  attention** — each token attends only to nearby/summary tokens → beats the **n²** cost; **formula-based
  positions** (sinusoidal / RoPE) → beats the **fixed positional table**.)
- Is it a stored memory the model keeps between requests? (No — it's **how much text is re-fed in each step**;
  the model reads the slab fresh every time.)
- A conversation grows longer than the window — what happens to the earliest tokens? (They're **dropped
  entirely**, not "remembered less." A hard cliff, not a fade.)

## Questions we worked through

- **Q: In your own words, what is a context window, and when a conversation outgrows it, do the earliest tokens
  get "remembered a little less" or something stronger?** A: it's the memory the model has when getting a
  prediction request; once the conversation is longer than the window, the first tokens just don't exist.
  *(Correct — reached for the strong version. Sharpened one word: it's not a stored "memory" carried between
  requests but **how much text is re-fed in each step**.)*
- **Q: Why can't builders just make the window infinite? Give the main cost and *why* it grows that way.** A:
  the cost grows by **n²**, which gets very big as length grows; and the **positional table is a fixed-length
  learned table from training**, so positions beyond its slots can't be used. *(Both reasons correct. Stepped
  down to isolate the "why n²": **how many tokens does each token get compared against?** → "the other words
  too" → pinned it: each of the n tokens is scored against ~n tokens, so n × n = n².)*
- **Q (top rung): A conversation outgrows the window — (1) what does the system do by default and what's the
  user-visible effect, and (2) when builders want a bigger window, which two Part-2 costs must they defeat, and
  how?** A: (1) the system **trims the first tokens**, so instructions the user gave at the start are no longer
  there; (2) defeat the **n²** cost by **not watching all tokens, only the near ones**; defeat the **fixed
  positional table** by **using a formula instead of a table**. *(Fully correct on all three — cleared the top
  of the ladder unaided. Added only: the smarter "summarize instead of drop" variant, and the names efficient
  attention / sinusoidal / RoPE.)*

## What's next / depends on

- **Depends on:** [The sliding window](../05-training-llms/01-next-token-prediction.md) (context = the tokens
  before the target), [Tokens & tokenization](../03-language-as-data/02-tokens-and-tokenization.md) (the window
  is measured in tokens), [Self-attention](../04-the-transformer/02-self-attention-qkv.md) (the n² scoring that
  caps length), [Positional information](../04-the-transformer/04-positional-information.md) (learned table vs.
  sinusoidal formula).
- **This concept is complete.** Next in Module 6: **Fine-tuning** — taking a pre-trained model and training it
  further on a smaller, targeted dataset to specialize it.
