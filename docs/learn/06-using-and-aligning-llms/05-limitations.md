# Limitations — hallucination and the other ceilings

## In one sentence

An LLM **hallucinates** — confidently states falsehoods — because its only job is to produce the most
*plausible* next token, with no built-in sense of truth and no way to know what it doesn't know.

## Why it matters

Hallucination is the single most important limitation to understand when *using* LLMs: it's why you must verify
anything factual, and why an LLM should never be trusted blindly for citations, numbers, or claims. Knowing it
falls out of the core design — not a fixable glitch — tells you it will always be *present*, and shapes how you
use the model (verify, ground it in sources, give it tools).

## The intuition

### Part 1 — hallucination: why the model confidently makes things up

LLMs sometimes state, with total confidence, something flatly false — invent a book that doesn't exist, cite a
fake court case, give a wrong date without a flicker of doubt. This is **hallucination**. The question that
matters is *why*, and the cause is baked into everything we've learned.

The model's one and only job: **predict the most plausible next token** — not the *true* one, the **plausible**
one that statistically fits. Connect that to a factual question:

```
   "The author of the novel 'The Silver Tide' is ___"
```

If no such novel exists, or the model never saw who wrote it, the model **has no true/false check** and no slot
that says *"I don't know."* All it can do is its usual job: produce the **most plausible continuation** — and
the most plausible continuation of "The author of … is" is **a plausible-sounding author's name.** So it
confidently emits one, invented from the statistical shape of language. To the model, a fluent made-up answer
and a fluent true answer look **identical** — both are just high-probability token sequences.

Root cause, one line: **the model is optimized for *plausibility*, not *truth* — no built-in notion of fact, no
"I don't know," so when it lacks the real answer it fills the gap with something that merely *sounds* right.**

Two things make it worse:

- **It was *trained* to always produce fluent text.** Pre-training rewards confident continuation; nothing
  rewarded "I'm not sure." (RLHF helps a little — it can reward hedging — but can't install a truth-sense that
  isn't there.)
- **Knowledge is distributed & lossy (Module 5).** Facts aren't exact records in a database — they're smeared
  across billions of weights, so recall is *reconstructive*, like human memory, and reconstruction can blend
  into a confident-but-wrong answer.

The reframe: hallucination isn't a *bug* a patch removes — it's a **direct consequence of what the model is** (a
plausibility machine, not a truth machine). You can *reduce* it; you can't design it away. And you can't simply
"tell it to be truthful," because there's **no truth-signal inside it to consult** — true and plausible-made-up
look the same to it.

### Part 2 — the other limitations

Hallucination is the headline, but a few more ceilings matter just as much — and each traces back to something
we've already learned.

**1. Knowledge cutoff (frozen in time).** The model's knowledge is whatever was in its **training data**, baked
into the weights at training. After training, it learns nothing new on its own, so it doesn't know events after
its **cutoff date** — no today's news or stock price. *Why:* knowledge lives in **fixed weights**, and using the
model (inference) doesn't change them.

**2. Shaky reasoning & arithmetic.** Ask "4,839 × 2,176?" and it may answer confidently wrong. *Why:* it isn't
*calculating* — it's **predicting plausible tokens**. It never runs an arithmetic algorithm; it pattern-matches
what a product "looks like." Common sums it's seen often → right; novel multi-step problems → it can drift,
because plausible-looking ≠ logically-correct. Same root as hallucination.

**3. Bias from training data.** It learned from human text, absorbing the **patterns** in it — including
stereotypes and slanted views. "Garbage in, garbage out" (Module 5) applies to **values** too: a lean in the
data becomes a lean in the model. Cleaning and RLHF reduce it but can't fully remove what's woven through the
statistics.

**4. No memory beyond the context window.** Straight from the previous concept: the model has **no memory
between conversations**, and within one sees only what fits the **context window**. Exceed the window or start a
new chat and it's gone — it didn't "learn" it (learning = changing weights, which inference doesn't do); it only
ever *re-reads* what's in front of it.

The unifying thread: **the model is a fixed pile of weights that predicts plausible tokens over a limited
window.** Every limitation is that one sentence from a different angle — frozen knowledge (fixed weights), bad
math (plausible not computed), bias (learned from the data), no memory (weights don't change at inference; the
window is finite).

### Part 3 — mitigations, and why it's never fully "solved"

The limitations are **built into what the model is**, so we can't remove them — we **work around** them. The
trick in almost every case is the same: **stop relying on the frozen weights alone; give the model an external
source of truth at inference time.**

**1. Retrieval / RAG (Retrieval-Augmented Generation).** The big one — it attacks hallucination *and* the
knowledge cutoff. Instead of answering from memory, **first fetch relevant real documents** (a search index,
company files, a database) and **paste them into the context window** with the question. Now the model isn't
recalling from lossy weights — it's **reading the actual source text** and answering from that.

```
   question ─► search for relevant documents ─► put them in the context window ─► model answers FROM the text
```

Why it helps: it converts an error-prone *memory* task into a *reading-comprehension* task (which the model is
good at, and the text can be today's), and it lets the model **cite sources** so you can check it. RAG =
**Retrieval** (fetch docs) **Augmented** (add to the prompt) **Generation** (then generate).

**2. Tools / function-calling.** For what the model is bad at — exact arithmetic, live data, running code —
don't let it *guess*; let it **call a real tool** (calculator, web search, database, Python interpreter). The
model decides *when* to call, the tool returns a real result, and the model uses it. Bad-at-math is solved by
handing the math to a calculator.

**3. Grounding & verification habits.** Prompt it to **say when it's unsure**, ask it to **quote sources**, and
for anything that matters, **a human verifies.** RLHF can reward honesty/hedging, so it admits uncertainty more.

**The crucial honesty.** None of these change what the model fundamentally *is.* RAG, tools, and grounding all
work by **feeding better stuff into the context window** or **outsourcing the part the model is bad at.**
Underneath it's *still* a plausibility machine predicting tokens — so it can still misread a retrieved document,
call a tool wrong, or hallucinate in the gaps. These mitigations **shrink** the problem; they don't **delete**
it. That's why "always verify important outputs" never stops being the rule.

Takeaway for the whole concept: an LLM is an extraordinary **plausible-language generator** — not a database, a
calculator, or a truth oracle. Used *as* a language engine, with real sources and tools wired in for the rest,
it's incredibly powerful. Trusted *as* an oracle, it will confidently fail.

## Check yourself

- What is hallucination? (Confidently stating false things — inventing facts, citations, dates.)
- Why does it happen? (The model predicts the most **plausible** next token, not the true one; it has **no
  true/false check** and no "I don't know," so it fills gaps with plausible-sounding filler.)
- Why isn't "just make it tell the truth" a simple fix? (There's **no truth-signal inside** to switch on — a
  true answer and a fluent made-up one are the same kind of thing to it: both high-probability token sequences.)
- What makes it worse? (Training rewarded confident fluency, not hedging; and knowledge is **distributed &
  lossy**, so recall is reconstructive and can come out confidently wrong.)
- Why is there a knowledge cutoff? (Knowledge lives in **frozen weights** set at training; inference doesn't
  change them, so the model is stuck at its training date.)
- Why is the model bad at exact arithmetic? (It **predicts plausible tokens**, it doesn't run a calculation —
  so a novel large product is just "what an answer looks like," which can be wrong.)
- Why is it biased? (It learned the **statistics of human text**; a lean in the data becomes a lean in the
  predictions — values in, values out.)
- Why no memory across chats? (Chatting is **inference**, which doesn't change weights; it only re-reads what's
  in the finite **context window**.)
- What is RAG and why does it help? (**Retrieval-Augmented Generation**: fetch real documents → add to the
  prompt → answer from that text. It turns a lossy **recall** task into a **reading-comprehension** task and
  lets the model cite sources.)
- What do tools/function-calling fix? (They outsource what the model is bad at — math, live data, code — to a
  **real calculator/search/interpreter** instead of guessing.)
- Why don't mitigations fully solve the limitations? (They feed better text in or outsource the weak part, but
  **don't change the engine** — underneath it's still a plausibility machine, so it can misread or misuse. They
  **shrink** the failures, not delete them → always verify.)

## Questions we worked through

- **Q: Why does an LLM hallucinate? Tie it to the model's one job, and say why "just make it tell the truth"
  isn't a simple fix.** A: the model does what it was trained to do — predict the next word from other tokens;
  it has no true/false check, doesn't know what the truth is, only knows the weight numbers and predictions.
  *(Correct on the core — plausibility not truth, no fact-check. Made the "not a simple fix" piece explicit:
  there's no truth-signal inside to switch on; true vs. fluent-made-up look identical to the model.)*
- **Q: Pick two of {knowledge cutoff, shaky math, bias, no memory} and trace each back to how the model
  works.** A: **shaky math** — the model doesn't compute, it predicts the next best word from learned data;
  **bias** — it's trained on whatever data it's given, so biased data → biased predictions. *(Both correct and
  well-traced. Added the other two for completeness: cutoff = knowledge in frozen weights; no memory = inference
  doesn't change weights + finite window.)*
- **Q (top rung): (1) What is RAG and why does it help? (2) Why do mitigations never fully "solve" the
  limitations — what stays true?** A: (1) RAG = retrieve documents, add to the prompt, generate — it grounds
  the answer in current/specific data so the model predicts from truth supplied for that question; (2) what
  stays true is the model is a *model*, not a DB / calculator / oracle — and when it fails, it fails with
  confidence. *(Cleared the top unaided. Sharpened: RAG turns *recall* into *reading-comprehension*; mitigations
  work **outside** the model (better context / outsourced tools) and don't change the engine, so they shrink but
  never delete the failures.)*

## What's next / depends on

- **Depends on:** [Next-token prediction](../05-training-llms/01-next-token-prediction.md) (predicting
  *plausible*, not *true*), [What "learning" adjusts](../05-training-llms/03-what-learning-adjusts.md)
  (distributed, lossy knowledge), [RLHF](04-instruction-tuning-and-rlhf.md) (can reward hedging, can't install
  a truth-sense).
- **Also leans on:** [Datasets & scale](../05-training-llms/02-datasets-and-scale.md) (garbage/values in →
  out; bias), [Context windows](02-context-windows.md) (no memory beyond the window), [Inference &
  sampling](01-inference-and-sampling.md) (inference doesn't change weights).
- **This concept is complete — and so is Module 6.** Next: a **bridge recap** of the foundations the **Module 7
  capstone** (build a tiny GPT) leans on, then assemble, train, and run a tiny character-level GPT.
