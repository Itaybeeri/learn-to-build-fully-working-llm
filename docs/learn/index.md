# Learn to Build a Fully Working LLM — Learning Index

Your master map. Every concept is a small file; read them in order. Each concept lists the
**parts** you work through with the tutor. Stop and resume anytime — **your personal progress
lives in your private, gitignored `.progress/` folder**: the tutor keeps `.progress/progress.md`
(where you are right now) and `.progress/my-index.md` (your own copy of this map, checked off
concept by concept as you learn) up to date for you. This index itself stays pristine for
everyone — open `.progress/my-index.md` whenever you want to see how far you've come.

> **How depth works:** intuition first, then the *key* math with every symbol explained
> from zero. No prior ML assumed. Modules 2–6 each add a small **runnable demo**; Module 7
> assembles them into a tiny working GPT.
>
> **Glossary:** running list of terms (weight, parameter, token, vector, …) lives in
> [`glossary.md`](glossary.md).
>
> **New here?** Open this folder in Claude Code and run `/start` — the tutor will greet you,
> ask a couple of quick questions, and begin at Module 0. Come back later and run `/continue`
> to pick up exactly where you left off. (Your progress is saved privately in `.progress/`.)

---

## Module 0 — Orientation
- [What is an LLM? (the big picture)](00-orientation/01-what-is-an-llm.md)
    - Part 1 — next-word prediction & the generation loop
    - Part 2 — the output is a probability over every possible word
- [How an LLM learns, and where its "knowledge" lives](00-orientation/02-how-llms-learn-and-where-knowledge-lives.md)
    - Part 1 — learning by the guessing game (practice, then nudge)
    - Part 2 — knowledge is dials (weights), not stored text
- Getting Python ready (for the demos in Module 2+) — Python 3.12 recommended

## Module 1 — Foundations *(the gentle math)*
- [Numbers & vectors — what a vector *represents*](01-foundations/01-numbers-and-vectors.md)
    - Part 1 — why one number isn't enough (map → vector)
    - Part 2 — vectors describe *things*; dimension
- [Matrices & matrix multiplication (intuition)](01-foundations/02-matrices-and-matrix-multiplication.md)
    - Part 1 — a matrix is a stack of vectors
    - Part 2 — the dot product (a weighted sum)
    - Part 3 — matrix × vector = a stack of dot products (re-expressing meaning)
    - Part 4 — why thousands of them = a model
- [Functions & gradients (intuition)](01-foundations/03-functions-and-gradients.md)
    - Part 1 — a function (in → rule → out)
    - Part 2 — the error / loss meter (the valley)
    - Part 3 — slope = the gradient
    - Part 4 — the update rule (gradient descent)
- [Probability basics](01-foundations/04-probability-basics.md)
    - Part 1 — what a single probability is (the 0–1 scale)
    - Part 2 — a probability distribution (sums to 1)
    - Part 3 — probability is how we score "how wrong"

## Module 2 — Neural networks *(+ runnable demo)*
- [What a neuron is](02-neural-networks/01-what-a-neuron-is.md)
    - Part 1 — a neuron is a weighted sum (+ where inputs vs. weights come from)
    - Part 2 — the activation function (the shaping "switch")
- [Layers & the forward pass](02-neural-networks/02-layers-and-the-forward-pass.md)
    - Part 1 — a layer is a stack of neurons (one matrix × vector + activation)
    - Part 2 — the forward pass (data flowing layer to layer)
- [Loss — measuring "wrong"](02-neural-networks/03-loss-measuring-wrong.md)
    - Part 1 — loss is one number; squared error
    - Part 2 — cross-entropy (loss for probability outputs)
    - Part 3 — averaging into one batch loss
- [Gradient descent & backprop](02-neural-networks/04-gradient-descent-and-backprop.md)
    - Part 1 — gradient descent over many weights (the sign trick)
    - Part 2 — backpropagation (passing blame backward)
- [The training loop](02-neural-networks/05-the-training-loop.md)
    - Part 1 — the loop itself (tiny steps, repeated)
    - Part 2 — batches & epochs
- [Runnable demo — a tiny network that learns XOR from scratch](02-neural-networks/demo/README.md)
- [**Interactive Learning Portal**](portal/README.md) — open `portal/index.html` in a browser:
  watch the XOR network learn step-by-step (forward → loss → backward → update), with every
  number explained. (A tab per module.)

## Module 3 — Language as data *(+ runnable demo)*
- [Why representing language is hard](03-language-as-data/01-why-representing-language-is-hard.md)
    - Part 1 — the mismatch: networks eat numbers, language carries meaning
    - Part 2 — why numbering words (1,2,3…) fails
    - Part 3 — what a good representation must do
- [Tokens & tokenization](03-language-as-data/02-tokens-and-tokenization.md)
    - Part 1 — what a token is; why "one token per word" fails
    - Part 2 — the other extreme: characters
    - Part 3 — sub-words: the sweet spot
- [Vocabularies](03-language-as-data/03-vocabularies.md)
    - Part 1 — what a vocabulary is; frequency decides
    - Part 2 — BPE: merge the most frequent pair
    - Part 3 — what you end up with (the merge rulebook)
- [Embeddings — words as vectors](03-language-as-data/04-embeddings-words-as-vectors.md)
    - Part 1 — the embedding lookup table (ID = address, vector = meaning)
    - Part 2 — positioned by meaning, learned by training
    - Part 3 — why training arranges them by meaning
- [**Portal — Module 3 tab**](portal/modules/module3.js) — Word map (embeddings + `king − man + woman ≈ queen`) & Tokenizer (BPE)

## Module 4 — The Transformer *(+ runnable demo)*
- [The attention idea](04-the-transformer/01-the-attention-idea.md)
    - Part 1 — why a fixed embedding isn't enough (context-free → context-aware)
    - Part 2 — the idea: score by similarity, blend by relevance
    - Part 3 — the weighted-average mechanic (a blend that sums to 1)
- [**Portal — Module 4 tab**](portal/modules/module4.js) — Attention blend: pick a subject, watch
  "eat" score every word (dot product), soften to weights (softmax), blend the vectors, and flip the
  next-word guess between meat & grass
- [Self-attention (Q/K/V)](04-the-transformer/02-self-attention-qkv.md)
    - Part 1 — one vector can't do three jobs (split into query, key, value)
    - Part 2 — three learned weight matrices make Q, K, V (learned/general/shared)
    - Part 3 — the full mechanic (score = Q·K, softmax, blend = weighted sum of V)
- [Multi-head attention](04-the-transformer/03-multi-head-attention.md)
    - Part 1 — why one head isn't enough (one head = one relationship)
    - Part 2 — the mechanism (h heads in parallel, concatenate + combine)
    - Part 3 — what heads specialize in & why it helps
- [Positional information](04-the-transformer/04-positional-information.md)
    - Part 1 — attention is blind to word order (a bag of words, not a sequence)
    - Part 2 — the fix: add a position vector to each token embedding (what + where)
    - Part 3 — how position vectors are made (learned table vs. sinusoidal waves)
- [A transformer block (feed-forward, residuals, layer norm)](04-the-transformer/05-a-transformer-block.md)
    - Part 1 — two sub-layers: attention (mix) + feed-forward (think)
    - Part 2 — residual connections (add the input back, don't replace)
    - Part 3 — layer normalization (subtract mean to recenter, divide by spread to re-scale)
    - Part 4 — stacking blocks = a full Transformer (embed+position → N blocks → final norm → logits → softmax)

## Module 5 — Training LLMs *(+ runnable demo)*
- [Next-token prediction](05-training-llms/01-next-token-prediction.md)
    - Part 1 — the text labels itself (self-supervised; the next word is a free label)
    - Part 2 — the sliding window (one sentence = many examples)
    - Part 3 — predicting well forces learning grammar, facts, reasoning (the task is the *pressure*)
- [Datasets & scale](05-training-llms/02-datasets-and-scale.md)
    - Part 1 — where the text comes from (web/books/Wikipedia/code) & how much (trillions of tokens); self-supervision makes it assemblable
    - Part 2 — data quality: garbage in, garbage out; cleaning, dedup, safety filtering (quality > quantity)
    - Part 3 — scale: data + parameters + compute (grown together), scaling laws, emergent abilities
- [What "learning" actually adjusts](05-training-llms/03-what-learning-adjusts.md)
    - Part 1 — the inventory of learnable weight groups (embeddings, Q/K/V, W_O, FFN, γ/β, positional, unembedding); all nudged together
    - Part 2 — what each group learns (embeddings=meaning, Q/K/V=relationships, FFN=processing+facts)
    - Part 3 — knowledge is distributed (no single "Paris" weight); architecture fixed, only weight values learned
- [Overfitting vs. generalization](05-training-llms/04-overfitting-vs-generalization.md)
    - Part 1 — memorizing vs. learning (two ways to lower the same loss; train loss is only a proxy)
    - Part 2 — detecting it: train/validation split & the two diverging loss curves
    - Part 3 — causes (capacity · data · time) & remedies (right-size · more/clean data · early stopping · regularization)

## Module 6 — Using & aligning LLMs
- [Inference & sampling (temperature, top-k/p)](06-using-and-aligning-llms/01-inference-and-sampling.md)
    - Part 1 — inference & the sampling problem (greedy/argmax is deterministic → weighted sampling)
    - Part 2 — temperature (the randomness dial: low sharpens → facts, high flattens → creative)
    - Part 3 — top-k / top-p (high temp lifts the junk tail → trim it: fixed count vs. adaptive nucleus)
- [Context windows](06-using-and-aligning-llms/02-context-windows.md)
    - Part 1 — what a context window is (max token span; a hard ceiling, not a fading memory)
    - Part 2 — why the limit exists (attention is n² — every token scores every token; fixed positional table)
    - Part 3 — what happens at the edges (truncate/slide & drop oldest; summarize; enlarge via efficient attention + formula positions)
- [Fine-tuning](06-using-and-aligning-llms/03-fine-tuning.md)
    - Part 1 — the two-stage idea (pre-train → generalist, fine-tune → specialist; why it's cheap)
    - Part 2 — how it works mechanically (same loop, small targeted data, small learning rate, few steps)
    - Part 3 — trade-offs (catastrophic forgetting = overwriting; LoRA/PEFT freezes base, trains small add-on)
- [Instruction tuning & RLHF (alignment)](06-using-and-aligning-llms/04-instruction-tuning-and-rlhf.md)
    - Part 1 — the problem: a base model completes text, it doesn't obey (alignment is the missing piece)
    - Part 2 — instruction tuning / SFT (fine-tune on human-written instruction → answer pairs; reshapes the instinct, still next-token prediction)
    - Part 3 — RLHF (Move 1: reward model from A-vs-B human comparisons; Move 2: reinforcement — generate → score → nudge)
- [Limitations (hallucination, etc.)](06-using-and-aligning-llms/05-limitations.md)
    - Part 1 — hallucination (plausibility not truth; no fact-check, no "I don't know"; a design consequence)
    - Part 2 — other limitations (knowledge cutoff, shaky reasoning/math, bias, no memory; all = "fixed weights predicting plausible tokens")
    - Part 3 — mitigations & why it's never fully solved (RAG, tools, grounding; they shrink not delete — always verify)

## Module 7 — Capstone: build a tiny GPT
*Built twice, same architecture: **7A** in PyTorch (autograd handles backprop), then **7B** in pure NumPy from
scratch (forward + backward by hand). 7A first = quick, testable, and a correctness reference for 7B.*
- [Overview & plan](07-capstone-tiny-gpt/01-overview-and-plan.md)
- [Stage 1 — data & character tokenizer](07-capstone-tiny-gpt/02-data-and-char-tokenizer.md) *(shared by 7A & 7B)* — 65-char vocab, encode/decode, sliding-window (x,y) batches; runs (`demo/tiny_gpt_pytorch.py`); **portal:** Module 7 tab → Stage 1 live tokenizer
- [Stage 2 — the model](07-capstone-tiny-gpt/03-the-model.md) (token + positional embeddings → N Transformer blocks → final norm → logits; **causal mask** = only look back; 159,937 params; untrained loss ≈ ln(65)); **portal:** Module 7 → Stage 2 forward-pass diagram
- [Stage 3 — training](07-capstone-tiny-gpt/04-training.md) — the Module-2 loop on the GPT (four steps: batch → forward + cross-entropy → `loss.backward()` → AdamW `optimizer.step()`), watching train & val loss fall together; **portal:** Module 7 → Stage 3 loss curve
- [Stage 4 — generation](07-capstone-tiny-gpt/05-generation.md) — the autoregressive loop (forward → last-position scores → temperature+softmax → sample → append → repeat); prompt `"ROMEO:"` → Shakespeare-*shaped* text from the 160k-param model; **portal:** Module 7 → Stage 4 (autoregressive-loop stepper + temperature sampler)
- [Scale-up](07-capstone-tiny-gpt/06-scale-up.md) — turn the size knobs up (`--big` preset, ~2.7M params) for readable samples; on a CUDA GPU the same code produces **real English words** instead of letter-soup — Module 5's scaling lesson made tangible
- [7B — the same tiny GPT in pure NumPy from scratch](07-capstone-tiny-gpt/07-numpy-from-scratch.md) (hand-written forward + backward, grad-checked to ~1e-7; trains & generates; 7A is the reference)
    - Part 1 — the backward contract + numeric gradient-check + the Linear layer
    - Part 2 — embedding, LayerNorm, ReLU, softmax + cross-entropy
    - Part 3 — causal self-attention forward + backward (per-head + multi-head)
    - Part 4 — the model (Block → TinyGPT)
    - Part 5 — training loop (manual backprop + hand-written Adam)
    - Part 6 — generation
