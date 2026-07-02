# Glossary

A running list of terms, growing as we learn. `✅ = already covered` · `⬜ = coming up`.
Short, plain-language definitions; the full explanation lives in each term's concept note.
**Grouped by the module that introduces each term** (same organization as the index, the notes,
and the portal). A term is listed once, under the module where it's first taught.

## Module 0 — Orientation

| Term | Status | Plain meaning |
|---|---|---|
| **LLM (Large Language Model)** | ✅ | A system that predicts the next chunk of text, looping to write whole answers. |
| **Next-word / next-token prediction** | ✅ | The one job an LLM does: given the text so far, guess what comes next. |
| **Training** | ✅ | Teaching the model by making it practice next-word prediction on huge text and nudging its numbers to be less wrong. |
| **Weight / parameter** | ✅ | One of the model's adjustable numbers ("dials"); together they ARE the model. "7B-parameter" = 7 billion such dials. |
| **Model** | ✅ | The trained pile of weights — what's left after training; the text is thrown away. |

## Module 1 — Foundations *(the gentle math)*

| Term | Status | Plain meaning |
|---|---|---|
| **Vector** | ✅ | An ordered list of numbers, each slot with a fixed meaning; how meanings and "dials" are represented. |
| **Dimension** | ✅ | How many numbers are in a vector (`[3,5]` = 2-D; 300 numbers = 300-D); each is one independent direction/attribute. |
| **Component / entry** | ✅ | One of the individual numbers inside a vector. |
| **Matrix** | ✅ | A grid of numbers (rows × columns); a stack of vectors. The model's weights are matrices. |
| **Dot product** | ✅ | Multiply matching slots of two equal-length vectors and sum → one number; measures how much they align. |
| **Matrix multiplication** | ✅ | Dotting each matrix row with an input vector → a new vector; the matrix *transforms* (re-expresses) the input. |
| **Function** | ✅ | A machine that turns an input into an output via a rule; written `f(x) = rule`. |
| **Error / Loss** | ✅ | A function from a knob (weight) setting to a single number: how wrong the model is. Small = good. |
| **Slope / Gradient** | ✅ | How much the output changes when you nudge an input; points uphill, so step the opposite way to lower error. Gradient = slope with many inputs. |
| **Learning rate** | ✅ | The step size in the update rule `w_new = w_old − rate × slope`; too big overshoots, too small crawls. |
| **Gradient descent** | ✅ | The loop "measure slope → step opposite → repeat" that walks the weights to the bottom of the error valley. |
| **Probability (P)** | ✅ | A number 0–1 saying how likely something is; the model gives one to every possible next word (they sum to 1). |
| **Probability distribution** | ✅ | A probability for every possible outcome, all summing to exactly 1 — a fixed budget the outcomes compete for. An LLM's output is one of these over its vocabulary. |

## Module 2 — Neural networks

| Term | Status | Plain meaning |
|---|---|---|
| **Neuron** | ✅ | A unit that does a weighted sum of its inputs (a dot product → one raw score), then passes that score through an activation function. The building block of every layer. |
| **Activation function** | ✅ | The shaping step after a neuron's weighted sum. Two jobs: **fences** (squash the unbounded score into a readable range so it means something) and **bend** (a non-linear kink/curve that stops stacked layers from collapsing into one and lets the network learn curves). |
| **Linear / Non-linear** | ✅ | *Linear* = one single straight line describes the whole function (same step everywhere; chains of them collapse to one step). *Non-linear* = the relationship changes between regions (a corner or curve); adding 1 to the input moves the output by different amounts in different places. |
| **Step function** | ✅ | The bluntest activation: score ≥ 0 → 1 (fire), else 0 (silent). A hard on/off switch — historical, too blunt for confidence. |
| **Sigmoid (σ)** | ✅ | A smooth S-shaped activation, `σ(z) = 1/(1+e^(−z))`, squashing any score into (0,1): very negative → ~0, zero → 0.5, very positive → ~1. Steep in the middle, flat at the edges; output reads as a probability/confidence. |
| **ReLU** | ✅ | The simplest popular activation: negative score → 0, otherwise pass the score through unchanged. One sharp corner makes it non-linear; cheap and the most-used in real networks. |
| **Layer** | ✅ | Several neurons side by side, all reading the same input vector (each with its own weights); turns a vector into a new vector. Its weights = one matrix (rows = neurons = outputs, columns = inputs). |
| **Forward pass** | ✅ | Running the model: data flows in as a vector, each layer does `W×input` + activation, hands its output to the next layer, on to the final answer `ŷ`. No learning — just producing a guess. |
| **Hidden layer** | ✅ | Any layer between input and output; its output vector is "hidden" (not the input, not the final answer) and lives inside the network. |
| **Width / Depth** | ✅ | Width = how many neurons in a layer (how many patterns it can attend to); depth = how many layers. Both are capacity dials with tradeoffs (too small underfits, too big is costly/overfits). |
| **Hyperparameter** | ✅ | A setting *you choose before training* (width, depth, learning rate), as opposed to weights, which training *learns*. Tuned by experiment, judged on a validation set. |
| **Squared error / MSE** | ✅ | Loss for a single-number prediction: `(ŷ − y)²`. Kills the sign and punishes big misses disproportionately; averaged over examples = Mean Squared Error. |
| **Cross-entropy / Negative log-likelihood** | ✅ | The loss for probability outputs (used to train LLMs): `−log(p_correct)`, where `p_correct` is the probability the model gave the true word. 0 when perfect (p=1), huge when confidently wrong (p→0). |
| **Logarithm (log)** | ✅ | A function used in cross-entropy; on inputs in (0,1) it turns them negative and dives to −∞ as the input → 0. `log(1)=0`, `log(0.1)≈−2.3`. |
| **Gradient (full-network)** | ✅ | The bundle of slopes — one per weight — each saying how the loss reacts to nudging that weight. Backprop's output; the update step turns it into nudges. |
| **Backpropagation** | ✅ | The efficient algorithm that computes every weight's slope in one backward pass, sending the loss's "blame" back through the layers and multiplying local layer-to-layer effects (chain rule), reusing downstream results. |
| **Chain rule** | ✅ | The "multiply the link-by-link effects to get the end-to-end effect" rule; the engine of backprop (no calculus needed — just chain the local effects). |
| **Nudge vs. slope** | ✅ | Slope = measurement (how loss reacts to a weight); nudge = the action applied, `−(rate × slope)`. Backprop produces slopes; the update applies nudges. |
| **Training loop** | ✅ | The repeated cycle forward → loss → backprop → update; each turn nudges weights a hair downhill. Run millions of times until the loss flattens. |
| **Batch / mini-batch** | ✅ | The small handful of examples (e.g. 32) processed in one loop turn = one weight update. Averages out single-example noise while staying fast. |
| **Epoch** | ✅ | One full pass through the entire training dataset (all its batches). Training usually runs many epochs, reshuffling each time. |
| **Stochastic gradient descent (SGD)** | ✅ | Gradient descent using a random batch each step rather than the whole dataset; the standard way networks are trained. |

## Module 3 — Language as data

| Term | Status | Plain meaning |
|---|---|---|
| **Token** | ✅ | The discrete chunk of text a model actually reads/predicts. Real LLMs use *sub-word* tokens: common words stay whole, rare/new words split into reusable pieces (down to characters), so a modest list spells anything. |
| **Tokenization** | ✅ | The process of cutting raw text into tokens before the model reads them. Whole-words → huge vocab + out-of-vocabulary; characters → meaning lost + long sequences; sub-words → the sweet spot. |
| **Out-of-vocabulary (OOV)** | ✅ | A word the model has no token/vector for, so it can't represent it. Sub-word tokenization removes the problem (any word builds from known pieces). |
| **Vocabulary** | ✅ | The model's fixed, numbered list of every token it knows. A token's **ID** = its position in the list. Built once from data, fixed size (~30k–100k). |
| **Byte Pair Encoding (BPE)** | ✅ | The method that learns the vocabulary: start from single characters, repeatedly merge the most frequent adjacent pair into a new token until the list hits the target size. The ordered merges are re-applied to tokenize new text. |
| **Token ID** | ✅ | The integer index of a token in the vocabulary; tokenized text becomes a list of these IDs, which then map to embedding vectors. |
| **Embedding** | ✅ | The learned vector a token's ID looks up in the embedding table — the meaning-carrying representation the network reads. Similar words get nearby vectors (distance = relatedness); directions carry meaning (king − man + woman ≈ queen). |
| **Embedding table / matrix** | ✅ | The lookup table of one vector per vocabulary token (vocab_size × d). Just more learned weights; arranged by training. The token ID is the row address. |
| **Embedding dimension (d)** | ✅ | How many numbers are in each embedding vector (e.g. 256, 768) — a design choice. |

## Module 4 — The Transformer

| Term | Status | Plain meaning |
|---|---|---|
| **Attention** | ✅ | The step that lets each word look at the others, score who's relevant, and rewrite itself as a blend of them — turning a context-free embedding into a context-aware one. |
| **Self-attention (Q/K/V)** | ✅ | Attention made precise: each word emits a query (W_Q·x), key (W_K·x), value (W_V·x); score = query·key, softmax → weights, new vector = weighted sum of values. The matrices are learned. |
| **Query / Key / Value** | ✅ | Three learned re-expressions of a word's embedding: query = what I seek, key = how I'm found, value = what I contribute. Score on key, take the value. |
| **Softmax** | ✅ | Squashes a list of scores into positive weights that sum to exactly 1 — used to turn attention scores into blend weights (and to turn final scores into a probability distribution). |
| **Head** | ✅ | One complete self-attention computation — one set of W_Q/W_K/W_V. |
| **Multi-head attention** | ✅ | Several heads run in parallel, each its own Q/K/V learning a different relationship; their outputs are concatenated and combined by a learned output matrix W_O into one rich vector per word. |
| **Positional information / encoding** | ✅ | A position vector added onto each token's embedding (right after lookup, before attention) so the order-blind attention sum can tell word order apart. Made by a learned table (capped length) or a fixed sinusoidal wave formula (any length). |
| **Transformer block** | ✅ | The repeatable building unit of a Transformer: an attention sub-layer (words *mix* info between each other) then a feed-forward sub-layer (each word *thinks* on its own), each wrapped with a residual connection and layer norm. Stacking many identical blocks = the Transformer body. |
| **Feed-forward network (FFN)** | ✅ | The "think" sub-layer of a transformer block: the Module 2 layers (matrix → activation → matrix) applied to **each word's vector independently**, with the **same shared weights** at every position. Transforms the info attention gathered. |
| **Residual / skip connection** | ✅ | `output = input + sublayer(input)` — a sub-layer *adds* its result on top of its input instead of replacing it. Keeps the original meaning intact and gives backprop a straight "highway," so very deep stacks stay trainable; each sub-layer only learns a small correction. |
| **Layer normalization (LayerNorm)** | ✅ | A reset placed before each sub-layer that re-centers a word's vector (subtract its mean μ) and re-scales it (divide by its spread σ) so the numbers stay at a stable, standard volume and don't drift huge/tiny across the deep stack. Two learned dials (γ scale, β shift) let training pick the final scale. |
| **Logits** | ✅ | The raw, un-normalized scores the final layer produces — one per vocabulary token (final vector × the unembedding matrix). Softmax turns them into the next-token probability distribution. |
| **Transformer** | ✅ | The neural-network architecture LLMs are built from: input (embed + positional) → a stack of N identical transformer blocks → output (final layer norm → logits → softmax = next-token distribution). The whole of Module 4. |

## Module 5 — Training LLMs

| Term | Status | Plain meaning |
|---|---|---|
| **Self-supervised learning** | ✅ | Training where the data provides its own labels — no humans tag anything. For an LLM the label is just the *next word*, already present in the text, so all raw text is instantly a labeled dataset (no labeling bottleneck → train on ~everything). |
| **Label (training)** | ✅ | The "correct answer" paired with a training input. In next-token prediction the label is the actual next token in the text itself. |
| **Underfitting / Overfitting** | ✅ | Underfit = too simple, bad even on training data. Overfit = memorizes the training text, poor on unseen data (training loss is only a *proxy* — a high-capacity model can game it by memorizing). Sweet spot = good on both. |
| **Generalization** | ✅ | Learning patterns/rules that still work on inputs the model never saw (the real goal of training), as opposed to memorizing specific examples. The opposite end of the spectrum from overfitting. |
| **Validation set (held-out / dev set)** | ✅ | A chunk of data set aside before training and **never trained on**, so it stays a fair stand-in for unseen text. Measuring loss on it reveals generalization; the train/validation loss gap reveals overfitting. |
| **Early stopping** | ✅ | Halting training at the lowest **validation** loss — before it turns back up — so the model is kept at its best on unseen text instead of overfitting the training text. |
| **Regularization** | ✅ | Umbrella term for tricks that make memorizing *harder* (so the model generalizes) — e.g. **dropout** (randomly switch off neurons during training) and **weight decay** (gently pull weights toward zero). |

## Module 6 — Using & aligning LLMs

| Term | Status | Plain meaning |
|---|---|---|
| **Inference** | ✅ | Running the finished, trained model to generate text (as opposed to *training* it). Each step: produce the next-token distribution → sample one token → append → repeat. |
| **Sampling** | ✅ | Choosing one actual next token from the model's predicted probability distribution. Plain sampling = a **weighted die** (each token's chance = its probability); shaped by temperature, top-k, top-p. |
| **Greedy decoding / argmax** | ✅ | The simplest pick: always take the single highest-probability token. **Deterministic** (same prompt → identical output) and prone to bland repetition loops. |
| **Temperature** | ✅ | A dial that reshapes the distribution before sampling: **low** sharpens it (top word dominates → near-greedy, reliable — facts), **high** flattens it (long shots get a real chance → creative/risky). temp→0 = greedy. |
| **Top-k / Top-p (nucleus)** | ✅ | Trim the junk tail before sampling so high temperature stays coherent. **Top-k** keeps a fixed number of the most-likely tokens; **top-p** keeps the smallest set whose probabilities sum to p (e.g. 0.9) — top-k is a fixed headcount, top-p **adapts** to the model's confidence. |
| **Context window / context length** | ✅ | The **maximum number of tokens** the model can attend over in one forward pass (prompt + generated output share the budget). A hard ceiling baked into the model (GPT-2 ≈ 1,024; modern 8K–1M+); not a stored memory but **how much text is re-fed in each step**. Tokens past it are dropped entirely. |
| **Pre-training** | ✅ | The giant, expensive first training run (trillions of tokens, months, huge cost) that produces a broad **generalist** model — the run from Module 5. |
| **Fine-tuning** | ✅ | Taking a pre-trained model and training it a **little further on a small, targeted dataset** to specialize it (medical, legal, a brand voice). Cheap because the model already knows language; by default nudges **all** weights gently on little data for few steps. |
| **Catastrophic forgetting** | ✅ | When aggressive fine-tuning **overwrites** the weight values that stored the model's general knowledge (no backup → it's gone), so it loses everyday abilities. Defenses: small learning rate, few steps, mixed dataset, early stopping. |
| **PEFT / LoRA** | ✅ | Parameter-efficient fine-tuning: **freeze** the original model and train only a small add-on (a few million new weights) instead of all billions. Cheap, storage-light (a tiny swappable file per specialty), and the frozen base **can't be overwritten** → no catastrophic forgetting. |
| **Base model** | ✅ | The raw pre-trained model — powerful at next-token prediction but "unaligned": knows a lot, follows nothing. The assistant you chat with is built on top of it via alignment. |
| **Alignment** | ✅ | Retraining a base model so its behavior matches **what humans actually want** (treat input as an instruction; be helpful, honest, safe) rather than just the raw statistics of web text. Done via instruction tuning + RLHF. |
| **Instruction tuning / SFT** | ✅ | **Supervised fine-tuning**: fine-tune the base model on thousands of human-written **(instruction → good response) pairs** so it learns to *answer* requests. Still plain next-token prediction — only the data changed — and it **generalizes** the "instruction → comply" pattern to unseen instructions. |
| **RLHF** | ✅ | **Reinforcement Learning from Human Feedback**: align beyond SFT. **Move 1** — humans pick the better of two answers; those comparisons train a **reward model**. **Move 2** — the assistant generates → reward model scores → nudge weights toward higher reward (reinforcement). Shapes the model by human preference at scale. |
| **Reward model** | ✅ | A **separate** network trained on human A-vs-B comparisons; given a prompt + response it outputs a **score** = how good a human would judge it. Turns fuzzy human taste into a number the assistant can be optimized against — no human needed in the loop. |
| **Hallucination** | ✅ | When an LLM confidently states something false (invents a book, citation, or date). Cause: it predicts the most **plausible** next token, not the true one — no built-in truth-check and no "I don't know," so it fills gaps with plausible-sounding filler. A consequence of the design, reducible but not patchable away. |
| **Knowledge cutoff** | ✅ | The model only knows what was in its training data, frozen into the weights at training time; inference doesn't update them, so it can't know events after its cutoff date. |
| **RAG (Retrieval-Augmented Generation)** | ✅ | Mitigation for hallucination/cutoff: **retrieve** relevant real documents, **augment** the prompt by pasting them into the context window, then **generate** an answer **from that text**. Turns a lossy recall task into reading-comprehension and lets the model cite sources. |
| **Tools / function-calling** | ✅ | Letting the model **call a real tool** (calculator, web search, database, code interpreter) for what it's bad at, instead of guessing — the model decides when to call, the tool returns a real result. |

## Module 7 — Capstone: build a tiny GPT

| Term | Status | Plain meaning |
|---|---|---|
| **Character-level model** | ✅ | A model whose tokens are single characters (vocab = the unique chars, here 65). Simplest possible tokenizer; the model must build words/grammar from scratch, so it needs more scale to read well. |
| **Causal mask** | ✅ | The one new piece vs. Module 4: before softmax, set every attention score to the **future** to −∞ so its weight becomes 0. A position may attend only to itself and the past — so a next-token predictor can't peek at the answer. |
| **Logits** | ✅ | The raw (un-normalized) scores the model outputs per position — one per vocabulary token. Softmax turns them into the next-token probability distribution. |
| **Autograd** | ✅ | A framework's automatic differentiation: it records the forward ops and computes every gradient for you when you call `loss.backward()` (used in 7A/PyTorch). 7B removes it and writes each backward by hand. |
| **Forward / backward (of a layer)** | ✅ | The two functions every layer implements: **forward** computes the output (and stashes what's needed); **backward** takes `dL/d(output)` and returns `dL/d(input)` while accumulating `dL/d(weights)`. Chaining backwards = backprop. |
| **Numeric gradient check** | ✅ | A correctness test for hand-written gradients: nudge one value by ±ε and estimate the slope as `(loss(x+ε)−loss(x−ε))/(2ε)`; if it matches the analytic gradient (to ~1e-7), the math is right. |
| **Adam / AdamW** | ✅ | A smarter gradient-descent optimizer: per-weight adaptive step sizes + momentum (a running average of past gradients) + bias correction. Faster, steadier than plain SGD; the standard for training transformers. |

> Keep this current: when a `⬜` term is taught, flip it to `✅` and tighten the definition;
> add new rows under the right **module section** as terms appear.
