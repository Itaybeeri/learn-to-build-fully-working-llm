/* ============================================================================
   Concept Map tab — the whole curriculum as one connected diagram.

   Every key concept is a box, badged & colored by the MODULE that teaches it,
   arranged left→right in learning order. Arrows show how concepts relate.
   Hover a box to light up only its connections (the "what touches this?" view);
   click a box for its definition + which note to read.

   Self-registers with the portal engine via Portal.register({...}).
   Vanilla SVG; reuses Portal.el / Portal.svgEl. Opens from file://.
   ============================================================================ */
(function () {
  "use strict";

  // ---- modules: color + label (learning order = left→right) ----------------
  const MODULES = [
    { id: 0, name: "Module 0 · Orientation",     color: "#b07cff" },
    { id: 1, name: "Module 1 · Foundations",     color: "#4ea1ff" },
    { id: 2, name: "Module 2 · Neural networks", color: "#46d39a" },
    { id: 3, name: "Module 3 · Language as data",color: "#ffd24e" },
    { id: 4, name: "Module 4 · The Transformer", color: "#ff8db4" },
    { id: 5, name: "Module 5 · Training LLMs",   color: "#ff9d5c" },
    { id: 6, name: "Module 6 · Using & aligning", color: "#c98aff" }
  ];

  // ---- the concepts (curated spine). def seeded from the glossary -----------
  const NODES = [
    // M0
    { id: "llm",        label: "LLM",                m: 0, note: "00-orientation/01-what-is-an-llm.md",
      def: "Large Language Model — a system that predicts the next chunk of text, looping to write whole answers." },
    { id: "nextword",   label: "next-token prediction", m: 0, note: "00-orientation/01-what-is-an-llm.md",
      def: "The one job an LLM does: given the text so far, guess what comes next." },
    { id: "weight",     label: "weight (dial)",      m: 0, note: "00-orientation/02-how-llms-learn-and-where-knowledge-lives.md",
      def: "One adjustable number. Together, all the weights ARE the model — its knowledge lives here, not in stored text." },
    { id: "training",   label: "training",           m: 0, note: "00-orientation/02-how-llms-learn-and-where-knowledge-lives.md",
      def: "The process that nudges every weight to make the model's next-token guesses less wrong." },
    // M1
    { id: "vector",     label: "vector",             m: 1, note: "01-foundations/01-numbers-and-vectors.md",
      def: "An ordered list of numbers; how a meaning (or a set of dials) is represented." },
    { id: "dotproduct", label: "dot product",        m: 1, note: "01-foundations/02-matrices-and-matrix-multiplication.md",
      def: "Multiply two vectors slot-by-slot and add → one number. Big = they point the same way = similar." },
    { id: "matrix",     label: "matrix",             m: 1, note: "01-foundations/02-matrices-and-matrix-multiplication.md",
      def: "A grid of numbers; a stack of vectors. The model's weights are matrices." },
    { id: "probdist",   label: "probability dist.",  m: 1, note: "01-foundations/04-probability-basics.md",
      def: "A probability for every outcome, all summing to exactly 1." },
    { id: "loss",       label: "loss",               m: 1, note: "02-neural-networks/03-loss-measuring-wrong.md",
      def: "One number for how wrong a guess is (0 = perfect). Cross-entropy is the loss for probabilities." },
    { id: "graddescent",label: "gradient descent",   m: 1, note: "01-foundations/03-functions-and-gradients.md",
      def: "Nudge each weight opposite its slope to lower the loss; repeat. w ← w − rate × slope." },
    // M2
    { id: "neuron",     label: "neuron",             m: 2, note: "02-neural-networks/01-what-a-neuron-is.md",
      def: "A weighted sum of inputs (a dot product) followed by an activation. The building block of every layer." },
    { id: "activation", label: "activation (bend)",  m: 2, note: "02-neural-networks/01-what-a-neuron-is.md",
      def: "The bend (ReLU/sigmoid) after the weighted sum. Non-linear, so stacked layers don't collapse into one." },
    { id: "layer",      label: "layer",              m: 2, note: "02-neural-networks/02-layers-and-the-forward-pass.md",
      def: "A stack of neurons = one matrix × vector + activation. Turns a vector into a new vector." },
    { id: "forward",    label: "forward pass",       m: 2, note: "02-neural-networks/02-layers-and-the-forward-pass.md",
      def: "Data flowing layer to layer, input to output. This is the model 'running'." },
    { id: "backprop",   label: "backpropagation",    m: 2, note: "02-neural-networks/04-gradient-descent-and-backprop.md",
      def: "Sends the loss's 'blame' backward through the layers to get every weight's slope, in one pass." },
    { id: "trainloop",  label: "training loop",      m: 2, note: "02-neural-networks/05-the-training-loop.md",
      def: "Forward → loss → backprop → update, repeated millions of times over batches and epochs." },
    // M3
    { id: "token",      label: "token",              m: 3, note: "03-language-as-data/02-tokens-and-tokenization.md",
      def: "The chunk of text the model actually reads/predicts — usually a reusable sub-word piece." },
    { id: "tokenization",label: "tokenization (BPE)",m: 3, note: "03-language-as-data/03-vocabularies.md",
      def: "Cut text into tokens by replaying a learned, ranked list of merge rules (BPE)." },
    { id: "vocabulary", label: "vocabulary",         m: 3, note: "03-language-as-data/03-vocabularies.md",
      def: "The fixed, numbered list of all known tokens. A token's ID = its position in the list." },
    { id: "embedding",  label: "embedding",          m: 3, note: "03-language-as-data/04-embeddings-words-as-vectors.md",
      def: "The learned vector a token's ID looks up. Similar words get nearby vectors; directions carry meaning." },
    // M4
    { id: "attentionidea", label: "attention idea",  m: 4, note: "04-the-transformer/01-the-attention-idea.md",
      def: "Each word looks at all the others, scores who's relevant, and rewrites itself as a blend of them." },
    { id: "qkv",        label: "Q / K / V",          m: 4, note: "04-the-transformer/02-self-attention-qkv.md",
      def: "Three learned re-expressions of each word's embedding (W_Q/W_K/W_V): what I seek, how I'm found, what I give." },
    { id: "score",      label: "score (q·k)",        m: 4, note: "04-the-transformer/02-self-attention-qkv.md",
      def: "A word's query dotted with every word's key → how relevant each other word is." },
    { id: "softmax",    label: "softmax",            m: 4, note: "04-the-transformer/02-self-attention-qkv.md",
      def: "Squashes raw scores into weights that sum to 1 — the blend weights." },
    { id: "selfattention", label: "self-attention",  m: 4, note: "04-the-transformer/02-self-attention-qkv.md",
      def: "score q·k → softmax → blend the values into one new, context-aware vector. Every word at once." },
    { id: "multihead",  label: "multi-head attention", m: 4, note: "04-the-transformer/03-multi-head-attention.md",
      def: "Several self-attention heads in parallel, each its own Q/K/V learning a different relationship; outputs concatenated and combined by a learned matrix W_O." },
    { id: "positional", label: "positional info",   m: 4, note: "04-the-transformer/04-positional-information.md",
      def: "A position vector added to each token's embedding so attention (an order-blind sum) can tell word order apart — 'dog bites man' ≠ 'man bites dog'." },
    { id: "tblock",     label: "transformer block",  m: 4, note: "04-the-transformer/05-a-transformer-block.md",
      def: "The repeatable unit: attention sub-layer (mix info between words) then feed-forward sub-layer (each word thinks alone), each wrapped with a residual + layer norm. Stacking many = the Transformer." },
    { id: "feedforward",label: "feed-forward (FFN)", m: 4, note: "04-the-transformer/05-a-transformer-block.md",
      def: "The 'think' sub-layer: the Module 2 layers (matrix → activation → matrix) applied to each word's vector on its own, with shared weights. Transforms what attention gathered." },
    { id: "residual",   label: "residual connection",m: 4, note: "04-the-transformer/05-a-transformer-block.md",
      def: "output = input + sublayer(input). A sub-layer ADDS its change on top instead of replacing — keeps the original intact and gives backprop a highway, so deep stacks stay trainable." },
    { id: "layernorm",  label: "layer norm",         m: 4, note: "04-the-transformer/05-a-transformer-block.md",
      def: "A reset before each sub-layer: subtract the mean (recenter to 0) and divide by the spread (re-scale to standard volume) so numbers don't drift huge/tiny. γ/β are learned scale/shift dials." },
    // M5
    { id: "selfsup",    label: "self-supervised",    m: 5, note: "05-training-llms/01-next-token-prediction.md",
      def: "Training where the data labels itself — the next word IS the answer, already in the text. No human labels, so all raw text becomes training data." },
    { id: "dataset",    label: "dataset & scale",    m: 5, note: "05-training-llms/02-datasets-and-scale.md",
      def: "The fuel: a colossal pile of raw text — mostly the open web (Common Crawl) plus books, Wikipedia, code — measured in trillions of tokens. Assemblable thanks to self-supervision; trainable thanks to compute." },
    { id: "distributed",label: "distributed weights", m: 5, note: "05-training-llms/03-what-learning-adjusts.md",
      def: "What 'learning' adjusts: all the weight groups (embeddings=meaning, Q/K/V=relationships, FFN=patterns/facts) nudged together. Knowledge is spread across thousands of weights — no single 'Paris' weight — and only the values change, never the fixed architecture." },
    { id: "overfit",    label: "overfit vs. generalize", m: 5, note: "05-training-llms/04-overfitting-vs-generalization.md",
      def: "Training only lowers loss on the text shown — a proxy. A high-capacity model can game it by MEMORIZING (overfitting: great on training data, weak on new), or learn transferable patterns (GENERALIZING). The real goal is performance on unseen text." },
    // M6
    { id: "sampling",   label: "inference & sampling", m: 6, note: "06-using-and-aligning-llms/01-inference-and-sampling.md",
      def: "Inference = running the trained model to generate text. Its final step outputs a probability distribution; SAMPLING picks one token from it. Greedy (always the top token) is deterministic & repetitive; weighted sampling adds variety; TEMPERATURE dials how random (low=sharp/reliable, high=flat/creative); TOP-K / TOP-P trim the junk tail so high temp stays coherent." },
    { id: "ctxwindow",  label: "context window", m: 6, note: "06-using-and-aligning-llms/02-context-windows.md",
      def: "The maximum number of TOKENS the model can attend over in one forward pass (prompt + generated output share the budget). A hard ceiling baked into the model (GPT-2 ≈ 1,024; modern 8K–1M+); not a stored memory but how much text is re-fed in each step. Tokens past it are dropped entirely — a hard cliff, not a fade." },
    { id: "finetune",   label: "fine-tuning", m: 6, note: "06-using-and-aligning-llms/03-fine-tuning.md",
      def: "PRE-TRAINING is the giant run (trillions of tokens, months, huge $) → a broad GENERALIST. FINE-TUNING trains that model a little further on a small targeted dataset to make a SPECIALIST (medical, legal, brand voice). Cheap because the model already knows language — it's a small adjustment on top, not a fresh start." },
    { id: "forgetting", label: "catastrophic forgetting", m: 6, note: "06-using-and-aligning-llms/03-fine-tuning.md",
      def: "Aggressive fine-tuning OVERWRITES the weight values that stored the model's general knowledge (it lived in those weights — no backup), so it suddenly fails everyday tasks. Defenses: small learning rate, few steps, a mixed dataset, early stopping. LoRA sidesteps it by FREEZING the base." },
    { id: "lora",       label: "PEFT / LoRA", m: 6, note: "06-using-and-aligning-llms/03-fine-tuning.md",
      def: "Parameter-efficient fine-tuning: FREEZE the original model and train only a tiny add-on (a few million weights) instead of all billions. Cheap, storage-light (a small swappable file per specialty), and the frozen base can't be overwritten → no catastrophic forgetting." },
    { id: "alignment",  label: "alignment", m: 6, note: "06-using-and-aligning-llms/04-instruction-tuning-and-rlhf.md",
      def: "A BASE MODEL only knows how to continue text, so it ignores instructions (told 'write a poem' it may list more prompts). ALIGNMENT retrains it to treat input as a request and answer the way humans want — helpful, honest, safe — via instruction tuning (SFT) + RLHF." },
    { id: "sft",        label: "instruction tuning (SFT)", m: 6, note: "06-using-and-aligning-llms/04-instruction-tuning-and-rlhf.md",
      def: "Supervised fine-tuning: fine-tune the base model on thousands of human-written (instruction → good response) pairs so it learns to ANSWER requests. Still plain next-token prediction — only the diet of text changed — and it generalizes the 'instruction → comply' pattern to unseen instructions. Ceiling: can only teach what humans write out, and 'good' is easier to judge than demonstrate → RLHF." },
    { id: "rlhf",       label: "RLHF", m: 6, note: "06-using-and-aligning-llms/04-instruction-tuning-and-rlhf.md",
      def: "Reinforcement Learning from Human Feedback — alignment past SFT, built on 'good is easy to judge, hard to demonstrate.' Move 1: humans pick the better of two answers → those comparisons train a separate REWARD MODEL (prompt + response → a 'how good' score, imitating human taste). Move 2: the assistant generates → reward model scores → nudge weights toward higher reward → repeat (reinforcement). A leash keeps it near the SFT model. Result: shaped by human preference at scale." },
    { id: "hallucination", label: "hallucination", m: 6, note: "06-using-and-aligning-llms/05-limitations.md",
      def: "When an LLM confidently states false things (invents a book, citation, date). It predicts the most PLAUSIBLE next token, not the true one — no built-in truth-check and no 'I don't know,' so it fills gaps with plausible-sounding filler. True and fluent-made-up look identical to it. A consequence of the design (plausibility machine, not truth machine), reducible but not patchable away." },
    { id: "rag",        label: "RAG / tools", m: 6, note: "06-using-and-aligning-llms/05-limitations.md",
      def: "Mitigations that give the model an external source of truth at inference. RAG (Retrieval-Augmented Generation): fetch relevant real documents, paste them into the context window, answer FROM that text — turns lossy recall into reading-comprehension, attacks hallucination + cutoff, allows citations. TOOLS/function-calling: let it call a real calculator, search, or code interpreter for what it's bad at. They work OUTSIDE the model, so they shrink the failures but never delete them — always verify." }
  ];

  // ---- relationships (from → to, short label) ------------------------------
  const EDGES = [
    ["llm", "nextword", "does"],
    ["nextword", "loss", "scored by"],
    ["training", "weight", "tunes"],
    ["trainloop", "training", "is the"],
    ["vector", "dotproduct", "compared by"],
    ["vector", "matrix", "stacks into"],
    ["weight", "matrix", "stored as"],
    ["dotproduct", "neuron", "powers"],
    ["weight", "neuron", "has"],
    ["neuron", "activation", "applies"],
    ["neuron", "layer", "stacks into"],
    ["matrix", "layer", "is a"],
    ["layer", "forward", "flows through"],
    ["forward", "loss", "scored by"],
    ["loss", "backprop", "blamed by"],
    ["backprop", "graddescent", "feeds"],
    ["graddescent", "trainloop", "step of"],
    ["probdist", "loss", "cross-entropy"],
    ["softmax", "probdist", "produces"],
    ["tokenization", "token", "produces"],
    ["tokenization", "vocabulary", "builds"],
    ["vocabulary", "embedding", "ID → vector"],
    ["embedding", "vector", "is a"],
    ["weight", "embedding", "is"],
    ["embedding", "attentionidea", "fed into"],
    ["embedding", "qkv", "becomes"],
    ["weight", "qkv", "is"],
    ["attentionidea", "selfattention", "made precise"],
    ["qkv", "score", "q·k"],
    ["dotproduct", "score", "is a"],
    ["score", "softmax", "softened by"],
    ["softmax", "selfattention", "weights"],
    ["selfattention", "multihead", "× many in parallel ="],
    ["qkv", "multihead", "each head has own"],
    ["embedding", "positional", "+ position"],
    ["positional", "selfattention", "gives order to"],
    ["multihead", "tblock", "is the 'mix' in"],
    ["feedforward", "tblock", "is the 'think' in"],
    ["layer", "feedforward", "built from"],
    ["activation", "feedforward", "bends inside"],
    ["residual", "tblock", "wraps each sub-layer of"],
    ["layernorm", "tblock", "stabilizes each sub-layer of"],
    ["backprop", "residual", "flows freely through"],
    ["tblock", "llm", "stacked = body of"],
    ["nextword", "selfsup", "trained by"],
    ["training", "selfsup", "is a kind of"],
    ["selfsup", "loss", "scored by"],
    ["selfsup", "token", "predicts the next"],
    ["selfsup", "dataset", "makes assemblable"],
    ["dataset", "training", "fuels"],
    ["token", "dataset", "measured in"],
    ["training", "distributed", "adjusts"],
    ["weight", "distributed", "spread across"],
    ["embedding", "distributed", "is one group of"],
    ["qkv", "distributed", "is one group of"],
    ["feedforward", "distributed", "is one group of"],
    ["loss", "overfit", "low ≠ goal"],
    ["training", "overfit", "risks"],
    ["distributed", "overfit", "capacity enables"],
    ["dataset", "overfit", "more/cleaner data fights"],
    ["nextword", "sampling", "generates by"],
    ["softmax", "sampling", "distribution fed to"],
    ["probdist", "sampling", "picks one from"],
    ["llm", "sampling", "run at inference by"],
    ["token", "ctxwindow", "measured in"],
    ["selfsup", "ctxwindow", "context capped by"],
    ["positional", "ctxwindow", "limits its length"],
    ["sampling", "ctxwindow", "each step reads"],
    ["dataset", "finetune", "pre-training reused by"],
    ["trainloop", "finetune", "same loop, briefly"],
    ["distributed", "finetune", "nudges the same weights"],
    ["finetune", "forgetting", "risks"],
    ["distributed", "forgetting", "overwrites stored knowledge"],
    ["overfit", "forgetting", "early stopping defends"],
    ["finetune", "lora", "made cheap by"],
    ["lora", "forgetting", "freezing avoids"],
    ["nextword", "alignment", "base model only does"],
    ["finetune", "alignment", "built out of"],
    ["alignment", "sft", "first stage"],
    ["finetune", "sft", "is fine-tuning on"],
    ["sft", "rlhf", "polished by"],
    ["alignment", "rlhf", "second stage"],
    ["trainloop", "rlhf", "reward replaces fixed loss"],
    ["nextword", "hallucination", "plausible-not-true causes"],
    ["distributed", "hallucination", "lossy recall worsens"],
    ["rlhf", "hallucination", "reduces but can't remove"],
    ["hallucination", "rag", "mitigated by"],
    ["ctxwindow", "rag", "fills the window with sources"]
  ];

  // ---- fuller explanations (shown in the detail panel on click) ------------
  const LONG = {
    llm: "A Large Language Model reads the text so far and predicts the next chunk, then appends it and repeats — that loop is how it writes whole answers. It isn't looking anything up; each step it computes a fresh guess from billions of learned weights.",
    nextword: "Everything an LLM does reduces to one task: given the text so far, produce a probability for every possible next token. Generation is just doing this over and over — each guess appended and fed back in.",
    weight: "A weight is a single tunable number. A model is nothing but a giant pile of them (billions), and its 'knowledge' lives entirely in their values — not in any stored text. Training is the act of setting them well. Embeddings, the Q/K/V matrices, and the neuron weights are all just weights.",
    training: "Training shows the model text, lets it guess the next token, measures how wrong it was (the loss), and nudges every weight a hair to be less wrong — repeated millions of times. It's ONE process that tunes all the weights at once, and it happens before you ever use the model.",
    vector: "A vector is an ordered list of numbers where each slot has a fixed meaning. It's how we represent a thing (like a word's meaning) or a set of dials as something a network can do arithmetic on. 'Dimension' = how many numbers are in it.",
    dotproduct: "Multiply two equal-length vectors slot-by-slot and add the results into one number. It comes out large when the two vectors are big in the same slots (they point the same way → similar) and near zero when they don't overlap. This single operation is the model's similarity meter — and attention scores are dot products.",
    matrix: "A matrix is a grid of numbers — equivalently, a stack of vectors. Multiplying a matrix by a vector is just taking a dot product with each row, which re-expresses (transforms) the input. The model's weights are stored as matrices.",
    probdist: "A probability distribution assigns a number between 0 and 1 to every possible outcome, with all of them summing to exactly 1 — a fixed budget the outcomes compete for. An LLM's output over its whole vocabulary is one of these.",
    loss: "The loss is a single number measuring how wrong a guess is — 0 means perfect, bigger means worse. For probability outputs (like next-token), the loss is cross-entropy: it reads the probability the model gave the CORRECT answer and punishes putting weight on wrong ones. The entire point of training is to push this number down.",
    graddescent: "To lower the loss, gradient descent finds each weight's slope (which way nudging it moves the loss) and steps the weight the opposite way by a small amount: w ← w − rate × slope. Done across all weights, repeatedly, it walks the model downhill toward better values.",
    neuron: "A neuron multiplies each input by a weight, sums them (a dot product) and adds a bias, then passes the result through an activation. The weighted sum is its 'opinion'; the activation shapes it. Stack many neurons and you get a network.",
    activation: "After the weighted sum, a neuron applies a non-linear bend (ReLU or sigmoid). Without it, stacking layers would collapse into a single straight-line step — the bend is exactly what lets a deep network learn curved, complex patterns.",
    layer: "A layer is a row of neurons all reading the same input — which is exactly one matrix times the input vector, followed by the activation. It turns one vector into a new vector, re-describing the input in terms of the questions its neurons ask.",
    forward: "The forward pass is the model running: the input vector flows through each layer in turn, every layer reshaping it, until the final layer produces the output (e.g. next-token probabilities). No learning happens here — it's pure computation of a guess, and it runs both during training and every time you use the model.",
    backprop: "Backpropagation efficiently works out how much each weight contributed to the loss by sending the 'blame' backward from the output through the layers in one pass (the chain rule). Its output is the gradient — every weight's slope — which gradient descent then uses to nudge the weights.",
    trainloop: "The training loop is the repeated cycle: forward pass → measure loss → backprop → nudge weights. Run over small batches of data and many epochs (full passes over the data), millions of tiny steps accumulate into a trained model.",
    token: "Text isn't fed in as whole words or single letters but as tokens — usually sub-word pieces (so 'running' might become 'run' + 'ning'). This keeps the vocabulary modest while letting any word, even one never seen, be spelled from known pieces.",
    tokenization: "Tokenization cuts raw text into tokens. The common method, BPE, is learned once: start from characters and repeatedly merge the most frequent adjacent pair into a new token. Tokenizing new text just replays those merge rules in their learned order, so the same word always splits the same way.",
    vocabulary: "The vocabulary is the fixed, numbered list of every token the model knows (often tens of thousands). After tokenizing, text becomes a list of these ID numbers, and each ID is then turned into an embedding vector.",
    embedding: "Each token ID looks up a row in the embedding table — a learned vector representing the word's meaning. Training arranges these so similar words land near each other, and even directions carry meaning (king − man + woman ≈ queen). This is the numeric form the network actually reads; the ID is just the address.",
    attentionidea: "A raw embedding is context-free — 'bank' is the same vector by a river or in a bank account. Attention fixes this: each word scores how relevant every other word is (by similarity) and rewrites itself as a weighted blend of them, becoming context-aware.",
    qkv: "Self-attention gives each word three learned re-expressions of its embedding: a query (what I'm looking for), a key (how I'm found), and a value (what I contribute). They're produced by multiplying the embedding by three learned matrices W_Q, W_K, W_V — and learning those matrices is exactly what makes attention powerful, letting it find the right neighbours (e.g. a verb finding its subject) rather than just similar words.",
    score: "To decide relevance, a word's query is dot-producted with every word's key. A high score means 'your key matches what I'm searching for.' Scoring query·key (instead of raw embedding · embedding) is what lets the learned matrices steer attention to the right words.",
    softmax: "Softmax squashes the raw relevance scores into positive weights that add up to exactly 1 — the blend weights. High scores take most of the weight, low scores almost none, so the result is a genuine weighted average rather than a pile-up.",
    selfattention: "Self-attention runs the full mechanic for every word at once: score query·key against all words, softmax those into weights, then blend the value vectors into one new context-aware vector. It is the core engine of the Transformer.",
    multihead: "A single self-attention 'head' (one set of W_Q/W_K/W_V) can only learn one notion of 'what's relevant' — e.g. a verb finding its subject. But a sentence has many relationships at once (verb→subject, verb→object, adjective→noun). Multi-head attention runs several heads in parallel, each with its own Q/K/V free to specialize in a different relationship; their per-word outputs are concatenated and combined by a learned output matrix W_O into one rich vector. Nobody assigns the specialties — training makes them emerge.",
    positional: "Attention's blend is a weighted SUM of the other words' values, and a sum is order-blind (3+5 = 5+3) — so on its own it can't tell 'dog bites man' from 'man bites dog' (a bag of words, not a sequence). The fix: add a position vector onto each token's embedding right after lookup, before attention, so the same word at a different slot becomes a different vector — order now survives the sum. The position vectors are made either by a LEARNED table (simple, but capped to a trained max length) or by a fixed SINUSOIDAL wave formula ('odometer wheels' at different speeds — works for any length, even positions never seen).",
    tblock: "A transformer block is the one repeatable unit that gets stacked to form the body of an LLM. It runs two sub-layers in order: first multi-head attention lets words MIX information between each other (communication), then a small feed-forward network lets each word THINK about what it gathered (computation). Each sub-layer is wrapped with a residual connection (add, don't replace) and a layer norm (keep numbers stable). Stacking many identical blocks lets the model refine its understanding of each word a little more at every level — like re-reading a sentence several times.",
    feedforward: "The feed-forward network (FFN) is the 'think' half of a block — and it's nothing new: it's exactly the Module 2 layers (matrix → activation bend → matrix). Two twists in how it's used: it runs on EACH word's vector independently (no looking at neighbours — attention already did that), and it reuses the SAME shared weights at every position. Where attention only gathers and averages (a weighted sum), the FFN actually transforms that gathered info into something more useful for the prediction.",
    residual: "A residual (skip) connection makes a sub-layer ADD its result on top of its input instead of replacing it: output = input + sublayer(input). Two payoffs. (1) The original meaning always survives — a clean straight path carries the word's identity up the whole tower, and a block that has nothing useful to add can output ≈0 and pass the vector through untouched. (2) That straight path is a HIGHWAY for backprop's blame signal, so it reaches even the earliest blocks without shrinking to nothing. This single trick is what made very deep networks trainable.",
    layernorm: "As a vector flows up the tower, every residual ADDS something on top of it — dozens of times — so its numbers drift: they can balloon huge, collapse tiny, or get lopsided, and activations/gradients only behave in a moderate range. Layer norm resets the vector before each sub-layer with two moves across its slots: SUBTRACT the mean (recenter it to 0 — fixes 'too high/low overall') and DIVIDE by the spread σ (re-scale to a standard swing-size of ~1 — fixes 'swings too big/small'), keeping the relative pattern intact. Two learned dials, γ (scale) and β (shift), then let training pick the final scale. Because the residual re-adds drift after every step, a layer norm guards EACH sub-layer — two per block.",
    selfsup: "Ordinary 'supervised' learning needs humans to label every example (photo → 'cat'), which caps how much data you can use. Next-token prediction is SELF-supervised: the label is just the next word, which is already sitting in the text. So any sentence — 'the cat sat on the ___' → 'mat' — is a free training example with its answer built in, and all the text ever written becomes a labeled dataset with no human tagging. That removal of the labeling bottleneck is exactly why LLMs can train at internet scale.",
    dataset: "An LLM's training set is a colossal pile of raw text gathered biggest-first: the open web (crawled archives like Common Crawl) is the bulk, plus books (long, well-edited), Wikipedia (clean facts), and public code (so it can program). We don't count files or sentences but TOKENS — the sub-word chunks the model reads — and modern models train on TRILLIONS of them (GPT-3 ≈ 300 billion; newer ones 1–15+ trillion), more than a person could read in thousands of lifetimes. Two distinct things make this possible: self-supervision makes such a giant LABELED set assemblable with zero human tagging, and massive compute (GPUs) makes it feasible to actually train on it. Raw web is mostly junk, so the corpus is heavily CLEANED (quality-filtered, deduplicated, safety-filtered) — garbage in, garbage out, and quality often beats raw quantity. SCALE is the era's big lesson: three dials — data, parameters (model size), compute — grown together; scaling laws make 'bigger = lower loss' smooth and predictable, and some abilities EMERGE suddenly only past a size threshold.",
    sampling: "Training is over; now we USE the model — that's INFERENCE. But the model never emits a word: its final step (logits → softmax) produces a PROBABILITY DISTRIBUTION over the whole vocabulary, and something must pick ONE token from it to put on screen. That pick is SAMPLING, repeated token-by-token to generate text. The simplest rule, GREEDY/argmax (always take the highest-probability token), is deterministic — same prompt → identical output every time — and tends to fall into bland repetition loops. The fix is to roll a WEIGHTED DIE: pick each token with a chance equal to its probability (Paris 0.70 → ~70% of the time), buying variety without chaos. TEMPERATURE then dials HOW random: low temp sharpens the distribution toward the top word (near-greedy, reliable — good for facts), high temp flattens it so long shots get a real chance (creative, risky — good for stories). One catch: the distribution has a huge TAIL of tens of thousands of tiny-probability junk tokens whose total is substantial, and high temperature LIFTS that tail — so the die can land on gibberish. TOP-K and TOP-P fix this by trimming the tail before sampling: top-k keeps a fixed number of top tokens, top-p (nucleus) keeps the smallest set whose probabilities sum to p (e.g. 0.9). Top-k is a fixed headcount; top-p ADAPTS to the model's confidence (fewer tokens when sure, more when unsure) — which is why it's usually preferred.",
    rag: "Because the limitations are built into what the model IS, we don't remove them — we work around them, almost always by giving the model an external source of truth at inference instead of relying on the frozen weights. RAG (Retrieval-Augmented Generation) is the big one and attacks hallucination AND the knowledge cutoff: instead of answering from memory, first FETCH relevant real documents (a search index, company files, a database), paste them into the context window with the question, and let the model answer FROM that text. It converts an error-prone MEMORY task into a READING-COMPREHENSION task — which the model is reliably good at, and the text can be today's — and it lets the model cite sources you can check. TOOLS / function-calling handle what the model is bad at: for exact arithmetic, live data, or running code, let it call a real calculator, web search, or interpreter rather than guess. GROUNDING habits help too: prompt it to flag uncertainty, quote sources, and have a human verify. The crucial honesty: none of these change what the model fundamentally is — they feed better stuff into the window or outsource the weak part. Underneath it's STILL a plausibility machine predicting tokens, so it can misread a retrieved document or misuse a tool. The mitigations SHRINK the problem; they never DELETE it — which is why 'always verify important outputs' never stops being the rule.",
    hallucination: "HALLUCINATION is when an LLM states something false with total confidence — inventing a book that doesn't exist, citing a fake court case, giving a wrong date without a flicker of doubt. The cause ties straight back to the model's one job: predict the most PLAUSIBLE next token, not the TRUE one. Ask it 'the author of the novel X is ___' for a novel it never saw, and it has no true/false check and no slot that says 'I don't know' — so it does the only thing it can, emit the most plausible continuation, which is a plausible-sounding author's name made up from the statistical shape of language. To the model a fluent made-up answer and a fluent true answer are the SAME kind of thing: both are just high-probability token sequences. That's also why you can't simply 'tell it to be truthful' — there's no truth-signal inside to switch on. Two things worsen it: pre-training rewarded confident fluency, not 'I'm unsure'; and knowledge is distributed and lossy (smeared across billions of weights), so recall is reconstructive, like memory, and can blend into a confident-but-wrong answer. The key reframe: hallucination is not a bug a patch removes — it's a direct consequence of what the model fundamentally is, a plausibility machine, not a truth machine. You can reduce it (retrieval, tools, grounding), but it falls out of the core design.",
    rlhf: "RLHF — REINFORCEMENT LEARNING FROM HUMAN FEEDBACK — is the polish that takes a merely instruction-following model and makes it feel like a careful, aligned assistant. It exists because SFT has a ceiling: for open-ended requests, 'good' is easy to JUDGE but hard to DEMONSTRATE — nobody can author the single perfect story for every prompt. RLHF turns that around in two moves. MOVE 1 (reward model): instead of writing answers, humans COMPARE — the model generates two responses to a prompt and a person picks which is better; tens of thousands of these comparisons train a SEPARATE network, the REWARD MODEL, that takes a prompt + response and outputs one number = how good a human would judge it. This is the key trick: it turns fuzzy human taste into a score a machine can optimize, with no human needed in the loop afterward. MOVE 2 (reinforcement learning): the assistant generates a response, the reward model SCORES it, and we nudge the assistant's weights to make high-scoring responses more likely and low-scoring ones less likely — generate → score → nudge, repeated. The model is never shown 'the right answer'; it's rewarded for producing good ones and learns to chase the reward. A guard rail keeps it from drifting too far from the SFT model (else it finds weird, high-reward gibberish). The result: a model shaped by human preference at scale, learning the hard-to-write-down qualities — be helpful, honest, polite, refuse harm.",
    sft: "INSTRUCTION TUNING, or SUPERVISED FINE-TUNING (SFT), is the first and most direct way to align a base model — and it's just fine-tuning aimed at a special dataset. You collect thousands of human-written (instruction → good response) pairs — 'Write a poem about the sea' → an actual poem; 'Explain photosynthesis to a 10-year-old' → a simple explanation; including examples that refuse harmful requests — and fine-tune the base model on them. It's called SUPERVISED because every example carries the demonstrated correct answer (a human supervised it). The mechanism is STILL plain next-token prediction — nothing new is bolted on; only the diet of text changed. Having practiced thousands of cases where an instruction is followed by a helpful answer, the model's instinct for 'what comes after an instruction' flips from 'more prompts' to 'the answer', and it GENERALIZES the pattern 'instruction → comply' to instructions it never saw in training. SFT gets most of the way to a usable assistant, but it has a ceiling: it can only teach what humans bothered to write out, and for open-ended requests 'good' is hard to demonstrate yet easy to judge — humans can't author the perfect story for every prompt, but they can easily say which of two answers is better. SFT can't use that comparison feedback, and that gap is exactly what RLHF fills.",
    alignment: "After pre-training, the model has exactly one skill: predict the next token to CONTINUE text. That's a brilliant autocomplete, but autocomplete is not an assistant — 'continue this text' and 'do what I'm asking' are different jobs. Hand a raw BASE MODEL an instruction like 'Write a poem about the sea' and it may just continue with MORE prompts ('Write a poem about the city…'), because on the web that line usually sits in a list of homework prompts, so the statistically likely continuation is more prompts, not a poem. The model has no concept of 'command' or 'request'; it only knows what text usually follows other text. ALIGNMENT closes the gap between 'continue text like the internet' and 'treat my input as a request and answer helpfully, honestly, and safely'. It's done in two stages built on fine-tuning: INSTRUCTION TUNING (supervised fine-tuning on human-written instruction→answer examples) teaches the format of obeying, and RLHF (learning from human preferences) polishes helpfulness and safety beyond what those examples can cover.",
    forgetting: "CATASTROPHIC FORGETTING is the main danger of fine-tuning. The model's general knowledge — grammar, facts, reasoning — was stored IN the weight values themselves during pre-training, and there is no separate backup. When you fine-tune aggressively on a narrow dataset (big learning rate, too many steps), the training changes those same weights to fit the new data, OVERWRITING the values that encoded the general skills. So the model gets great at the specialty but suddenly fails at everyday tasks it used to handle — and it's 'catastrophic' because the drop can be sudden and severe, not a gentle trade. The defenses are all about keeping the change gentle: a SMALL learning rate and FEW steps (small nudges overwrite little), a MIXED dataset that blends general data with the specialty so the model keeps practicing both, and EARLY STOPPING (from the overfitting note) to halt before it drifts too far. The cleanest sidestep is LoRA, which freezes the original weights so they can't be overwritten at all.",
    lora: "Full fine-tuning is far cheaper than pre-training, but it still updates ALL of a large model's weights — billions of numbers — and you'd need a full copy of the model for each specialty. PARAMETER-EFFICIENT FINE-TUNING (PEFT), of which LoRA is the best-known, fixes this: FREEZE the entire original model and train only a tiny set of new add-on weights bolted alongside it (a few million, not billions). Two wins fall out. First, it's cheap and storage-light — each specialty is a small file you can swap in and out of the same frozen base. Second, it sidesteps catastrophic forgetting entirely: because the original weights never change, the general knowledge they hold literally cannot be overwritten — you ADD a specialty on top rather than rewriting the base. This is why PEFT/LoRA is so popular for adapting big models cheaply and safely.",
    finetune: "The giant Module 5 training run — trillions of tokens, months of compute, enormous cost — is PRE-TRAINING, and it produces a broad GENERALIST that knows grammar, facts, and reasoning a bit of everything. But often you want a SPECIALIST: a model great at medical Q&A, fluent in legal contracts, or that always writes in your company's voice. You do NOT train a fresh model from scratch for that — it would cost a fortune and waste effort relearning language itself. Instead you FINE-TUNE: take the already pre-trained model and train it a little further on a small, targeted dataset, nudging its weights toward the specialty. It's dramatically cheaper because it stands on the shoulders of everything already in the weights — specializing is a small adjustment on top, not a fresh start. By default fine-tuning updates ALL the weights, but only gently, on little data, for few steps (a parameter-efficient variant updates only a small subset). This is why most teams never pre-train: they download an open pre-trained model and fine-tune it.",
    ctxwindow: "The CONTEXT WINDOW (or context length) is the maximum number of TOKENS the model can attend over in a single forward pass — the prompt plus everything generated so far, which both spend from the same budget. It's a hard ceiling baked into the model: GPT-2 held ~1,024 tokens, modern models reach 8K, 128K, even a million+. A key reframing: it is NOT a memory the model stores between requests — it's how much text gets RE-FED into the model on each step. Every prediction, the model is handed a fresh slab of up to N tokens and reads it from scratch, so 'what it remembers' = 'what fits in the slab this time'. When the conversation grows past the window, the earliest tokens are dropped ENTIRELY — not remembered a little less, but simply not fed in. A hard cliff, not a fade.",
    overfit: "Training does exactly one thing — push down the loss on the TEXT WE SHOW the model. But that's only a PROXY for what we want: good predictions on text the model has never seen. A model with billions of weights can lower training loss two very different ways. It can MEMORIZE the exact examples (an effective lookup table) — training loss drops to near zero, but change one word and it fails; that's OVERFITTING (great on training data, weak on new data). Or it can learn the underlying PATTERN ('the capital of X is ___' → X's capital), a rule that transfers to inputs it never saw — that's GENERALIZING. Like two exam students: one memorizes the 50 practice answers (lost on new numbers), the other learns the method (the method transfers). DETECT it by withholding a VALIDATION SET (data never trained on, a fair stand-in for unseen text) and watching two loss curves: early on both fall together, but when overfitting starts they SPLIT — training loss keeps dropping while validation loss bottoms out and RISES. Three CAUSES, each with a remedy: model capacity too big for the data (right-size the model), too little / too repetitive data (more clean, deduplicated data), training too long (EARLY STOPPING at the validation low). Plus REGULARIZATION — tricks that make memorizing harder: dropout (randomly switch off neurons) and weight decay (pull weights toward zero).",
    distributed: "'Knowledge lives in the weights' made concrete. A Transformer's learnable numbers come in distinct groups — the embedding table (meaning of each word), the Q/K/V matrices and W_O (relationships: which words attend to which), the two shared feed-forward matrices (processing + much of the stored patterns/facts), the LayerNorm γ/β dials, and the unembedding — and training nudges ALL of them together with one gradient descent. Crucially, a fact like 'capital of France → Paris' is NOT in a single weight: it's a distributed representation spread across thousands of weights acting together, and each weight serves many facts. And training only changes the VALUES in these fixed slots — the architecture (layers, heads, wiring) is decided before training and never changes."
  };

  // ---- layout --------------------------------------------------------------
  const W = 1340, H = 880, PADX = 150, PADY = 64, BOXW = 158, BOXH = 46;
  function computePositions() {
    MODULES.forEach((mod, ci) => {
      const colNodes = NODES.filter(n => n.m === mod.id);
      const x = PADX + ci * ((W - 2 * PADX) / (MODULES.length - 1));
      colNodes.forEach((n, i) => {
        n.x = x;
        n.y = colNodes.length === 1 ? H / 2
          : PADY + (H - 2 * PADY) * (i / (colNodes.length - 1));
      });
    });
  }
  // border point of a node box on the line toward (tx,ty)
  function edgePoint(n, tx, ty) {
    const dx = tx - n.x, dy = ty - n.y;
    const hw = BOXW / 2 + 2, hh = BOXH / 2 + 2;
    const sx = dx === 0 ? Infinity : hw / Math.abs(dx);
    const sy = dy === 0 ? Infinity : hh / Math.abs(dy);
    const s = Math.min(sx, sy);
    return { x: n.x + dx * s, y: n.y + dy * s };
  }

  // ---- registration --------------------------------------------------------
  Portal.register({
    id: "concept-map",
    tab: "Concept Map",
    title: "Concept Map — the whole curriculum, connected",
    intro: "Every concept you've learned, as one connected map. Boxes are colored by the module " +
           "that teaches them and laid out left→right in learning order; arrows show how they relate.",
    render: function (mount, P) {
      computePositions();
      const byId = {}; NODES.forEach(n => byId[n.id] = n);
      // adjacency for hover-highlight
      const adj = {}; NODES.forEach(n => adj[n.id] = new Set([n.id]));
      EDGES.forEach(([a, b]) => { adj[a].add(b); adj[b].add(a); });

      const panel = P.el("div", { class: "panel" });

      // guide
      const guide = P.el("details", { class: "guide" });
      guide.innerHTML =
        '<summary>What am I looking at?</summary>' +
        '<div class="guide-body">' +
          'This is the <b>whole curriculum as one picture</b>. Each <b>box is a concept</b>; its ' +
          '<b>color and column tell you which module</b> teaches it (left→right = the order you learn them). ' +
          '<b>Arrows are relationships</b> — e.g. <code>dot product → powers → neuron</code>, ' +
          '<code>embedding → fed into → attention idea</code>.<br><br>' +
          '<b>Hover a box</b> to dim everything except it and what it directly connects to (the ' +
          '"what touches this?" view). <b>Click a box</b> to read its definition and jump to the note. ' +
          'Notice some arrows reach <i>backward</i> across modules — e.g. <code>embedding → is a → vector</code>, ' +
          '<code>score → is a → dot product</code> — that\'s the curriculum building on its own foundations.' +
        '</div>';
      panel.appendChild(guide);

      // legend
      const legend = P.el("div", { class: "legend" });
      legend.innerHTML = MODULES.map(m =>
        `<span class="item"><span class="swatch" style="background:${m.color}"></span> ${m.name}</span>`).join("") +
        '<span class="item">hover = show its links · click = definition</span>';
      panel.appendChild(legend);

      // layout: map (left) + detail panel (right)
      const layout = P.el("div", { class: "layout cmap-layout" });
      const stage = P.el("div", {});
      const side = P.el("div", {});
      layout.appendChild(stage); layout.appendChild(side);
      panel.appendChild(layout);

      // zoom toolbar
      let zoom = 1;
      const zlevel = P.el("span", { class: "zlevel" }, ["100%"]);
      function applyZoom() {
        svg.style.width = (zoom * 100) + "%";
        zlevel.textContent = Math.round(zoom * 100) + "%";
      }
      const toolbar = P.el("div", { class: "cmap-toolbar" });
      toolbar.appendChild(P.el("button", { class: "zbtn", title: "Zoom out", onclick: () => { zoom = Math.max(0.6, +(zoom - 0.2).toFixed(2)); applyZoom(); } }, ["−"]));
      toolbar.appendChild(zlevel);
      toolbar.appendChild(P.el("button", { class: "zbtn", title: "Zoom in (map gets bigger; scroll to pan)", onclick: () => { zoom = Math.min(2.6, +(zoom + 0.2).toFixed(2)); applyZoom(); } }, ["+"]));
      toolbar.appendChild(P.el("button", { class: "zbtn wide", title: "Reset zoom to fit", onclick: () => { zoom = 1; applyZoom(); } }, ["Reset"]));
      toolbar.appendChild(P.el("span", { class: "hint", style: "margin:0 0 0 4px" }, ["bigger text? zoom in, then scroll the map"]));
      stage.appendChild(toolbar);

      const mapBox = P.el("div", { class: "cmap-scroll" });
      stage.appendChild(mapBox);

      const detail = P.el("div", { class: "explain" });
      detail.innerHTML = '<div class="phase-tag">concept</div>Click any box to see what it means and which note teaches it. Hover to trace its connections.';
      side.appendChild(detail);

      // ---- build the SVG ----
      const svg = P.svgEl("svg", { class: "net cmap", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });
      const defs = P.svgEl("defs");
      defs.innerHTML = '<marker id="cm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5b6b7d"></path></marker>';
      svg.appendChild(defs);

      // edges
      const edgeEls = [];
      EDGES.forEach(([a, b, label]) => {
        const A = byId[a], B = byId[b];
        const p1 = edgePoint(A, B.x, B.y), p2 = edgePoint(B, A.x, A.y);
        const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
        // gentle curve; bigger bow for backward (right→left) edges
        const backward = B.x < A.x;
        const bow = backward ? 46 : 16;
        const path = P.svgEl("path", {
          d: `M ${p1.x} ${p1.y} Q ${mx} ${my - bow} ${p2.x} ${p2.y}`,
          fill: "none", stroke: "#3a4757", "stroke-width": 1.6,
          "marker-end": "url(#cm-arrow)", class: "cm-edge"
        });
        path.dataset.a = a; path.dataset.b = b;
        svg.appendChild(path);
        const lbl = P.svgEl("text", { x: mx, y: my - bow - 3, class: "cm-edge-label", "text-anchor": "middle" });
        lbl.textContent = label; lbl.dataset.a = a; lbl.dataset.b = b;
        svg.appendChild(lbl);
        edgeEls.push({ path, lbl, a, b });
      });

      // nodes
      const nodeEls = {};
      NODES.forEach(n => {
        const mod = MODULES[n.m];
        const g = P.svgEl("g", { class: "cm-node", transform: `translate(${n.x - BOXW / 2}, ${n.y - BOXH / 2})` });
        const rect = P.svgEl("rect", { x: 0, y: 0, width: BOXW, height: BOXH, rx: 9,
          fill: "#1a212b", stroke: mod.color, "stroke-width": 2 });
        g.appendChild(rect);
        // module accent bar
        g.appendChild(P.svgEl("rect", { x: 0, y: 0, width: 6, height: BOXH, rx: 3, fill: mod.color }));
        // badge
        const badge = P.svgEl("text", { x: BOXW - 8, y: 15, class: "cm-badge", "text-anchor": "end", fill: mod.color });
        badge.textContent = "M" + n.m; g.appendChild(badge);
        // label (wrap to 2 lines if long)
        const t = P.svgEl("text", { x: BOXW / 2, y: BOXH / 2 + 4, class: "cm-label", "text-anchor": "middle" });
        t.textContent = n.label; g.appendChild(t);

        g.addEventListener("mouseenter", () => highlight(n.id));
        g.addEventListener("mouseleave", clearHighlight);
        g.addEventListener("click", () => select(n));
        svg.appendChild(g);
        nodeEls[n.id] = g;
      });
      mapBox.innerHTML = ""; mapBox.appendChild(svg); applyZoom();

      function highlight(id) {
        const keep = adj[id];
        NODES.forEach(n => nodeEls[n.id].classList.toggle("dim", !keep.has(n.id)));
        edgeEls.forEach(e => {
          const on = (e.a === id || e.b === id);
          e.path.classList.toggle("hot", on);
          e.path.classList.toggle("dim", !on);
          e.lbl.classList.toggle("show", on);
        });
      }
      function clearHighlight() {
        NODES.forEach(n => nodeEls[n.id].classList.remove("dim"));
        edgeEls.forEach(e => { e.path.classList.remove("hot", "dim"); e.lbl.classList.remove("show"); });
      }
      function select(n) {
        const mod = MODULES[n.m];
        // every edge touching this node, written as a clear directional sentence
        const touching = EDGES.filter(([a, b]) => a === n.id || b === n.id);
        const conns = touching.map(([a, b, l]) => {
          const aN = a === n.id ? `<b>${byId[a].label}</b>` : byId[a].label;
          const bN = b === n.id ? `<b>${byId[b].label}</b>` : byId[b].label;
          return `${aN} <span style="color:var(--accent)">—${l}→</span> ${bN}`;
        });
        detail.innerHTML =
          `<div class="phase-tag" style="background:${mod.color};color:#08121f">${mod.name}</div>` +
          `<div style="font-size:19px;font-weight:700;margin:4px 0 8px">${n.label}</div>` +
          `<div style="line-height:1.55">${LONG[n.id] || n.def}</div>` +
          (conns.length ? `<div class="hint" style="margin-top:12px"><b>How it connects (arrow = direction):</b><br>${conns.join("<br>")}</div>` : "") +
          `<div class="hint" style="margin-top:12px">📖 Read the full note: <code>docs/learn/${n.note}</code></div>`;
        highlight(n.id);
      }

      host_append(mount, panel);
    }
  });

  function host_append(mount, panel) { mount.appendChild(panel); }
})();
