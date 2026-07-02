# Learn to Build an LLM — Learning Portal

An interactive, browser-based companion to the manual. One page, a **tab per module**, each tab
holding that module's *visible* demos with plain-language explanations. Watch the concepts
happen — neurons lighting up, data flowing forward, blame flowing backward, loss dropping.

## How to open it

**Double-click `serve.cmd`** (Windows). It starts a tiny local web server, serves this folder at
**http://localhost:8000/**, and opens the portal in your browser. Keep that window open while you
learn; close it (or press Ctrl+C) to stop the server. On PowerShell you can instead run
`.\serve.ps1`.

> **Why a server and not just the file?** The demos work fine opened directly as a file
> (`index.html`), but later modules will load real data (a text corpus for tokenization, training
> data for the tiny GPT). Browsers block `fetch()` of local files over `file://`, so we serve the
> folder over `http://localhost` to keep every demo working as the curriculum grows.

## What's here now

- **Module 2 — Neural Networks** tab, with three demos:
  1. **Single neuron** — drag inputs/weights/bias; watch the weighted sum and sigmoid update live.
  2. **XOR network (step-by-step)** — the 2-2-1 network drawn live. Press **Next step ▶** to walk
     one example through *forward → loss → backward → update*, with every number narrated. Use
     **+100 / +500 epochs** to train fast and watch the loss chart plunge. **Reset** starts from a
     fresh (known-good) random seed.
  3. **"Should I go outside?"** — a single neuron on meaningful inputs (temperature, raining,
     weekend); flip the situation and watch the decision change.

- **Module 3 — Language as data** tab, with two demos:
  1. **Word map (embeddings)** — words as points in a 2-D meaning space; similar words cluster.
     Do vector arithmetic (`king − man + woman ≈ queen`, shown step by step) and hover any dot for
     its vector and nearest neighbour.
  2. **Tokenizer (BPE)** — type a word and watch it split into sub-word tokens by replaying a
     ranked merge list, with the merged pair highlighted at each step; unknown words fall back to
     small pieces (never out-of-vocabulary).

- **Concept Map** tab — the whole curriculum as one connected diagram: every key concept is a box
  (colored & badged by the module that teaches it, laid out left→right in learning order), with
  arrows showing relationships. Hover a box to light up only its connections; click for its
  definition + which note to read. A "see the whole landscape" study aid.

## How it's built (for future modules)

- `index.html` — the shell (tab bar + script includes).
- `portal.css` — all styling.
- `portal.js` — the **engine**: tab registry, the tiny-network math (`forward`, `loss`,
  `backprop`, `applyUpdate` — the same logic as the Python demo), and the SVG/canvas renderers.
- `modules/module2.js` — the Module 2 tab; registers itself via `Portal.register({...})`.

**Adding Module 3 later** = create `modules/module3.js` that calls `Portal.register({...})` and
add one `<script>` line to `index.html`. The shell builds its tab automatically.

## Verifying the math

The engine's net-math is the same as the from-scratch Python demo and learns XOR to ~0 loss
(predictions approach 0 / 1 / 1 / 0). The default and all Reset seeds are chosen to converge
within a few hundred epochs at learning rate 1.0.
