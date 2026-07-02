# Learn to Build a Fully Working LLM

**A from-scratch course that takes you from "what is a vector?" all the way to building, training, and
running your own tiny GPT — taught by an AI tutor that teaches you Socratically and remembers where you
left off.**

Most "build a GPT" material is static text that assumes you already know the math and the ML. This is the
opposite: it assumes **nothing**. Every term is defined from zero, intuition comes before the math, and
you never rely on an idea that hasn't been taught yet. By the end you will have built a working
character-level GPT **twice** — once in PyTorch, once in **pure NumPy with every gradient written by
hand** — trained it, and watched it generate text.

> Newer to ML and to higher math? That's exactly who this is for.

---

## What makes it different

- 🎓 **An AI Socratic tutor, not a wall of text.** It teaches a concept, then *asks you a question*. If
  you're stuck, it doesn't dump the answer — it guides you with hints and easier questions until *you* get
  there.
- 💾 **It remembers where you left off.** Stop any time; run `/continue` and pick up mid-concept. Your
  progress is saved privately on your own machine.
- 🔬 **See it happen.** A no-install browser **portal** lets you watch a neuron fire, data flow forward,
  loss drop, attention blend words together, and your tiny GPT tokenize, train, and generate.
- 🛠️ **You build the real thing.** The capstone is a working tiny GPT — including a from-scratch NumPy
  version where you implement every forward and backward pass yourself (gradient-checked to ~1e-7).

---

## Quickstart

You'll need [**Claude Code**](https://claude.com/claude-code) (the tutor runs inside it) and, for the
Module 2+ demos, **Python 3.12**.

```bash
git clone https://github.com/Itaybeeri/learn-to-build-fully-working-llm.git
cd learn-to-build-fully-working-llm
```

Then open the folder in Claude Code and run:

```
/start
```

The tutor will welcome you, ask a couple of quick questions, and begin at Module 0. Come back any time and
run **`/continue`** to resume, or **`/recap`** to be quizzed on what you've learned.

**Prefer to just read?** The whole course is plain Markdown — open
[`docs/learn/index.md`](docs/learn/index.md) and read front-to-back.

**Want to see the visuals?** Open [`docs/learn/portal/index.html`](docs/learn/portal/index.html) in a
browser (or run `docs/learn/portal/serve.cmd` on Windows).

---

## What you'll learn (the roadmap)

| Module | You'll understand… |
|---|---|
| **0 · Orientation** | What an LLM actually is, and where its "knowledge" lives |
| **1 · Foundations** | Vectors, matrices & the dot product, functions & gradients, probability — the gentle math, from zero |
| **2 · Neural networks** | Neurons, layers, the forward pass, loss, backprop, the training loop *(+ a network that learns XOR from scratch)* |
| **3 · Language as data** | Tokenization, vocabularies, BPE, and embeddings — turning words into vectors |
| **4 · The Transformer** | Attention, self-attention (Q/K/V), multi-head attention, positional info, and the full transformer block |
| **5 · Training LLMs** | Next-token prediction, datasets & scale, what "learning" adjusts, overfitting vs. generalization |
| **6 · Using & aligning LLMs** | Sampling (temperature, top-k/p), context windows, fine-tuning, instruction tuning & RLHF, limitations |
| **7 · Capstone** | **Build a tiny GPT** — in PyTorch, then again in pure NumPy from scratch; train it, scale it up, and generate text |

Full details and links to every note are in [`docs/learn/index.md`](docs/learn/index.md).

---

## How progress works (and why contributors are safe)

Your learning progress is stored in a private **`.progress/`** folder that is **gitignored** — it never
shows up in `git status` and can never end up in a pull request. So you can learn through the whole course
*and* contribute improvements without your progress ever leaking. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Contributing

Clearer explanations, fixes, and new demos are very welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[MIT](LICENSE) — free to use, learn from, and build on.

---

## Author

Created by **Itay Beeri**.

- GitHub: [@Itaybeeri](https://github.com/Itaybeeri)
- LinkedIn: [itaybeeri](https://www.linkedin.com/in/itaybeeri/)

If this helped you understand how LLMs really work, a ⭐ on the repo is appreciated — and I'd love to hear
about it on LinkedIn.
