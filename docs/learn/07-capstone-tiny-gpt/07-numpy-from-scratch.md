# 7B — the same GPT in pure NumPy from scratch (backprop by hand)

> **Status: ✅ complete — built, gradient-checked, trained, and generating.** The whole GPT is
> reimplemented in `demo/tiny_gpt_numpy.py` with **no autograd** — every layer has a hand-written
> `forward` and `backward`. The full model's gradients match a numeric check to **~1e-7**, and the
> model trains (loss ~4.65 → ~2.07) and generates Shakespeare-shaped text, exactly like 7A.

## In one sentence

7B rebuilds the exact same character-level GPT as 7A, but we compute every gradient ourselves —
each layer implements its own `forward` and `backward`, verified against a numeric gradient check,
so nothing about learning is a black box.

## Why it matters

This is the deepest possible understanding of an LLM. In 7A, PyTorch's `loss.backward()` computed
all gradients invisibly. Writing them by hand turns Module 2's backprop from a story into something
we *build* — through attention, LayerNorm, softmax and all — and proves we understand exactly how a
transformer learns. 7A is the correctness reference.

## The intuition

### Part 1 — the backward contract, the safety net, and Linear

Every layer is **two functions**: `forward(input) → output` (compute output, stash what backward
needs) and `backward(grad_output) → grad_input` (given `dL/d(output)`, return `dL/d(input)` to pass
back, and accumulate `dL/d(weights)`). The chain rule threads these — Module 2's *pass the blame
backward*, as code.

**Safety net — numeric gradient check.** Nudge one entry by a tiny `ε` and measure the loss:
`(loss(x+ε) − loss(x−ε)) / (2ε)`. If our analytic gradient matches, it's right. We check every piece.

**Linear** `y = x@W + b`: `dW = xᵀ@dy`, `db = sum(dy)`, `dx = dy@Wᵀ`. Grad-checked to ~1e-11.

### Part 2 — Embedding, LayerNorm, ReLU, cross-entropy

- **Embedding** (a lookup table). forward: pick rows `W[ids]`. backward: **scatter-add** the incoming
  gradient into those rows (`np.add.at`), because a row used k times gets k contributions. No gradient
  flows to the integer ids.
- **LayerNorm** (normalize each vector to mean-0/var-1 over the last axis, then scale `γ`/shift `β`).
  forward: `x̂ = (x−μ)/√(σ²+ε)`, `y = γx̂ + β`. backward: `dγ = Σ dy·x̂`, `dβ = Σ dy`, and the input
  gradient `dx = (1/√(σ²+ε)) · (dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂))` with `dx̂ = dy·γ` (means over the
  last axis). The subtractions are because changing one input shifts the mean and variance too.
- **ReLU**: forward `max(0,x)`; backward passes the gradient only where the input was positive.
- **softmax + cross-entropy** (fused, for the clean gradient). forward: `p = softmax(logits)`,
  `loss = −mean(log p[correct])`. backward: **`dlogits = (p − onehot)/N`** — "predicted minus
  target," the single most important gradient in the whole model.

### Part 3 — causal self-attention

Per **head**: project `q,k,v = x@W`; scores `= q·kᵀ · (1/√hs)`; **causal-mask** the future to −∞;
`a = softmax(scores)`; `out = a·v`. The backward, in order:
`dv = aᵀ·dout`; `da = dout·vᵀ`; **softmax backward** `dscores = a·(da − Σ(da·a))`; re-apply the mask
and the scale; `dq = dscores·k`, `dk = dscoresᵀ·q`; then back through the three linears, summing their
input gradients. **Multi-head** runs several heads, concatenates their outputs, and mixes with an
output projection `W_O`; its backward splits the gradient back to each head.

### Part 4 — the model

- **FeedForward**: `Linear → ReLU → Linear` (expand ×4, bend, shrink); backward chains the three.
- **Block**: `x = x + attention(LayerNorm(x))`, then `x = x + feedforward(LayerNorm(x))`. The residual
  `+` means backward **adds** the gradient flowing around each sub-layer to the gradient through it.
- **TinyGPT**: token embedding + position embedding → N blocks → final LayerNorm → output Linear →
  logits. backward runs the whole thing in reverse; the position-embedding gradient is the token
  gradient **summed over the batch** (positions are shared across the batch).

### Part 5 — the training loop (manual backprop)

The Module-2 loop, with our own backprop: `get_batch → forward → cross_entropy → zero_grad →
model.backward(dlogits) → optimizer.step()`. A hand-written **Adam** optimizer (per-weight adaptive
step + momentum + bias correction) updates every `Param`. Each `Param` holds `.data` and `.grad`;
`backward` fills the grads, Adam consumes them. After training, the weights are saved to
`tiny_gpt_numpy.npz` (like 7A's `.pt`); re-running loads the checkpoint and skips straight to
generation.

### Part 6 — generation

Identical autoregressive loop to 7A: crop context to `BLOCK_SIZE`, forward, take the last position's
logits, temperature + softmax, **sample** a char, append, repeat.

## What we ran (verified)

```
model parameters : 30,017
gradient check - full model, analytic vs numeric (float64):
  worst relative diff across all parameter tensors = 8.93e-08   ALL OK

training for 2000 steps (loss falling from ~ln(65)=4.17):
  step     0   train loss 4.654   val loss 4.646
  step  2000   train loss 2.071   val loss 2.132

generation (prompt 'ROMEO:'):
  ROMEO:
  Ory becacty;
  You forsoo; mid feat my wellf.
  ...  and king ... year ... tear ...
```

The whole GPT learns and speaks with **every gradient written by hand** — and the ~1e-7 grad-check is
the proof the hand-written calculus is correct.

## Check yourself

- What two methods must every layer provide, and what does each return? (forward: output + stash;
  backward: dL/d(input), and accumulate dL/d(weights).)
- Why does the Embedding backward use scatter-add? (A row used multiple times must sum all its
  gradient contributions.)
- What is the fused softmax+cross-entropy gradient? (`(p − onehot)/N` — predicted minus target.)
- In a residual block `x = x + sublayer(norm(x))`, what does backward do at the `+`? (Adds the
  gradient flowing straight through to the gradient coming back through the sub-layer.)
- How do we know all this hand-written calculus is correct? (Numeric gradient check: analytic vs.
  finite-difference matches to ~1e-7.)

## What's next / depends on

- **Depends on:** [Gradient descent & backprop](../02-neural-networks/04-gradient-descent-and-backprop.md),
  [Stage 2 — the model](03-the-model.md), and 7A ([`tiny_gpt_pytorch.py`]) as the reference.
- **This completes the capstone.** From "what is a vector" to a transformer trained with hand-written
  backprop — the full path is done.
