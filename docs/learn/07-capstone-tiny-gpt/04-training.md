# Stage 3 — training (the loss falls)

> **Status: ✅ complete — code written, run, and mastery check cleared.** We run the **Module 2 training loop** on
> the GPT: ① `get_batch` → ② `logits, loss = model(x, y)` (forward + cross-entropy) → ③ `loss.backward()`
> (backprop — PyTorch computes all ~160k gradients automatically) → ④ `optimizer.step()` (nudge every weight
> downhill), repeated `MAX_ITERS` times. New pieces vs. Module 2: the **optimizer (AdamW)** — a smarter
> gradient descent (adaptive per-weight step + momentum) — and watching **train *and* validation** loss
> together (Module 5: together-down = learning; val rising = overfitting). **Verified run:** loss fell from
> ≈ ln(65) ≈ 4.4 down to **train 1.86 / val 1.94** over 3000 steps; the two curves fall *together* with only a
> small gap → healthy learning, no real overfitting. Trained weights saved to `tiny_gpt.pt` for Stage 4.

## The loop (it's Module 2, on the GPT)

```
   repeat MAX_ITERS times:
     ① x, y = get_batch("train")          grab chars + their next chars
     ② logits, loss = model(x, y)          forward pass + cross-entropy loss
     ③ optimizer.zero_grad(); loss.backward()   backprop: all gradients, automatically
     ④ optimizer.step()                    nudge every weight downhill
```

Three things worth naming:

- **`loss.backward()` is Module 2's backprop, automated.** We hand-derived gradients for a 9-weight XOR net;
  here PyTorch's **autograd** computes gradients for all ~160,000 weights through attention and layer norm. This
  is the convenience we chose for 7A — and exactly the calculus we'll write *by hand* in 7B.
- **The optimizer — AdamW.** Module 2 nudged each weight by `−learning_rate × gradient` (plain gradient
  descent). AdamW is the same idea, better engineered: it adapts the step size per-weight and carries a little
  **momentum**, so training is faster and steadier. Same spirit: *take a step downhill.*
- **Train *and* validation loss (Module 5).** Every `EVAL_INTERVAL` steps we average loss over both splits.
  Falling **together** = genuine learning; **val rising while train falls** = overfitting. Here they tracked
  each other closely → healthy.

## What we ran (verified)

```
training for 3000 steps (loss falling from ~ln(65)=4.17):
  step     0   train loss 4.409   val loss 4.396     <- random guessing
  step   500   train loss 2.308   val loss 2.316
  step  1000   train loss 2.140   val loss 2.173
  step  1500   train loss 2.044   val loss 2.087
  step  2000   train loss 1.966   val loss 2.039
  step  2500   train loss 1.900   val loss 1.987
  step  3000   train loss 1.864   val loss 1.964     <- trained

saved trained weights -> tiny_gpt.pt
```

The drop from **4.4 → 1.9** *is* the training loop working — the model went from guessing uniformly among 65
characters to confidently predicting plausible next characters. (A char-level loss near 1.9 already produces
real words and Shakespeare-ish structure, which we'll see in Stage 4. Scaling up later pushes it lower.)

## Where each concept lives in the code (`tiny_gpt_pytorch.py`)

| Concept | In the code |
|---|---|
| The four-step loop (Module 2) | `train()` |
| Forward + cross-entropy loss | `logits, loss = model(xb, yb)` |
| Backprop (Module 2), automated | `loss.backward()` |
| The weight nudge (gradient descent) | `optimizer.step()` (AdamW) |
| Train/val loss watch (Module 5) | `estimate_loss()` |

## Check yourself

- What four steps make up the training loop? (get a batch → forward + loss → `backward()` (backprop) →
  optimizer step.)
- What does `loss.backward()` do, and which module did we build it by hand in? (Computes the gradient of the
  loss w.r.t. every weight — **backprop** from Module 2 — automatically.)
- What is the optimizer (AdamW) doing, and how does it relate to Module 2? (Nudging each weight downhill — a
  smarter gradient descent with per-weight adaptive steps + momentum.)
- Why watch validation loss too? (To catch **overfitting**, Module 5 — if val rises while train falls, the
  model is memorizing, not generalizing. Here they fell together = healthy.)

## Questions we worked through

- **Q (top rung): Walk through the training loop's four steps, then explain `loss.backward()` (and which
  module built it by hand), what AdamW does (vs. plain gradient descent), and why we watch validation loss.**
  A: ① `get_batch` (grab `xb` = char chunks + `yb` = the same chunks shifted by one = the next-char answers,
  the self-supervised "free labels"); ② forward + cross-entropy loss; ③ `loss.backward()` = **backprop**,
  the gradient for every weight — built by hand in **Module 2** on the 9-weight XOR net; ④ `optimizer.step()`
  = AdamW nudges each weight downhill (a smarter gradient descent: per-weight adaptive step + momentum).
  Watch **validation** loss to catch **overfitting** — if val rises while train falls, the model is
  memorizing, not generalizing.
- *Ladder:* steps 2–4, AdamW, `backward()`=gradient, and val-rising=memorizing usually come through on
  their own. Two gaps to fill: (1) step ① is `get_batch` — the model needs input data (`xb`) and its
  next-char targets (`yb`) *before* it can guess; (2) backprop was built by hand in **Module 2**. Also
  clarify: `backward()` *computes* the gradients; the *optimizer* applies them.
- *Watch a real run live* (loss ~4.4 → ~1.86, train/val together) and it's worth recapping
  cross-entropy (`−log p_correct`; ln 65 ≈ 4.4 = uniform-guessing baseline) and where the **159,937**
  parameters come from (sum of every matrix/embedding size; fixed by the hyperparameters; the weights all
  exist from the start as random numbers — training changes their *values*, not their count).

## What's next / depends on

- **Depends on:** [The training loop](../02-neural-networks/05-the-training-loop.md),
  [Gradient descent & backprop](../02-neural-networks/04-gradient-descent-and-backprop.md),
  [Loss / cross-entropy](../02-neural-networks/03-loss-measuring-wrong.md),
  [Overfitting vs. generalization](../05-training-llms/04-overfitting-vs-generalization.md) (train/val curves).
- **Next:** Stage 4 — generation (load `tiny_gpt.pt`, feed a prompt, sample the next char with
  temperature/top-k, append, repeat — and read what it writes).
