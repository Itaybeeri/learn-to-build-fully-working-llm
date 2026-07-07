# Scale-up — a bigger model for readable text (run on a CUDA GPU)

## In one sentence

**Scaling up** means turning the size knobs up — a wider embedding, more layers, a longer context, more
training — so the same architecture learns real words and grammar instead of only surface texture.

## Why it matters

This is Module 5's lesson made tangible. Stage 4 showed the 160k-param model learned Shakespeare's *shape*
(formatting, letter patterns) but not real words — because it was too small and undertrained, **not** because
the data was bad. Scaling up is the fix, and seeing the *same code* produce dramatically better text with
bigger knobs is the whole point.

## What changes (the knobs)

The `--big` preset (vs. the baseline) — nothing structural changes, only the sizes:

| Knob | Baseline | `--big` | Effect |
|---|---|---|---|
| `BLOCK_SIZE` (context) | 32 | 128 | sees 4× more history |
| `N_EMBD` (vector width) | 64 | 192 | 3× more room to represent meaning |
| `N_HEAD` (attention heads) | 4 | 6 | more relationships in parallel |
| `N_LAYER` (blocks) | 3 | 6 | 2× deeper — more "refine" steps |
| `MAX_ITERS` (steps) | 3000 | 5000 | more training |
| `LEARNING_RATE` | 1e-3 | 3e-4 | smaller, steadier steps for a bigger model |
| **parameters** | **159,937** | **~2,715,713** | **~17× bigger** |

The code is **device-agnostic** (`device = "cuda" if torch.cuda.is_available() else "cpu"`), so it uses a GPU
automatically wherever one is present — no code change needed.

## How to run it on a CUDA GPU

On a machine with an NVIDIA GPU:

```bash
# 1. get the latest code
git pull

# 2. install the CUDA build of PyTorch (pick the command for your CUDA version from pytorch.org;
#    example for CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. confirm the GPU is visible
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # -> cuda: True

# 4. train the scale-up model (auto-uses the GPU; ~2 min) and print a sample
cd docs/learn/07-capstone-tiny-gpt/demo
python tiny_gpt_pytorch.py --big
```

It saves `tiny_gpt_big.pt` (gitignored) and prints a fresh sample from the prompt `"ROMEO:"`. Re-running
loads the saved weights and just generates (skips training).

## Results (RTX 3070, CUDA — ~few minutes)

```
config           : BIG (scale-up)
device           : cuda
corpus / vocab   : 1,115,394 chars / 65 unique
model parameters : 2,715,713
forward pass     : x (32, 128) -> logits (32, 128, 65)   (B, T, vocab)

training for 5000 steps (loss falling from ~ln(65)=4.17):
  step     0   train loss 4.358   val loss 4.351
  step   500   train loss 2.235   val loss 2.254
  step  1000   train loss 1.912   val loss 2.012
  step  1500   train loss 1.740   val loss 1.872
  step  2000   train loss 1.636   val loss 1.797
  step  2500   train loss 1.562   val loss 1.747
  step  3000   train loss 1.513   val loss 1.695
  step  3500   train loss 1.473   val loss 1.674
  step  4000   train loss 1.447   val loss 1.644
  step  4500   train loss 1.421   val loss 1.625
  step  5000   train loss 1.393   val loss 1.601
```

**Reading it:** loss started at ~ln(65) = 4.17 (a model guessing uniformly among 65 chars) and fell to
**train 1.393 / val 1.601**. Train and val fall *together* the whole way — no overfitting (Module 5); the small
train–val gap (~0.2) is the healthy sign of a model that's learned patterns, not memorized. Compare the
baseline (Stage 3): **train 1.86 / val 1.96** — the bigger model reaches a clearly lower loss.

**Sample (big model, prompt `"ROMEO:"`, temperature 1.0):**

```
ROMEO:
I was prove your words grace; and in your gest;
But I have not plain'd not sonior
Of our purpose and sorrows in yet for the glister,
And the provil, or your honour brother is myself,;
That till be a fraither of belse this;
Sea shame worth we sufficeld gaves--but were hands speak life:
For an fy the darence as my poor,
Heads death in lies me court, and a gentlemen
From foar me hold councry him. By twill be her better
In shorten shall may him to hill down; breath,
In this shall be to for his game
```

**Before vs. after.** The Stage-4 baseline (160k params) learned Shakespeare's *shape* — line breaks, a
speaker-name-then-colon layout, plausible letter runs — but almost no real words. The `--big` model (2.7M
params) now produces mostly **real English words** ("grace", "purpose", "sorrows", "honour brother",
"gentlemen", "death", "breath", "court"), real contractions ("plain'd"), and sentence-like punctuation. It's
still not *meaningful* Shakespeare — the grammar wanders and some words are invented ("sonior", "fraither") —
because 2.7M params on ~1M characters is still tiny. But the jump from "letter texture" to "English words" from
nothing but **turning the size knobs up** is exactly Module 5's scaling lesson, made tangible on our own model.

## Check yourself

- The output improved but the code didn't change structurally — what did we change, and why did that help?
  (Only the size knobs — wider/deeper/longer-context + more training; bigger capacity lets it learn grammar
  and words, not just surface statistics. Module 5.)
- Why does the *same script* run on a GPU on one machine and a CPU on another with no edits? (It picks the
  device at runtime: `cuda` if a CUDA GPU is available, else `cpu`.)

## What's next / depends on

- **Depends on:** [Stage 3 — training](04-training.md), [Stage 4 — generation](05-generation.md),
  [Datasets & scale](../05-training-llms/02-datasets-and-scale.md) (scaling laws & emergence).
- **Next:** nothing — this was the last knob to turn on **7A**. The same tiny GPT is also rebuilt in pure NumPy
  from scratch in [**7B**](07-numpy-from-scratch.md) (hand-written forward + backward), which completes the
  capstone and the curriculum.

> **How it was run:** on an NVIDIA RTX 3070, PyTorch `2.6.0+cu124` (the CUDA build) installed into an isolated
> virtual environment so the machine's default `torch ... +cpu` stayed untouched. `torch.cuda.is_available()`
> was `True`, so the device-agnostic code used the GPU automatically. The 5000-step run took a few minutes
> (versus ~70 min on CPU).
