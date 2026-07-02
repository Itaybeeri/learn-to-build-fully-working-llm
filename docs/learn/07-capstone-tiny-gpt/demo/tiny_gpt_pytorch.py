"""
Tiny GPT (7A) — a character-level GPT in PyTorch.

Built stage by stage alongside the Module 7 notes:
  Stage 1 — data & character tokenizer        (this file, so far)
  Stage 2 — the model (embeddings + blocks)   (coming)
  Stage 3 — training loop                      (coming)
  Stage 4 — generation / sampling              (coming)

Run:  python tiny_gpt_pytorch.py
"""

import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hyperparameters (the knobs). Two presets:
#   default  — small + fast, trains on a CPU in ~2 min (matches the Stage 2/3 notes).
#   --big    — the scale-up: bigger context/embedding/depth for more readable text.
#              Comfortable on a GPU; slower but still runnable on a CPU.
# Run:  python tiny_gpt_pytorch.py          (baseline, -> tiny_gpt.pt)
#       python tiny_gpt_pytorch.py --big    (scale-up, -> tiny_gpt_big.pt)
# ----------------------------------------------------------------------------
BIG = "--big" in sys.argv

if BIG:
    BLOCK_SIZE = 128      # context window: 4x more chars of history than the baseline
    BATCH_SIZE = 32       # bigger batches = steadier gradients
    N_EMBD     = 192      # 3x wider vectors: much more room to represent meaning
    N_HEAD     = 6        # more attention heads (head size stays 32)
    N_LAYER    = 6        # 2x deeper: 6 Transformer blocks stacked
    DROPOUT    = 0.2      # a touch more regularization for the bigger model
    LEARNING_RATE = 3e-4  # smaller step: bigger models train more stably at a lower rate
    MAX_ITERS     = 5000  # more steps
    EVAL_INTERVAL = 500
    EVAL_ITERS    = 200
    CKPT_NAME = "tiny_gpt_big.pt"
else:
    BLOCK_SIZE = 32       # context window: how many chars the model sees at once
    BATCH_SIZE = 16       # how many independent chunks we train on per step
    N_EMBD     = 64       # embedding size: the length of each token's vector
    N_HEAD     = 4        # number of attention heads (run in parallel)
    N_LAYER    = 3        # how many Transformer blocks we stack
    DROPOUT    = 0.1      # regularization (Module 5): randomly zero some activations while training
    LEARNING_RATE = 1e-3  # step size for the optimizer (Module 1/2)
    MAX_ITERS     = 3000  # how many training steps to run
    EVAL_INTERVAL = 500   # how often to measure train/val loss
    EVAL_ITERS    = 200   # how many batches to average when measuring loss
    CKPT_NAME = "tiny_gpt.pt"
SEED       = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"   # auto-uses a CUDA GPU if the CUDA torch build is installed
torch.manual_seed(SEED)
print(f"config           : {'BIG (scale-up)' if BIG else 'baseline'}")

# ----------------------------------------------------------------------------
# Stage 1 — data & character tokenizer
# ----------------------------------------------------------------------------
import os
HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "input.txt"), encoding="utf-8").read()

# the vocabulary = every unique character, sorted for a stable order
chars = sorted(set(text))
vocab_size = len(chars)

# the two lookup tables (the "address book": char <-> integer id)
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> string

def encode(s):
    "a string -> a list of integer token ids"
    return [stoi[c] for c in s]

def decode(ids):
    "a list of integer token ids -> a string"
    return "".join(itos[i] for i in ids)

# the whole corpus as one long tensor of ids, split into train / validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    """Grab BATCH_SIZE random chunks. x = chars, y = the same chunks shifted
    by one (each target is the *next* char — the sliding-window free label)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))          # random start positions
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])             # (BATCH_SIZE, BLOCK_SIZE)
    y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])     # shifted by one
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------
# Stage 2 — the model
# Each class maps to a Module 4 concept (noted in comments).
# ----------------------------------------------------------------------------

class Head(nn.Module):
    """One self-attention head (Module 4: self-attention Q/K/V).
    Scores = Q·K, causal-masked so a position only sees the past, softmax, blend V."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)   # the three learned projections
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # a lower-triangular matrix of 1s — the causal mask (1 = allowed, 0 = future)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)
        # score every token against every token, scaled (Module 4: Q·K / sqrt(d))
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5    # (B, T, T)
        # CAUSAL MASK: blank out the future (upper triangle) before softmax
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                       # rows sum to 1 (Module 4: softmax)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v                                        # blend = weighted sum of V


class MultiHeadAttention(nn.Module):
    """h heads in parallel, concatenated, then projected back (Module 4: multi-head)."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(N_EMBD, N_EMBD)        # W_O: combine the heads
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # stack heads side by side
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """The 'think' sub-layer: Linear -> ReLU bend -> Linear (Module 4: feed-forward; Module 2 layers)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),   # expand
            nn.ReLU(),                       # the non-linear bend (Module 2)
            nn.Linear(4 * N_EMBD, N_EMBD),   # back down
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: attention (mix) + feed-forward (think),
    each wrapped in a residual add and preceded by layer norm (Module 4)."""
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa  = MultiHeadAttention(N_HEAD, head_size)
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)     # layer norm (Module 4)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))        # residual: add the input back (Module 4)
        x = x + self.ff(self.ln2(x))        # residual
        return x


class TinyGPT(nn.Module):
    """The whole model: embeddings (+ position) -> N blocks -> final norm -> logits."""
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)    # Module 3: the embedding table
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, N_EMBD)    # Module 4: positional info (learned)
        self.blocks    = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f      = nn.LayerNorm(N_EMBD)                # final layer norm
        self.lm_head   = nn.Linear(N_EMBD, vocab_size)       # unembedding -> logits over the vocab

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                            # (B, T, N_EMBD)  "what"
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, N_EMBD)  "where"
        x = tok + pos                                        # add what + where (Module 4)
        x = self.blocks(x)                                   # run the Transformer blocks
        x = self.ln_f(x)
        logits = self.lm_head(x)                             # (B, T, vocab_size)
        loss = None
        if targets is not None:                              # loss comes alive in Stage 3
            B, T, Cv = logits.shape
            loss = F.cross_entropy(logits.view(B * T, Cv), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Stage 4 — the autoregressive loop (Module 0's generation loop, made real).
        Predict the next char, sample one, append it, repeat. `idx` = (B, T) of context ids."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]              # crop context to the last BLOCK_SIZE chars
            logits, _ = self(idx_cond)                   # forward pass
            logits = logits[:, -1, :] / temperature      # keep only the LAST position; temperature (Module 6)
            if top_k is not None:                        # optional top-k: keep only the k best chars
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)            # scores -> probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)   # sample one char (not just the max)
            idx = torch.cat((idx, idx_next), dim=1)      # append and loop
        return idx


# ----------------------------------------------------------------------------
# Stage 3 — training
# ----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model):
    """Average loss over EVAL_ITERS batches for train and val (no gradients).
    Two numbers we watch: falling together = learning; val rising = overfitting (Module 5)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model):
    """The Module 2 training loop, on the GPT: forward -> loss -> backward -> step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)   # a smarter gradient descent
    for it in range(MAX_ITERS + 1):
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"  step {it:>5}   train loss {losses['train']:.3f}   val loss {losses['val']:.3f}")
        xb, yb = get_batch("train")            # ① a batch
        _, loss = model(xb, yb)                 # ② forward + cross-entropy loss
        optimizer.zero_grad(set_to_none=True)   #    clear last step's gradients
        loss.backward()                         # ③ backprop (all gradients, automatically)
        optimizer.step()                        # ④ nudge every weight downhill
    return model


# ----------------------------------------------------------------------------
# Run: summary → train (or load saved weights) → Stage 4 generation.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"device           : {device}")
    print(f"corpus / vocab   : {len(text):,} chars / {vocab_size} unique")

    model = TinyGPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters : {n_params:,}")

    xb, yb = get_batch("train")
    logits, _ = model(xb)
    print(f"forward pass     : x {tuple(xb.shape)} -> logits {tuple(logits.shape)}   (B, T, vocab)\n")

    ckpt = os.path.join(HERE, CKPT_NAME)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded trained weights <- {CKPT_NAME}  (skipping training)")
    else:
        print(f"training for {MAX_ITERS} steps (loss falling from ~ln(65)={torch.log(torch.tensor(65.0)).item():.2f}):")
        train(model)
        torch.save(model.state_dict(), ckpt)
        print(f"\nsaved trained weights -> {CKPT_NAME}")

    # ---- Stage 4 — generation (the autoregressive loop) --------------------
    print("\n" + "=" * 60)
    print("Stage 4 - generation (feed a prompt, sample char by char):")
    print("=" * 60)
    prompt = "ROMEO:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=500, temperature=1.0, top_k=None)
    print(decode(out[0].tolist()))
