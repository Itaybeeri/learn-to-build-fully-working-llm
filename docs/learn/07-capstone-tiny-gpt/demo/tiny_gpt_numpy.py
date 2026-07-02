"""
Tiny GPT (7B) — the SAME character-level GPT as 7A, but in pure NumPy from scratch.

No autograd: every layer implements its own forward AND backward pass by hand, so we
compute every gradient ourselves. 7A (tiny_gpt_pytorch.py) is the correctness reference.

Contents (built alongside 07-numpy-from-scratch.md):
  Part 1 — the backward contract + numeric gradient check + Linear
  Part 2 — Embedding, LayerNorm, ReLU, softmax + cross-entropy
  Part 3 — causal self-attention (per-head) + multi-head
  Part 4 — the model (Block -> TinyGPT)
  Part 5 — the training loop (manual backprop, Adam)
  Part 6 — generation (autoregressive sampling)

Run:  python tiny_gpt_numpy.py            (grad-check, then train + save tiny_gpt_numpy.npz + generate;
                                           re-running loads the checkpoint and just generates)
      python tiny_gpt_numpy.py --check    (just run the gradient checks)
"""

import os
import sys
import numpy as np

rng = np.random.default_rng(1337)

# ----------------------------------------------------------------------------
# Hyperparameters — tiny so pure-NumPy trains on a CPU in a couple of minutes.
# ----------------------------------------------------------------------------
BLOCK_SIZE = 16
BATCH_SIZE = 16
N_EMBD     = 32
N_HEAD     = 4
N_LAYER    = 2
MAX_ITERS     = 2000
EVAL_INTERVAL = 500
EVAL_ITERS    = 50
LEARNING_RATE = 3e-3

# ----------------------------------------------------------------------------
# Shared data & character tokenizer (same corpus as 7A).
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
_text = open(os.path.join(HERE, "input.txt"), encoding="utf-8").read()
chars = sorted(set(_text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[int(i)] for i in ids)

_data = np.array(encode(_text), dtype=np.int64)
_n = int(0.9 * len(_data))
_train, _val = _data[:_n], _data[_n:]

def get_batch(split):
    d = _train if split == "train" else _val
    ix = rng.integers(0, len(d) - BLOCK_SIZE, size=BATCH_SIZE)
    x = np.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = np.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x, y


# ============================================================================
# A tiny parameter wrapper: holds a weight array `.data` and its gradient `.grad`.
# ============================================================================
class Param:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
# Part 1 — Linear:  y = x @ W + b   (handles any leading dims via reshape)
# ============================================================================
class Linear:
    def __init__(self, n_in, n_out, bias=True):
        self.W = Param(rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, n_out)))
        self.b = Param(np.zeros(n_out)) if bias else None
        self.params = [self.W] + ([self.b] if bias else [])

    def forward(self, x):
        self.shape = x.shape
        self.x2 = x.reshape(-1, self.shape[-1])        # flatten leading dims
        y = self.x2 @ self.W.data
        if self.b is not None:
            y = y + self.b.data
        return y.reshape(*self.shape[:-1], self.W.data.shape[1])

    def backward(self, dy):
        dy2 = dy.reshape(-1, self.W.data.shape[1])
        self.W.grad += self.x2.T @ dy2
        if self.b is not None:
            self.b.grad += dy2.sum(axis=0)
        dx2 = dy2 @ self.W.data.T
        return dx2.reshape(self.shape)


# ============================================================================
# Part 2 — Embedding, LayerNorm, ReLU, cross-entropy
# ============================================================================
class Embedding:
    """A lookup table. forward: pick rows. backward: scatter-add into those rows."""
    def __init__(self, n_rows, n_cols):
        self.W = Param(rng.normal(0, 0.02, size=(n_rows, n_cols)))
        self.params = [self.W]

    def forward(self, ids):
        self.ids = ids
        return self.W.data[ids]

    def backward(self, dout):
        np.add.at(self.W.grad, self.ids, dout)     # rows can repeat -> accumulate
        return None                                # ids are integers; no gradient flows to them


class LayerNorm:
    """Normalize each vector to mean 0 / var 1 (over the last axis), then scale+shift."""
    def __init__(self, dim, eps=1e-5):
        self.g = Param(np.ones(dim))
        self.b = Param(np.zeros(dim))
        self.eps = eps
        self.params = [self.g, self.b]

    def forward(self, x):
        self.mu = x.mean(axis=-1, keepdims=True)
        self.xc = x - self.mu
        self.var = (self.xc ** 2).mean(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.xhat = self.xc / self.std
        return self.g.data * self.xhat + self.b.data

    def backward(self, dy):
        D = dy.shape[-1]
        axes = tuple(range(dy.ndim - 1))
        self.g.grad += (dy * self.xhat).sum(axis=axes)
        self.b.grad += dy.sum(axis=axes)
        dxhat = dy * self.g.data
        # the standard LayerNorm input gradient
        dx = (1.0 / self.std) * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )
        return dx


class ReLU:
    params = []
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, dy):
        return dy * self.mask


def cross_entropy(logits, targets):
    """logits (B,T,V), targets (B,T). Returns (loss, dlogits)."""
    B, T, V = logits.shape
    z = logits.reshape(B * T, V)
    t = targets.reshape(B * T)
    p = softmax(z, axis=-1)
    n = B * T
    loss = -np.log(p[np.arange(n), t] + 1e-12).mean()
    dz = p.copy()
    dz[np.arange(n), t] -= 1.0
    dz /= n
    return loss, dz.reshape(B, T, V)


# ============================================================================
# Part 3 — causal self-attention
# ============================================================================
_TRIL = np.tril(np.ones((BLOCK_SIZE, BLOCK_SIZE))).astype(bool)

class Head:
    """One causal self-attention head: score q·k, mask the future, softmax, blend v."""
    def __init__(self, head_size):
        self.key   = Linear(N_EMBD, head_size, bias=False)
        self.query = Linear(N_EMBD, head_size, bias=False)
        self.value = Linear(N_EMBD, head_size, bias=False)
        self.hs = head_size
        self.params = self.key.params + self.query.params + self.value.params

    def forward(self, x):
        B, T, C = x.shape
        self.T = T
        self.k = self.key.forward(x)          # (B,T,hs)
        self.q = self.query.forward(x)
        self.v = self.value.forward(x)
        self.scale = self.hs ** -0.5
        scores = (self.q @ self.k.transpose(0, 2, 1)) * self.scale   # (B,T,T)
        self.msk = _TRIL[:T, :T]
        scores = np.where(self.msk, scores, -1e9)                    # blank the future
        self.a = softmax(scores, axis=-1)                            # (B,T,T)
        return self.a @ self.v                                       # (B,T,hs)

    def backward(self, dout):                                        # dout (B,T,hs)
        dv = self.a.transpose(0, 2, 1) @ dout                        # (B,T,hs)
        da = dout @ self.v.transpose(0, 2, 1)                        # (B,T,T)
        # softmax backward (per row): ds = a * (da - sum(da*a))
        dscores = self.a * (da - (da * self.a).sum(axis=-1, keepdims=True))
        dscores = np.where(self.msk, dscores, 0.0) * self.scale
        dq = dscores @ self.k                                        # (B,T,hs)
        dk = dscores.transpose(0, 2, 1) @ self.q                     # (B,T,hs)
        dx = self.query.backward(dq) + self.key.backward(dk) + self.value.backward(dv)
        return dx


class MultiHead:
    def __init__(self, num_heads, head_size):
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = Linear(N_EMBD, N_EMBD)
        self.hs = head_size
        self.params = [p for h in self.heads for p in h.params] + self.proj.params

    def forward(self, x):
        self.outs = [h.forward(x) for h in self.heads]
        cat = np.concatenate(self.outs, axis=-1)                     # (B,T,C)
        return self.proj.forward(cat)

    def backward(self, dy):
        dcat = self.proj.backward(dy)                               # (B,T,C)
        dx = None
        for i, h in enumerate(self.heads):
            chunk = dcat[..., i * self.hs:(i + 1) * self.hs]
            dxi = h.backward(chunk)
            dx = dxi if dx is None else dx + dxi
        return dx


class FeedForward:
    def __init__(self):
        self.l1 = Linear(N_EMBD, 4 * N_EMBD)
        self.relu = ReLU()
        self.l2 = Linear(4 * N_EMBD, N_EMBD)
        self.params = self.l1.params + self.l2.params

    def forward(self, x):
        return self.l2.forward(self.relu.forward(self.l1.forward(x)))

    def backward(self, dy):
        return self.l1.backward(self.relu.backward(self.l2.backward(dy)))


# ============================================================================
# Part 4 — the model
# ============================================================================
class Block:
    """attention (mix) + feed-forward (think), each with a pre-LayerNorm and residual."""
    def __init__(self):
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHead(N_HEAD, head_size)
        self.ff = FeedForward()
        self.ln1 = LayerNorm(N_EMBD)
        self.ln2 = LayerNorm(N_EMBD)
        self.params = self.ln1.params + self.sa.params + self.ln2.params + self.ff.params

    def forward(self, x):
        x = x + self.sa.forward(self.ln1.forward(x))
        x = x + self.ff.forward(self.ln2.forward(x))
        return x

    def backward(self, dy):
        # second residual: x = a + ff(ln2(a))
        da = dy + self.ln2.backward(self.ff.backward(dy))
        # first residual: a = x + sa(ln1(x))
        dx = da + self.ln1.backward(self.sa.backward(da))
        return dx


class TinyGPT:
    def __init__(self):
        self.token_emb = Embedding(vocab_size, N_EMBD)
        self.pos_emb = Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = [Block() for _ in range(N_LAYER)]
        self.ln_f = LayerNorm(N_EMBD)
        self.head = Linear(N_EMBD, vocab_size)
        self.params = (self.token_emb.params + self.pos_emb.params
                       + [p for b in self.blocks for p in b.params]
                       + self.ln_f.params + self.head.params)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_emb.forward(idx)                  # (B,T,C)
        pos = self.pos_emb.forward(np.arange(T))           # (T,C)
        x = tok + pos                                      # broadcast add
        for b in self.blocks:
            x = b.forward(x)
        x = self.ln_f.forward(x)
        return self.head.forward(x)                        # (B,T,V)

    def backward(self, dlogits):
        dx = self.head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for b in reversed(self.blocks):
            dx = b.backward(dx)
        self.pos_emb.backward(dx.sum(axis=0))              # position grad summed over batch
        self.token_emb.backward(dx)
        return None

    def zero_grad(self):
        for p in self.params:
            p.grad[...] = 0.0

    def save(self, path):
        # store every parameter tensor under a zero-padded key so load order is exact
        np.savez(path, **{f"p{i:03d}": p.data for i, p in enumerate(self.params)})

    def load(self, path):
        d = np.load(path)
        for i, p in enumerate(self.params):
            p.data[...] = d[f"p{i:03d}"]
        d.close()


# ============================================================================
# Part 5 — Adam optimizer + the training loop (manual backprop)
# ============================================================================
class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), eps=1e-8):
        self.params = params; self.lr = lr; self.b1, self.b2 = betas; self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def estimate_loss(model):
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits = model.forward(x)
            loss, _ = cross_entropy(logits, y)
            losses.append(loss)
        out[split] = float(np.mean(losses))
    return out


def train(model):
    opt = Adam(model.params, lr=LEARNING_RATE)
    for it in range(MAX_ITERS + 1):
        if it % EVAL_INTERVAL == 0:
            l = estimate_loss(model)
            print(f"  step {it:>5}   train loss {l['train']:.3f}   val loss {l['val']:.3f}")
        xb, yb = get_batch("train")
        logits = model.forward(xb)                 # forward
        loss, dlogits = cross_entropy(logits, yb)  # loss
        model.zero_grad()
        model.backward(dlogits)                    # backprop (all by hand)
        opt.step()                                 # update
    return model


# ============================================================================
# Part 6 — generation (autoregressive sampling)
# ============================================================================
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model.forward(idx_cond)
        logits = logits[:, -1, :] / temperature            # (B,V)
        if top_k is not None:
            kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = np.where(logits < kth, -1e9, logits)
        probs = softmax(logits, axis=-1)
        nxt = np.array([rng.choice(vocab_size, p=probs[b]) for b in range(idx.shape[0])])
        idx = np.concatenate([idx, nxt[:, None]], axis=1)
    return idx


# ============================================================================
# Gradient check — the whole model, analytic vs numeric, on a tiny random batch.
# ============================================================================
def grad_check():
    print("gradient check - full model, analytic vs numeric (float64):")
    model = TinyGPT()
    B, T = 2, 6
    idx = rng.integers(0, vocab_size, size=(B, T))
    yb = rng.integers(0, vocab_size, size=(B, T))

    def loss_now():
        logits = model.forward(idx)
        return cross_entropy(logits, yb)[0]

    # analytic gradients
    logits = model.forward(idx)
    _, dl = cross_entropy(logits, yb)
    model.zero_grad()
    model.backward(dl)

    eps = 1e-5
    worst = 0.0
    # sample a few entries from every parameter tensor and compare
    for k, p in enumerate(model.params):
        flat = p.data.reshape(-1)
        gflat = p.grad.reshape(-1)
        n_samp = min(4, flat.size)
        idxs = rng.choice(flat.size, size=n_samp, replace=False)
        rels = []
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + eps; hi = loss_now()
            flat[j] = orig - eps; lo = loss_now()
            flat[j] = orig
            num = (hi - lo) / (2 * eps)
            rel = abs(num - gflat[j]) / (abs(num) + abs(gflat[j]) + 1e-12)
            rels.append(rel)
        worst = max(worst, max(rels))
    print(f"  worst relative diff across all parameter tensors = {worst:.2e}   "
          f"{'ALL OK' if worst < 1e-4 else 'MISMATCH'}")
    return worst


if __name__ == "__main__":
    print(f"corpus / vocab   : {len(_text):,} chars / {vocab_size} unique")
    _m = TinyGPT()
    n_params = sum(p.data.size for p in _m.params)
    print(f"model parameters : {n_params:,}\n")

    grad_check()

    if "--check" in sys.argv:
        sys.exit(0)

    ckpt = os.path.join(HERE, "tiny_gpt_numpy.npz")
    if os.path.exists(ckpt):
        _m.load(ckpt)
        print("\nloaded trained weights <- tiny_gpt_numpy.npz  (skipping training)")
    else:
        print(f"\ntraining for {MAX_ITERS} steps (loss falling from ~ln({vocab_size})={np.log(vocab_size):.2f}):")
        train(_m)
        _m.save(ckpt)
        print("\nsaved trained weights -> tiny_gpt_numpy.npz")

    print("\n" + "=" * 60)
    print("generation (prompt 'ROMEO:'):")
    print("=" * 60)
    ctx = np.array([encode("ROMEO:")], dtype=np.int64)
    out = generate(_m, ctx, max_new_tokens=400, temperature=1.0)
    print(decode(out[0]))
