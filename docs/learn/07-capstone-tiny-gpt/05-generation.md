# Stage 4 — generation (the model speaks)

## In one sentence

**Generation** runs the trained model in a loop — predict the next character, sample one from the predicted
probabilities, append it to the context, and repeat — so the model writes new text one character at a time.

## Why it matters

This is the moment the whole capstone pays off: the model we built (Stage 2) and trained (Stage 3) actually
**produces text**. It's also the exact loop every real LLM uses to answer you — just at a vastly larger scale.

## The intuition

### Generation is an autoregressive loop (back to Module 0)

The model does one thing: given some text, output a probability for the next character. **Generation just runs
that in a loop, feeding each new character back in** — "autoregressive" = it consumes its own output:

```
   context: "ROMEO"
     → model → next-char probs → sample ":"  → context "ROMEO:"
     → model → next-char probs → sample "\n" → context "ROMEO:\n"
     → ... repeat for N characters ...
```

Four steps per character (see `TinyGPT.generate`):

1. **Forward pass** on the current context → logits (raw scores) for every position.
2. **Keep only the last position's** scores — that's the prediction for the *next* char.
3. **Temperature + softmax** → a probability distribution over the 65 chars. Temperature (Module 6) controls
   adventurousness: `<1` sharper/safer, `>1` flatter/wilder, `=1` as-trained. (Optional **top-k** first: keep
   only the k highest-scoring chars, zero the rest.)
4. **Sample** one char from that distribution (`torch.multinomial`) — *not* just the max, so the text stays
   varied and alive — then **append** it and loop.

Two details that matter:
- **Crop the context** to the last `BLOCK_SIZE` (32) chars each step — the model can only look back that far.
- **Sample, don't argmax** — always taking the single most-likely char produces repetitive, stuck output.

## The gentle code

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    self.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]            # crop to the context window
        logits, _ = self(idx_cond)                 # forward pass
        logits = logits[:, -1, :] / temperature    # last position only; temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)          # scores -> probabilities
        idx_next = torch.multinomial(probs, 1)     # sample one char
        idx = torch.cat((idx, idx_next), dim=1)    # append and loop
    return idx
```

## What we ran (verified)

Prompt `"ROMEO:"`, 500 new chars, temperature 1.0:

```
ROMEO:
Ven trusir:
O, I say me tebled your shall cay thy not
soold it I the are on
...
HENRSARIO:
Then kraint; Not;
I he prisanch lord, enks scamed, a you lord cous with derve would hidgh...
```

Gibberish word-by-word, but it clearly learned Shakespeare's **shape** from scratch: `NAME:` speaker
headings, line breaks, English-textured words (`your shall`, `Thou`, `lord`, `dead`, `fate`), and sensible
punctuation. That is a cross-entropy loss of ~1.86 made visible. A bigger model + more training (the
scale-up stage) is what turns this texture into real words and sentences.

## Check yourself

- Why is generation called a *loop*, and what does "autoregressive" mean? (Predict next char → append →
  predict again; "autoregressive" = the model feeds its own output back in as input.)
- Why sample from the probabilities instead of always taking the most likely char? (Argmax gets repetitive
  and stuck; sampling keeps the text varied — temperature/top-k tune how adventurous.)
- Why crop the context to `BLOCK_SIZE`? (The model was built with a fixed context window / positional table
  of that size; it can't look back further.)
- The text is gibberish word-by-word but got the *format* right — what does that tell us it learned?
  (The statistics of the data — spelling, formatting, rhythm — even without real meaning; more scale adds the
  meaning.)

## Questions we worked through

- **Q1 — what does "autoregressive" mean?** A: the model feeds its own output back in — predict the next
  char, **append** it to the context, **run the model again** on the longer context, repeat. ("auto" = self,
  "regressive" = feeding back.)
- **Q2 — why sample instead of always taking the most-likely char?** A: so we don't get the identical output
  every time; there are always alternatives to the top pick, and temperature tunes how likely they are.
  (Argmax → repetitive/stuck.)
- **Q3 — the output nailed the play *format* but not real words; what did it learn, and what fixes the
  gibberish?** A (reached after correcting a misconception): the **data was clean** (real Shakespeare), so it
  isn't "garbage in." The model learned only the *surface statistics* (formatting, letter patterns) because
  it's **tiny (160k params) and barely trained (~2 min)**. Char-level makes it a bit harder (each token is a
  meaningless letter), but even char-level models write real English *when big enough* — so the real fix is
  **scale**: more parameters + more training + more data (Module 5). *(Misconception corrected: gibberish was
  blamed on "gibberish input" — but the input was clean; the limit is model size/training.)*

## What's next / depends on

- **Depends on:** [Stage 2 — the model](03-the-model.md) & [Stage 3 — training](04-training.md) (a trained
  model to sample from), [Inference & sampling](../06-using-and-aligning-llms/01-inference-and-sampling.md)
  (temperature, top-k/p), [What is an LLM?](../00-orientation/01-what-is-an-llm.md) (the generation loop).
- **Next:** Scale-up — a bigger model trained on a CUDA GPU for more readable samples; then
  **7B** — the same tiny GPT in pure NumPy from scratch (hand-written forward + backward).
