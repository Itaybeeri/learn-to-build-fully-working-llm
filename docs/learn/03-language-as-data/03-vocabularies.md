# Vocabularies

## In one sentence

A **vocabulary** is the model's fixed, numbered list of every token it knows, **learned once from
data** by starting from single characters and repeatedly merging the most frequent adjacent pair
into a new token (Byte Pair Encoding) until the list reaches a chosen size.

## Why it matters

Sub-word tokenization (last concept) needs a list of pieces — but *which* pieces? `running` →
`[run][ning]` or `[runn][ing]`? The vocabulary answers that. It's the bridge that turns text into
the ID numbers the network actually consumes, and it's built **before** training, so getting it
right shapes everything downstream.

## The intuition

### Part 1 — what a vocabulary is

A **vocabulary** is the model's complete, fixed list of every token it knows — a numbered table:

```
   id 0 → "<pad>"   id 1 → "the"   id 2 → "cat"   id 3 → "ing"   id 4 → "un"   …   id 49999 → "qux"
```

- **Every token has an ID** = its position in the list (0, 1, 2, …). After tokenizing, "the cat"
  becomes ID numbers `[1, 2]`; those IDs are what get turned into embedding vectors. The vocabulary
  is the bridge: **text → tokens → ID numbers → vectors.**
- **Fixed size, chosen in advance** (commonly ~30k–100k). That's the entire universe of tokens the
  model will ever use.

**Where does the list come from?** Not by hand. The vocabulary is **learned from data, once, up
front**, by scanning a big pile of text. The deciding property is **frequency**: pieces that appear
*often* (`the`, `ing`, `un`, common words) earn a slot; rare strings (`qzx`) don't. And **single
characters are always included** as the base, so anything can be spelled as a fallback — no
out-of-vocabulary, ever. The common multi-character pieces get built up on top of that base
(Part 2).

### Part 2 — BPE: build the list by merging frequent pairs

> **Two phases, and merging appears in both — don't confuse them.**
> **Phase 1 — BUILD (once, on a training corpus):** *create* the merge rules by counting pair
> frequencies. Output = a frozen, ranked list of merges + the vocabulary.
> **Phase 2 — USE (every time, on any text):** *replay* those frozen rules in order to tokenize.
> In Phase 1 we **decide** the merges; in Phase 2 we just **obey** them. Phase 2 works identically
> whether a word was in the training text or is brand-new — which is why nothing is ever OOV.
> (Part 2 below is Phase 1; the "what you end up with" section is Phase 2.)

The real method, **Byte Pair Encoding (BPE)**, *starts small and merges up* (this is **Phase 1**):

1. **Start from characters** — split every word into single characters; the vocabulary begins as
   just those characters (this guarantees everything stays spellable).
2. **Count every adjacent pair** of tokens across all the text.
3. **Merge the single most frequent pair** into one new token; add it to the vocabulary.
4. **Repeat** — recount, merge the next most frequent pair — until the vocabulary hits the chosen
   size.

Worked example. Text is mostly these words (with frequencies): `hug ×10`, `pug ×5`, `hugs ×4`.

```
   split:   h u g (×10)    p u g (×5)    h u g s (×4)     vocab: { h,u,g,p,s }

   pairs:   "u g" = 10+5+4 = 19  ← winner    "h u" = 10+4 = 14    "p u" = 5    "g s" = 4
   merge u+g → "ug":   h ug (×10)   p ug (×5)   h ug s (×4)        vocab: { …, ug }

   pairs:   "h ug" = 10+4 = 14  ← winner     "p ug" = 5     "ug s" = 4
   merge h+ug → "hug":  hug (×10)   p ug (×5)   hug s (×4)         vocab: { …, ug, hug }

   pairs:   "p ug" = 5  ← winner             "hug s" = 4
   merge p+ug → "pug":  hug (×10)   pug (×5)   hug s (×4)          vocab: { …, ug, hug, pug }

   pairs:   "hug s" = 4  ← only pair left
   merge hug+s → "hugs": hug (×10)  pug (×5)  hugs (×4)           vocab: { …, ug, hug, pug, hugs }
```

Now all three words are single tokens — each became whole *because it was frequent enough to earn
merges*. Frequent sequences get merged into their own tokens; rare ones stay as small pieces.
Always merge whatever adjacent pair is most frequent *right now*.

### Part 3 — what you end up with

After thousands of merges on real text, the finished vocabulary is a mix of: **single characters**
(the always-present base), **common fragments** (`ing`, `un`, `ed`), and **common whole words**
(`the`, `hug`) — stopping once it hits the chosen size.

The crucial part: **the ordered list of merges *is* the tokenizer's rulebook.** You keep the merges
*in the order they happened* (`ug`, then `hug`, then `pug`, …). To tokenize **new** text, split it
into characters and **re-apply those merges, in the same order** — which gives one deterministic
result every time. That's why unseen words still work:

```
   new word "thug":  t h u g  →(apply "ug") t h ug  →(apply "hug") t hug  →  [t] [hug]
```

It was tokenized using merges learned from *other* words, falling back to characters where no merge
applies — so **never out-of-vocabulary.** Then each final token becomes its **ID number**, and
(next concept) each ID gets an **embedding vector**. The vocabulary is the finished bridge:
`text → tokens → IDs → vectors`.

**Why a word splits the way it does (and not some other way).** The split is *not* a free choice;
the ranked rule list forces it. Tokenizing rule: at each step, apply the **highest-priority merge
that exists**, and repeat. Example — why `"lower"` becomes `low + er` and never `lo + wer`. Say the
learned merges, in rank order, are:

```
   rule 1: e+r → er     rule 2: l+o → lo     rule 3: lo+w → low     rule 4: low+er → lower

   l o w e r
   → rule 1 (e+r→er):    l o w er
   → rule 2 (l+o→lo):    lo w er
   → rule 3 (lo+w→low):  low er
   → rule 4 (low+er):    lower          result: [lower]
```

`lo + wer` is impossible because there is **no rule `w+er → wer`** — that pair was never frequent
enough to earn a merge, so `wer` simply isn't a token. The granularity is decided entirely by what
was frequent in the training data, frozen into the ranked merge list — which is also why the same
word always tokenizes the same way.

## A picture in words

```
   BUILD (once, from data):     characters → merge most-frequent pair → repeat → vocab (fixed size)
   USE (on any new text):       characters → re-apply the saved merges in order → tokens → IDs
```

Frequency decides granularity: common things end up whole tokens, rare things stay small pieces,
characters catch everything else.

## Check yourself

- What is a vocabulary, and what is a token's ID? (The model's fixed list of known tokens; the ID
  is the token's position in that list.)
- Which pieces earn a slot, and what's always included? (The most frequent pieces; single
  characters are always kept as the base.)
- Describe BPE in one line. (Start from characters; repeatedly merge the most frequent adjacent
  pair into a new token until the list reaches the target size.)
- How does the tokenizer handle a word it never saw? (Re-apply the saved merges in order, composing
  it from known sub-pieces down to characters — never out-of-vocabulary.)

## Questions we worked through

- **Q: Out of all possible character-sequences, which pieces should earn a slot in the vocabulary?**
  A: The most **frequent** ones (common words and pieces like `ing`, `un`). Single characters are
  always kept as the base so anything is still spellable (no out-of-vocabulary).
- **Q: After merging `ug` then `hug` (text: `hug ×10`, `p ug ×5`, `hug s ×4`), what merges next?**
  A: `p`+`ug` → `pug`, count 5 (beats `hug`+`s` = 4). The already-merged `h`+`ug`=14 is gone — you
  always merge the most frequent pair *remaining*.
- **Q: How does the tokenizer turn an unseen word into tokens?**
  A: It re-applies the learned merges in order, composing the word from known sub-pieces (down to
  characters if needed) — so any word can be represented, never out-of-vocabulary.

## What's next / depends on

- **Depends on:** [Tokens & tokenization](02-tokens-and-tokenization.md) (sub-word units).
- **Next:** *Embeddings — words as vectors* — give each token ID a learned vector positioned by
  meaning (the king/queen space). Likely the first **Module 3 portal tab** (a visual of words in
  2-D).
