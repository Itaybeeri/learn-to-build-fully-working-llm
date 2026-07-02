# Tokens & tokenization

## In one sentence

A **token** is the discrete chunk of text a model actually reads, and **tokenization** is the
process of cutting raw text into those chunks — done with *sub-word* pieces so a fixed, modest
list can spell *any* word, including ones never seen before.

## Why it matters

Last concept showed we must turn words into vectors. But which "words"? Language has endless
variety — new words, typos, run/runs/running — so we can't keep a master list of every possible
word. Tokenization is the practical answer: it decides the *units* we embed. Get it wrong and the
model either can't read new text at all, or needs an impossibly huge vocabulary. Every LLM begins
by tokenizing its input, so this is the true first step of the pipeline.

## The intuition

### Part 1 — what a token is, and why "one token per word" fails

A network reads text as **discrete units**: it cuts the text into pieces, then looks up a vector
for each piece.

- Each piece the model reads is a **token**.
- Cutting text into tokens is **tokenization**.
- The fixed list of all tokens the model knows is its **vocabulary**; every token in it has its own
  embedding vector.

```
   "The cat sat"  ──tokenize──►  [The] [cat] [sat]  ──embed──►  vectors  ──►  network
```

**First obvious idea: one token = one whole word** (split on spaces). Intuitive, but it fails two
ways against language's endless variety:

1. **Out-of-vocabulary (new words).** Any word not on the list — a name, a typo ("teh"), a coined
   word — has **no vector**, so the model can't represent it or relate it to anything.
2. **Vocabulary explosion.** You'd need *millions* of slots, and still never be complete: new words
   appear constantly, every form is a separate slot (`run`, `runs`, `running`, `ran` share no
   connection), every typo is unknown, and that's one language only.

So whole-words give a list that can be neither complete nor small. (Next: the opposite extreme —
characters — then the sweet spot, sub-words.)

### Part 2 — the other extreme: characters

Swing the opposite way: **one token = one character.** The vocabulary is just letters, digits, and
punctuation — roughly **100 tokens** for English.

```
   "cat"      →  [c] [a] [t]
   "running"  →  [r] [u] [n] [n] [i] [n] [g]
```

This **fixes both** whole-word problems: nothing is ever out-of-vocabulary (any word, name, typo,
or coined word is spellable from ~100 characters), and the vocabulary is tiny and complete. But it
introduces **two new** problems:

1. **The meaning unit is gone.** A lone letter `c` carries almost no meaning. The token `cat` was a
   meaningful unit; now the model must *reassemble* "cat" from three meaningless letters — much
   harder work.
2. **Sequences get long.** "The cat sat on the mat" is 6 word-tokens but 22 character-tokens. Since
   a model processes every token and relates each one to all the others, 3–4× more tokens means
   much more compute, slower runs, and more memory (and the token-to-token cost grows worse than
   linearly — see attention, Module 4).

So each extreme fixes the other's flaw but breaks something new:

```
   WHOLE WORDS          CHARACTERS
   ✓ meaning kept       ✗ meaning lost
   ✗ huge vocab         ✓ tiny vocab
   ✗ new words = OOV    ✓ anything spellable
   ✓ short sequences    ✗ very long sequences
```

### Part 3 — sub-words: the sweet spot

The fix is the middle ground: **sub-word tokens.** Keep common words and useful pieces as single
tokens, and split rare/long/new words into smaller *reusable* pieces.

```
   common word:   "the", "cat", "run"     →  one token each
   rarer word:    "running"               →  [run] [ning]
                  "unhappiness"            →  [un] [happi] [ness]
   unknown word:  "blorptastic"           →  [bl] [or] [pt] [astic]   (worst case: down to letters)
```

This captures the best of both extremes:

- **Fixed, modest vocabulary** (typically ~30k–100k tokens) — not millions, not 100.
- **Never out-of-vocabulary** — any word can be built from smaller pieces, falling back to single
  characters in the worst case. Nothing is ever unknown.
- **Meaning units kept** — common words stay whole, and recurring pieces like `un`, `ing`, `ness`
  become reusable tokens that carry real grammatical meaning.
- **Short-ish sequences** — ordinary text tokenizes to near word-length, not character-length.

Bonus: it repairs an earlier complaint. Whole-words treated `run`, `runs`, `running` as unrelated
slots; sub-words share the piece `run` across all of them, so the relationship is built in. This
sub-word approach (BPE, WordPiece, and friends) is what real LLMs use; *how* the piece list is
learned from data is the next concept, **Vocabularies**.

## A picture in words

```
   "The cat sat on the mat"

   whole words →  [The][cat][sat][on][the][mat]    6 tokens · huge vocab · new words = OOV
   characters  →  [T][h][e][ ][c][a][t]...         22 tokens · tiny vocab · meaning lost
   sub-words   →  [The][cat][sat][on][the][mat]    ~6 tokens · modest vocab · nothing unknown
                  (rare word "unhappiness" → [un][happi][ness])
```

Sub-words sit in the middle: small-enough list, nothing ever unknown, meaning chunks kept, and
short sequences.

## Check yourself

- What is a token? What is tokenization? (The unit the model reads; the process of cutting text
  into those units.)
- Why does "one token per whole word" fail? (Out-of-vocabulary new words; a millions-of-slots
  vocab that's never complete.)
- Why do single-character tokens fail, even though they fix those two? (Meaning unit lost;
  sequences get long → more compute/memory.)
- How do sub-words beat both extremes? (Common words/pieces stay whole — meaning kept, sequences
  short, modest vocab; rare/new words split into pieces down to characters — nothing unknown.)

## Questions we worked through

- **Q: If every token is a whole word, what breaks on real text — for a new word, and for vocab
  size?**
  A: A new/typo/coined word has no vector (out-of-vocabulary) — the model can't represent it. And
  the vocabulary would need millions of slots and still never be complete (new words, every word
  form, typos, other languages).
- **Q: Single-character tokens fix vocab size and OOV — so what new problems do they create?**
  A: The meaning unit is lost (a lone letter means little; the model must reassemble words from
  letters), and sequences get 3–4× longer → more compute, slower, more memory.
- **Q: How does sub-word tokenization beat both the whole-word and the character problems?**
  A: Common words stay single tokens (meaning kept, sequences short) and the list stays at tens of
  thousands (not millions); rare/new words split into reusable pieces down to characters, so
  nothing is ever out-of-vocabulary.

## What's next / depends on

- **Depends on:** [Why representing language is hard](01-why-representing-language-is-hard.md)
  (text → meaningful numbers; the endless-variety hurdle).
- **Next:** *Vocabularies* — how the sub-word piece list is actually built/learned from data
  (BPE) — then *Embeddings*.
