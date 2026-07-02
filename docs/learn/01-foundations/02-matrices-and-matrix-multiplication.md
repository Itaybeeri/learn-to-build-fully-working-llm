# Matrices & matrix multiplication — the transformation machine

## In one sentence

A **matrix** is a grid of numbers (rows × columns), and **multiplying** it by a vector
takes that vector in one set of meanings and rewrites it in a new one — the single
operation that happens inside every layer of an LLM.

## Why it matters

A vector held *one* thing's meaning. But a model has to **act** on meanings: take a word's
vector and turn it into something more useful, over and over, until it can predict the next
word. That "take a vector, produce a new vector" step is *exactly* matrix multiplication. The
model's weights ARE matrices; running the model is mostly multiplying by them. Master this
one operation and the rest of the network is just it, repeated and stacked.

## The intuition

### A matrix is a stack of vectors

A vector is one list, `[3, 5, 9]`. A **matrix** is a rectangle of numbers — several vectors
stacked as rows:

```
        height  weight  age
Alice  [  170     65    28  ]
Bob    [  180     80    35  ]
Carol  [  160     55    22  ]   ← a 3 × 3 matrix (3 rows, 3 columns)
```

Size is always stated **rows × columns** (rows first). Here each **row** is one thing being
described (a person); each **column** is one attribute across all of them (all the heights).

### The atom: the dot product (a weighted sum)

Before multiplying matrices, learn the one move they're built from. Take a movie and your
personal taste, both length-4:

```
movie = [funny, scary, romantic, length] = [8, 2, 5, 120]
taste = [3, -1, 2, 0]   ("like funny, dislike scary, like romantic, ignore length")
```

Multiply **matching slot against matching slot**, then add it all up:

```
(8×3) + (2×−1) + (5×2) + (120×0) = 24 − 2 + 10 + 0 = 32
```

That is the **dot product**. Two rules: both vectors must be the **same length** (every slot
needs a partner), and the output is **one single number**. What the number *means*: how
strongly the two vectors align — here, how well the movie fits your taste.

### Matrix × vector = a stack of dot products

Now give several people their own taste, stacked as rows of a matrix, and keep one movie:

```
            funny scary roman length
Alice taste [  3    -1    2     0  ]
Bob   taste [  1     2    0     0  ]        movie = [4, 9, 1, 90]
Carol taste [  0     0    1     1  ]
```

**Multiplying the matrix by the movie** means: dot the movie with **each row in turn**. Each
row yields one number:

```
Alice:  (4×3)+(9×−1)+(1×2)+(90×0) =  5
Bob:    (4×1)+(9×2)+(1×0)+(90×0)  = 22
Carol:  (4×0)+(9×0)+(1×1)+(90×1)  = 91     → output vector [5, 22, 91]
```

A vector went **in** (length 4) and a different vector came **out** (length 3). The matrix is
a **machine**: feed it a vector, get a transformed vector back.

### The point: a matrix re-expresses information

By itself the output `[5, 22, 91]` is just "each person's score for this movie" — a ranking.
The deeper reason we care is the **shapes**:

```
movie (length 4)  →  [ MATRIX ]  →  new vector (length 3)
raw qualities         the machine     "Alice-ness, Bob-ness, Carol-ness"
```

The same movie is now written in a **new, more useful language**. No facts were added; the
information was **re-expressed** along axes we care about more. That is the whole job of a
layer:

> A matrix takes a vector written in one set of meanings and rewrites it in a new set.

In an LLM, swap "people's tastes" for the model's **weights** and "movie qualities" for a
**word's meaning vector**, and one matrix-multiply twists the word's representation toward
whatever the next decision needs.

### Why thousands of them = a model

No single re-expression can jump from raw text to a correct answer — the gap is too big. So
the model stacks many **modest** twists, each making the next step easier, the way a person
goes `letters → words → "it's a geography question" → "wants a capital" → "Paris"`. Each
arrow is one matrix-multiply; the **composition of all of them** spans the distance from raw
symbols to a meaningful prediction. A single transformation is dumb; a long chain of them
behaves like understanding.

## The gentle math

- **Notation & size:** a matrix is named with a capital letter, `M`, and sized **rows ×
  columns**. A 2×3 matrix has 2 rows, 3 columns.
- **Matrix × vector, formally:** output entry *i* = the dot product of **row *i*** of the
  matrix with the input vector.
  ```
  M = [ 2  0  1 ]     v = [1, 4, 2]
      [ 1  1  0 ]
  row 1 · v = 2·1 + 0·4 + 1·2 = 4
  row 2 · v = 1·1 + 1·4 + 0·2 = 5      →  M·v = [4, 5]
  ```
- **The shape rule (the one constraint):** the matrix's **column count** must equal the input
  vector's **length** (each row needs one partner per input slot). The matrix's **row count**
  becomes the output's length.
  ```
  (rows × inner) · (inner) → (rows)
     2   ×  3       3          2
                ^^^ these must match
  ```

## A picture in words

```
        INPUT vector            MATRIX (a pile of rows)           OUTPUT vector
       (one description)   →    each row dots with input    →   (re-described)
        [ a b c ]                [ row1 ]·in = out1               [ out1 ]
                                 [ row2 ]·in = out2               [ out2 ]
                                 [ row3 ]·in = out3               [ out3 ]
   meaning in OLD language        the transformation         meaning in NEW language
```

## Questions we worked through

- **Q: The *sign* of a dot product — compute `[1,0]·[1,0]`, `[1,0]·[0,1]`, `[1,0]·[-1,0]`,
  and say what each tells you.**
  A: `+1`, `0`, `−1`. A **big positive** dot product means the two vectors point the **same
  way** (aligned), **zero** means they're **unrelated** (sideways), **negative** means they
  point **opposite** ways. In one word, the dot product is a **similarity score** between two
  vectors — the seed of attention (a word scores every other word by dotting their vectors).
- **Q: A matrix for 5 movies × 4 qualities — what size, and what's a row?**
  A: 5 × 4 (rows first). Each row is one movie; each column is one quality across all movies.
- **Q: Compute `[4,9,1,90] · [3,−1,2,0]`.**
  A: 12 − 9 + 2 + 0 = **5**. (Low, because the movie is mostly scary, which taste penalizes.)
- **Q: Why does the output vector `[5,22,91]` matter — it's just scores?**
  A: By itself it's only a ranking. Its real value is that the matrix **re-expressed** the
  movie in a new, more useful language — and that re-expression, stacked thousands of times,
  is what a model does.
- **Q: Why thousands of transformations instead of one?**
  A: One matrix can only do a modest twist; the gap from raw text to answer is huge. Stacking
  many modest twists composes into something that spans it.
- **Q: Compute `M·v` for the 2×3 example.**
  A: rows dot v → [4, 5], length 2 (rows of M = length of output).
- **Q: With `movie=[1,4,2]` and Row B=`[1,1,0]` giving 5 — what does the 5 mean?**
  A: How much *that person (Row B)* would enjoy *this movie* — the row is the question, the
  movie is what's judged, the number is the verdict. (Not "how scary the movie is.")
- **Q: When a layer asks "is this royalty?" of the word "king" and gets a high number, what
  was measured?**
  A: How strongly "king" matches that question — high = the word is about royalty.
- **Q: Why does "king" stay a *vector* after every layer instead of collapsing to one number?**
  A: Each layer asks *many* questions at once; one number per question, and many numbers = a
  list = a vector. There's never a single "plain answer."
- **Q: Why isn't "is this a powerful human ruler?" a good *first* question on the raw word?**
  A: It's a *compound* idea (royalty AND person AND power). A single step can only check
  simple ingredients; compound ideas must be built from simpler answers first — like making
  pasta and sauce before asking "is this good lasagna?"
- **Q: Reuse vs. stacking — what's the difference?**
  A: *Reuse* = one matrix's weights are shared across many different words (why "is it
  royalty?" works for king *and* tomato). *Stacking* = many matrices in sequence, each taking
  the previous layer's output, refining the *same* word into a richer description.
- **Q: Why stack many matrices instead of one?**
  A: Built up like floors of a building — each layer answers deeper questions that lower
  layers can't, because it stands on their answers (royalty + person → "powerful ruler").

## Check yourself

1. **What is the dot product, and what shape is its result?**
   → Multiply matching slots and sum them; the result is a single number.
2. **What must be true to take a dot product / matrix-multiply?**
   → The two vectors must be the same length; for a matrix, its column count must equal the
   input vector's length.
3. **What does matrix × vector produce, and what decides the output's length?**
   → A new vector; its length is the matrix's number of rows.
4. **In one line, why is matrix multiplication the heart of an LLM?**
   → It re-expresses a vector's meaning in a more useful form, and a stack of such
   re-expressions is what turns raw text into a next-word prediction.

## What's next / depends on

- **Depends on:** [Numbers & vectors](01-numbers-and-vectors.md) (a matrix is stacked
  vectors; multiplication is repeated dot products).
- **Next:** *Functions & gradients (intuition)* — once a matrix transforms a vector, how do
  we measure whether the result is "good," and how do we nudge the matrix's numbers to make
  it better? That nudging is how a model **learns**.
