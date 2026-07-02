# Instruction tuning & RLHF — turning a text-predictor into a helpful assistant

> **Status: complete.** ① **the problem** — after pre-training (and even ordinary fine-tuning) the model can
> do exactly **one** thing: **predict the next token to continue text**. That's a brilliant autocomplete, but
> autocomplete ≠ assistant. Give a **base model** an instruction like *"Write a poem about the sea"* and it may
> just continue with *more* prompts ("Write a poem about the city…"), because on the web that line usually
> appears in a **list of homework prompts** — the likely continuation is more prompts, not a poem. The model
> has **no concept of "command" or "request"**; it only knows what text usually follows other text. Closing the
> gap between *"continue text like the internet"* and *"do what I'm asking, helpfully/honestly/safely"* is
> **alignment** — and instruction tuning (②) + RLHF (③) are how it's done. ② **instruction tuning (SFT)** —
> the first, most direct fix: it's just **fine-tuning** aimed at a special dataset of thousands of human-written
> **(instruction → good response) pairs**. "Supervised" fine-tuning because each example carries the
> demonstrated correct answer (a human supervised it). The mechanism is **still plain next-token prediction** —
> nothing new bolted on; only the *diet of text* changed. Having practiced thousands of cases where an
> instruction is followed by a helpful answer, the model's **instinct** for "what follows an instruction" flips
> from *"more prompts"* to *"the answer,"* and it **generalizes** — it learned the *pattern* "instruction →
> comply," so it answers instructions never seen in training. SFT gets most of the way to an assistant, but has
> a ceiling: it can only teach what humans bothered to write out, and "good" is hard to *demonstrate* yet easy
> to *judge* — the gap RLHF (③) fills. ③ **RLHF (Reinforcement Learning from Human Feedback)** — built on the
> asymmetry "good is easy to *judge*, hard to *demonstrate*." **Move 1 — the reward model:** instead of writing
> perfect answers, humans **compare** — the SFT model generates two responses to a prompt and a human picks
> which is better; tens of thousands of these comparisons train a **separate** network (the **reward model**)
> that takes a prompt + response and outputs a **number** = how good a human would judge it. It turns fuzzy
> human taste into a *score a machine can chase*, with no human in the loop afterward. **Move 2 —
> reinforcement learning:** the assistant generates a response, the reward model **scores** it, and we **nudge
> the assistant's weights toward higher-scoring responses** (generate → score → nudge → repeat). The model is
> never shown "the right answer" — it's *rewarded* for good ones. Result: the assistant is shaped by **human
> preference at scale**, learning hard-to-write-down qualities (helpful, honest, polite, refuses harm). A leash
> keeps it from drifting too far from the SFT model (else it finds weird high-reward gibberish).

## In one sentence

A pre-trained **base model** only knows how to continue text, so it ignores instructions; **alignment** —
**instruction tuning** plus **RLHF** — retrains it to treat your input as a request and answer the way humans
actually want.

## Why it matters

This is the step that separates a raw language model from ChatGPT-style assistants. Without it you have a
powerful autocomplete that knows a lot but follows nothing; with it you have something that takes an
instruction and responds helpfully, honestly, and safely. Everything that *feels* like "the model understands
what I want" comes from alignment, not from pre-training alone.

## The intuition

### Part 1 — the problem: a base model completes text, it doesn't obey

After pre-training (Module 5), and even after ordinary fine-tuning, the model has learned exactly **one** skill:
**predict the next token** to continue text. It's a phenomenal autocomplete — but autocomplete is not an
assistant, because *"continue this text"* and *"do what I'm asking"* are different jobs.

```
   You type:   "Write a poem about the sea."

   What you WANT (an assistant):
       "Here is a poem about the sea:
        The waves roll in, the gulls take flight..."

   What a raw base model might do (just continue the text):
       "Write a poem about the sea.
        Write a poem about the mountains.
        Write a poem about the city.
        (Homework assignment, due Friday.)"
```

The raw model isn't being difficult — it's doing its job *perfectly*. On the web, a line like "Write a poem
about the sea." most often appears in a **list of homework prompts**, so the statistically likely continuation
is *more prompts*, not a poem. The model has **no notion that you want it to comply**; it only knows what text
**usually follows** other text.

The gap:

- **What it learned to do:** continue text the way the internet continues.
- **What we want it to do:** treat your text as an **instruction** and respond **helpfully, honestly, safely.**

Closing that gap is **alignment** — making the model's behavior match **what humans actually want**, not just
the raw statistics of web text. The raw pre-trained model is the **base model** (powerful but "unaligned" —
knows a lot, follows nothing); the assistant you chat with is built on top of it via instruction tuning (②) and
RLHF (③).

### Part 2 — instruction tuning (supervised fine-tuning / SFT)

The first and most direct fix is something you already understand — it's just **fine-tuning** (Part ③ of the
previous concept) aimed at a special dataset.

**The idea:** show the model thousands of worked examples of *"here's an instruction, here's a good response,"*
and fine-tune on them. The model is still doing plain next-token prediction — but now the text it learns to
continue is **instruction-then-good-answer**, over and over, so it learns that *the way to continue an
instruction is to answer it.*

The dataset is **hand-written by humans** (paid annotators). Each example is a pair:

```
   INSTRUCTION:  "Write a poem about the sea."
   RESPONSE:     "The waves roll in, the gulls take flight,
                  the tide retreats into the night..."

   INSTRUCTION:  "Explain photosynthesis to a 10-year-old."
   RESPONSE:     "Plants make their own food using sunlight..."

   INSTRUCTION:  "Translate 'good morning' into French."
   RESPONSE:     "Bonjour."
```

Thousands of these, across every kind of task — writing, explaining, summarizing, refusing harmful requests.
Because we *demonstrate the correct answers*, this is **supervised fine-tuning (SFT)** — "supervised" because
every example comes with the right answer attached (a human supervised it), exactly like the labeled-data idea.

**What actually changes.** Mechanically it's the *same* next-token prediction — nothing new bolted on. But the
**data reshapes the model's instinct**: where the base model's instinct for "Write a poem about the sea." was
*"continue with more prompts,"* the SFT'd model's instinct becomes *"produce a poem."* It **generalizes** from
the thousands of demonstrations to instructions it has **never seen** — it learned the *pattern* "instruction →
comply," not memorized specific answers.

```
   base model         ──SFT on (instruction → good answer) pairs──►   instruction-following model
   "continue the text"                                                "answer the request"
```

This gets most of the way to a usable assistant, but it has a ceiling that sets up Part 3:

1. **It can only teach what humans can write out.** Someone had to *author* a gold answer for every instruction
   type — expensive, and it can't cover everything.
2. **"Good" is hard to demonstrate, easier to judge.** For open-ended requests ("write me a story") there's no
   single correct answer, and authoring the *perfect* one for thousands of prompts is impractical. But humans
   can easily look at two responses and say *"this one's better."* SFT can't use that kind of feedback — and
   that's the gap **RLHF** fills.

### Part 3 — RLHF (learning from human preferences)

SFT left a ceiling: it can only teach answers humans **write out**, and for open-ended requests "good" is **easy
to judge but hard to demonstrate.** RLHF — **Reinforcement Learning from Human Feedback** — is built precisely
around that asymmetry, in two moves.

**Move 1 — turn human judgments into a reward model.** Instead of asking humans to *write* perfect answers, ask
them to **compare**. Take a prompt, have the SFT model generate **two** answers, and a human picks **which is
better**:

```
   PROMPT: "Explain why the sky is blue."
       Answer A: "It just is, everyone knows that."
       Answer B: "Sunlight is made of many colors; air scatters the
                  blue ones more, so the sky looks blue."
   Human picks:  B is better   ✔
```

Collect tens of thousands of these comparisons, then train a **separate** neural network — the **reward
model** — whose job is: *given a prompt and a response, output a single number ("reward") = how good a human
would judge it.* The comparisons are its training data; it learns to **imitate human taste**, turning a fuzzy
preference into a **number a machine can optimize.** Humans **judge**; the reward model **turns judging into a
score** — and from then on no human is needed in the loop.

**Move 2 — train the assistant to maximize that reward.** Go back to the SFT model (the assistant) and
fine-tune it further — not on fixed answers, but on the reward: it generates a response, the **reward model
scores it**, and we **nudge the assistant's weights to make high-scoring responses more likely** and low-scoring
ones less likely. Generate → score → nudge, repeated. This "act, get a score, adjust to get more score" loop is
**reinforcement learning** — the model isn't shown the right answer, it's *rewarded* for producing good ones.

```
   assistant generates a response ─► reward model scores it ─► nudge weights toward higher-scoring responses ─► repeat
```

The payoff: the assistant is shaped by **human preference at scale**, not a fixed answer-book. It learns the
hard-to-write-down qualities — be helpful, honest, polite, refuse harmful requests — because those are what
humans rewarded. That's the final polish that makes a model feel like a careful, aligned assistant.

> **One guard rail.** While chasing reward, the model is held back from drifting *too* far from the SFT model —
> otherwise it could find weird, high-reward gibberish. So it improves on human preference while staying close
> to sensible language. (No math needed — just know there's a leash.)

## Check yourself

- What single skill does a base model have? (Predict the next token to **continue text** — nothing else.)
- Why might it list more prompts instead of writing the poem you asked for? (It has **no concept of a
  command/request**; "Write a poem…" usually appears in lists of prompts on the web, so the likely continuation
  is *more prompts*. It's continuing text, not obeying.)
- What is the missing piece called? (**Alignment** — retraining the model to treat input as an instruction and
  respond the way humans want.)
- What is instruction tuning / SFT, and what's in its dataset? (Fine-tuning on thousands of human-written
  **(instruction → good response) pairs**; "supervised" because each example has the right answer attached.)
- If SFT is still just next-token prediction, what changed the behavior? (Only the **diet of text** — practicing
  instruction-then-answer reshapes the model's instinct from "more prompts" to "the answer," and it
  **generalizes** the pattern to unseen instructions.)
- What's SFT's ceiling (that RLHF addresses)? (It can only teach answers humans **write out**; and "good" is
  hard to *demonstrate* but easy to *judge* — SFT can't learn from "this answer is better than that one.")
- What is the reward model and where does it come from? (A **separate** network trained on tens of thousands of
  human **A-vs-B comparisons**; it takes a prompt + response and outputs a **score** = how good a human would
  judge it — imitating human taste so no human is needed in the loop.)
- How is the assistant trained in RLHF's second move? (It generates a response → the **reward model scores
  it** → **nudge its weights toward higher-scoring** responses → repeat. That act-score-adjust loop is
  **reinforcement learning**; the model is rewarded for good answers, never shown a fixed "right" one.)
- Why a guard rail / leash? (To stop the model drifting **too far** from the SFT model and finding weird,
  high-reward gibberish — it improves on preference while staying close to sensible language.)

## Questions we worked through

- **Q: A base model is great at next-token prediction, yet told "Write a poem about the sea" it might list more
  prompts. Why — what is it actually doing, and what's missing?** A: the model does what it was trained to do —
  predict the next word; it doesn't understand the line as a command/request, so it just predicts likely
  continuation (more prompts); the missing part is **alignment**. *(Fully correct — named the cause (no concept
  of command) and the missing piece (alignment) unaided.)*
- **Q: What is instruction tuning / SFT — what's in the dataset, and how does it change behavior even though
  the mechanism is still next-token prediction?** A: SFT is supervised tuning; the dataset is human-made
  samples; it tunes the model to understand what's expected of it when asked to write a poem or story. *(Core
  correct — supervised, human-written data, learns to comply. Sharpened: the data is **(instruction → answer)
  pairs**; the mechanism stays plain next-token prediction (only the *diet* changed); and it **generalizes**
  the "instruction → comply" pattern to unseen instructions.)*
- **Q (top rung): (1) Why isn't SFT enough? (2) RLHF's two moves — what is the reward model, where does it come
  from, and how is the assistant trained with it?** A: (1) for open-ended questions "good" is easy to judge,
  hard to demonstrate, so humans vote A-vs-B; (2) first pass jumped to "fine-tune again on the good answers"
  (effectively just more SFT). Stepped down to isolate the reward model: *we have thousands of A-vs-B judgments
  — what could we build to score a brand-new answer automatically?* → "the reward model, and nudge the
  assistant's weights toward high-score responses." Climbed back to the full answer cleanly: **Move 1** train a
  separate **reward model** from the comparisons (prompt + response → score, imitating human taste); **Move 2**
  the assistant generates → reward model scores → nudge weights toward higher reward → repeat (reinforcement
  learning). *(Cleared the top once the reward-model step landed.)*

## What's next / depends on

- **Depends on:** [Next-token prediction](../05-training-llms/01-next-token-prediction.md) (the one skill a base
  model has), [Fine-tuning](03-fine-tuning.md) (alignment is built out of fine-tuning).
- **Also leans on:** [How LLMs learn](../00-orientation/02-how-llms-learn-and-where-knowledge-lives.md) (a
  separate network — the reward model — learns to score from data, just like any model).
- **This concept is complete.** Next in Module 6: **Limitations** (hallucination, etc.) — why even a
  well-aligned model can confidently state false things, and the other ceilings on what LLMs can do.
