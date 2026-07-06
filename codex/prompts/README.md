# Codex custom prompts for the LLM course (legacy fallback)

> **You almost certainly don't need this.** The `/start`, `/continue`, and `/recap` commands now ship as
> Codex **skills** in [`../../.agents/skills/`](../../.agents/skills/) — checked into the repo and
> auto-discovered, with nothing to copy. Invoke them with `$start` · `$continue` · `$recap`, or just say
> "hi" and let the auto-start in [`../../AGENTS.md`](../../AGENTS.md) begin. OpenAI now recommends skills
> over custom prompts (custom prompts are deprecated), so the skills are the primary path.

These three files are the **older** way to get the same commands, kept only as a fallback for Codex
versions that don't yet support repo-scoped skills. They're entirely optional.

## Install (fallback only)

Codex discovers custom prompts from `~/.codex/prompts/`. Copy these files there:

```bash
mkdir -p ~/.codex/prompts
cp codex/prompts/start.md ~/.codex/prompts/
cp codex/prompts/continue.md ~/.codex/prompts/
cp codex/prompts/recap.md ~/.codex/prompts/
```

On Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.codex\prompts" | Out-Null
Copy-Item codex\prompts\*.md "$env:USERPROFILE\.codex\prompts\"
```

Then in Codex type `/start`, `/continue`, or `/recap`.

> These are the same instructions as `.claude/commands/*.md`, `.agents/workflows/*.md`, and
> `.agents/skills/*/SKILL.md`, kept in sync by hand. If you edit the tutoring flow, update all of them.
> The real source of truth for how the tutor behaves is [`../../CLAUDE.md`](../../CLAUDE.md) /
> [`../../AGENTS.md`](../../AGENTS.md).
