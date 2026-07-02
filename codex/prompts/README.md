# Codex custom prompts for the LLM course

These three files give Codex the same `/start`, `/continue`, and `/recap` commands the course uses in
Claude Code. They're **optional** — thanks to the auto-start behavior in
[`../../AGENTS.md`](../../AGENTS.md), a learner can just open the folder in Codex, say "hi," and begin.
The commands are a convenience for jumping straight to a mode.

## Install

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

> These are the same instructions as `.claude/commands/*.md`, kept in sync by hand. If you edit the
> tutoring flow, update both. The real source of truth for how the tutor behaves is
> [`../../CLAUDE.md`](../../CLAUDE.md) / [`../../AGENTS.md`](../../AGENTS.md).
