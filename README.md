# LLM Skill Files for the MIST Ecosystem

This repository contains **Agent Skills** for Claude Code. Each gives Claude
deep, structured knowledge about one tool in the MIST ecosystem so it can answer
questions, debug issues and guide workflows without re-reading the source or the
docs every time.

## Available Skills

| Skill                                 | Tool                              | Description                           |
| ------------------------------------- | --------------------------------- | ------------------------------------- |
| [`mist-expert`][mist]                 | [MIST][mist-repo]                 | 3D medical image segmentation         |
| [`misfit-expert`][misfit]             | [MISFIT][misfit-repo]             | Imaging foundation-model pretraining  |

[mist]: mist-expert/SKILL.md
[misfit]: misfit-expert/SKILL.md
[mist-repo]: https://github.com/mist-medical/MIST
[misfit-repo]: https://github.com/mist-medical/MISFIT

## Usage

Copy or symlink a skill directory into your skills folder — `~/.claude/skills/`
for every project on the machine, or `<project>/.claude/skills/` to share it
with that repo's collaborators:

```bash
git clone https://github.com/mist-medical/skills.git
cp -r skills/mist-expert ~/.claude/skills/
```

Skills are discovered when a session starts, so restart Claude Code after adding
one. `/skills` lists what is loaded.

From there a skill applies in **two** ways:

- **Automatically** — Claude reads the `description` and loads the skill when
  your request matches it. Paste a MIST traceback or ask about `config.json` and
  the expert context is there without you naming it.
- **Explicitly** — invoke it by name with `/mist-expert`, as before.

The automatic path is the reason these are skills rather than slash commands: it
does not require the user to already know the skill exists.

### Other assistants

The skill body is plain markdown and portable to other coding agents.

- **OpenAI Codex** — Codex uses the same skill format: a directory containing a
  `SKILL.md` with `name`/`description` frontmatter, loaded on demand. Copy a
  skill directory into `~/.codex/skills/` (personal) or the project's
  `.agents/skills/` (shared, version-controlled). Effectively drop-in.
- **Gemini CLI** — has no on-demand skill format; it loads `GEMINI.md` context
  files instead. Reference a skill from a `GEMINI.md` (Gemini supports
  `@`-imports, e.g. `@mist-expert/SKILL.md`) or paste the body in. Like Cline,
  this is always in context, so install only what a project needs.
- **Cline** — copy `SKILL.md` into the project's `.clinerules/` (renaming it
  after the skill). Cline rules are always in context rather than loaded on
  demand, so install only the ones a given project needs.

## About the Tools

- **[MIST][mist-repo]** (Medical Imaging Segmentation Toolkit) — a simple,
  scalable, end-to-end framework for 3D medical image segmentation, handling
  everything from raw NIfTI files to trained models and evaluated predictions.
  Full documentation is at
  [mist-medical.readthedocs.io](https://mist-medical.readthedocs.io/).
- **[MISFIT][misfit-repo]** (Medical Imaging Semantic Foundation Toolkit) —
  pretrains 3D imaging foundation models via masked autoencoding on unlabeled
  NIfTI files, producing a SwinUNETR-V2 encoder that can transfer into MIST for
  segmentation fine-tuning.
