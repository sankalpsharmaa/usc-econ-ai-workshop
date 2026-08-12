# Workshop environment setup

One command gets your machine ready for the AI workshop sessions. It takes
roughly 15–30 minutes, most of it downloads. **Do this before the session**, not
during it.

The script asks before installing anything, tells you what it found already on
your machine, and never overwrites something you already have without asking
first.

---

## macOS

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/sankalpsharmaa/usc-econ-ai-workshop/jyl/setup/install.sh)"
```

> The `/bin/bash -c "$(...)"` shape is worth the extra punctuation. With
> `curl ... | bash`, bash starts executing the script as it arrives, so a
> download that gets cut off halfway runs half a script. Command substitution
> downloads the whole file first and only then runs it.

## Windows

In PowerShell (no administrator rights needed):

```powershell
irm https://raw.githubusercontent.com/sankalpsharmaa/usc-econ-ai-workshop/jyl/setup/install.ps1 | iex
```

---

## Before you run it

You are about to run a script off the internet, which is worth exactly one
moment of suspicion — the same suspicion the rest of this workshop will teach
you to apply to AI agents running commands on your behalf. Read it first:

```bash
curl -fsSL https://raw.githubusercontent.com/sankalpsharmaa/usc-econ-ai-workshop/jyl/setup/install.sh | less
```

It downloads this `setup/` directory and runs `bin/setup.sh`. Nothing is hidden,
and nothing is minified.

---

## What it installs

Everything is optional. You will be asked about each one, and you can answer `?`
at any prompt to see the exact commands that would run.

| | macOS | Windows |
|---|---|---|
| Package manager | Homebrew | winget (built in) |
| Editor | Cursor | Cursor |
| Agent CLIs | Claude Code, Codex, `cursor-agent` | Claude Code, Codex |
| Terminal | cmux | Windows Terminal |
| Prompt | oh-my-posh | oh-my-posh |
| Python | uv + Python 3.12 | uv + Python 3.12 |
| R | R via `rig`, plus `renv` | R, plus `renv` |
| Julia | juliaup | juliaup |
| Publishing | Quarto | Quarto |
| Version control | git, GitHub CLI | git, GitHub CLI |
| Build | GNU Make | GNU Make |

Your existing Python and R are never modified. uv installs its own interpreters
in its own directory; `rig` and `juliaup` manage versions side by side. If you
already have a working setup, keep it — say "keep existing" when asked.

---

## Options

```
--all           Install everything missing, no per-component questions
--check         Report what you have and exit. Changes nothing.
--dry-run       Show every command that would run, without running any
--only a,b,c    Only consider these components
--skip x,y      Never consider these components
```

On Windows these are PowerShell switches: `-All`, `-Check`, `-DryRun`,
`-Only a,b`, `-Skip x,y`.

Component ids: `clt brew git gh make node uv r julia quarto cursor cursor_cli
claude codex cmux omp` (macOS); `winget pwsh terminal git gh make node uv r
julia quarto cursor claude codex omp` (Windows).

---

## Checking your setup later

```bash
setup/bin/doctor.sh          # macOS
.\setup\bin\doctor.ps1       # Windows
```

Prints the same table the installer starts with. **If something is wrong on the
day, run this and send us the output** — it is the fastest way for us to see
what is going on.

---

## After it finishes

The script cannot log you in. Three things are left:

1. **Restart your terminal.** PATH and prompt changes do not apply to the window
   you ran the installer in.

2. **Sign in to each tool**, one at a time:

   ```
   claude               follow the browser prompt
   codex                sign in with ChatGPT
   cursor-agent login   Cursor is free for students
   gh auth login        choose HTTPS, then browser sign-in
   ```

   Claude Code's free tier cannot use browser sign-in — you need a
   Pro/Max/Team plan or an `ANTHROPIC_API_KEY`. USC IT can grant ChatGPT and
   Codex access if you do not already have it.

3. **Add your API keys.** From the repository root:

   ```bash
   cp .env.example .env
   ```

   Then fill in `OPENAI_API_KEY`. Neil's labelling demos read it from there.
   `.env` is gitignored; never commit real keys.

---

## Running several agents at once

`bin/agents.sh` (macOS) and `bin/agents.ps1` (Windows) set up the parallel-agent
workflow the later sessions demonstrate:

```bash
setup/bin/agents.sh did-spec robustness tables
```

That creates three git worktrees — three separate checkouts of the same
repository on three branches, sharing one `.git` — and opens a workspace for
each. Point one agent at each. Two agents editing the same working tree will
overwrite each other's work; worktrees are how you avoid that.

Review and merge with ordinary git:

```bash
git diff main..agent/tables
git worktree remove ../usc-econ-ai-workshop-tables
```

---

## If something goes wrong

The installer keeps going when one component fails, so you get everything else.
Failures are listed at the end along with a log path
(`~/.econ-ai-setup/install-*.log`).

Re-run only what failed:

```bash
setup/bin/setup.sh --only quarto,julia
```

**University-managed laptops** sometimes block `winget`, unsigned PowerShell, or
Homebrew's installer. If that happens, run `--check` to get the list of what is
missing and install those by hand — the `?` option at each prompt prints the
exact command. Bring the log to the session and we will work through it.

---

## Status

**The macOS path has been exercised end to end** on Apple Silicon — detection,
`--check`, `--dry-run`, the conflict prompts, and the config merge.

**The Windows path has never been run.** It was written against documentation
and has had no syntax check and no execution. In particular, every `winget`
package id in `lib-ps/Components.psm1` is unconfirmed, as is whether each honours
`--scope user`. Before this goes to participants, someone needs to sit at a
Windows 11 machine and:

1. `winget search <id>` for each id in the catalogue,
2. run `.\setup\bin\setup.ps1 -DryRun -All`,
3. run a real install on a fresh account,
4. repeat on Windows 10 if any participant is likely to be on it.

Do the same shakedown on a genuinely fresh macOS account rather than a developer
machine, since detection behaves differently when nothing is installed.

---

## Layout

```
setup/
├── install.sh / install.ps1   bootstrap: fetches this directory, then runs bin/setup
├── bin/
│   ├── setup.sh / .ps1        the installer itself
│   ├── doctor.sh / .ps1       report only, changes nothing
│   └── agents.sh / .ps1       git worktree + workspace per agent
├── lib/                       macOS: ui, registry, helpers, component catalogue
├── lib-ps/                    Windows: the same, as PowerShell modules
└── assets/                    Cursor settings and extensions, prompt theme, .env template
```

Each component declares `detect`, `plan`, `install`, `extras`, and `verify`.
`detect` never changes anything, which is what makes `--check` and `--dry-run`
trustworthy. Adding a tool means adding one entry to the catalogue
(`lib/components.sh` or `lib-ps/Components.psm1`).
