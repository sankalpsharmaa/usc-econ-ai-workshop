# Workshop setup scripts

One script per platform. Each one installs the workshop's common floor: Claude Code, Cursor CLI, Codex, uv, Python, R, Julia, git, and the standard data-science packages for Python, R, and Julia.

Both scripts only add things. If you already have a tool, the script reports it and moves on. Nothing is upgraded, removed, or overwritten, and your system Python is left alone. Python packages go into a workshop-only virtual environment at `~/econ-ai-workshop/.venv`, and Julia packages into a workshop-only project at `~/econ-ai-workshop/julia`.

Every step runs on its own. If one fails, the script logs it and keeps going, then prints a table of what happened.

## macOS

```
bash setup-mac.sh --check    # see what is missing, install nothing
bash setup-mac.sh            # install what is missing
```

## Windows

Open PowerShell, then:

```
powershell -ExecutionPolicy Bypass -File setup-windows.ps1 -Check
powershell -ExecutionPolicy Bypass -File setup-windows.ps1
```

## After it finishes

Close the terminal, open a new one, then check:

```
claude --version
uv --version
Rscript --version
julia --version
```

Activate the Python environment with `source ~/econ-ai-workshop/.venv/bin/activate` on macOS, or `~\econ-ai-workshop\.venv\Scripts\Activate.ps1` on Windows. A Jupyter kernel named "Python (econ workshop)" is registered for you.

## Logs

`~/econ-ai-workshop/setup-mac.log` or `~/econ-ai-workshop/setup-windows.log`. Send that file if something fails.

## Options

| Flag | Effect |
|-|-|
| `--check` / `-Check` | Report what is missing, change nothing |
| `--yes` / `-Yes` | Never ask a question; skip anything that needs one |
| `WORKSHOP_DIR=...` / `-WorkshopDir` | Put the workshop folder somewhere else |
| `PY_VERSION=...` / `-PyVersion` | Use a Python version other than 3.12 |
