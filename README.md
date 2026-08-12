# USC Economics AI Workshops (Spring 2026)

Hands-on workshops on AI tools for economics research, led by PhD students. Sessions are designed for attendees to apply skills in real time.

## Schedule

| Date | Time | Location |
|------|------|----------|
| February 3, 2026 | 4:00–5:00 PM | KAP 319 |
| March 24, 2026 | 4:00–5:00 PM | KAP 319 |

## Sessions

### Workshop 1: February 3

**Neil He**: Using GPT as a Data-Labeling Tool for Economics Research  
Tutorial on using GPT via API to convert unstructured text into analysis-ready variables. Covers API calls, schema design, batching, and export.

**Daniil Sherstnev**: Retrieval-Augmented Generation (RAG) for LLMs  
Framework for grounding LLM outputs in a user-provided knowledge base to reduce hallucinations and enable access to local files and niche literature.

### Workshop 2: March 24

**Joshua Levy**: Cursor for Economists: An AI-First IDE from RA to Referee  
Using Cursor across the research lifecycle: project scaffolding, data collection, custom ML tools, and producing reproducible exhibits.

**Sankalp Sharma**: Agentic Web Scraping with Claude Code  
Building web scrapers without writing code manually. Covers project initialization, dependency management, error handling, and supervising AI-generated code.

## Materials

Each session folder contains slides and code examples. Video recordings are available in the this [Dropbox](https://www.dropbox.com/scl/fo/re27ib58js9g63bw4h32g/ACrxNTiZhjEzat7H-00NwW8/20260203?rlkey=8f5sqqwdih13pizf1l2dmuo7l&e=1&subfolder_nav_tracking=1&dl=0) folder.

## Prerequisites

**Run the setup script before you arrive.** It installs everything the sessions
need — Python, R, Julia, Cursor, Claude Code, Codex, git, and the rest — asking
before each one and leaving anything you already have alone. Budget 15–30
minutes, mostly downloads.

macOS:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/sankalpsharmaa/usc-econ-ai-workshop/jyl/setup/install.sh)"
```

Windows (PowerShell, no administrator rights needed):

```powershell
irm https://raw.githubusercontent.com/sankalpsharmaa/usc-econ-ai-workshop/jyl/setup/install.ps1 | iex
```

Already set up, or want to check? `setup/bin/doctor.sh` (macOS) or
`setup\bin\doctor.ps1` (Windows) reports what you have and changes nothing.

See [`setup/README.md`](setup/README.md) for what it installs, how to install
pieces by hand on a managed laptop, and what to do if something fails. You will
also need API keys — copy `.env.example` to `.env` and fill it in.

Individual session folders may add their own notes.

## Contact

Organized by David Schonholzer. For questions, contact the workshop organizers or open an issue.
