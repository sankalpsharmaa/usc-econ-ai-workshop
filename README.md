# USC Economics AI Workshops

Hands-on sessions on AI tools for economics research, led by PhD students. Attendees apply the skills in real time rather than watching a lecture. Organized by David Schonholzer.

## Fall 2026: AI Bootcamp

Two afternoons, August 24 and 25, 2026, two 50-minute sessions each day. Aimed at first-year PhD students and open to everyone else. Taught by Sankalp Sharma and Joshua Levy.

**Day 1, August 24.** Why a chatbot becomes "agentic", a shared vocabulary for the tools (context, `CLAUDE.md`, skills, rules, hooks, harness), and a live data-scraping demo built on SEC EDGAR.

**Day 2, August 25.** Gentzkow and Shapiro (2014) reread for a world where the research assistant is a bot, testing research code, git and version control, project directory structure, and building Claude artifacts to learn first-year material.

Slides and demo code: [`fall-2026/`](fall-2026/)

## Spring 2026: Workshop series

| Date | Time | Location |
|-|-|-|
| February 3, 2026 | 4:00–5:00 PM | KAP 319 |
| March 24, 2026 | 4:00–5:00 PM | KAP 319 |

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

Slides and code: [`spring-2026/`](spring-2026/)

## Materials

Each term folder holds the slides and code for its sessions. Video recordings are in this [Dropbox](https://www.dropbox.com/scl/fo/re27ib58js9g63bw4h32g/ACrxNTiZhjEzat7H-00NwW8/20260203?rlkey=8f5sqqwdih13pizf1l2dmuo7l&e=1&subfolder_nav_tracking=1&dl=0) folder.

## Prerequisites

Python 3.10 or newer, and API keys for whichever services a session uses.

[`setup/`](setup/) has one install script per platform. Each installs the common floor (Claude Code, Cursor CLI, Codex, uv, Python, R, Julia, git, and the standard data-science packages) and only adds what is missing:

```bash
bash setup/setup-mac.sh --check     # see what is missing, install nothing
bash setup/setup-mac.sh             # install it
```

Windows users run `setup/setup-windows.ps1` instead. See [`setup/README.md`](setup/README.md) for details and logs.

## Contact

For questions, contact the workshop organizers or open an issue.
