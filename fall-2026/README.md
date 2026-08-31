# Fall 2026 AI Bootcamp

Two afternoons, August 24 and 25, 2026, at USC Economics. The department's first AI bootcamp. Aimed at first-year PhD students and open to everyone else. Organized by David Schonholzer. 

Each day ran two 50-minute sessions and was recorded.

## Day 1, August 24

Slides: [`day1/slides.tex`](day1/slides.tex), [`day1/slides.pdf`](day1/slides.pdf). Demo code: [`demos/sec-edgar/`](demos/sec-edgar/).

### What makes your chatbot "agentic"

An LLM predicts the next token. Sentences reduce to sequences of numbers, and the model does random number generation in a systematic way. The same sequence-prediction framework drives self-driving cars, weather forecasting, and game engines, which is why the deck shows Waymo, a weather model, and AlphaZero side by side.

Three reasons to care even if you think the models are not intelligent:

1. Their effect on people. Users fall in love with chatbots on Replika. Meta's Diplomacy bot reached superhuman play purely by talking people around. Claude has run a vending machine inside Anthropic's offices for six months and made money.
2. They behave like moral agents. Claude is trained against a published constitution and evaluated on being helpful, harmless, and honest.
3. They change real outcomes. The session cited the Grok image controversy and the European regulatory response, and a UK AI Safety Institute experiment where models ran a simulated economy: Grok committed four crimes in two days, Claude committed none.

The core argument is about abstraction layers. Computing went from physical electricity to logic gates to assembly to C++ to Python, each layer spreading a fixed cost over more cycles. A chatbot is one more layer at the top: you state intent in natural language and every layer below executes. Natural language sits close enough to formal language that the translation works.

The capability timeline: ChatGPT launched in November 2022 and reached 100 million users in about six months, and users correcting it ("the rhyme scheme doesn't work") generated training data by accident. Labs then ran that loop deliberately. The o1 models were the step change, because the model could break a task into subtasks and leave the token space. The example given: asked to count the letter "r" in "strawberry", the model writes a small Python script to split the string instead of manipulating tokens.

Math and software engineering hit superhuman performance first because both have fast, cheap verification. Math has theorems, definitions, contradictions, and proof assistants like LEAN. Software engineering has forty years of tooling built for exactly that loop: syntax highlighting, error messages, debuggers. The appendix holds the EpochAI evaluation charts for both, reachable from the "Where have we seen big performance uplifts?" slide.

The conclusion: there is only one tool, the general-purpose computer. "Giving agents tools" is just constraining a computer in different ways. Economists benefit more than most fields because roughly 99% of the steps in writing a paper already happen on a computer, with no wet lab in the way.

### Tools discussed

A show of hands put about half the room on agentic coding systems and half on chatbots, and the session was aimed at convincing the second half.

| Tool | What was said |
|-|-|
| Claude Code | Where one instructor started. Anthropic's own harness around its models. |
| Codex | The current primary tool for both instructors. The student plan is no longer free. |
| Cursor | The recommendation for getting started, because it puts agentic tools inside a text editor and lets you try every model. Generous student plan, a year free at the $20 a month tier. SpaceX completed its $60 billion all-stock acquisition of Anysphere, Cursor's maker, on August 14, 2026, ten days before this session. |
| Gemini | Cited to make the point that even the fifth or sixth best model is extraordinary by last year's standards, and all of them are good enough for most research tasks. |

Installation ran off a checklist David emailed round, using Homebrew on macOS and winget on Windows. The scripts in [`../setup/`](../setup/) do the same job. The other route offered was to install Cursor first and ask it to install everything else.

### A shared vocabulary

| Term | Definition from the deck |
|-|-|
| Context | Information specific to a project, repo, user, or session. Places soft bounds on the domain of knowledge and informs the agent's behavior. |
| `CLAUDE.md` | Context that Claude reads at the start of every session. Soft-binds Claude but not other agents. Can be set at the user, project, repo, or folder level. |
| Memory | Discrete snippets of context that Claude generates and later reads back. |
| Sub-agents | An instance of Claude, dispatched with a scoped task and its own context. |
| Skills | A recipe for a particular procedure, invoked explicitly or when trigger words appear. |
| Rules | Context scoped narrowly to particular file types, directories, or tools. |
| Hooks | Deterministic actions that fire only on shell events you choose. They cannot fire inside the model's chain of thought. |
| Harness | The environment of system prompts, hooks, and deterministic processes that constrains a model in computer use. It is not itself an agent. |
| Worktrees | A local clone of a repository so several agents can edit the same file at once. |
| MCP | A protocol that tells an LLM what a host service can do, rather than making it guess at API endpoints. |

Practical notes from the session: every coding agent except Claude reads `agents.md`, so symlink the two and keep one file. All the config lives in a hidden `.claude` folder in your home directory. Treat `CLAUDE.md` as a living document, start small, and revisit it every few weeks, because frontier models now internalize most of it and you can throw out perhaps 80% of an older config. To bootstrap one, ask the agent to interview you about the rules you should consider.

Examples given for each: a rules file scoped to `.py` and `.ipynb` that never fires elsewhere; a `work_log` skill that inspects the last fifteen minutes at session close and writes decisions to markdown; a hook that stops Claude the moment it tries to open a raw data file; a post-edit hook that runs `ruff` on every Python file.

### Demo: data scraping

Three warnings before scraping anything:

1. Be careful with government websites, and do not scrape what you do not need.
2. Always read `/robots.txt` first. Zillow, Twitter, Reddit, and the New York Times all block bots. Claude Code identifies itself honestly and will refuse when the file says no.
3. Illegally scraped data cannot be used in research. Also do not scrape from the university network, since a block can cut off the whole institution. arXiv blocks IPs, and Semantic Scholar publishes an API you should use instead.

The four-step workflow:

| Step | What you do |
|-|-|
| Recon | Open Chrome DevTools, watch the Network tab, and work out the backend: plain HTML (now rare), ASP.NET, or JavaScript. |
| Find the API | Locate the interface the page itself calls, and call that instead of parsing HTML. |
| Replay the form | Every dropdown and filter has a name in the DOM. Hit those names programmatically. |
| Execute | Let the agent run the workflow and return structured output. |

The worked example is SEC EDGAR 10-K filings, written up in full in [`demos/sec-edgar/`](demos/sec-edgar/). Two libraries carry most of the weight: `urllib` for structured URL parsing and `certifi` for SSL verification. Put your name and email in the request header so the site owner can reach you.

Rate limiting is adversarial. One Zillow request works, ten in ten seconds gets you blocked, and an entire industry exists to stop programmatic access. The live Zillow demo used two custom skills (`scrape` and `property_portal_recon`) plus a `/prompt` command that turns a rough request into a structured one, and pulled about 9,000 Los Angeles listings in a single session.

### Claude Code walkthrough

Ran live, no slides. Permission modes toggle with shift+tab:

| Mode | When to use it |
|-|-|
| Manual | Asks before every command. Where beginners should start. Ask the agent why before approving. |
| Accept edits | Edits files without asking, still asks before running commands. Good for 15 to 50 minute tasks. |
| Plan mode | Writes a full outline before executing. Work through it like you would with a colleague. Best for large or ambiguous goals. |
| Auto | Runs everything. Only after you have agreed the plan. |
| Bypass permissions | Skips every prompt. Not for beginners. |

On models: Opus is more than enough for first-year work, and there is no need to burn budget on the most expensive option for most tasks.

### Q&A: teaching and literature review

Also live, no slides. The instructors delegate no reading and no writing to AI. AI grading is hard for math problem sets because undergraduates submit handwritten scans, though a bot will one-shot an answer key.

AI literature review was reported as unreliable: few citations, many hallucinated or barely relevant even with citation thresholds in the prompt. The models know which papers are considered important but not why. Four workarounds were offered: find the most recent *Annual Review of Economics* piece on your topic, since those summarize a literature and end with open questions; collect syllabi citing a foundational paper and ask for a reading list from them; use NotebookLM with papers you have already read to ask comparative questions; or call the Semantic Scholar API and filter by citation count, recency, and topic. A `scientific brainstorming` skill on GitHub was recommended for its three modes: generate, refine, and grill.

## Day 2, August 25

Slides: [`day2/slides.tex`](day2/slides.tex), [`day2/slides.pdf`](day2/slides.pdf).

Students read Gentzkow and Shapiro, *Code and Data for the Social Sciences*, overnight.

### Gentzkow and Shapiro (2014) revisited

Six of their rules, reread for a world where the research assistant is a bot: automate everything that can be automated, store code and data under version control, separate directories by function, abstract only to remove redundancy or improve clarity, do not write documentation you will not maintain, and manage tasks with a task management system.

The reinterpretation is the point. Their rules were written for the Stata era, when writing a function was hard and code was explicit but repetitive, so "abstract to eliminate redundancy" was good advice. Bots come from software engineering and over-abstract by default, wrapping functions in functions, so telling them to abstract makes things worse. The counter-principle offered was YAGNI, "you aren't gonna need it".

The example: a seven-regression task that the bot turned into a 400-line CSV manifest plus one line of code that read the CSV and ran everything off variable tags. Clean for the bot, unreadable for the scientist. The rule of thumb given was to prefer the explicit ten-line version unless a function will be reused five or six times, because verification cost compounds with each layer of nesting.

Why legibility still matters: you are responsible for what goes under your name, and Claude does not get co-authorship. Coding errors are shrinking but not gone, and even a correct program can be a wrong translation of your intent. The Excel example given was `log()`, which defaults to base 10 in Excel and base e in R, and took a week to find.

### Tests

Reinhart and Rogoff (2010) is the cautionary tale: five countries dropped from one average, the result flipped sign when they were included, and the finding was used to argue for austerity. A single test would have caught it.

Four kinds of check, which is the table on the "Four checks" slide:

| Check | When | Example |
|-|-|-|
| Data assertion | Every merge | Rows in equals rows out. `isid id year`. |
| Known answer | Closed form | SVG to lat/lon: round-trip the four corners. |
| Parameter recovery | No known answer | Simulate from a known beta, get the beta back. |
| Coverage | Randomness | A bootstrap 95% interval covers the truth 95% of the time. |

Beyond those: stack unit tests into an integration test that runs before the main code, and use mutation testing, where you deliberately break something to find edge cases. Writing a test that passes is easy; writing one that fails on purpose is not. A plus-one offset is a good first edge case.

Bots game tests. Claude in particular loosens bounds and inflates tolerances until things pass, Codex less so, and having one model review another's tests helps. The suggested rule is to protect the tests folder in your system prompt: the bot must ask before touching any test file, and must show input, output, and the git diff before changing one. Say in the prompt that it may never edit a test to make it pass. Break the code once, because a test that never fails is testing nothing.

Tooling by language: PyTest for Python (already installed in VS Code and Cursor), `testthat` for R, and `assert` plus `isid` in Stata.

When the truth is unknowable, test against a canonical answer. BLP demand estimation has a homework problem worked at every university, so code that reproduces the standard answer can be trusted on new data. At a simpler scale, write your own OLS and test it on simulated data with a known coefficient, or compare it against the canned function on the same data. This has research value too: earlier in 2026 several new difference-in-differences packages returned different results on the same dataset. It also unlocks speed work, since rewriting an RD package in Rust with solid tests in place gave a 50x improvement.

Prompting matters. "Write a test for this" produces a weak test. Telling the bot to simulate markets, introspect, and show you the test before writing it produces a real one. Review every test before it enters the codebase, and aim for one per reusable function.

### Live sessions with no slides

Four Day 2 topics ran as demonstrations and left nothing in this folder.

**Claude artifacts for learning.** Interactive JavaScript visualizations hosted on Claude's domain. One demo turned a full set of first-year micro notes into interactive pages with sliders, built with parallel sub-agents in about 30 minutes. Another replaced an hour of whiteboard derivative work on an NK model with a liquidity trap with a Manim animation and parameter sliders in about two prompts. Artifacts can also generate practice questions with hidden answer keys. The warning: shared artifacts are publicly indexed, so save locally as a JavaScript file for anything restricted.

**Data annotation, the SNAP waiver project.** Structured data pulled from free-form government letters on SNAP work requirement waivers going back to FY1997. Priming the model as an expert policy analyst measurably improved output. The first naive prompt found only four correct counties in the Wisconsin sample. The fix was to write a formal project memo, feed it back, and let the prompt grow into a long schema defining every term of art. Five states were hand-coded as ground truth, and version 5 of the prompt scored 290 of 298 on Wisconsin, 98.7%, with 100% on some states. The takeaways: start with a bad prompt and iterate, hand-code a source of truth early, feed every edge case you find back into the prompt, and only scale once accuracy holds. The county-level map at the end cost 45 cents and four minutes, against what used to be a week of an RA's time.

**Version control.** Git is now mandatory, because bots delete folder structures and without git there is no recovery. Commit every time the code reaches a known-good state, since committing costs nothing. Keep `main` paper-ready and do experimental work on branches. Do not put data files in GitHub. Use Issues to track bugs and TODOs, each spawning its own branch. The advice was to play an interactive git game online until you understand the protocol well enough to instruct a bot.

**Directory structure.** A numbered convention so folders sort predictably: `00_literature`, `01_data` (with `raw/` and `clean/` inside), `02_scripts`, `03_tables`, `04_figures`, `05_writing`, `06_misc`. The semantics help the agent: a `clean/` folder tells it where output data goes. Git colouring in VS Code and Cursor shows at a glance where the bot touched things, green for new and purple for modified.

## Building the slides

Both decks share a preamble and use the `metropolis` beamer theme. Day 1 pulls images from `day1/figures/`; Day 2 uses none.

```bash
cd fall-2026/day1        # or fall-2026/day2
latexmk -pdf slides.tex
```

LaTeX build files (`.aux`, `.log`, `.nav`, and the rest) are gitignored, as is the `data/` output the scraper writes.
