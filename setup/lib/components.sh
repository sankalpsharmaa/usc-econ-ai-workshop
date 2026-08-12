#!/usr/bin/env bash
# components.sh --- the macOS component catalogue.
#
# Every component implements:
#   c_<id>_detect   sets STATUS/VERSION/CPATH/NOTE. Must never mutate anything.
#   c_<id>_plan     prints the exact commands and install location.
#   c_<id>_install  performs the install.
#   c_<id>_extras   optional add-on (renv, a managed Python, ...).
#   c_<id>_verify   0/1, used by both the post-install report and doctor mode.
#
# Ordering below is also install order, so prerequisites come first.

PY_MIN="3.10"        # floor stated in the workshop materials
PY_TARGET="3.12"     # version we install via uv

register clt        "Xcode CLI Tools" "Compilers and headers Homebrew builds against" ""      ""
register brew       "Homebrew"        "Package manager for macOS"                     "clt"   ""
register git        "git"             "Version control"                               "brew"  ""
register gh         "GitHub CLI"      "Authenticate and work with GitHub from the terminal" "brew" ""
register make       "GNU Make"        "Build automation for reproducible pipelines"   "brew"  ""
register node       "Node.js"         "JavaScript runtime some agent CLIs depend on"  "brew"  ""
register uv         "uv"              "Fast Python package and version manager"       ""      "Python ${PY_TARGET}"
register r          "R"               "Statistical computing environment"             "brew"  "renv"
register julia      "Julia"           "Numerical computing language"                  "brew"  ""
register quarto     "Quarto"          "Publishing system for notebooks and papers"    "brew"  ""
register cursor     "Cursor"          "AI-native editor (free for students)"          "brew"  "editor extensions"
register cursor_cli "Cursor CLI"      "Cursor's terminal agent (cursor-agent)"        ""      ""
register claude     "Claude Code"     "Anthropic's terminal coding agent"             ""      ""
register codex      "Codex CLI"       "OpenAI's terminal coding agent"                "node"  ""
register cmux       "cmux"            "Terminal built for running AI agents in parallel" "brew" ""
register omp        "oh-my-posh"      "Informative shell prompt (git branch, env, timing)" "" "prompt theme"

# ---------------------------------------------------------------------------
# Xcode Command Line Tools
# ---------------------------------------------------------------------------
c_clt_detect() {
  if xcode-select -p >/dev/null 2>&1; then
    set_status clt ok
    set_cpath  clt "$(xcode-select -p 2>/dev/null)"
    set_version clt "$(first_version pkgutil --pkg-info=com.apple.pkg.CLTools_Executables)"
  else
    set_status clt missing
  fi
}
c_clt_plan() {
  say "      xcode-select --install"
  say "      ${C_DIM}Opens Apple's installer dialog. ~1-2 GB, a few minutes.${C_RESET}"
}
c_clt_install() {
  # This hands off to a GUI dialog; we cannot drive it, so we wait and explain.
  run xcode-select --install || true
  warn "Apple's installer window handles this one."
  if [[ "$DRY_RUN" != "1" ]]; then
    local waited=0
    while ! xcode-select -p >/dev/null 2>&1; do
      (( waited >= 900 )) && { err "Timed out waiting for Xcode CLI Tools."; return 1; }
      sleep 10; waited=$(( waited + 10 ))
    done
  fi
}
c_clt_verify() { xcode-select -p >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Homebrew
# ---------------------------------------------------------------------------
c_brew_detect() {
  local b; b="$(brew_bin)"
  if [[ -n "$b" ]]; then
    set_status brew ok
    set_cpath  brew "$b"
    set_version brew "$(first_version "$b" --version)"
  else
    set_status brew missing
  fi
}
c_brew_plan() {
  say "      /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
  say "      ${C_DIM}Installs to /opt/homebrew (Apple Silicon) or /usr/local (Intel).${C_RESET}"
  say "      ${C_DIM}Will ask for your password once.${C_RESET}"
}
c_brew_install() {
  run_shell '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/tty' || return 1
  activate_brew || return 1
  # Persist brew's environment for future shells.
  local prefix; prefix="$(brew_prefix)"
  ensure_rc_block "$(user_shell_rc)" "econ-ai-setup homebrew" \
    "eval \"\$(${prefix}/bin/brew shellenv)\""
}
c_brew_verify() { [[ -n "$(brew_bin)" ]]; }

# ---------------------------------------------------------------------------
# git --- macOS ships a git via the CLI Tools; brew's is newer. Both coexist.
# ---------------------------------------------------------------------------
c_git_detect() {
  if have git; then
    local v p; v="$(first_version git --version)"; p="$(where git)"
    set_version git "$v"; set_cpath git "$p"
    if [[ "$p" == /usr/bin/git ]]; then
      set_status git conflict
      set_note   git "Apple's git; Homebrew's is newer and updates independently"
    else
      set_status git ok
    fi
  else
    set_status git missing
  fi
}
c_git_plan() { say "      brew install git"; say "      ${C_DIM}Leaves Apple's /usr/bin/git untouched.${C_RESET}"; }
c_git_install() { brew_install git; }
c_git_verify() { have git; }

# ---------------------------------------------------------------------------
# GitHub CLI
# ---------------------------------------------------------------------------
c_gh_detect() { detect_simple gh gh --version; }
c_gh_plan() { say "      brew install gh"; say "      ${C_DIM}Log in afterwards with: gh auth login${C_RESET}"; }
c_gh_install() { brew_install gh; }
c_gh_verify() { have gh; }

# ---------------------------------------------------------------------------
# GNU Make --- macOS ships 3.81 (2006). Brew's installs as `gmake`.
# ---------------------------------------------------------------------------
c_make_detect() {
  if have gmake; then
    set_status make ok; set_version make "$(first_version gmake --version)"; set_cpath make "$(where gmake)"
  elif have make; then
    local v; v="$(first_version make --version)"
    set_version make "$v"; set_cpath make "$(where make)"
    if ver_ge "$v" "4.0"; then set_status make ok
    else
      set_status make outdated
      set_note   make "Apple ships GNU Make 3.81; 4.x adds features Makefiles often assume"
    fi
  else
    set_status make missing
  fi
}
c_make_plan() {
  say "      brew install make"
  say "      ${C_DIM}Installs as 'gmake' so Apple's /usr/bin/make is untouched.${C_RESET}"
}
c_make_install() { brew_install make; }
c_make_verify() { have gmake || have make; }

# ---------------------------------------------------------------------------
# Node --- needed for the Codex npm package and npx-based tools.
# ---------------------------------------------------------------------------
c_node_detect() {
  if have node; then
    local v; v="$(first_version node --version)"
    set_version node "$v"; set_cpath node "$(where node)"
    # Codex requires Node 22+. Checking against a lower floor would report a
    # machine as ready and then fail at the Codex install instead.
    if ver_ge "$v" "22.0.0"; then set_status node ok
    else
      set_status node outdated
      set_note   node "Codex needs Node 22 or newer; upgrading is safe and does not remove ${v}"
    fi
  else
    set_status node missing
  fi
}
c_node_plan() { say "      brew install node"; }
c_node_install() { brew_install node; }
c_node_verify() { have node; }

# ---------------------------------------------------------------------------
# uv + a managed Python
#
# The system Python at /usr/bin/python3 is macOS-managed and must never be
# touched. uv installs its own interpreters under ~/.local/share/uv, which is
# why "install alongside" is always safe here.
# ---------------------------------------------------------------------------
c_uv_detect() {
  if have uv; then
    set_status uv ok; set_version uv "$(first_version uv --version)"; set_cpath uv "$(where uv)"
    return
  fi
  # No uv. An existing python3 is worth reporting, but it is NOT a conflict:
  # uv installs its own interpreters under ~/.local/share/uv and leaves every
  # other Python alone. Calling it a conflict would make --all skip uv on any
  # machine that has a python3, which is all of them.
  set_status uv missing
  if have python3; then
    local v p; v="$(first_version python3 --version)"; p="$(where python3)"
    set_version uv "$v"; set_cpath uv "$p"
    if [[ "$p" == /usr/bin/python3 ]]; then
      set_note uv "your python3 ${v} is macOS system Python; uv installs separately and leaves it alone"
    elif ! ver_ge "$v" "$PY_MIN"; then
      set_note uv "your python3 ${v} is older than the ${PY_MIN} the workshop needs; uv will install ${PY_TARGET} alongside it"
    else
      set_note uv "your python3 ${v} at ${p} stays as your default"
    fi
  fi
}
c_uv_plan() {
  say "      curl -LsSf https://astral.sh/uv/install.sh | sh"
  say "      uv python install ${PY_TARGET}"
  say "      ${C_DIM}Installs to ~/.local/bin and ~/.local/share/uv.${C_RESET}"
  say "      ${C_DIM}Does not change your system or existing Python.${C_RESET}"
}
c_uv_install() {
  run_shell 'curl -LsSf https://astral.sh/uv/install.sh | sh' || return 1
  prepend_path "$HOME/.local/bin"
}
c_uv_extras() {
  have uv || prepend_path "$HOME/.local/bin"
  run uv python install "$PY_TARGET"
}
c_uv_verify() { have uv || [[ -x "$HOME/.local/bin/uv" ]]; }

# ---------------------------------------------------------------------------
# R --- installed through r-rig, the R Installation Manager.
#
# Careful: the Homebrew formula named `rig` is an unrelated fake-data generator.
# The R version manager is `r-rig`, and the two conflict over the same binary.
# ---------------------------------------------------------------------------
c_r_detect() {
  if have R; then
    set_status r ok; set_version r "$(first_version R --version)"; set_cpath r "$(where R)"
  elif [[ -x /Library/Frameworks/R.framework/Resources/bin/R ]]; then
    set_status r ok
    set_cpath r /Library/Frameworks/R.framework/Resources/bin/R
    set_version r "$(first_version /Library/Frameworks/R.framework/Resources/bin/R --version)"
    set_note r "installed but not on PATH"
  else
    set_status r missing
  fi
}
c_r_plan() {
  say "      brew install r-rig      ${C_DIM}(the R Installation Manager)${C_RESET}"
  say "      rig add release         ${C_DIM}(installs current R)${C_RESET}"
  say "      ${C_DIM}rig lets you hold several R versions side by side, which${C_RESET}"
  say "      ${C_DIM}matters for reproducing an old project.${C_RESET}"
}
c_r_install() {
  brew_install r-rig || return 1
  run rig add release
}
c_r_extras() {
  # renv into the user library, so no sudo and no system-library pollution.
  run_shell 'Rscript -e '"'"'if (!requireNamespace("renv", quietly=TRUE)) install.packages("renv", repos="https://cloud.r-project.org")'"'"''
}
c_r_verify() { have R || have Rscript; }

# ---------------------------------------------------------------------------
# Julia --- via juliaup, the official version multiplexer.
# ---------------------------------------------------------------------------
c_julia_detect() { detect_simple julia julia --version; }
c_julia_plan() {
  say "      brew install juliaup"
  say "      juliaup add release"
  say "      ${C_DIM}juliaup manages Julia versions; ~/.julia holds packages.${C_RESET}"
}
c_julia_install() {
  brew_install juliaup || return 1
  run juliaup add release
}
c_julia_verify() { have julia || have juliaup; }

# ---------------------------------------------------------------------------
# Quarto
# ---------------------------------------------------------------------------
c_quarto_detect() { detect_simple quarto quarto --version; }
c_quarto_plan() { say "      brew install --cask quarto"; }
c_quarto_install() { brew_cask_install quarto; }
c_quarto_verify() { have quarto; }

# ---------------------------------------------------------------------------
# Cursor --- the editor, plus the language extensions that make R and Python
# usable on first launch.
# ---------------------------------------------------------------------------
CURSOR_APP="/Applications/Cursor.app"
CURSOR_SETTINGS="$HOME/Library/Application Support/Cursor/User/settings.json"

c_cursor_detect() {
  if [[ -d "$CURSOR_APP" ]]; then
    set_status cursor ok; set_cpath cursor "$CURSOR_APP"
    local plist="$CURSOR_APP/Contents/Info.plist"
    [[ -f "$plist" ]] && set_version cursor \
      "$(defaults read "$plist" CFBundleShortVersionString 2>/dev/null || true)"
    [[ -s "$CURSOR_SETTINGS" ]] && set_note cursor "existing settings.json will be merged, not replaced"
  else
    set_status cursor missing
  fi
}
c_cursor_plan() {
  say "      brew install --cask cursor"
  say "      ${C_DIM}Extensions (asked separately): Python, Jupyter, R, Julia, Quarto, Ruff${C_RESET}"
}
c_cursor_install() { brew_cask_install cursor; }

c_cursor_extras() {
  local cli="$CURSOR_APP/Contents/Resources/app/bin/cursor"
  if [[ ! -x "$cli" ]]; then
    if have cursor; then cli="$(where cursor)"
    else warn "Cursor's command line helper not found; skipping extensions."; return 1; fi
  fi

  local ext
  while IFS= read -r ext; do
    [[ -z "$ext" || "$ext" == \#* ]] && continue
    say "      installing extension: ${ext}"
    run "$cli" --install-extension "$ext" --force
  done <"$ASSET_DIR/cursor-extensions.txt"

  merge_cursor_settings
}

# Merges our keys into an existing settings.json without clobbering the user's
# choices. Any key they have already set wins; we only add what is absent.
merge_cursor_settings() {
  local ours="$ASSET_DIR/cursor-settings.json"
  local dir; dir="$(dirname "$CURSOR_SETTINGS")"

  if [[ ! -s "$CURSOR_SETTINGS" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      say "      ${C_DIM}would write:${C_RESET} $CURSOR_SETTINGS"
    else
      mkdir -p "$dir"; cp "$ours" "$CURSOR_SETTINGS"
    fi
    ok "Wrote Cursor settings."
    return 0
  fi

  local bak; bak="$(backup_file "$CURSOR_SETTINGS")"
  [[ -n "$bak" ]] && info "Backed up your settings to ${bak}"

  if [[ "$DRY_RUN" == "1" ]]; then
    say "      ${C_DIM}would merge our keys into${C_RESET} $CURSOR_SETTINGS"
    return 0
  fi

  # python3 is guaranteed present on macOS, so this needs no extra dependency.
  # Comments are legal in VS Code-family JSON, hence the tolerant parse.
  /usr/bin/python3 - "$CURSOR_SETTINGS" "$ours" <<'PY' || { warn "Could not merge settings; your file is unchanged."; return 1; }
import json, re, sys

def load(path):
    with open(path) as fh:
        raw = fh.read()
    raw = re.sub(r'^\s*//.*$', '', raw, flags=re.M)       # line comments
    raw = re.sub(r',(\s*[}\]])', r'\1', raw)              # trailing commas
    return json.loads(raw) if raw.strip() else {}

target, source = sys.argv[1], sys.argv[2]
try:
    current = load(target)
except Exception as exc:
    sys.exit(f"unparseable settings.json: {exc}")

added = {k: v for k, v in load(source).items() if k not in current}
current.update(added)

with open(target, "w") as fh:
    json.dump(current, fh, indent=2)
    fh.write("\n")

for key in added:
    print(f"      added {key}")
PY
  ok "Merged Cursor settings (your existing keys were kept)."
}
c_cursor_verify() { [[ -d "$CURSOR_APP" ]]; }

# ---------------------------------------------------------------------------
# Cursor CLI (cursor-agent)
# ---------------------------------------------------------------------------
c_cursor_cli_detect() { detect_simple cursor_cli cursor-agent --version; }
c_cursor_cli_plan() {
  say "      curl https://cursor.com/install -fsSL | bash"
  say "      ${C_DIM}Log in afterwards with: cursor-agent login${C_RESET}"
}
c_cursor_cli_install() {
  run_shell 'curl https://cursor.com/install -fsSL | bash' || return 1
  prepend_path "$HOME/.local/bin"
}
c_cursor_cli_verify() { have cursor-agent || [[ -x "$HOME/.local/bin/cursor-agent" ]]; }

# ---------------------------------------------------------------------------
# Claude Code --- the official installer, matching the workshop slides.
# ---------------------------------------------------------------------------
c_claude_detect() { detect_simple claude claude --version; }
c_claude_plan() {
  say "      curl -fsSL https://claude.ai/install.sh | bash"
  say "      ${C_DIM}Sign in on first run. Note the free tier cannot use browser${C_RESET}"
  say "      ${C_DIM}sign-in; you need Pro/Max/Team or an ANTHROPIC_API_KEY.${C_RESET}"
}
c_claude_install() {
  run_shell 'curl -fsSL https://claude.ai/install.sh | bash' || return 1
  prepend_path "$HOME/.local/bin"
}
c_claude_verify() { have claude || [[ -x "$HOME/.local/bin/claude" ]]; }

# ---------------------------------------------------------------------------
# Codex CLI
#
# The package is scoped: `npm i -g codex` installs an unrelated package.
# ---------------------------------------------------------------------------
c_codex_detect() { detect_simple codex codex --version; }
c_codex_plan() {
  say "      npm install -g @openai/codex"
  say "      ${C_DIM}Scoped package name matters: 'codex' alone is something else.${C_RESET}"
}
c_codex_install() { run npm install -g @openai/codex; }
c_codex_verify() { have codex; }

# ---------------------------------------------------------------------------
# cmux --- macOS only (Ghostty-based), requires macOS 14+.
# ---------------------------------------------------------------------------
c_cmux_detect() {
  if [[ -d /Applications/cmux.app ]]; then
    set_status cmux ok; set_cpath cmux /Applications/cmux.app
  else
    local major; major="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
    if [[ -n "$major" ]] && (( major < 14 )); then
      set_status cmux conflict
      set_note   cmux "needs macOS 14+; you are on $(sw_vers -productVersion)"
    else
      set_status cmux missing
    fi
  fi
}
c_cmux_plan() {
  say "      brew install --cask cmux"
  say "      ${C_DIM}Terminal with one tab per agent, each showing git branch and${C_RESET}"
  say "      ${C_DIM}status, plus a notification when an agent is waiting on you.${C_RESET}"
}
c_cmux_install() { brew_cask_install cmux; }
c_cmux_verify() { [[ -d /Applications/cmux.app ]]; }

# ---------------------------------------------------------------------------
# oh-my-posh
#
# The Homebrew tap is no longer trusted by default (brew refuses it without an
# explicit `brew trust`), so we use the vendor's install script instead.
# ---------------------------------------------------------------------------
OMP_DIR="$HOME/.local/bin"
OMP_THEME_DIR="$HOME/.config/econ-ai"

c_omp_detect() {
  if have oh-my-posh; then
    set_status omp ok; set_version omp "$(first_version oh-my-posh --version)"; set_cpath omp "$(where oh-my-posh)"
  elif [[ -x "$OMP_DIR/oh-my-posh" ]]; then
    set_status omp ok; set_cpath omp "$OMP_DIR/oh-my-posh"
  else
    set_status omp missing
  fi
}
c_omp_plan() {
  say "      curl -s https://ohmyposh.dev/install.sh | bash -s -- -d ${OMP_DIR}"
  say "      ${C_DIM}The prompt theme is asked about separately, since it edits${C_RESET}"
  say "      ${C_DIM}$(user_shell_rc).${C_RESET}"
}
c_omp_install() {
  run_shell "curl -s https://ohmyposh.dev/install.sh | bash -s -- -d '${OMP_DIR}'" || return 1
  prepend_path "$OMP_DIR"
}
c_omp_extras() {
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$OMP_THEME_DIR"
    cp "$ASSET_DIR/econ-ai.omp.json" "$OMP_THEME_DIR/econ-ai.omp.json"
  fi
  ensure_rc_block "$(user_shell_rc)" "econ-ai-setup prompt" \
"export PATH=\"${OMP_DIR}:\$PATH\"
if command -v oh-my-posh >/dev/null 2>&1; then
  eval \"\$(oh-my-posh init \${SHELL##*/} --config '${OMP_THEME_DIR}/econ-ai.omp.json')\"
fi"
  info "Prompt configured in $(user_shell_rc). Restart your terminal to see it."
}
c_omp_verify() { have oh-my-posh || [[ -x "$OMP_DIR/oh-my-posh" ]]; }
