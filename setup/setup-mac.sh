#!/usr/bin/env bash
#
# USC Econ AI Workshop, Fall 2026 - macOS setup
#
# Installs the common floor: Claude Code, Cursor CLI, Codex, uv, Python, R, Julia,
# git, and the standard data-science packages for Python, R, and Julia.
#
# It only ADDS things. If you already have a tool, it says so and moves on.
# It never upgrades, never removes, and never touches your system Python.
#
# Usage:
#   bash setup-mac.sh            install what is missing
#   bash setup-mac.sh --check    report what is missing, install nothing
#   bash setup-mac.sh --yes      never ask, skip anything that needs a question
#
set -uo pipefail

WORKSHOP_DIR="${WORKSHOP_DIR:-$HOME/econ-ai-workshop}"
PY_VERSION="${PY_VERSION:-3.12}"
CHECK_ONLY=0
ASSUME_YES=0

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --help|-h) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

mkdir -p "$WORKSHOP_DIR"
LOG="$WORKSHOP_DIR/setup-mac.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== USC econ AI workshop setup, macOS, $(date) ==="
echo "Log file: $LOG"
echo "Workshop folder: $WORKSHOP_DIR"
[ "$CHECK_ONLY" = 1 ] && echo "MODE: --check, nothing will be installed"

RESULTS=()
have() { command -v "$1" >/dev/null 2>&1; }
record() { RESULTS+=("$1|$2|$3"); }
banner() { printf '\n--- %s ---\n' "$1"; }

# install_if_missing <binary> <friendly name> <function to run>
install_if_missing() {
  local bin="$1" name="$2" fn="$3"
  banner "$name"
  if have "$bin"; then
    echo "Already installed: $(command -v "$bin")"
    record "HAVE" "$name" "$(command -v "$bin")"
    return 0
  fi
  if [ "$CHECK_ONLY" = 1 ]; then
    echo "MISSING, would install"
    record "MISSING" "$name" "would install"
    return 0
  fi
  echo "Installing..."
  if "$fn"; then
    record "INSTALLED" "$name" "ok"
  else
    record "FAILED" "$name" "see $LOG"
    echo "FAILED. Not fatal, moving on."
  fi
}

ask() {  # ask "question" -> 0 for yes
  [ "$ASSUME_YES" = 1 ] && return 1
  [ -t 0 ] || return 1
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy] ]]
}

# ---------------------------------------------------------------- prerequisites

banner "Command line tools (git, curl)"
if have git; then
  echo "git: $(git --version)"
  record "HAVE" "git" "$(command -v git)"
else
  if [ "$CHECK_ONLY" = 1 ]; then
    record "MISSING" "git" "run: xcode-select --install"
  else
    echo "git is missing. Opening Apple's installer window."
    xcode-select --install 2>/dev/null || true
    record "MANUAL" "git" "finish the Xcode tools popup, then rerun"
  fi
fi

banner "Homebrew"
if have brew; then
  echo "Homebrew: $(brew --version | head -1)"
  record "HAVE" "Homebrew" "$(command -v brew)"
elif [ "$CHECK_ONLY" = 1 ]; then
  record "MISSING" "Homebrew" "needed for R and Codex"
elif ask "Homebrew is missing. Install it? (asks for your Mac password)"; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    && record "INSTALLED" "Homebrew" "ok" || record "FAILED" "Homebrew" "see $LOG"
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [ -x "$p" ] && eval "$("$p" shellenv)"
  done
else
  echo "Skipping Homebrew. R and Codex will be skipped too."
  record "SKIPPED" "Homebrew" "declined"
fi

# ------------------------------------------------------------------ agent tools

install_claude() { curl -fsSL https://claude.ai/install.sh | bash; }
install_cursor() { curl -fsS https://cursor.com/install | bash; }
install_codex() {
  if have brew; then brew install --cask codex
  elif have npm; then npm install -g @openai/codex
  else echo "Needs Homebrew or npm."; return 1; fi
}
install_uv() { curl -LsSf https://astral.sh/uv/install.sh | sh; }

install_if_missing claude       "Claude Code" install_claude
install_if_missing cursor-agent "Cursor CLI"  install_cursor
install_if_missing codex        "Codex CLI"   install_codex
install_if_missing uv           "uv"          install_uv

# new installers drop binaries here; make them visible to the rest of this run
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------- runtime

banner "R"
if have Rscript; then
  echo "R: $(Rscript -e 'cat(R.version.string)' 2>/dev/null)"
  record "HAVE" "R" "$(command -v Rscript)"
elif [ "$CHECK_ONLY" = 1 ]; then
  record "MISSING" "R" "would install"
elif have brew; then
  brew install --cask r && record "INSTALLED" "R" "ok" || record "FAILED" "R" "see $LOG"
else
  echo "No Homebrew. Download R from https://cran.r-project.org/bin/macosx/"
  record "MANUAL" "R" "https://cran.r-project.org/bin/macosx/"
fi

banner "Julia (via juliaup)"
if have julia || have juliaup; then
  echo "Julia already installed: $(command -v julia || command -v juliaup)"
  record "HAVE" "Julia" "$(command -v julia || command -v juliaup)"
elif [ "$CHECK_ONLY" = 1 ]; then
  record "MISSING" "Julia" "would install"
else
  curl -fsSL https://install.julialang.org | sh -s -- --yes \
    && record "INSTALLED" "Julia" "ok" || record "FAILED" "Julia" "see $LOG"
fi
export PATH="$HOME/.juliaup/bin:$PATH"

# --------------------------------------------------------------------- packages

REQ="$WORKSHOP_DIR/requirements-workshop.txt"
cat > "$REQ" <<'EOF'
# Workshop Python floor. Edit and rerun `uv pip install` to add your own.
numpy
pandas
polars
pyarrow
scipy
statsmodels
linearmodels
pyfixest
scikit-learn
matplotlib
seaborn
plotnine
jupyterlab
ipykernel
openpyxl
requests
beautifulsoup4
tqdm
pytest
radon
ruff
EOF

banner "Python $PY_VERSION and data-science packages"
if [ "$CHECK_ONLY" = 1 ]; then
  echo "Would create $WORKSHOP_DIR/.venv and install $(grep -cv '^#' "$REQ") packages"
  record "MISSING" "Python packages" "would install into $WORKSHOP_DIR/.venv"
elif have uv; then
  uv python install "$PY_VERSION"
  [ -d "$WORKSHOP_DIR/.venv" ] || uv venv "$WORKSHOP_DIR/.venv" --python "$PY_VERSION"
  if uv pip install --python "$WORKSHOP_DIR/.venv" -r "$REQ"; then
    record "INSTALLED" "Python packages" "$WORKSHOP_DIR/.venv"
  else
    record "FAILED" "Python packages" "see $LOG"
  fi
  "$WORKSHOP_DIR/.venv/bin/python" -m ipykernel install --user \
    --name econ-workshop --display-name "Python (econ workshop)" >/dev/null 2>&1 || true
else
  record "SKIPPED" "Python packages" "uv not available"
fi

banner "R packages"
R_SCRIPT="$WORKSHOP_DIR/install-r-packages.R"
cat > "$R_SCRIPT" <<'EOF'
# Installs only what is missing, into your personal R library.
pkgs <- c("tidyverse", "data.table", "fixest", "modelsummary", "sandwich",
          "lmtest", "haven", "arrow", "here", "renv", "knitr", "rmarkdown")
lib <- Sys.getenv("R_LIBS_USER")
if (lib == "") lib <- file.path(path.expand("~"), "R", "workshop-library")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) == 0) {
  cat("All R packages already present.\n")
} else {
  cat("Installing:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, lib = lib, repos = "https://cloud.r-project.org")
}
EOF
if [ "$CHECK_ONLY" = 1 ]; then
  record "MISSING" "R packages" "would run $R_SCRIPT"
elif have Rscript; then
  Rscript "$R_SCRIPT" && record "INSTALLED" "R packages" "ok" || record "FAILED" "R packages" "see $LOG"
else
  record "SKIPPED" "R packages" "R not available"
fi

banner "Julia packages"
JL_SCRIPT="$WORKSHOP_DIR/install-julia-packages.jl"
cat > "$JL_SCRIPT" <<EOF
# Installs into a workshop-only project, so your global Julia setup is untouched.
using Pkg
proj = raw"$WORKSHOP_DIR/julia"
mkpath(proj); Pkg.activate(proj)
want = ["DataFrames", "CSV", "StatsBase", "GLM", "Distributions", "Optim", "Plots"]
present = keys(Pkg.project().dependencies)
add = filter(p -> !(p in present), want)
isempty(add) ? println("All Julia packages already present.") : Pkg.add(add)
EOF
if [ "$CHECK_ONLY" = 1 ]; then
  record "MISSING" "Julia packages" "would run $JL_SCRIPT"
elif have julia; then
  julia "$JL_SCRIPT" && record "INSTALLED" "Julia packages" "ok" || record "FAILED" "Julia packages" "see $LOG"
else
  record "SKIPPED" "Julia packages" "julia not on PATH yet, rerun after opening a new terminal"
fi

# ------------------------------------------------------------------------- PATH

banner "PATH"
RC="$HOME/.zshrc"; [ -n "${BASH_VERSION:-}" ] && [ "$SHELL" = "/bin/bash" ] && RC="$HOME/.bash_profile"
LINE='export PATH="$HOME/.local/bin:$HOME/.juliaup/bin:$PATH"  # econ workshop'
if [ "$CHECK_ONLY" = 1 ]; then
  grep -qF 'econ workshop' "$RC" 2>/dev/null && echo "PATH line already in $RC" || echo "Would add a PATH line to $RC"
elif grep -qF 'econ workshop' "$RC" 2>/dev/null; then
  echo "PATH line already in $RC, leaving it alone."
else
  printf '\n%s\n' "$LINE" >> "$RC"
  echo "Added a PATH line to $RC"
fi

# ---------------------------------------------------------------------- summary

echo
echo "======================= SUMMARY ======================="
printf '%-10s %-20s %s\n' "STATUS" "TOOL" "DETAIL"
for row in "${RESULTS[@]}"; do
  IFS='|' read -r s n d <<< "$row"
  printf '%-10s %-20s %s\n' "$s" "$n" "$d"
done
echo "======================================================="
echo
echo "Next steps:"
echo "  1. Close this terminal and open a new one."
echo "  2. Check it worked:  claude --version && uv --version && Rscript --version"
echo "  3. Your Python setup: source $WORKSHOP_DIR/.venv/bin/activate"
echo "  4. Log saved at: $LOG"
