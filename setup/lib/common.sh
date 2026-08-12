#!/usr/bin/env bash
# common.sh --- helpers shared by every macOS component.

DRY_RUN="${DRY_RUN:-0}"

have() { command -v "$1" >/dev/null 2>&1; }

# where <cmd> --- resolved path, or empty.
where() { command -v "$1" 2>/dev/null || true; }

# first_version <cmd> [args...] --- runs the command and extracts the first
# dotted version number it prints. Tolerates tools that write to stderr.
#
# Update nags are filtered out first. Several of these CLIs (juliaup, npm, gh,
# claude) print "a new version, 1.2.3, is available" alongside their own
# version, and a naive first-match grep reports the *available* version as the
# installed one --- which makes the diagnostic table quietly wrong.
first_version() {
  local out
  out="$("$@" 2>&1 | head -5)" || true
  printf '%s' "$out" \
    | grep -viE 'available|update|upgrade|newer|new version|latest' \
    | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' \
    | head -1
}

# ver_ge <a> <b> --- true when version a >= version b. Pads missing segments,
# so "3.12" >= "3.10" compares as 3.12.0 >= 3.10.0 rather than lexically.
ver_ge() {
  local a="$1" b="$2" i
  local -a pa pb
  IFS=. read -r -a pa <<<"$a"
  IFS=. read -r -a pb <<<"$b"
  for (( i = 0; i < 3; i++ )); do
    local x="${pa[$i]:-0}" y="${pb[$i]:-0}"
    x="${x//[^0-9]/}"; y="${y//[^0-9]/}"
    x="${x:-0}"; y="${y:-0}"
    (( 10#$x > 10#$y )) && return 0
    (( 10#$x < 10#$y )) && return 1
  done
  return 0
}

# run <command...> --- the single choke point for anything that mutates the
# machine. Honours --dry-run and tees output into the log.
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    say "      ${C_DIM}would run:${C_RESET} $*"
    return 0
  fi
  _log_raw "+ $*"
  if [[ -n "$LOG_FILE" ]]; then
    "$@" >>"$LOG_FILE" 2>&1
  else
    "$@" >/dev/null 2>&1
  fi
}

# run_shell <string> --- for pipelines that genuinely need a shell (the vendor
# install one-liners). Same dry-run and logging contract as run().
run_shell() {
  if [[ "$DRY_RUN" == "1" ]]; then
    say "      ${C_DIM}would run:${C_RESET} $1"
    return 0
  fi
  _log_raw "+ $1"
  if [[ -n "$LOG_FILE" ]]; then
    bash -c "$1" >>"$LOG_FILE" 2>&1
  else
    bash -c "$1" >/dev/null 2>&1
  fi
}

# detect_simple <id> <command> [version-args...]
#
# The common "is this CLI here and working?" check. A command that exists but
# reports no version is treated as broken rather than fine: shim wrappers left
# behind by other tools sit on PATH, exit 0, and print an error instead of a
# version. Reporting those as installed would send someone to the workshop
# believing they have a tool they do not have.
detect_simple() {
  local id="$1" cmd="$2"; shift 2
  local args=( "$@" ); [[ ${#args[@]} -eq 0 ]] && args=( --version )

  if ! have "$cmd"; then
    set_status "$id" missing
    return 0
  fi

  local p v; p="$(where "$cmd")"; v="$(first_version "$cmd" "${args[@]}")"
  set_cpath "$id" "$p"

  if [[ -z "$v" ]]; then
    set_status "$id" conflict
    set_note   "$id" "found at ${p} but it did not report a version — likely a broken wrapper, not a real install"
    return 0
  fi

  set_version "$id" "$v"
  set_status  "$id" ok
}

# --- Homebrew ---------------------------------------------------------------
# Apple Silicon and Intel use different prefixes, and a freshly installed brew
# is not yet on PATH for the current process.
brew_prefix() {
  if [[ -x /opt/homebrew/bin/brew ]]; then printf '%s' /opt/homebrew
  elif [[ -x /usr/local/bin/brew ]];  then printf '%s' /usr/local
  fi
}

brew_bin() {
  local p; p="$(brew_prefix)"
  [[ -n "$p" ]] && printf '%s' "$p/bin/brew"
}

# Make brew usable in this process immediately after installing it.
activate_brew() {
  local b; b="$(brew_bin)"
  [[ -n "$b" ]] || return 1
  eval "$("$b" shellenv)" 2>/dev/null || true
  return 0
}

brew_formula_installed() {
  local b; b="$(brew_bin)"; [[ -n "$b" ]] || return 1
  "$b" list --formula --versions "$1" >/dev/null 2>&1
}

brew_cask_installed() {
  local b; b="$(brew_bin)"; [[ -n "$b" ]] || return 1
  "$b" list --cask --versions "$1" >/dev/null 2>&1
}

brew_install()      { local b; b="$(brew_bin)"; run "$b" install "$@"; }
brew_cask_install() { local b; b="$(brew_bin)"; run "$b" install --cask "$@"; }

# --- Files ------------------------------------------------------------------
# Timestamped backup. Echoes the backup path so callers can tell the user.
backup_file() {
  local f="$1" stamp bak
  [[ -e "$f" ]] || return 0
  stamp="$(date +%Y%m%d-%H%M%S)"
  bak="${f}.bak.${stamp}"
  if [[ "$DRY_RUN" == "1" ]]; then
    say "      ${C_DIM}would back up:${C_RESET} $f -> $bak"
  else
    cp -p "$f" "$bak" 2>/dev/null || return 1
  fi
  printf '%s' "$bak"
}

# ensure_rc_block <file> <marker> <content>
# Idempotently maintains a delimited block in a shell rc file. Re-running the
# installer replaces the block rather than stacking duplicates, which is the
# usual way these scripts corrupt people's shell configs.
ensure_rc_block() {
  local file="$1" marker="$2" content="$3"
  local begin="# >>> ${marker} >>>"
  local end="# <<< ${marker} <<<"

  if [[ "$DRY_RUN" == "1" ]]; then
    say "      ${C_DIM}would update block '${marker}' in${C_RESET} $file"
    return 0
  fi

  mkdir -p "$(dirname "$file")"
  touch "$file"

  if grep -qF "$begin" "$file" 2>/dev/null; then
    # Drop the existing block, keeping everything around it intact.
    local tmp; tmp="$(mktemp)"
    awk -v b="$begin" -v e="$end" '
      index($0, b) { skip = 1 }
      !skip        { print }
      index($0, e) { skip = 0 }
    ' "$file" >"$tmp" && mv "$tmp" "$file"
  fi

  {
    printf '\n%s\n' "$begin"
    printf '%s\n' "$content"
    printf '%s\n' "$end"
  } >>"$file"
}

# The rc file for the user's login shell.
user_shell_rc() {
  case "${SHELL:-}" in
    */zsh)  printf '%s' "$HOME/.zshrc" ;;
    */bash) printf '%s' "$HOME/.bash_profile" ;;
    *)      printf '%s' "$HOME/.zshrc" ;;   # macOS default since Catalina
  esac
}

# Ensure a directory is on PATH for the remainder of this process, so a tool we
# just installed can be used by a later component without a shell restart.
prepend_path() {
  case ":$PATH:" in
    *":$1:"*) ;;
    *) PATH="$1:$PATH"; export PATH ;;
  esac
}
