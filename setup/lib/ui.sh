#!/usr/bin/env bash
# ui.sh --- terminal I/O primitives for the workshop installer.
#
# The single most important thing in this file is that every prompt reads from
# /dev/tty rather than stdin. When the installer is delivered as
# `curl ... | bash`, the script's stdin *is* the pipe carrying the script, so a
# bare `read` consumes script bytes instead of keystrokes. Reading the terminal
# device directly sidesteps that entirely.

# --- Colour -----------------------------------------------------------------
# Honour NO_COLOR (https://no-color.org) and degrade cleanly when not a tty.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m';  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''
  C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_CYAN=''
fi

# --- Logging ----------------------------------------------------------------
# LOG_FILE is set by the orchestrator. Everything printed to the user is also
# appended there, minus escape codes, so a participant can mail us the log.
LOG_FILE="${LOG_FILE:-}"

_log_raw() {
  [[ -n "$LOG_FILE" ]] || return 0
  # Strip ANSI escapes so the log file stays greppable.
  printf '%s\n' "$*" | sed $'s/\033\\[[0-9;]*m//g' >>"$LOG_FILE"
}

say()   { printf '%s\n' "$*"; _log_raw "$*"; }
info()  { say "${C_BLUE}  i${C_RESET}  $*"; }
ok()    { say "${C_GREEN}  ✓${C_RESET}  $*"; }
warn()  { say "${C_YELLOW}  !${C_RESET}  $*"; }
err()   { say "${C_RED}  ✗${C_RESET}  $*" >&2; }
step()  { say ""; say "${C_BOLD}$*${C_RESET}"; }

# Prints a rule the width of the terminal (capped, so it stays readable on
# very wide windows).
rule() {
  local width=${COLUMNS:-0} i out=''
  [[ $width -gt 0 ]] || width=$(tput cols 2>/dev/null || echo 80)
  (( width > 78 )) && width=78
  for (( i = 0; i < width; i++ )); do out+='─'; done
  say "${C_DIM}${out}${C_RESET}"
}

banner() {
  say ""
  say "${C_BOLD}${C_CYAN}  USC Economics · AI Research Environment Setup${C_RESET}"
  say "${C_DIM}  Installs the toolchain for the agentic-AI workshop sessions.${C_RESET}"
  say ""
}

# --- Interactivity ----------------------------------------------------------
# HAVE_TTY is probed once at startup. Anything that prompts must check it.
HAVE_TTY=0
if [[ -r /dev/tty && -w /dev/tty ]] && : 2>/dev/null >/dev/tty; then
  HAVE_TTY=1
fi

# ASSUME_YES short-circuits every prompt to its default. Set by --yes/--all.
ASSUME_YES="${ASSUME_YES:-0}"

# Explain the piping problem and bail. Called when we need input and cannot
# reach a terminal --- far better than hanging forever on a blocked read.
die_no_tty() {
  err "This installer is interactive, but it cannot reach your terminal."
  say ""
  say "  That usually means it was started with a pipe, like:"
  say "      ${C_DIM}curl -fsSL <url> | bash${C_RESET}"
  say ""
  say "  When a script is piped, its input is the download rather than your"
  say "  keyboard, so it cannot ask you anything. Use this form instead:"
  say ""
  say "      ${C_BOLD}/bin/bash -c \"\$(curl -fsSL <url>)\"${C_RESET}"
  say ""
  say "  Or run it without prompts:  ${C_BOLD}<url> --all${C_RESET}"
  exit 2
}

# ask_yn <question> <default:y|n>
# Returns 0 for yes, 1 for no.
ask_yn() {
  local question="$1" default="${2:-n}" hint reply
  [[ "$default" == "y" ]] && hint="[Y/n]" || hint="[y/N]"

  if [[ "$ASSUME_YES" == "1" ]]; then
    [[ "$default" == "y" ]] && return 0 || return 1
  fi
  [[ "$HAVE_TTY" == "1" ]] || die_no_tty

  while true; do
    printf '%s' "  ${question} ${C_DIM}${hint}${C_RESET} " >/dev/tty
    IFS= read -r reply </dev/tty || reply=''
    reply="$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    [[ -z "$reply" ]] && reply="$default"
    _log_raw "  ${question} ${hint} -> ${reply}"
    case "$reply" in
      y|yes) return 0 ;;
      n|no)  return 1 ;;
      *) printf '%s\n' "  ${C_YELLOW}Please answer y or n.${C_RESET}" >/dev/tty ;;
    esac
  done
}

# ask_choice <prompt> <default> <key:label> ...
# Echoes the chosen key on stdout. Keys are matched case-insensitively.
ask_choice() {
  local prompt="$1" default="$2"; shift 2
  local -a keys=() labels=()
  local pair key label reply

  for pair in "$@"; do
    keys+=( "${pair%%:*}" )
    labels+=( "${pair#*:}" )
  done

  say ""
  local i
  for (( i = 0; i < ${#keys[@]}; i++ )); do
    key="${keys[$i]}"; label="${labels[$i]}"
    if [[ "$key" == "$default" ]]; then
      say "    ${C_BOLD}[${key}]${C_RESET} ${label} ${C_GREEN}(recommended)${C_RESET}"
    else
      say "    ${C_BOLD}[${key}]${C_RESET} ${label}"
    fi
  done
  say ""

  if [[ "$ASSUME_YES" == "1" ]]; then
    printf '%s' "$default"; return 0
  fi
  [[ "$HAVE_TTY" == "1" ]] || die_no_tty

  while true; do
    printf '%s' "  ${prompt} ${C_DIM}[${default}]${C_RESET} " >/dev/tty
    IFS= read -r reply </dev/tty || reply=''
    reply="$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    [[ -z "$reply" ]] && reply="$default"
    _log_raw "  ${prompt} -> ${reply}"
    for key in "${keys[@]}"; do
      if [[ "$reply" == "$(printf '%s' "$key" | tr '[:upper:]' '[:lower:]')" ]]; then
        printf '%s' "$key"; return 0
      fi
    done
    printf '%s\n' "  ${C_YELLOW}Not one of the options.${C_RESET}" >/dev/tty
  done
}

# pause_for_enter <message> --- used between long phases so people can read.
pause_for_enter() {
  [[ "$ASSUME_YES" == "1" ]] && return 0
  [[ "$HAVE_TTY" == "1" ]] || return 0
  printf '%s' "  ${C_DIM}${1:-Press Enter to continue}${C_RESET} " >/dev/tty
  IFS= read -r _ </dev/tty || true
  printf '\n' >/dev/tty
}

# --- Status table -----------------------------------------------------------
# Column widths are fixed so the scan table and the final report line up
# visually, which makes the before/after diff easy to read.
table_header() {
  say ""
  printf_row "${C_BOLD}COMPONENT${C_RESET}" "${C_BOLD}STATUS${C_RESET}" "${C_BOLD}FOUND${C_RESET}"
  rule
}

# printf_row <name> <status> <detail>
# Padding is computed on the visible text, so colour codes are added after.
printf_row() {
  local name="$1" status="$2" detail="$3"
  local plain_name plain_status
  plain_name="$(_strip "$name")"
  plain_status="$(_strip "$status")"
  local pad_n=$(( 18 - ${#plain_name} ))   ; (( pad_n < 1 )) && pad_n=1
  local pad_s=$(( 12 - ${#plain_status} )) ; (( pad_s < 1 )) && pad_s=1
  say "  ${name}$(_spaces $pad_n)${status}$(_spaces $pad_s)${detail}"
}

_strip()  { printf '%s' "$1" | sed $'s/\033\\[[0-9;]*m//g'; }
_spaces() { printf '%*s' "$1" ''; }

# Renders a component status as a coloured word.
status_word() {
  case "$1" in
    ok)       printf '%s' "${C_GREEN}ok${C_RESET}" ;;
    missing)  printf '%s' "${C_DIM}missing${C_RESET}" ;;
    conflict) printf '%s' "${C_YELLOW}conflict${C_RESET}" ;;
    outdated) printf '%s' "${C_YELLOW}outdated${C_RESET}" ;;
    failed)   printf '%s' "${C_RED}failed${C_RESET}" ;;
    skipped)  printf '%s' "${C_DIM}skipped${C_RESET}" ;;
    *)        printf '%s' "$1" ;;
  esac
}
