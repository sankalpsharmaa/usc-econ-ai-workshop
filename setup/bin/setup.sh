#!/usr/bin/env bash
# setup.sh --- macOS orchestrator for the workshop environment installer.
#
# Runs six phases: preflight, scan, selection, conflict confirmation, plan
# review, execute, report. Nothing mutates the machine before Phase 5, and
# Phase 4 is the last chance to back out.

set -uo pipefail   # deliberately not -e: one failed component must not abort
                   # the other fifteen. Failures are collected and reported.

SETUP_ROOT="${SETUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LIB_DIR="$SETUP_ROOT/lib"
ASSET_DIR="$SETUP_ROOT/assets"

# --- Arguments --------------------------------------------------------------
DRY_RUN=0; ASSUME_YES=0; CHECK_ONLY=0; ONLY_IDS=""; SKIP_IDS=""
LOG_FILE=""; WANT_ALL=0

usage() {
  cat <<'EOF'
USC Economics AI workshop --- environment installer

  --all            Install everything missing, without per-component prompts
  --only a,b,c     Only consider these components
  --skip x,y       Never consider these components
  --check          Report what is installed and exit. Changes nothing.
  --dry-run        Walk the whole flow, print every command, change nothing
  --yes            Accept the default answer at every prompt
  --log <path>     Write the transcript here
  -h, --help       This message

Component ids:
  clt brew git gh make node uv r julia quarto cursor cursor_cli
  claude codex cmux omp
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)     WANT_ALL=1; shift ;;
    --only)    ONLY_IDS="${2:-}"; shift 2 ;;
    --skip)    SKIP_IDS="${2:-}"; shift 2 ;;
    --check)   CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes)     ASSUME_YES=1; shift ;;
    --log)     LOG_FILE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

# --- Log --------------------------------------------------------------------
if [[ -z "$LOG_FILE" && "$CHECK_ONLY" != "1" ]]; then
  mkdir -p "$HOME/.econ-ai-setup"
  LOG_FILE="$HOME/.econ-ai-setup/install-$(date +%Y%m%d-%H%M%S).log"
fi
[[ -n "$LOG_FILE" ]] && : >"$LOG_FILE"

export DRY_RUN ASSUME_YES ONLY_IDS SKIP_IDS LOG_FILE ASSET_DIR

# shellcheck source=../lib/ui.sh
. "$LIB_DIR/ui.sh"
# shellcheck source=../lib/registry.sh
. "$LIB_DIR/registry.sh"
# shellcheck source=../lib/common.sh
. "$LIB_DIR/common.sh"
# shellcheck source=../lib/components.sh
. "$LIB_DIR/components.sh"

FAILED_IDS=""; INSTALLED_IDS=""

# ===========================================================================
# Phase 0 --- Preflight
# ===========================================================================
phase_preflight() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This script is for macOS. On Windows use install.ps1 in PowerShell."
    exit 1
  fi

  local osver major
  osver="$(sw_vers -productVersion 2>/dev/null || echo '0')"
  major="${osver%%.*}"
  if [[ -n "$major" ]] && (( major < 12 )); then
    err "macOS ${osver} is too old. Cursor needs 12+, cmux needs 14+."
    exit 1
  fi

  # A tty is required for the interactive path. Say so now rather than
  # hanging on the first prompt.
  if [[ "$HAVE_TTY" != "1" && "$ASSUME_YES" != "1" && "$WANT_ALL" != "1" && "$CHECK_ONLY" != "1" ]]; then
    die_no_tty
  fi

  local avail
  avail="$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -n "$avail" ]] && (( avail < 10 )); then
    warn "Only ${avail} GB free. A full install needs roughly 10 GB."
    ask_yn "Continue anyway?" n || exit 1
  fi

  if ! curl -fsS --max-time 10 -o /dev/null https://github.com 2>/dev/null; then
    err "Cannot reach github.com. Check your network, VPN, or proxy."
    exit 1
  fi

  # Homebrew is a prerequisite for most of the catalogue; on Apple Silicon it
  # also lives at a prefix that is not on the default PATH.
  activate_brew || true
  prepend_path "$HOME/.local/bin"
}

# ===========================================================================
# Phase 1 --- Scan. Also the whole of --check.
# ===========================================================================
phase_scan() {
  step "Checking what you already have"
  local id
  for id in "${REG_IDS[@]}"; do
    set_status "$id" missing
    call_component detect "$id" || true
  done

  table_header
  for id in "${REG_IDS[@]}"; do
    is_filtered_in "$id" || continue
    local st ver p detail
    st="$(get_status "$id")"; ver="$(get_version "$id")"; p="$(get_cpath "$id")"
    detail=""
    [[ -n "$ver" ]] && detail="$ver"
    [[ -n "$p" ]] && detail="$(printf '%-9s %s' "$detail" "$p")"
    [[ -z "$detail" ]] && detail="${C_DIM}—${C_RESET}"
    printf_row "$(reg_name "$id")" "$(status_word "$st")" "$detail"
    local note; note="$(get_note "$id")"
    [[ -n "$note" ]] && say "  ${C_DIM}                                 ↳ ${note}${C_RESET}"
  done
  say ""
}

# ===========================================================================
# Phase 2 --- Selection
# ===========================================================================

# Ask about one component. Sets ACTION to install, skip, or keep.
select_one() {
  local id="$1" st name extras
  st="$(get_status "$id")"; name="$(reg_name "$id")"; extras="$(reg_extras "$id")"

  if [[ "$st" == "ok" ]]; then
    set_action "$id" skip
    return 0
  fi

  # Conflicts and outdated installs get the dedicated Phase 3 treatment.
  if [[ "$st" == "conflict" || "$st" == "outdated" ]]; then
    confirm_conflict "$id"
  else
    if [[ "$WANT_ALL" == "1" ]]; then
      set_action "$id" install
    else
      say ""
      say "  ${C_BOLD}${name}${C_RESET} — $(reg_desc "$id")"
      local answer
      answer="$(ask_choice "Install ${name}?" y \
        "y:Yes, install it" \
        "n:No, skip it" \
        "?:Show exactly what this will do")"
      if [[ "$answer" == "?" ]]; then
        say ""
        call_component plan "$id" || say "      ${C_DIM}(no details available)${C_RESET}"
        answer="$(ask_choice "Install ${name}?" y "y:Yes, install it" "n:No, skip it")"
      fi
      [[ "$answer" == "y" ]] && set_action "$id" install || set_action "$id" skip
    fi
  fi

  # Extras are always a separate question --- "R and its associated tooling"
  # should not be a single yes.
  if [[ -n "$extras" && "$(get_action "$id")" == "install" ]]; then
    if [[ "$WANT_ALL" == "1" ]]; then
      set_extras_wanted "$id" 1
    elif ask_yn "Also set up ${extras}?" y; then
      set_extras_wanted "$id" 1
    else
      set_extras_wanted "$id" 0
    fi
  fi
}

# ===========================================================================
# Phase 3 --- Conflict confirmation
# ===========================================================================
confirm_conflict() {
  local id="$1" name note
  name="$(reg_name "$id")"; note="$(get_note "$id")"

  say ""
  say "  ${C_YELLOW}⚠  ${name} — something is already here${C_RESET}"
  say "     Found:    $(get_version "$id")  $(get_cpath "$id")"
  [[ -n "$note" ]] && say "     ${C_DIM}${note}${C_RESET}"
  say ""
  say "     Proposed:"
  call_component plan "$id" || true

  if [[ "$WANT_ALL" == "1" ]]; then
    # --all means "install what is missing", never "overwrite what is there".
    # Anything ambiguous stays untouched unless a human says otherwise.
    set_action "$id" keep
    info "Leaving ${name} as it is (--all does not overwrite)."
    return 0
  fi

  local answer
  answer="$(ask_choice "What would you like to do?" i \
    "i:Install ours alongside — your existing copy is left in place" \
    "k:Keep only what you already have" \
    "s:Skip this component entirely")"

  case "$answer" in
    i) set_action "$id" install ;;
    *) set_action "$id" keep ;;
  esac
}

phase_select() {
  step "What would you like to install?"

  local mode
  if [[ "$WANT_ALL" == "1" ]]; then
    mode=a
  else
    mode="$(ask_choice "Choose" a \
      "a:Install everything missing" \
      "s:Choose component by component" \
      "d:Just show me the report, change nothing" \
      "q:Quit")"
  fi

  case "$mode" in
    q) say ""; info "Nothing was changed."; exit 0 ;;
    d) CHECK_ONLY=1; return 0 ;;
    a) WANT_ALL=1 ;;
  esac

  local id
  for id in "${REG_IDS[@]}"; do
    if ! is_filtered_in "$id"; then set_action "$id" skip; continue; fi
    select_one "$id"
  done

  # A component whose prerequisite was declined cannot proceed. Catch that here
  # rather than letting it fail forty lines into the install.
  for id in "${REG_IDS[@]}"; do
    [[ "$(get_action "$id")" == "install" ]] || continue
    local missing_dep
    if ! missing_dep="$(deps_satisfied "$id")"; then
      warn "$(reg_name "$id") needs $(reg_name "$missing_dep"), which you skipped. Dropping it."
      set_action "$id" skip
    fi
  done
}

# ===========================================================================
# Phase 4 --- Plan review
# ===========================================================================
phase_review() {
  local id count=0
  step "Here is exactly what will happen"
  for id in "${REG_IDS[@]}"; do
    [[ "$(get_action "$id")" == "install" ]] || continue
    count=$(( count + 1 ))
    local extra_note=""
    [[ "$(get_extras_wanted "$id")" == "1" ]] && extra_note=" ${C_DIM}+ $(reg_extras "$id")${C_RESET}"
    say "  ${C_GREEN}+${C_RESET} $(reg_name "$id")${extra_note}"
  done

  if (( count == 0 )); then
    say ""
    ok "Nothing to do — you are already set up."
    return 1
  fi

  say ""
  say "  ${C_DIM}Log: ${LOG_FILE}${C_RESET}"
  say ""

  if [[ "$DRY_RUN" == "1" ]]; then
    step "Dry run — commands that would be executed"
    for id in "${REG_IDS[@]}"; do
      [[ "$(get_action "$id")" == "install" ]] || continue
      say "  ${C_BOLD}$(reg_name "$id")${C_RESET}"
      call_component plan "$id" || true
    done
    say ""
    ok "Dry run complete. Nothing was changed."
    exit 0
  fi

  ask_yn "Proceed?" y || { say ""; info "Nothing was changed."; exit 0; }
  return 0
}

# ===========================================================================
# Phase 5 --- Execute
# ===========================================================================
phase_execute() {
  step "Installing"
  local id
  for id in "${REG_IDS[@]}"; do
    [[ "$(get_action "$id")" == "install" ]] || continue
    local name; name="$(reg_name "$id")"
    say ""
    say "  ${C_BOLD}${name}${C_RESET}"

    if call_component install "$id"; then
      if [[ "$(get_extras_wanted "$id")" == "1" ]]; then
        call_component extras "$id" || warn "${name}: the optional extra did not complete."
      fi
      if call_component verify "$id"; then
        ok "${name} installed."
        set_result "$id" done
        INSTALLED_IDS="$INSTALLED_IDS $id"
      else
        # Installed without error but not detectable --- almost always PATH.
        warn "${name} installed but is not on PATH yet. A new terminal should fix it."
        set_result "$id" done
        INSTALLED_IDS="$INSTALLED_IDS $id"
      fi
    else
      err "${name} failed. See the log for details."
      set_result "$id" failed
      FAILED_IDS="$FAILED_IDS $id"
    fi
  done
}

# ===========================================================================
# Phase 6 --- Report
# ===========================================================================
phase_report() {
  step "Where things stand"
  local id
  for id in "${REG_IDS[@]}"; do
    set_status "$id" missing
    call_component detect "$id" || true
  done

  table_header
  for id in "${REG_IDS[@]}"; do
    is_filtered_in "$id" || continue
    local st; st="$(get_status "$id")"
    [[ "$(get_result "$id")" == "failed" ]] && st=failed
    local detail; detail="$(get_version "$id")"
    [[ -z "$detail" ]] && detail="${C_DIM}—${C_RESET}"
    printf_row "$(reg_name "$id")" "$(status_word "$st")" "$detail"
  done

  step "Three things left, which only you can do"
  say ""
  say "  ${C_BOLD}1. Restart your terminal${C_RESET} — or open a new tab."
  say "     PATH and prompt changes do not apply to this window."
  say ""
  say "  ${C_BOLD}2. Sign in to each tool${C_RESET} — run these one at a time:"
  say ""
  have claude       && say "       ${C_CYAN}claude${C_RESET}              then follow the browser prompt"
  say "                           ${C_DIM}the free tier cannot use browser sign-in; you${C_RESET}"
  say "                           ${C_DIM}need Pro/Max/Team or an ANTHROPIC_API_KEY${C_RESET}"
  have codex        && say "       ${C_CYAN}codex${C_RESET}               sign in with ChatGPT"
  say "                           ${C_DIM}USC IT can grant access if you do not have it${C_RESET}"
  have cursor-agent && say "       ${C_CYAN}cursor-agent login${C_RESET}  Cursor is free for students"
  have gh           && say "       ${C_CYAN}gh auth login${C_RESET}       choose HTTPS and browser sign-in"
  say ""
  say "  ${C_BOLD}3. Add your API keys${C_RESET} — copy the template and fill it in:"
  say ""
  say "       ${C_CYAN}cp .env.example .env${C_RESET}"
  say "     Neil's labelling demos read OPENAI_API_KEY from that file."
  say ""
  rule

  if [[ -n "${FAILED_IDS// /}" ]]; then
    say ""
    err "These did not install:${FAILED_IDS}"
    say "  Send ${LOG_FILE} to the workshop organisers and we will sort it out."
    say "  Re-run just the failures with:"
    say "      ${C_CYAN}$0 --only $(printf '%s' "${FAILED_IDS# }" | tr ' ' ',')${C_RESET}"
    say ""
    return 1
  fi

  say ""
  ok "You are ready for the workshop."
  say "  ${C_DIM}Check anything at any time with: $0 --check${C_RESET}"
  say ""
  return 0
}

# ===========================================================================
main() {
  banner
  phase_preflight
  phase_scan

  if [[ "$CHECK_ONLY" == "1" ]]; then
    say "  ${C_DIM}Report only — nothing was changed.${C_RESET}"
    say ""
    exit 0
  fi

  phase_select
  if [[ "$CHECK_ONLY" == "1" ]]; then
    say "  ${C_DIM}Report only — nothing was changed.${C_RESET}"
    exit 0
  fi

  phase_review || exit 0
  phase_execute
  phase_report
}

main "$@"
