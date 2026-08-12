#!/usr/bin/env bash
# install.sh --- macOS bootstrap for the USC Economics AI workshop environment.
#
# Run it like this:
#
#     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/OWNER/REPO/REF/setup/install.sh)"
#
# Why that shape rather than `curl ... | bash`: with a pipe, bash executes the
# script as it arrives, so a download cut short halfway runs half a script.
# Command substitution downloads the whole thing first and only then runs it.
#
# Prompting works under either form, because every prompt in this installer
# reads /dev/tty rather than stdin. What genuinely breaks prompting is having no
# controlling terminal at all --- CI, cron, some remote shells --- and that is
# what the check below detects.
#
# All this file does is fetch the setup directory and hand off to bin/setup.sh.
# Everything interesting lives there.

set -euo pipefail

REPO="${ECON_AI_REPO:-sankalpsharmaa/usc-econ-ai-workshop}"
REF="${ECON_AI_REF:-jyl}"

RED=''; YELLOW=''; BOLD=''; DIM=''; RESET=''
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED=$'\033[31m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
fi

die() { printf '%s\n' "${RED}error:${RESET} $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || \
  die "This is the macOS installer. On Windows, run install.ps1 in PowerShell."

# Fail loudly on the piped form rather than blocking forever on the first read.
# --all and --check are non-interactive, so they are still fine when piped.
wants_prompts=1
for arg in "$@"; do
  case "$arg" in --all|--check|--yes|--dry-run|-h|--help) wants_prompts=0 ;; esac
done

if [[ "$wants_prompts" == "1" ]] && ! { : >/dev/tty; } 2>/dev/null; then
  printf '%s\n' "${YELLOW}This installer needs to ask you some questions, but this${RESET}"
  printf '%s\n' "${YELLOW}session has no terminal attached, so it cannot.${RESET}"
  printf '\n'
  printf '%s\n' "Run it from a normal terminal window:"
  printf '\n'
  printf '%s\n' "  ${BOLD}/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/${REPO}/${REF}/setup/install.sh)\"${RESET}"
  printf '\n'
  printf '%s\n' "Or install everything without being asked anything:"
  printf '\n'
  printf '%s\n' "  ${BOLD}curl -fsSL https://raw.githubusercontent.com/${REPO}/${REF}/setup/install.sh | bash -s -- --all${RESET}"
  printf '\n'
  exit 2
fi

command -v curl >/dev/null 2>&1 || die "curl is required but was not found."
command -v tar  >/dev/null 2>&1 || die "tar is required but was not found."

WORK="$(mktemp -d "${TMPDIR:-/tmp}/econ-ai-setup.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

printf '%s\n' "${DIM}Fetching the setup files (${REPO} @ ${REF})...${RESET}"

TARBALL="https://codeload.github.com/${REPO}/tar.gz/${REF}"
curl -fsSL --retry 3 --max-time 120 "$TARBALL" -o "$WORK/src.tar.gz" \
  || die "Could not download ${TARBALL}
  Check your network, and that the branch or tag '${REF}' exists."

tar -xzf "$WORK/src.tar.gz" -C "$WORK" || die "Could not unpack the download."

# The tarball's top directory is named <repo>-<ref>, so find it rather than
# guessing --- refs containing a slash get rewritten in that name.
SETUP_ROOT="$(find "$WORK" -maxdepth 3 -type d -name setup -print -quit)"
[[ -n "$SETUP_ROOT" && -f "$SETUP_ROOT/bin/setup.sh" ]] \
  || die "The download did not contain setup/bin/setup.sh."

chmod +x "$SETUP_ROOT/bin/"*.sh 2>/dev/null || true

# exec so the installer inherits our terminal directly.
SETUP_ROOT="$SETUP_ROOT" exec /bin/bash "$SETUP_ROOT/bin/setup.sh" "$@"
