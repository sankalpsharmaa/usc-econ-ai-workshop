#!/usr/bin/env bash
# registry.sh --- component registration, ordering, and per-component state.
#
# Compatibility note: macOS still ships bash 3.2 as /bin/bash, which has no
# associative arrays. Everything here therefore uses indexed arrays plus
# eval-backed dynamic variables. Do not "modernise" this to `declare -A`
# without also changing the bootstrap to require bash 4+, which would mean
# installing bash before we can install anything else.

REG_IDS=()          # ordered component ids
REG_NAMES=()        # human-readable names
REG_DESCS=()        # one-line descriptions
REG_REQUIRES=()     # space-separated ids this component needs
REG_EXTRAS=()       # human label for the optional add-on, or empty

# Component ids may contain hyphens; variable names may not.
_slug() { printf '%s' "$1" | tr '-' '_'; }

# register <id> <name> <description> <requires> <extras-label>
register() {
  REG_IDS+=( "$1" )
  REG_NAMES+=( "$2" )
  REG_DESCS+=( "$3" )
  REG_REQUIRES+=( "${4:-}" )
  REG_EXTRAS+=( "${5:-}" )
}

# Index lookup. Echoes the array offset, or -1 when unknown.
reg_index() {
  local want="$1" i
  for (( i = 0; i < ${#REG_IDS[@]}; i++ )); do
    [[ "${REG_IDS[$i]}" == "$want" ]] && { printf '%s' "$i"; return 0; }
  done
  printf '%s' "-1"; return 1
}

reg_name()     { local i; i=$(reg_index "$1") && printf '%s' "${REG_NAMES[$i]}"; }
reg_desc()     { local i; i=$(reg_index "$1") && printf '%s' "${REG_DESCS[$i]}"; }
reg_requires() { local i; i=$(reg_index "$1") && printf '%s' "${REG_REQUIRES[$i]}"; }
reg_extras()   { local i; i=$(reg_index "$1") && printf '%s' "${REG_EXTRAS[$i]}"; }

# --- Per-component state ----------------------------------------------------
# Each component accumulates:
#   STATUS_<id>   ok | missing | conflict | outdated | failed | skipped
#   VERSION_<id>  detected version string
#   PATH_<id>     resolved location on disk
#   NOTE_<id>     free text shown beside the status (e.g. "system Python")
#   ACTION_<id>   install | extras | skip | keep   (decided during selection)
#   RESULT_<id>   done | failed | skipped          (set during execution)

set_state() { eval "$1_$(_slug "$2")=\$3"; }
get_state() { eval "printf '%s' \"\${$1_$(_slug "$2"):-}\""; }

set_status()  { set_state STATUS  "$1" "$2"; }
get_status()  { get_state STATUS  "$1"; }
set_version() { set_state VERSION "$1" "$2"; }
get_version() { get_state VERSION "$1"; }
set_cpath()   { set_state CPATH   "$1" "$2"; }
get_cpath()   { get_state CPATH   "$1"; }
set_note()    { set_state NOTE    "$1" "$2"; }
get_note()    { get_state NOTE    "$1"; }
set_action()  { set_state ACTION  "$1" "$2"; }
get_action()  { get_state ACTION  "$1"; }
set_result()  { set_state RESULT  "$1" "$2"; }
get_result()  { get_state RESULT  "$1"; }

# Whether the user opted into the component's extras (renv, a Python version...)
set_extras_wanted() { set_state EXTRAWANT "$1" "$2"; }
get_extras_wanted() { get_state EXTRAWANT "$1"; }

# --- Dispatch ---------------------------------------------------------------
# Components implement c_<slug>_{detect,plan,install,extras,verify}.
# detect/plan/verify are mandatory; extras only where an extras label exists.
call_component() {
  local verb="$1" id="$2"
  local fn="c_$(_slug "$id")_${verb}"
  if declare -f "$fn" >/dev/null 2>&1; then
    "$fn"
    return $?
  fi
  return 127
}

# --- Dependency resolution --------------------------------------------------
# Returns 0 when every prerequisite of <id> is either already present or
# selected for installation. Echoes the first unmet dependency otherwise.
deps_satisfied() {
  local id="$1" dep st act
  for dep in $(reg_requires "$id"); do
    st="$(get_status "$dep")"
    act="$(get_action "$dep")"
    # Present already, or queued to be installed: fine either way.
    [[ "$st" == "ok" || "$st" == "outdated" ]] && continue
    [[ "$act" == "install" ]] && continue
    printf '%s' "$dep"
    return 1
  done
  return 0
}

# Ids the user asked for via --only, or empty for "all".
ONLY_IDS="${ONLY_IDS:-}"
SKIP_IDS="${SKIP_IDS:-}"

# Whether a component is eligible given --only/--skip filters.
is_filtered_in() {
  local id="$1" tok
  if [[ -n "$SKIP_IDS" ]]; then
    for tok in $(printf '%s' "$SKIP_IDS" | tr ',' ' '); do
      [[ "$tok" == "$id" ]] && return 1
    done
  fi
  if [[ -n "$ONLY_IDS" ]]; then
    for tok in $(printf '%s' "$ONLY_IDS" | tr ',' ' '); do
      [[ "$tok" == "$id" ]] && return 0
    done
    return 1
  fi
  return 0
}
