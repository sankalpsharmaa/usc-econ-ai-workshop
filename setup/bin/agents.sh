#!/usr/bin/env bash
# agents.sh --- give each agent its own branch, its own directory, and its own
# workspace.
#
#     ./agents.sh did-spec robustness tables
#
# creates three git worktrees and opens one cmux workspace per worktree.
#
# The reason to bother: two agents editing the same working tree will overwrite
# each other's edits and produce a diff nobody can review. A worktree is a
# second checkout of the same repository on a different branch, sharing one
# .git directory --- so each agent gets an isolated set of files, and you merge
# the results through ordinary git.
#
# Clean up when you are done:  git worktree remove ../<repo>-<name>

set -euo pipefail

AGENT_CMD="${AGENT_CMD:-claude}"    # override to codex, cursor-agent, ...
CMUX_BIN="/Applications/cmux.app/Contents/Resources/bin/cmux"

if [[ $# -eq 0 ]]; then
  cat <<EOF
Usage: $(basename "$0") <name> [name...]

  Creates a git worktree and a branch for each name, then opens one workspace
  per worktree with '${AGENT_CMD}' ready to run.

  AGENT_CMD=codex $(basename "$0") tables    use a different agent
EOF
  exit 64
fi

git rev-parse --git-dir >/dev/null 2>&1 || {
  printf 'error: not inside a git repository.\n' >&2; exit 1
}

REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$REPO_ROOT")"
PARENT="$(dirname "$REPO_ROOT")"

# Worktrees are created as siblings of the repo, never inside it --- a worktree
# nested in its own parent confuses both git and every file watcher involved.
for name in "$@"; do
  branch="agent/${name}"
  dir="${PARENT}/${REPO_NAME}-${name}"

  if [[ -d "$dir" ]]; then
    printf '  = %s already exists, reusing it\n' "$dir"
  elif git show-ref --verify --quiet "refs/heads/${branch}"; then
    printf '  + worktree %s on existing branch %s\n' "$dir" "$branch"
    git worktree add "$dir" "$branch" >/dev/null
  else
    printf '  + worktree %s on new branch %s\n' "$dir" "$branch"
    git worktree add -b "$branch" "$dir" >/dev/null
  fi

  if [[ -x "$CMUX_BIN" ]]; then
    "$CMUX_BIN" "$dir" >/dev/null 2>&1 || printf '    (could not open a cmux workspace)\n'
  else
    printf '    open it with: cd %s && %s\n' "$dir" "$AGENT_CMD"
  fi
done

printf '\nWorktrees:\n'
git worktree list

cat <<EOF

Next:
  Run '${AGENT_CMD}' in each workspace and give it one task.
  Review each branch separately with: git diff main..agent/<name>
  Remove one when finished with:      git worktree remove ../${REPO_NAME}-<name>
EOF
