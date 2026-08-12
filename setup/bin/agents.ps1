# agents.ps1 --- give each agent its own branch, its own directory, and its own
# terminal pane.
#
#     .\agents.ps1 did-spec robustness tables
#
# creates three git worktrees and opens one Windows Terminal tab per worktree.
#
# The reason to bother: two agents editing the same working tree will overwrite
# each other's edits and produce a diff nobody can review. A worktree is a
# second checkout of the same repository on a different branch, sharing one
# .git directory --- so each agent gets an isolated set of files, and you merge
# the results through ordinary git.
#
# Clean up when you are done:  git worktree remove ..\<repo>-<name>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Names
)

$ErrorActionPreference = 'Stop'
$AgentCmd = if ($env:AGENT_CMD) { $env:AGENT_CMD } else { 'claude' }

if (-not $Names -or $Names.Count -eq 0) {
    Write-Host @"
Usage: .\agents.ps1 <name> [name...]

  Creates a git worktree and a branch for each name, then opens one Windows
  Terminal tab per worktree with '$AgentCmd' ready to run.

  `$env:AGENT_CMD='codex'; .\agents.ps1 tables    use a different agent
"@
    exit 64
}

& git rev-parse --git-dir 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host 'error: not inside a git repository.' -ForegroundColor Red; exit 1 }

$repoRoot = (& git rev-parse --show-toplevel).Trim()
$repoName = Split-Path -Leaf $repoRoot
$parent   = Split-Path -Parent $repoRoot

$created = @()

# Worktrees are created as siblings of the repo, never inside it --- a worktree
# nested in its own parent confuses both git and every file watcher involved.
foreach ($name in $Names) {
    $branch = "agent/$name"
    $dir = Join-Path $parent "$repoName-$name"

    if (Test-Path $dir) {
        Write-Host "  = $dir already exists, reusing it"
    } else {
        & git show-ref --verify --quiet "refs/heads/$branch" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  + worktree $dir on existing branch $branch"
            & git worktree add $dir $branch | Out-Null
        } else {
            Write-Host "  + worktree $dir on new branch $branch"
            & git worktree add -b $branch $dir | Out-Null
        }
    }
    $created += $dir
}

# One Windows Terminal tab per worktree, all in a single window.
if (Get-Command wt -ErrorAction SilentlyContinue) {
    $wtArgs = @()
    for ($i = 0; $i -lt $created.Count; $i++) {
        if ($i -gt 0) { $wtArgs += ';' }
        $wtArgs += @('new-tab', '--title', (Split-Path -Leaf $created[$i]), '-d', $created[$i])
    }
    try { & wt @wtArgs } catch { Write-Host '    (could not open Windows Terminal tabs)' -ForegroundColor Yellow }
} else {
    foreach ($dir in $created) { Write-Host "    open it with: cd $dir; $AgentCmd" }
}

Write-Host ''
Write-Host 'Worktrees:'
& git worktree list

Write-Host @"

Next:
  Run '$AgentCmd' in each tab and give it one task.
  Review each branch separately with: git diff main..agent/<name>
  Remove one when finished with:      git worktree remove ..\$repoName-<name>
"@
