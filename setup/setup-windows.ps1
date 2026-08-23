<#
    USC Econ AI Workshop, Fall 2026 - Windows setup

    Installs the common floor: Claude Code, Cursor CLI, Codex, uv, Python, R, Julia,
    git, and the standard data-science packages for Python, R, and Julia.

    It only ADDS things. If you already have a tool, it says so and moves on.
    It never upgrades, never removes, and never touches your system Python.

    Usage, in PowerShell:
      powershell -ExecutionPolicy Bypass -File setup-windows.ps1
      powershell -ExecutionPolicy Bypass -File setup-windows.ps1 -Check
#>

param(
    [switch]$Check,
    [switch]$Yes,
    [string]$WorkshopDir = (Join-Path $HOME 'econ-ai-workshop'),
    [string]$PyVersion   = '3.12'
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $WorkshopDir | Out-Null
$Log = Join-Path $WorkshopDir 'setup-windows.log'
try { Start-Transcript -Path $Log -Append | Out-Null } catch { }

Write-Host "=== USC econ AI workshop setup, Windows, $(Get-Date) ==="
Write-Host "Log file: $Log"
Write-Host "Workshop folder: $WorkshopDir"
if ($Check) { Write-Host "MODE: -Check, nothing will be installed" }

$Results = New-Object System.Collections.ArrayList

function Record($Status, $Name, $Detail) {
    [void]$Results.Add([PSCustomObject]@{ Status = $Status; Tool = $Name; Detail = $Detail })
}
function Write-Section($Text) { Write-Host "`n--- $Text ---" }
function Have($Cmd) { return [bool](Get-Command $Cmd -ErrorAction SilentlyContinue) }

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $extra   = @(
        (Join-Path $HOME '.local\bin'),
        (Join-Path $env:LOCALAPPDATA 'Programs\cursor-agent'),
        (Join-Path $env:APPDATA 'npm')
    ) -join ';'
    $env:Path = "$machine;$user;$extra"
}
Update-SessionPath

function Ask($Question) {
    if ($Yes) { return $false }
    $reply = Read-Host "$Question [y/N]"
    return ($reply -match '^[Yy]')
}

# install_if_missing: run the action, then confirm the command actually appeared
function Install-IfMissing {
    param([string]$Command, [string]$Name, [scriptblock]$Action)
    Write-Section $Name
    if (Have $Command) {
        $where = (Get-Command $Command).Source
        Write-Host "Already installed: $where"
        Record 'HAVE' $Name $where
        return
    }
    if ($Check) {
        Write-Host "MISSING, would install"
        Record 'MISSING' $Name 'would install'
        return
    }
    Write-Host "Installing..."
    try { & $Action } catch { Write-Host "Installer reported: $_" }
    Update-SessionPath
    if (Have $Command) {
        Record 'INSTALLED' $Name (Get-Command $Command).Source
    } else {
        Record 'FAILED' $Name "not on PATH yet, try a new terminal, then rerun"
        Write-Host "Did not appear on PATH. Not fatal, moving on."
    }
}

function Install-WithWinget($Id, $Source) {
    $wgArgs = @('install', '-e', '--id', $Id, '--accept-source-agreements', '--accept-package-agreements')
    if ($Source) { $wgArgs += @('--source', $Source) }
    & winget @wgArgs
}

# ---------------------------------------------------------------- prerequisites

Write-Section 'winget (Windows package manager)'
if (Have 'winget') {
    Write-Host "winget: $(winget --version)"
    Record 'HAVE' 'winget' 'ok'
} else {
    Write-Host "winget is missing. Install 'App Installer' from the Microsoft Store, then rerun."
    Record 'MANUAL' 'winget' 'https://apps.microsoft.com/detail/9nblggh4nns1'
}

Install-IfMissing 'git' 'Git' { Install-WithWinget 'Git.Git' }

# ------------------------------------------------------------------ agent tools

Install-IfMissing 'claude' 'Claude Code' {
    Invoke-RestMethod https://claude.ai/install.ps1 | Invoke-Expression
}
Install-IfMissing 'cursor-agent' 'Cursor CLI' {
    Invoke-RestMethod 'https://cursor.com/install?win32=true' | Invoke-Expression
}
Install-IfMissing 'codex' 'Codex CLI' {
    if (Have 'winget') { Install-WithWinget 'OpenAI.Codex' }
    elseif (Have 'npm') { npm install -g '@openai/codex' }
    else { Write-Host 'Needs winget or npm.' }
}
Install-IfMissing 'uv' 'uv' {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# ---------------------------------------------------------------------- runtime

function Find-Rscript {
    if (Have 'Rscript') { return (Get-Command 'Rscript').Source }
    $candidates = Get-ChildItem -Path "$env:ProgramFiles\R" -Filter 'Rscript.exe' -Recurse -ErrorAction SilentlyContinue
    if ($candidates) { return ($candidates | Sort-Object FullName -Descending | Select-Object -First 1).FullName }
    return $null
}

Write-Section 'R'
$Rscript = Find-Rscript
if ($Rscript) {
    Write-Host "Already installed: $Rscript"
    Record 'HAVE' 'R' $Rscript
} elseif ($Check) {
    Record 'MISSING' 'R' 'would install'
} elseif (Have 'winget') {
    Install-WithWinget 'RProject.R'
    Install-WithWinget 'RProject.Rtools'
    Update-SessionPath
    $Rscript = Find-Rscript
    if ($Rscript) { Record 'INSTALLED' 'R' $Rscript } else { Record 'FAILED' 'R' "see $Log" }
} else {
    Record 'MANUAL' 'R' 'https://cran.r-project.org/bin/windows/base/'
}

Write-Section 'Julia (via juliaup)'
if ((Have 'julia') -or (Have 'juliaup')) {
    Record 'HAVE' 'Julia' 'already installed'
    Write-Host 'Already installed.'
} elseif ($Check) {
    Record 'MISSING' 'Julia' 'would install'
} elseif (Have 'winget') {
    # community repo first: works on machines where the Microsoft Store is blocked
    Install-WithWinget 'Julialang.Juliaup'
    Update-SessionPath
    if (-not (Have 'julia')) {
        Write-Host 'Trying the Microsoft Store version instead...'
        & winget install --name Julia --id 9NJNWW8PVKMN -e --source msstore --accept-source-agreements --accept-package-agreements
        Update-SessionPath
    }
    if (Have 'julia') {
        Record 'INSTALLED' 'Julia' (Get-Command 'julia').Source
    } elseif (Have 'juliaup') {
        & juliaup add release
        Update-SessionPath
        if (Have 'julia') { Record 'INSTALLED' 'Julia' (Get-Command 'julia').Source }
        else { Record 'FAILED' 'Julia' 'juliaup installed but julia did not appear, open a new terminal and rerun' }
    } else {
        Record 'MANUAL' 'Julia' 'https://julialang.org/downloads/'
        Write-Host 'Automatic install did not finish. Download Julia from https://julialang.org/downloads/'
    }
} else {
    Record 'MANUAL' 'Julia' 'https://julialang.org/downloads/'
}

# --------------------------------------------------------------------- packages

$Req = Join-Path $WorkshopDir 'requirements-workshop.txt'
@'
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
'@ | Set-Content -Path $Req -Encoding UTF8

Write-Section "Python $PyVersion and data-science packages"
$Venv = Join-Path $WorkshopDir '.venv'
if ($Check) {
    Write-Host "Would create $Venv and install the packages listed in $Req"
    Record 'MISSING' 'Python packages' "would install into $Venv"
} elseif (Have 'uv') {
    & uv python install $PyVersion
    if (-not (Test-Path $Venv)) { & uv venv $Venv --python $PyVersion }
    & uv pip install --python $Venv -r $Req
    if ($LASTEXITCODE -eq 0) { Record 'INSTALLED' 'Python packages' $Venv }
    else { Record 'FAILED' 'Python packages' "see $Log" }
    & (Join-Path $Venv 'Scripts\python.exe') -m ipykernel install --user --name econ-workshop --display-name 'Python (econ workshop)' 2>$null
} else {
    Record 'SKIPPED' 'Python packages' 'uv not available'
}

Write-Section 'R packages'
$RScriptFile = Join-Path $WorkshopDir 'install-r-packages.R'
@'
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
'@ | Set-Content -Path $RScriptFile -Encoding UTF8

if ($Check) {
    Record 'MISSING' 'R packages' "would run $RScriptFile"
} elseif ($Rscript) {
    & $Rscript $RScriptFile
    if ($LASTEXITCODE -eq 0) { Record 'INSTALLED' 'R packages' 'ok' } else { Record 'FAILED' 'R packages' "see $Log" }
} else {
    Record 'SKIPPED' 'R packages' 'R not available'
}

Write-Section 'Julia packages'
$JlFile = Join-Path $WorkshopDir 'install-julia-packages.jl'
$JlBody = @'
# Installs into a workshop-only project, so your global Julia setup is untouched.
using Pkg
proj = raw"__PROJ__"
mkpath(proj); Pkg.activate(proj)
want = ["DataFrames", "CSV", "StatsBase", "GLM", "Distributions", "Optim", "Plots"]
present = keys(Pkg.project().dependencies)
add = filter(p -> !(p in present), want)
isempty(add) ? println("All Julia packages already present.") : Pkg.add(add)
'@
$JlBody.Replace('__PROJ__', (Join-Path $WorkshopDir 'julia')) | Set-Content -Path $JlFile -Encoding UTF8

if ($Check) {
    Record 'MISSING' 'Julia packages' "would run $JlFile"
} elseif (Have 'julia') {
    & julia $JlFile
    if ($LASTEXITCODE -eq 0) { Record 'INSTALLED' 'Julia packages' 'ok' } else { Record 'FAILED' 'Julia packages' "see $Log" }
} else {
    Record 'SKIPPED' 'Julia packages' 'julia not on PATH yet, rerun after opening a new terminal'
}

# ---------------------------------------------------------------------- summary

Write-Host "`n======================= SUMMARY ======================="
$Results | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "======================================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Close this window and open a new PowerShell."
Write-Host "  2. Check it worked:  claude --version ; uv --version"
Write-Host "  3. Your Python setup: $Venv\Scripts\Activate.ps1"
Write-Host "  4. Log saved at: $Log"

try { Stop-Transcript | Out-Null } catch { }

# exit 1 if anything actually failed, so CI and the -Check run can be scripted
$failed = @($Results | Where-Object { $_.Status -eq 'FAILED' })
if ($failed.Count -gt 0) { exit 1 } else { exit 0 }
