# Components.psm1 --- the Windows component catalogue and its helpers.
#
# Targets native Windows: winget with user scope wherever the package allows it,
# so no administrator rights and no reboot. WSL is deliberately not used.
#
# EVERY WINGET ID BELOW NEEDS VERIFYING ON A REAL WINDOWS MACHINE before this
# ships. They were written from documentation. Run `winget search <id>` for each
# and confirm both the id and whether it honours --scope user.

$script:DryRun = $false
$script:AssetDir = $null

function Set-ComponentContext {
    param([bool]$DryRun, [string]$AssetDir)
    $script:DryRun = $DryRun
    $script:AssetDir = $AssetDir
}

# --- Helpers ----------------------------------------------------------------

function Test-Have {
    param([string]$Name)
    [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-CmdPath {
    param([string]$Name)
    (Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1).Source
}

# Extracts the first dotted version the command prints, ignoring "an update is
# available" notices --- several of these CLIs print both, and matching the
# first number blindly reports the available version as the installed one.
function Get-FirstVersion {
    param([string]$Name, [string[]]$VersionArgs = @('--version'))
    try {
        $raw = & $Name @VersionArgs 2>&1 | Select-Object -First 5
    } catch { return $null }
    foreach ($line in $raw) {
        $text = [string]$line
        if ($text -match '(?i)available|update|upgrade|newer|new version|latest') { continue }
        if ($text -match '(\d+\.\d+(\.\d+)?)') { return $Matches[1] }
    }
    return $null
}

function Compare-Version {
    param([string]$Have, [string]$Want)
    if (-not $Have) { return $false }
    try { return ([version]("$Have.0.0".Split('.')[0..2] -join '.')) -ge ([version]("$Want.0.0".Split('.')[0..2] -join '.')) }
    catch { return $false }
}

# The single choke point for anything that changes the machine.
function Invoke-Step {
    param([string]$Display, [scriptblock]$Action)
    if ($script:DryRun) { Say "      would run: $Display" 'DarkGray'; return $true }
    Write-LogLine "+ $Display"
    try {
        & $Action
        return $LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE
    } catch {
        Write-LogLine "  ! $($_.Exception.Message)"
        return $false
    }
}

# winget install with user scope. Falls back to default scope when a package
# refuses user scope, which several vendor installers do.
function Install-Winget {
    param([string]$Id, [switch]$NoUserScope)
    $scopeArgs = if ($NoUserScope) { @() } else { @('--scope', 'user') }
    $display = "winget install --id $Id $($scopeArgs -join ' ') --silent"

    if ($script:DryRun) { Say "      would run: $display" 'DarkGray'; return $true }
    Write-LogLine "+ $display"

    $common = @('install', '--id', $Id, '--exact', '--silent',
                '--accept-package-agreements', '--accept-source-agreements',
                '--disable-interactivity')
    $out = & winget @common @scopeArgs 2>&1
    Write-LogLine ($out -join "`n")

    if ($LASTEXITCODE -eq 0) { return $true }

    if (-not $NoUserScope) {
        Say-Warn "$Id refused a user-scope install; retrying at the default scope."
        $out = & winget @common 2>&1
        Write-LogLine ($out -join "`n")
        return ($LASTEXITCODE -eq 0)
    }
    return $false
}

# Refresh PATH inside this process so a tool installed a moment ago is usable by
# the next component without restarting PowerShell.
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

# ConvertFrom-Json -AsHashtable only exists in PowerShell 6+, and this installer
# frequently runs under Windows PowerShell 5.1 --- pwsh is one of the things it
# installs. Convert by hand so the settings merge works on both.
function ConvertFrom-JsonToHashtable {
    param([string]$Json)
    if (-not $Json.Trim()) { return @{} }

    if ($PSVersionTable.PSVersion.Major -ge 6) {
        return ($Json | ConvertFrom-Json -AsHashtable)
    }

    $obj = $Json | ConvertFrom-Json
    $table = @{}
    foreach ($prop in $obj.PSObject.Properties) { $table[$prop.Name] = $prop.Value }
    return $table
}

function Backup-UserFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $bak = "$Path.bak.$stamp"
    if ($script:DryRun) { Say "      would back up: $Path -> $bak" 'DarkGray'; return $bak }
    Copy-Item -LiteralPath $Path -Destination $bak -Force
    return $bak
}

# Idempotent block in the PowerShell profile. Re-running replaces the block
# rather than appending a second copy.
function Set-ProfileBlock {
    param([string]$Marker, [string]$Content)
    $begin = "# >>> $Marker >>>"
    $end   = "# <<< $Marker <<<"

    if ($script:DryRun) { Say "      would update block '$Marker' in $PROFILE" 'DarkGray'; return }

    $dir = Split-Path -Parent $PROFILE
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if (-not (Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }

    $lines = Get-Content -LiteralPath $PROFILE
    $kept = @(); $skip = $false
    foreach ($line in $lines) {
        if ($line -eq $begin) { $skip = $true }
        if (-not $skip) { $kept += $line }
        if ($line -eq $end) { $skip = $false }
    }
    $kept += @('', $begin, $Content, $end)
    Set-Content -LiteralPath $PROFILE -Value $kept
}

# --- Component definitions --------------------------------------------------
# Each carries its own mutable state (Status/Version/CPath/Note/Action/Result),
# which keeps the orchestrator free of a parallel bookkeeping structure.

function New-Component {
    param(
        [string]$Id, [string]$Name, [string]$Desc,
        [string[]]$Requires = @(), [string]$Extras = '',
        [scriptblock]$Detect, [scriptblock]$Plan,
        [scriptblock]$Install, [scriptblock]$ExtrasAction, [scriptblock]$Verify
    )
    [PSCustomObject]@{
        Id = $Id; Name = $Name; Desc = $Desc
        Requires = $Requires; Extras = $Extras
        Detect = $Detect; Plan = $Plan; Install = $Install
        ExtrasAction = $ExtrasAction; Verify = $Verify
        Status = 'missing'; Version = ''; CPath = ''; Note = ''
        Action = 'skip'; Result = ''; ExtrasWanted = $false
    }
}

# Shared detector for "is this CLI present and actually working?". A command
# that exists but reports no version is treated as broken, not installed ---
# stale shims sit on PATH and exit cleanly while doing nothing.
function Set-SimpleDetection {
    param($C, [string]$Command, [string[]]$VersionArgs = @('--version'))
    if (-not (Test-Have $Command)) { $C.Status = 'missing'; return }
    $C.CPath = Get-CmdPath $Command
    $v = Get-FirstVersion -Name $Command -VersionArgs $VersionArgs
    if (-not $v) {
        $C.Status = 'conflict'
        $C.Note = "found at $($C.CPath) but it did not report a version - likely a broken wrapper, not a real install"
        return
    }
    $C.Version = $v
    $C.Status = 'ok'
}

$PY_TARGET = '3.12'
$PY_MIN    = '3.10'

function Get-ComponentCatalogue {
    $list = @()

    $list += New-Component -Id 'winget' -Name 'winget' -Desc 'Windows package manager' `
        -Detect { param($C) Set-SimpleDetection $C 'winget' @('--version') } `
        -Plan { Say '      Ships with Windows 11 as part of App Installer.'
                Say '      On Windows 10, install "App Installer" from the Microsoft Store.' } `
        -Install {
            # Deliberately not automated: installing winget itself means
            # side-loading an MSIX bundle, which is exactly the kind of thing
            # that should be a deliberate human action on a managed laptop.
            Say-Err 'winget cannot be installed automatically.'
            Say '  Install "App Installer" from the Microsoft Store, then re-run this script.'
            Say '  https://apps.microsoft.com/detail/9nblggh4nns1'
            return $false
        } `
        -Verify { Test-Have 'winget' }

    $list += New-Component -Id 'pwsh' -Name 'PowerShell 7' -Desc 'Modern PowerShell, what the agent CLIs expect' -Requires @('winget') `
        -Detect { param($C)
            if (Test-Have 'pwsh') { Set-SimpleDetection $C 'pwsh' @('--version') }
            else {
                $C.Status = 'missing'
                $C.Note = "you are on Windows PowerShell $($PSVersionTable.PSVersion.Major); 7 is a separate, side-by-side install"
            } } `
        -Plan { Say '      winget install --id Microsoft.PowerShell --scope user'
                Say '      Installs alongside Windows PowerShell 5.1; neither replaces the other.' } `
        -Install { Install-Winget -Id 'Microsoft.PowerShell' } `
        -Verify { Test-Have 'pwsh' }

    $list += New-Component -Id 'terminal' -Name 'Windows Terminal' -Desc 'Tabs, panes, and a usable console' -Requires @('winget') `
        -Detect { param($C)
            if (Test-Have 'wt') { $C.Status = 'ok'; $C.CPath = Get-CmdPath 'wt' }
            else { $C.Status = 'missing' } } `
        -Plan { Say '      winget install --id Microsoft.WindowsTerminal --scope user'
                Say '      Preinstalled on most Windows 11 machines.' } `
        -Install { Install-Winget -Id 'Microsoft.WindowsTerminal' } `
        -Verify { Test-Have 'wt' }

    $list += New-Component -Id 'git' -Name 'git' -Desc 'Version control' -Requires @('winget') `
        -Detect { param($C) Set-SimpleDetection $C 'git' @('--version') } `
        -Plan { Say '      winget install --id Git.Git --scope user'
                Say '      Also provides Git Bash, which supplies a POSIX shell.' } `
        -Install { Install-Winget -Id 'Git.Git' } `
        -Verify { Test-Have 'git' }

    $list += New-Component -Id 'gh' -Name 'GitHub CLI' -Desc 'Work with GitHub from the terminal' -Requires @('winget') `
        -Detect { param($C) Set-SimpleDetection $C 'gh' @('--version') } `
        -Plan { Say '      winget install --id GitHub.cli --scope user'
                Say '      Log in afterwards with: gh auth login' } `
        -Install { Install-Winget -Id 'GitHub.cli' } `
        -Verify { Test-Have 'gh' }

    $list += New-Component -Id 'make' -Name 'GNU Make' -Desc 'Build automation for reproducible pipelines' -Requires @('winget') `
        -Detect { param($C) Set-SimpleDetection $C 'make' @('--version') } `
        -Plan { Say '      winget install --id ezwinports.make --scope user'
                Say '      Git Bash also ships a make; either satisfies the workshop material.' } `
        -Install { Install-Winget -Id 'ezwinports.make' } `
        -Verify { Test-Have 'make' }

    $list += New-Component -Id 'node' -Name 'Node.js' -Desc 'Runtime the Codex CLI depends on' -Requires @('winget') `
        -Detect { param($C)
            Set-SimpleDetection $C 'node' @('--version')
            # Codex requires Node 22+. A lower floor would report the machine as
            # ready and then fail at the Codex install instead.
            if ($C.Status -eq 'ok' -and -not (Compare-Version $C.Version '22.0')) {
                $C.Status = 'outdated'
                $C.Note = "Codex needs Node 22 or newer; upgrading is safe and does not remove $($C.Version)"
            } } `
        -Plan { Say '      winget install --id OpenJS.NodeJS.LTS --scope user' } `
        -Install { Install-Winget -Id 'OpenJS.NodeJS.LTS' } `
        -Verify { Test-Have 'node' }

    $list += New-Component -Id 'uv' -Name 'uv' -Desc 'Fast Python package and version manager' -Extras "Python $PY_TARGET" `
        -Detect { param($C)
            if (Test-Have 'uv') { Set-SimpleDetection $C 'uv' @('--version'); return }
            # An existing Python is worth reporting but is NOT a conflict: uv
            # installs its own interpreters and leaves every other Python alone.
            # Treating it as a conflict would make -All skip uv on any machine
            # that already has a python, which is most of them.
            $C.Status = 'missing'
            if (Test-Have 'python') {
                $C.Version = Get-FirstVersion -Name 'python' -VersionArgs @('--version')
                $C.CPath = Get-CmdPath 'python'
                $C.Note = if (Compare-Version $C.Version $PY_MIN) {
                    "your Python $($C.Version) stays as your default"
                } else {
                    "your Python $($C.Version) is older than the $PY_MIN the workshop needs; uv will install $PY_TARGET alongside it"
                }
            } } `
        -Plan { Say '      winget install --id astral-sh.uv --scope user'
                Say "      uv python install $PY_TARGET"
                Say '      Installs its own interpreters; your existing Python is untouched.' } `
        -Install { $ok = Install-Winget -Id 'astral-sh.uv'; Update-SessionPath; $ok } `
        -ExtrasAction { Invoke-Step "uv python install $PY_TARGET" { & uv python install $PY_TARGET } } `
        -Verify { Test-Have 'uv' }

    $list += New-Component -Id 'r' -Name 'R' -Desc 'Statistical computing environment' -Requires @('winget') -Extras 'renv' `
        -Detect { param($C)
            if (Test-Have 'R') { Set-SimpleDetection $C 'R' @('--version'); return }
            $found = Get-ChildItem 'C:\Program Files\R' -Directory -ErrorAction SilentlyContinue |
                     Sort-Object Name -Descending | Select-Object -First 1
            if ($found) {
                $C.Status = 'ok'; $C.CPath = $found.FullName
                $C.Version = ($found.Name -replace '[^0-9.]', '')
                $C.Note = 'installed but not on PATH'
            } else { $C.Status = 'missing' } } `
        -Plan { Say '      winget install --id RProject.R --scope user'
                Say '      renv (asked separately) pins package versions per project.' } `
        -Install { $ok = Install-Winget -Id 'RProject.R'; Update-SessionPath; $ok } `
        -ExtrasAction {
            Invoke-Step 'Rscript -e install.packages("renv")' {
                & Rscript -e 'if (!requireNamespace("renv", quietly=TRUE)) install.packages("renv", repos="https://cloud.r-project.org")'
            } } `
        -Verify { (Test-Have 'R') -or (Test-Have 'Rscript') -or (Test-Path 'C:\Program Files\R') }

    $list += New-Component -Id 'julia' -Name 'Julia' -Desc 'Numerical computing language' -Requires @('winget') `
        -Detect { param($C) Set-SimpleDetection $C 'julia' @('--version') } `
        -Plan { Say '      winget install --id Julialang.Juliaup --scope user'
                Say '      juliaup manages Julia versions side by side.' } `
        -Install { $ok = Install-Winget -Id 'Julialang.Juliaup'; Update-SessionPath; $ok } `
        -Verify { (Test-Have 'julia') -or (Test-Have 'juliaup') }

    $list += New-Component -Id 'quarto' -Name 'Quarto' -Desc 'Publishing system for notebooks and papers' -Requires @('winget') `
        -Detect { param($C) Set-SimpleDetection $C 'quarto' @('--version') } `
        -Plan { Say '      winget install --id Posit.Quarto --scope user' } `
        -Install { Install-Winget -Id 'Posit.Quarto' } `
        -Verify { Test-Have 'quarto' }

    $list += New-Component -Id 'cursor' -Name 'Cursor' -Desc 'AI-native editor (free for students)' -Requires @('winget') -Extras 'editor extensions' `
        -Detect { param($C)
            $exe = Get-ChildItem "$env:LOCALAPPDATA\Programs\cursor\Cursor.exe" -ErrorAction SilentlyContinue
            if (Test-Have 'cursor') { $C.Status = 'ok'; $C.CPath = Get-CmdPath 'cursor' }
            elseif ($exe) { $C.Status = 'ok'; $C.CPath = $exe.FullName }
            else { $C.Status = 'missing'; return }
            $settings = "$env:APPDATA\Cursor\User\settings.json"
            if (Test-Path $settings) { $C.Note = 'existing settings.json will be merged, not replaced' } } `
        -Plan { Say '      winget install --id Anysphere.Cursor --scope user'
                Say '      Extensions (asked separately): Python, Jupyter, R, Julia, Quarto, Ruff' } `
        -Install { $ok = Install-Winget -Id 'Anysphere.Cursor'; Update-SessionPath; $ok } `
        -ExtrasAction { Install-CursorExtras } `
        -Verify { (Test-Have 'cursor') -or (Test-Path "$env:LOCALAPPDATA\Programs\cursor\Cursor.exe") }

    $list += New-Component -Id 'claude' -Name 'Claude Code' -Desc "Anthropic's terminal coding agent" `
        -Detect { param($C) Set-SimpleDetection $C 'claude' @('--version') } `
        -Plan { Say '      irm https://claude.ai/install.ps1 | iex'
                Say '      Native Windows install; WSL is not required.'
                Say '      The free tier cannot use browser sign-in - you need'
                Say '      Pro/Max/Team, or an ANTHROPIC_API_KEY.' } `
        -Install {
            $ok = Invoke-Step 'irm https://claude.ai/install.ps1 | iex' {
                Invoke-Expression (Invoke-RestMethod -Uri 'https://claude.ai/install.ps1')
            }
            Update-SessionPath; $ok } `
        -Verify { Test-Have 'claude' }

    $list += New-Component -Id 'codex' -Name 'Codex CLI' -Desc "OpenAI's terminal coding agent" -Requires @('node') `
        -Detect { param($C) Set-SimpleDetection $C 'codex' @('--version') } `
        -Plan { Say '      npm install -g @openai/codex'
                Say '      The scoped name matters: plain "codex" is a different package.' } `
        -Install {
            $ok = Invoke-Step 'npm install -g @openai/codex' { & npm install -g '@openai/codex' }
            Update-SessionPath; $ok } `
        -Verify { Test-Have 'codex' }

    $list += New-Component -Id 'omp' -Name 'oh-my-posh' -Desc 'Informative shell prompt (git branch, env, timing)' -Requires @('winget') -Extras 'prompt theme' `
        -Detect { param($C) Set-SimpleDetection $C 'oh-my-posh' @('--version') } `
        -Plan { Say '      winget install --id JanDeDobbeleer.OhMyPosh --scope user'
                Say "      The theme is asked separately, since it edits $PROFILE." } `
        -Install { $ok = Install-Winget -Id 'JanDeDobbeleer.OhMyPosh'; Update-SessionPath; $ok } `
        -ExtrasAction { Install-PromptTheme } `
        -Verify { Test-Have 'oh-my-posh' }

    return $list
}

# --- Cursor configuration ---------------------------------------------------

function Get-CursorCli {
    foreach ($candidate in @(
        "$env:LOCALAPPDATA\Programs\cursor\resources\app\bin\cursor.cmd",
        "$env:LOCALAPPDATA\Programs\cursor\resources\app\bin\cursor"
    )) { if (Test-Path $candidate) { return $candidate } }
    if (Test-Have 'cursor') { return (Get-CmdPath 'cursor') }
    return $null
}

function Install-CursorExtras {
    $cli = Get-CursorCli
    if (-not $cli) { Say-Warn "Cursor's command line helper was not found; skipping extensions."; return $false }

    $extFile = Join-Path $script:AssetDir 'cursor-extensions.txt'
    foreach ($line in Get-Content -LiteralPath $extFile) {
        $ext = $line.Trim()
        if (-not $ext -or $ext.StartsWith('#')) { continue }
        Say "      installing extension: $ext"
        Invoke-Step "cursor --install-extension $ext" { & $cli --install-extension $ext --force | Out-Null } | Out-Null
    }
    Merge-CursorSettings
    return $true
}

# Adds our keys to an existing settings.json without disturbing the user's own.
# Any key they have already set wins. Clobbering someone's editor config is the
# worst thing this installer could do, so it is the most defensive path here.
function Merge-CursorSettings {
    $target = "$env:APPDATA\Cursor\User\settings.json"
    $source = Join-Path $script:AssetDir 'cursor-settings.json'

    if (-not (Test-Path $target) -or -not (Get-Content -LiteralPath $target -Raw).Trim()) {
        if ($script:DryRun) { Say "      would write: $target" 'DarkGray'; return }
        $dir = Split-Path -Parent $target
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Copy-Item -LiteralPath $source -Destination $target -Force
        Say-Ok 'Wrote Cursor settings.'
        return
    }

    $bak = Backup-UserFile $target
    if ($bak) { Say-Info "Backed up your settings to $bak" }
    if ($script:DryRun) { Say "      would merge our keys into $target" 'DarkGray'; return }

    try {
        # VS Code-family settings allow comments and trailing commas; strip both
        # before parsing so a legal file does not look corrupt to us.
        $raw = Get-Content -LiteralPath $target -Raw
        $raw = [regex]::Replace($raw, '(?m)^\s*//.*$', '')
        $raw = [regex]::Replace($raw, ',(\s*[}\]])', '$1')
        $current = ConvertFrom-JsonToHashtable $raw
        $ours = ConvertFrom-JsonToHashtable (Get-Content -LiteralPath $source -Raw)

        $added = @()
        foreach ($key in $ours.Keys) {
            if (-not $current.ContainsKey($key)) { $current[$key] = $ours[$key]; $added += $key }
        }
        $current | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $target
        foreach ($key in $added) { Say "      added $key" }
        Say-Ok 'Merged Cursor settings (your existing keys were kept).'
    } catch {
        Say-Warn "Could not merge settings; your file is unchanged. ($($_.Exception.Message))"
    }
}

function Install-PromptTheme {
    $themeDir = "$env:USERPROFILE\.config\econ-ai"
    $theme = Join-Path $themeDir 'econ-ai.omp.json'
    if (-not $script:DryRun) {
        if (-not (Test-Path $themeDir)) { New-Item -ItemType Directory -Path $themeDir -Force | Out-Null }
        Copy-Item -LiteralPath (Join-Path $script:AssetDir 'econ-ai.omp.json') -Destination $theme -Force
    }
    Set-ProfileBlock -Marker 'econ-ai-setup prompt' -Content @"
if (Get-Command oh-my-posh -ErrorAction SilentlyContinue) {
    oh-my-posh init pwsh --config '$theme' | Invoke-Expression
}
"@
    Say-Info "Prompt configured in $PROFILE. Restart your terminal to see it."
    return $true
}

Export-ModuleMember -Function *
