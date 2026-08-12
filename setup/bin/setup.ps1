# setup.ps1 --- Windows orchestrator for the workshop environment installer.
#
# Mirrors the macOS flow: preflight, scan, selection, conflict confirmation,
# plan review, execute, report. Nothing mutates the machine before Phase 5,
# and Phase 4 is the last chance to back out.

[CmdletBinding()]
param(
    [switch]$All,        # install everything missing, no per-component prompts
    [string]$Only = '',  # comma-separated component ids
    [string]$Skip = '',
    [switch]$Check,      # report and exit, changing nothing
    [switch]$DryRun,     # walk the flow, print commands, change nothing
    [switch]$Yes,        # accept the default at every prompt
    [string]$LogPath = ''
)

$ErrorActionPreference = 'Continue'   # one failed component must not abort the
                                      # rest; failures are collected instead

$SetupRoot = if ($env:SETUP_ROOT) { $env:SETUP_ROOT }
             else { Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$AssetDir = Join-Path $SetupRoot 'assets'

Import-Module (Join-Path $SetupRoot 'lib-ps\UI.psm1') -Force
Import-Module (Join-Path $SetupRoot 'lib-ps\Components.psm1') -Force

# --- Log --------------------------------------------------------------------
if (-not $LogPath -and -not $Check) {
    $dir = Join-Path $env:USERPROFILE '.econ-ai-setup'
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $LogPath = Join-Path $dir ("install-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
if ($LogPath) { Set-Content -LiteralPath $LogPath -Value '' }

Set-UiLogFile $LogPath
Set-UiAssumeYes ([bool]($Yes -or $All))
Set-ComponentContext -DryRun ([bool]$DryRun) -AssetDir $AssetDir

$Components = Get-ComponentCatalogue
$OnlyIds = @($Only -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$SkipIds = @($Skip -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })

function Get-Component { param([string]$Id) $Components | Where-Object { $_.Id -eq $Id } | Select-Object -First 1 }

function Test-FilteredIn {
    param($C)
    if ($SkipIds -contains $C.Id) { return $false }
    if ($OnlyIds.Count -gt 0) { return ($OnlyIds -contains $C.Id) }
    return $true
}

# ===========================================================================
# Phase 0 --- Preflight
# ===========================================================================
function Invoke-Preflight {
    if (-not $IsWindows -and $PSVersionTable.PSVersion.Major -ge 6) {
        Say-Err 'This script is for Windows. On macOS run install.sh instead.'
        exit 1
    }

    $build = [Environment]::OSVersion.Version.Build
    if ($build -lt 17763) {
        Say-Err "Windows build $build is too old. Claude Code needs Windows 10 1809 or newer."
        exit 1
    }

    if (-not $Check -and -not $All -and -not $Yes) { Assert-Interactive }

    try {
        $free = (Get-PSDrive -Name ($env:SystemDrive.TrimEnd(':')) -ErrorAction Stop).Free / 1GB
        if ($free -lt 10) {
            Say-Warn ("Only {0:N1} GB free. A full install needs roughly 10 GB." -f $free)
            if (-not (Ask-YesNo -Question 'Continue anyway?' -Default 'n')) { exit 1 }
        }
    } catch {}

    try {
        $null = Invoke-WebRequest -Uri 'https://github.com' -UseBasicParsing -TimeoutSec 10 -Method Head
    } catch {
        Say-Err 'Cannot reach github.com. Check your network, VPN, or proxy.'
        exit 1
    }

    Update-SessionPath
}

# ===========================================================================
# Phase 1 --- Scan. Also the whole of -Check.
# ===========================================================================
function Invoke-Scan {
    Say-Step 'Checking what you already have'
    foreach ($c in $Components) {
        $c.Status = 'missing'; $c.Version = ''; $c.CPath = ''; $c.Note = ''
        try { & $c.Detect $c } catch { $c.Status = 'missing' }
    }

    Show-TableHeader
    foreach ($c in $Components) {
        if (-not (Test-FilteredIn $c)) { continue }
        $detail = if ($c.Version -and $c.CPath) { "{0,-9} {1}" -f $c.Version, $c.CPath }
                  elseif ($c.Version) { $c.Version }
                  elseif ($c.CPath) { $c.CPath }
                  else { '-' }
        Show-TableRow -Name $c.Name -Status (Format-StatusWord $c.Status) -Detail $detail
        if ($c.Note) { Say ("  {0}{1} {2}" -f (' ' * 31), [char]0x21B3, $c.Note) 'DarkGray' }
    }
    Say ''
}

# ===========================================================================
# Phase 3 --- Conflict confirmation (called from selection)
# ===========================================================================
function Confirm-Conflict {
    param($C)
    Say ''
    Say "  !  $($C.Name) - something is already here" 'Yellow'
    Say "     Found:    $($C.Version)  $($C.CPath)"
    if ($C.Note) { Say "     $($C.Note)" 'DarkGray' }
    Say ''
    Say '     Proposed:'
    & $C.Plan

    if ($All) {
        # -All means "install what is missing", never "overwrite what is here".
        # Anything ambiguous is left alone unless a human says otherwise.
        $C.Action = 'keep'
        Say-Info "Leaving $($C.Name) as it is (-All does not overwrite)."
        return
    }

    $answer = Ask-Choice -Prompt 'What would you like to do?' -Default 'i' -Options @(
        @{ Key = 'i'; Label = 'Install ours alongside - your existing copy is left in place' },
        @{ Key = 'k'; Label = 'Keep only what you already have' },
        @{ Key = 's'; Label = 'Skip this component entirely' }
    )
    $C.Action = if ($answer -eq 'i') { 'install' } else { 'keep' }
}

# ===========================================================================
# Phase 2 --- Selection
# ===========================================================================
function Select-One {
    param($C)

    if ($C.Status -eq 'ok') { $C.Action = 'skip'; return }

    if ($C.Status -in @('conflict', 'outdated')) {
        Confirm-Conflict $C
    }
    elseif ($All) { $C.Action = 'install' }
    else {
        Say ''
        Say "  $($C.Name) - $($C.Desc)"
        $answer = Ask-Choice -Prompt "Install $($C.Name)?" -Default 'y' -Options @(
            @{ Key = 'y'; Label = 'Yes, install it' },
            @{ Key = 'n'; Label = 'No, skip it' },
            @{ Key = '?'; Label = 'Show exactly what this will do' }
        )
        if ($answer -eq '?') {
            Say ''
            & $C.Plan
            $answer = Ask-Choice -Prompt "Install $($C.Name)?" -Default 'y' -Options @(
                @{ Key = 'y'; Label = 'Yes, install it' },
                @{ Key = 'n'; Label = 'No, skip it' }
            )
        }
        $C.Action = if ($answer -eq 'y') { 'install' } else { 'skip' }
    }

    # Extras are always a separate question - "R and its associated tooling"
    # should not be a single yes.
    if ($C.Extras -and $C.Action -eq 'install') {
        $C.ExtrasWanted = if ($All) { $true } else { Ask-YesNo -Question "Also set up $($C.Extras)?" -Default 'y' }
    }
}

function Invoke-Selection {
    Say-Step 'What would you like to install?'

    $mode = if ($All) { 'a' } else {
        Ask-Choice -Prompt 'Choose' -Default 'a' -Options @(
            @{ Key = 'a'; Label = 'Install everything missing' },
            @{ Key = 's'; Label = 'Choose component by component' },
            @{ Key = 'd'; Label = 'Just show me the report, change nothing' },
            @{ Key = 'q'; Label = 'Quit' }
        )
    }

    switch ($mode) {
        'q' { Say ''; Say-Info 'Nothing was changed.'; exit 0 }
        'd' { return $false }
        'a' { $script:All = $true; Set-UiAssumeYes $true }
    }

    foreach ($c in $Components) {
        if (-not (Test-FilteredIn $c)) { $c.Action = 'skip'; continue }
        Select-One $c
    }

    # A component whose prerequisite was declined cannot proceed. Catch it here
    # rather than letting it fail deep into the install.
    foreach ($c in $Components) {
        if ($c.Action -ne 'install') { continue }
        foreach ($depId in $c.Requires) {
            $dep = Get-Component $depId
            if (-not $dep) { continue }
            if ($dep.Status -in @('ok', 'outdated') -or $dep.Action -eq 'install') { continue }
            Say-Warn "$($c.Name) needs $($dep.Name), which you skipped. Dropping it."
            $c.Action = 'skip'
            break
        }
    }
    return $true
}

# ===========================================================================
# Phase 4 --- Plan review
# ===========================================================================
function Invoke-Review {
    Say-Step 'Here is exactly what will happen'
    $queued = @($Components | Where-Object { $_.Action -eq 'install' })

    foreach ($c in $queued) {
        $extra = if ($c.ExtrasWanted) { "  + $($c.Extras)" } else { '' }
        Say "  + $($c.Name)$extra" 'Green'
    }

    if ($queued.Count -eq 0) {
        Say ''
        Say-Ok 'Nothing to do - you are already set up.'
        return $false
    }

    Say ''
    Say "  Log: $LogPath" 'DarkGray'
    Say ''

    if ($DryRun) {
        Say-Step 'Dry run - commands that would be executed'
        foreach ($c in $queued) { Say "  $($c.Name)"; & $c.Plan }
        Say ''
        Say-Ok 'Dry run complete. Nothing was changed.'
        exit 0
    }

    if (-not (Ask-YesNo -Question 'Proceed?' -Default 'y')) {
        Say ''; Say-Info 'Nothing was changed.'; exit 0
    }
    return $true
}

# ===========================================================================
# Phase 5 --- Execute
# ===========================================================================
function Invoke-Execute {
    Say-Step 'Installing'
    foreach ($c in $Components) {
        if ($c.Action -ne 'install') { continue }
        Say ''
        Say "  $($c.Name)"

        $ok = $false
        try { $ok = [bool](& $c.Install) } catch { $ok = $false; Write-LogLine $_.Exception.Message }

        if (-not $ok) {
            Say-Err "$($c.Name) failed. See the log for details."
            $c.Result = 'failed'
            continue
        }

        if ($c.ExtrasWanted -and $c.ExtrasAction) {
            try { & $c.ExtrasAction | Out-Null }
            catch { Say-Warn "$($c.Name): the optional extra did not complete." }
        }

        Update-SessionPath
        $verified = $false
        try { $verified = [bool](& $c.Verify) } catch {}
        if ($verified) { Say-Ok "$($c.Name) installed." }
        else { Say-Warn "$($c.Name) installed but is not on PATH yet. A new terminal should fix it." }
        $c.Result = 'done'
    }
}

# ===========================================================================
# Phase 6 --- Report
# ===========================================================================
function Invoke-Report {
    Say-Step 'Where things stand'
    foreach ($c in $Components) {
        $c.Status = 'missing'; $c.Version = ''; $c.CPath = ''; $c.Note = ''
        try { & $c.Detect $c } catch {}
    }

    Show-TableHeader
    foreach ($c in $Components) {
        if (-not (Test-FilteredIn $c)) { continue }
        $status = if ($c.Result -eq 'failed') { 'failed' } else { $c.Status }
        $detail = if ($c.Version) { $c.Version } else { '-' }
        Show-TableRow -Name $c.Name -Status (Format-StatusWord $status) -Detail $detail
    }

    Say-Step 'Three things left, which only you can do'
    Say ''
    Say '  1. Restart your terminal - or open a new tab.'
    Say '     PATH and prompt changes do not apply to this window.'
    Say ''
    Say '  2. Sign in to each tool - run these one at a time:'
    Say ''
    if (Test-Have 'claude') { Say '       claude              then follow the browser prompt' 'Cyan' }
    Say '                           the free tier cannot use browser sign-in; you' 'DarkGray'
    Say '                           need Pro/Max/Team or an ANTHROPIC_API_KEY' 'DarkGray'
    if (Test-Have 'codex')  { Say '       codex               sign in with ChatGPT' 'Cyan' }
    Say '                           USC IT can grant access if you do not have it' 'DarkGray'
    if (Test-Have 'gh')     { Say '       gh auth login       choose HTTPS and browser sign-in' 'Cyan' }
    Say ''
    Say '  3. Add your API keys - copy the template and fill it in:'
    Say ''
    Say '       copy .env.example .env' 'Cyan'
    Say "     Neil's labelling demos read OPENAI_API_KEY from that file."
    Say ''
    Say-Rule

    $failed = @($Components | Where-Object { $_.Result -eq 'failed' })
    if ($failed.Count -gt 0) {
        Say ''
        Say-Err ("These did not install: " + (($failed | ForEach-Object { $_.Id }) -join ' '))
        Say "  Send $LogPath to the workshop organisers and we will sort it out."
        Say '  Re-run just the failures with:'
        Say ("      .\setup.ps1 -Only " + (($failed | ForEach-Object { $_.Id }) -join ',')) 'Cyan'
        Say ''
        return
    }

    Say ''
    Say-Ok 'You are ready for the workshop.'
    Say '  Check anything at any time with:  .\setup.ps1 -Check' 'DarkGray'
    Say ''
}

# ===========================================================================
Show-Banner
Invoke-Preflight
Invoke-Scan

if ($Check) {
    Say '  Report only - nothing was changed.' 'DarkGray'
    Say ''
    exit 0
}

if (-not (Invoke-Selection)) {
    Say '  Report only - nothing was changed.' 'DarkGray'
    exit 0
}

if (-not (Invoke-Review)) { exit 0 }
Invoke-Execute
Invoke-Report
