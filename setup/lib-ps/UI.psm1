# UI.psm1 --- console I/O primitives for the Windows installer.
#
# Unlike the bash side, PowerShell's `irm <url> | iex` runs in the caller's live
# session, so Read-Host reaches the keyboard normally and there is no /dev/tty
# equivalent to work around. The guard here is for non-interactive hosts
# (scheduled tasks, CI, some remoting sessions) where prompting would hang.

$script:LogFile = $null
$script:AssumeYes = $false

function Set-UiLogFile { param([string]$Path) $script:LogFile = $Path }
function Set-UiAssumeYes { param([bool]$Value) $script:AssumeYes = $Value }
function Get-UiAssumeYes { $script:AssumeYes }

function Write-LogLine {
    param([string]$Text)
    if ($script:LogFile) { Add-Content -LiteralPath $script:LogFile -Value $Text }
}

function Say {
    param([string]$Text = '', [string]$Color = $null)
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
    Write-LogLine $Text
}

function Say-Info { param([string]$T) Say "  i  $T" 'Cyan' }
function Say-Ok   { param([string]$T) Say "  $([char]0x2713)  $T" 'Green' }
function Say-Warn { param([string]$T) Say "  !  $T" 'Yellow' }
function Say-Err  { param([string]$T) Say "  x  $T" 'Red' }
function Say-Step { param([string]$T) Say ''; Say $T 'White' }

function Say-Rule {
    $width = 78
    try { $width = [Math]::Min(78, $Host.UI.RawUI.WindowSize.Width - 2) } catch {}
    if ($width -lt 20) { $width = 78 }
    Say ([string]([char]0x2500) * $width) 'DarkGray'
}

function Show-Banner {
    Say ''
    Say '  USC Economics - AI Research Environment Setup' 'Cyan'
    Say '  Installs the toolchain for the agentic-AI workshop sessions.' 'DarkGray'
    Say ''
}

# Non-interactive hosts cannot answer prompts. Detect and explain rather than
# blocking forever on Read-Host.
function Test-Interactive {
    if ($script:AssumeYes) { return $true }
    try {
        if ([Environment]::UserInteractive -eq $false) { return $false }
        $null = $Host.UI.RawUI.KeyAvailable
        return $true
    } catch { return $false }
}

function Assert-Interactive {
    if (Test-Interactive) { return }
    Say-Err 'This installer is interactive, but this session cannot accept input.'
    Say ''
    Say '  Run it from an interactive PowerShell window, or install everything'
    Say '  without prompts by adding -All:'
    Say ''
    Say '      irm <url> | iex; econ-ai-setup -All' 'White'
    Say ''
    exit 2
}

# Ask-YesNo -Question 'Install R?' -Default 'y'
function Ask-YesNo {
    param(
        [Parameter(Mandatory)][string]$Question,
        [ValidateSet('y','n')][string]$Default = 'n'
    )
    if ($script:AssumeYes) { return ($Default -eq 'y') }
    Assert-Interactive

    $hint = if ($Default -eq 'y') { '[Y/n]' } else { '[y/N]' }
    while ($true) {
        $reply = (Read-Host "  $Question $hint").Trim().ToLower()
        if ([string]::IsNullOrWhiteSpace($reply)) { $reply = $Default }
        Write-LogLine "  $Question $hint -> $reply"
        switch ($reply) {
            { $_ -in 'y','yes' } { return $true }
            { $_ -in 'n','no'  } { return $false }
            default { Say '  Please answer y or n.' 'Yellow' }
        }
    }
}

# Ask-Choice -Prompt 'Choose' -Default 'a' -Options @(@{Key='a';Label='...'}, ...)
function Ask-Choice {
    param(
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][string]$Default,
        [Parameter(Mandatory)][array]$Options
    )
    Say ''
    foreach ($opt in $Options) {
        $suffix = if ($opt.Key -eq $Default) { '  (recommended)' } else { '' }
        Say ("    [{0}] {1}{2}" -f $opt.Key, $opt.Label, $suffix)
    }
    Say ''

    if ($script:AssumeYes) { return $Default }
    Assert-Interactive

    $keys = $Options | ForEach-Object { $_.Key.ToLower() }
    while ($true) {
        $reply = (Read-Host "  $Prompt [$Default]").Trim().ToLower()
        if ([string]::IsNullOrWhiteSpace($reply)) { $reply = $Default }
        Write-LogLine "  $Prompt -> $reply"
        if ($keys -contains $reply) { return $reply }
        Say '  Not one of the options.' 'Yellow'
    }
}

# --- Status table -----------------------------------------------------------
function Show-TableHeader {
    Say ''
    Show-TableRow -Name 'COMPONENT' -Status 'STATUS' -Detail 'FOUND'
    Say-Rule
}

function Show-TableRow {
    param([string]$Name, [string]$Status, [string]$Detail)
    Say ("  {0,-18}{1,-12}{2}" -f $Name, $Status, $Detail)
}

function Format-StatusWord {
    param([string]$Status)
    switch ($Status) {
        'ok'       { 'ok' }
        'missing'  { 'missing' }
        'conflict' { 'conflict' }
        'outdated' { 'outdated' }
        'failed'   { 'failed' }
        'skipped'  { 'skipped' }
        default    { $Status }
    }
}

Export-ModuleMember -Function *
