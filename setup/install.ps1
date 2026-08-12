# install.ps1 --- Windows bootstrap for the USC Economics AI workshop.
#
# Run it like this, in PowerShell:
#
#     irm https://raw.githubusercontent.com/OWNER/REPO/REF/setup/install.ps1 | iex
#
# Unlike the macOS side, piping is fine here: `| iex` executes in your live
# session, so prompts still reach your keyboard.
#
# To pass options, download first and then call it:
#
#     irm https://.../install.ps1 -OutFile install.ps1
#     .\install.ps1 -All
#
# All this file does is fetch the setup directory and hand off to bin/setup.ps1.

[CmdletBinding()]
param(
    [switch]$All,
    [string]$Only = '',
    [string]$Skip = '',
    [switch]$Check,
    [switch]$DryRun,
    [switch]$Yes,
    [string]$LogPath = ''
)

$ErrorActionPreference = 'Stop'

$Repo = if ($env:ECON_AI_REPO) { $env:ECON_AI_REPO } else { 'sankalpsharmaa/usc-econ-ai-workshop' }
$Ref  = if ($env:ECON_AI_REF)  { $env:ECON_AI_REF }  else { 'jyl' }

function Die { param([string]$Message) Write-Host "error: $Message" -ForegroundColor Red; exit 1 }

if ($PSVersionTable.PSVersion.Major -ge 6 -and -not $IsWindows) {
    Die 'This is the Windows installer. On macOS run install.sh instead.'
}

# TLS 1.2 for Windows PowerShell 5.1, which does not negotiate it by default
# and otherwise fails against GitHub with an opaque connection error.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$work = Join-Path ([IO.Path]::GetTempPath()) ("econ-ai-setup-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $work -Force | Out-Null

try {
    Write-Host "Fetching the setup files ($Repo @ $Ref)..." -ForegroundColor DarkGray

    $tarUrl = "https://codeload.github.com/$Repo/zip/$Ref"
    $zip = Join-Path $work 'src.zip'
    try {
        Invoke-WebRequest -Uri $tarUrl -OutFile $zip -UseBasicParsing -TimeoutSec 120
    } catch {
        Die "Could not download $tarUrl`n  Check your network, and that the branch or tag '$Ref' exists."
    }

    Expand-Archive -LiteralPath $zip -DestinationPath $work -Force

    # The archive's top folder is named <repo>-<ref>, so find it rather than
    # guessing --- refs containing a slash get rewritten in that name.
    $setupRoot = Get-ChildItem -Path $work -Recurse -Directory -Filter 'setup' -Depth 2 |
                 Where-Object { Test-Path (Join-Path $_.FullName 'bin\setup.ps1') } |
                 Select-Object -First 1

    if (-not $setupRoot) { Die 'The download did not contain setup\bin\setup.ps1.' }

    $env:SETUP_ROOT = $setupRoot.FullName

    $forward = @{}
    if ($All)     { $forward['All'] = $true }
    if ($Check)   { $forward['Check'] = $true }
    if ($DryRun)  { $forward['DryRun'] = $true }
    if ($Yes)     { $forward['Yes'] = $true }
    if ($Only)    { $forward['Only'] = $Only }
    if ($Skip)    { $forward['Skip'] = $Skip }
    if ($LogPath) { $forward['LogPath'] = $LogPath }

    & (Join-Path $setupRoot.FullName 'bin\setup.ps1') @forward
}
finally {
    # Keep the temp copy when something failed, so the log and scripts survive
    # long enough to diagnose the problem.
    if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}
