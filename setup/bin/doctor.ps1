# doctor.ps1 --- report what is installed. Changes nothing, ever.
#
# Deliberately a thin wrapper: doctor mode is the installer's scan phase with an
# early exit, so there is exactly one detection code path to keep correct.
& (Join-Path $PSScriptRoot 'setup.ps1') -Check @args
