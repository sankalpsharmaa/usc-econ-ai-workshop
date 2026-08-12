#!/usr/bin/env bash
# doctor.sh --- report what is installed. Changes nothing, ever.
#
# Deliberately a thin wrapper: doctor mode is the installer's scan phase with an
# early exit, so there is exactly one detection code path to keep correct.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.sh" --check "$@"
