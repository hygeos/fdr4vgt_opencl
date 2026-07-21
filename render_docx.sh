#!/usr/bin/env bash
# Render atbd_extended.qmd to DOCX using the pixi/rattler Python environment,
# which has all required packages (matplotlib, xarray, nbformat, etc.).
set -e

PIXI_PY=$(pixi run which python 2>/dev/null)
if [ -z "$PIXI_PY" ]; then
    echo "ERROR: pixi is not available or not configured." >&2
    exit 1
fi

export QUARTO_PYTHON="$PIXI_PY"
echo "Using Python: $QUARTO_PYTHON"

cd "$(dirname "$0")/quarto"
quarto render atbd_extended.qmd --to docx "$@"

cd ..
"$QUARTO_PYTHON" scripts/fix_docx_equations.py quarto/atbd_extended.docx
