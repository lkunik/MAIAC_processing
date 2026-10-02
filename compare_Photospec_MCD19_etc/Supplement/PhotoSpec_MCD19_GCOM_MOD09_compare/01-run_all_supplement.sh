#!/usr/bin/env bash
#
# Runs each of the six per-site supplement scripts, which each save one
# panel PNG per comparison product and index to:
#   <plot_dir>/panels/<site_name>_<NDVI|CCI>_<product>_panel.png
# (01d DEJU also saves the standalone legend PNGs.)
#
# Run this first, then run 02_plot_supplement_product_panels.py to assemble
# the panels into the multi-panel supplement figures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "01a_compare_MCD19_GCOM_MOD09_US-NR1.py"
    "01b_compare_MCD19_GCOM_MOD09_OSBS.py"
    "01c_compare_MCD19_GCOM_MOD09_Ca-Obs.py"
    "01d_compare_MCD19_GCOM_MOD09_DEJU.py"
    "01e_compare_MCD19_GCOM_MOD09_US-UMB.py"
    "01f_compare_MCD19_GCOM_MOD09_US-Ne3.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "=== Running ${script} ==="
    python3 "${SCRIPT_DIR}/${script}"
done

echo "=== All supplement scripts finished ==="
