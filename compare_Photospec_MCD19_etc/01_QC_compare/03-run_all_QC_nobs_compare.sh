#!/usr/bin/env bash
#
# Runs each of the six per-site MCD19 QC-comparison scripts, which each
# save a two-axes NDVI panel and a two-axes CCI panel PNG to:
#   <plot_dir>/panels/<site_name>_NDVI_panel.png
#   <plot_dir>/panels/<site_name>_CCI_panel.png
#
# Run this first, then run 03_plot_QC_compare_panels.py to assemble the
# panels into the multi-panel summary figure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "03a_compare_MCD19_nobs_QC_options_DEJU.py"
    "03b_compare_MCD19_nobs_QC_options_Ca-Obs.py"
    "03c_compare_MCD19_nobs_QC_options_US-UMB.py"
    "03d_compare_MCD19_nobs_QC_options_US-NR1.py"
    "03e_compare_MCD19_nobs_QC_options_US-Ne3.py"
    "03f_compare_MCD19_nobs_QC_options_OSBS.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "=== Running ${script} ==="
    python3 "${SCRIPT_DIR}/${script}"
done

echo "=== All QC comparison scripts finished ==="
