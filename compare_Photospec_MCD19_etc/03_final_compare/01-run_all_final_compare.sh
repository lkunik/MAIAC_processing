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
    "01ab_compare_Photospec_MCD19_etc_DEJU_timeseries_batch.py"
    "01cd_compare_Photospec_MCD19_etc_Ca-Obs_timeseries_batch.py"
    "01ef_compare_Photospec_MCD19_etc_US-UMB_timeseries_batch.py"
    "01gh_compare_Photospec_MCD19_etc_US-NR1_timeseries_batch.py"
    "01ij_compare_FLOX_MCD19_etc_US-Ne3_timeseries_batch.py"
    "01kl_compare_Photospec_MCD19_etc_OSBS_timeseries_batch.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "=== Running ${script} ==="
    python3 "${SCRIPT_DIR}/${script}"
done

echo "=== All QC comparison scripts finished ==="
