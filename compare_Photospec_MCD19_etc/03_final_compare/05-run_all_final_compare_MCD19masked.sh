#!/usr/bin/env bash
#
# Runs each of the six per-site MCD19-masked final-comparison scripts, which each
# save a two-axes NDVI panel and a two-axes CCI panel PNG to:
#   <plot_dir>/MCD19masked/panels/<site_name>_NDVI_panel.png
#   <plot_dir>/MCD19masked/panels/<site_name>_CCI_panel.png
#
# Run this first, then run 05_plot_final_compare_panels_MCD19masked.py to assemble the
# panels into the multi-panel summary figure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "05ab_compare_Photospec_MCD19_etc_DEJU_timeseries_MCD19masked_batch.py"
    "05cd_compare_Photospec_MCD19_etc_Ca-Obs_timeseries_MCD19masked_batch.py"
    "05ef_compare_Photospec_MCD19_etc_US-UMB_timeseries_MCD19masked_batch.py"
    "05gh_compare_Photospec_MCD19_etc_US-NR1_timeseries_MCD19masked_batch.py"
    "05ij_compare_FLOX_MCD19_etc_US-Ne3_timeseries_MCD19masked_batch.py"
    "05kl_compare_Photospec_MCD19_etc_OSBS_timeseries_MCD19masked_batch.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "=== Running ${script} ==="
    python3 "${SCRIPT_DIR}/${script}"
done

echo "=== All MCD19-masked final comparison scripts finished ==="
