#!/usr/bin/env bash
#
# Runs each of the six per-site snow-masked final-comparison scripts, which each
# save a two-axes NDVI panel and a two-axes CCI panel PNG to:
#   <plot_dir>/SnowMasked/panels/<site_name>_NDVI_panel.png
#   <plot_dir>/SnowMasked/panels/<site_name>_CCI_panel.png
#
# Run this first, then run 06_plot_final_compare_panels_SnowMasked.py to assemble the
# panels into the multi-panel summary figure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "06ab_compare_Photospec_MCD19_etc_DEJU_timeseries_SnowMasked_batch.py"
    "06cd_compare_Photospec_MCD19_etc_Ca-Obs_timeseries_SnowMasked_batch.py"
    "06ef_compare_Photospec_MCD19_etc_US-UMB_timeseries_SnowMasked_batch.py"
    "06gh_compare_Photospec_MCD19_etc_US-NR1_timeseries_SnowMasked_batch.py"
    "06ij_compare_FLOX_MCD19_etc_US-Ne3_timeseries_SnowMasked_batch.py"
    "06kl_compare_Photospec_MCD19_etc_OSBS_timeseries_SnowMasked_batch.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "=== Running ${script} ==="
    python3 "${SCRIPT_DIR}/${script}"
done

echo "=== All snow-masked final comparison scripts finished ==="
