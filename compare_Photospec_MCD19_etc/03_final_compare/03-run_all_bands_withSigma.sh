#!/usr/bin/env bash
#
# Runs each of the six per-site MCD19 band-reflectance scripts, which each
# save a Band1 panel and a Band2 panel PNG to:
#   <plot_dir>/panels/<site_name>_Band1_sigma_panel.png
#   <plot_dir>/panels/<site_name>_Band2_sigma_panel.png
#
# Run this first, then run 03_plot_final_bands_panels_withSigma.py to
# assemble the panels into the multi-panel summary figure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "03ab_plot_MCD19_bands_withSigma_DEJU_timeseries_batch.py"
    "03cd_plot_MCD19_bands_withSigma_Ca-Obs_timeseries_batch.py"
    "03ef_plot_MCD19_bands_withSigma_US-UMB_timeseries_batch.py"
    "03gh_plot_MCD19_bands_withSigma_US-NR1_timeseries_batch.py"
    "03ij_plot_MCD19_bands_withSigma_US-Ne3_timeseries_batch.py"
    "03kl_plot_MCD19_bands_withSigma_OSBS_timeseries_batch.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "=== Running ${script} ==="
    python3 "${SCRIPT_DIR}/${script}"
done

echo "=== All band plotting scripts finished ==="
