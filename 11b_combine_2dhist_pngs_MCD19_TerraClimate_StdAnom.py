#!/usr/bin/env python3

#%%
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

#%%
# Configuration
avg_window_abbr = "JJA"
base_dir = Path(f"/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_annual/QCfilt_CloudFree/std_anomaly_{avg_window_abbr}")


# Variables to process
variables = ['CWD', 'PDSI', 'PET', 'SoilM', 'VPD']

# VI order: top-left, top-right, bottom-left, bottom-right
vi_order = ['CCI', 'NDVI', 'NDMI', 'NIRv']

#%%
for var in variables:
    var_dir = base_dir / var
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Load and plot each image
    for idx, vi in enumerate(vi_order):
        img_path = var_dir / f"TerraClimate_{var}_vs_{vi}_std_anomaly_{avg_window_abbr}.png"
        
        if img_path.exists():
            img = imread(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
        else:
            print(f"Warning: {img_path} not found")
    
    # Remove spacing between subplots
    plt.tight_layout(pad=0.1)
    
    # Save combined figure
    output_path = base_dir / f"TerraClimate_{var}_vs_MCD19_std_anomaly_{avg_window_abbr}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")

print("Done!")
# %%

#!/usr/bin/env python3

#%%
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

#%%
# Configuration
avg_window_abbr = "JJA"
base_dir = Path(f"/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_annual/QCfilt_CloudFree/std_anomaly_{avg_window_abbr}")

# Variables to process (rows, top to bottom)
variables = ['CWD', 'PDSI', 'SoilM', 'PET']

# VI order (columns, left to right)
vi_order = ['CCI', 'NDVI', 'NDMI', 'NIRv']

#%%
# Create figure with 4x4 subplots
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# Load and plot each image
for row_idx, var in enumerate(variables):
    var_dir = base_dir / var
    
    for col_idx, vi in enumerate(vi_order):
        img_path = var_dir / f"TerraClimate_{var}_vs_{vi}_std_anomaly_{avg_window_abbr}.png"
        
        if img_path.exists():
            img = imread(img_path)
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')
        else:
            print(f"Warning: {img_path} not found")

# Remove spacing between subplots
plt.tight_layout(pad=0.1)

# Save combined figure
output_path = base_dir / f"TerraClimate_all_vars_vs_MCD19_std_anomaly_{avg_window_abbr}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Created: {output_path}")
print("Done!")
# %%