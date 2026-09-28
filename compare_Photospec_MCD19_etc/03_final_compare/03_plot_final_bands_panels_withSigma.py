#%%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/final/'
panel_dir = os.path.join(plot_dir, 'panels')

# Top-to-bottom site order (matches panel letters a-l)
site_order = ['DEJU', 'Ca-Obs', 'US-UMB', 'US-NR1', 'US-Ne3', 'OSBS']

# Sites whose Band1 and Band2 panels each get their own full-width row
wide_sites = {'Ca-Obs'}

# Build row layout: each row is a list of 1 (full width) or 2 (paired) images
rows = []
for site_name in site_order:
    band1_img = mpimg.imread(os.path.join(panel_dir, f'{site_name}_Band1_sigma_panel.png'))
    band2_img = mpimg.imread(os.path.join(panel_dir, f'{site_name}_Band2_sigma_panel.png'))
    if site_name in wide_sites:
        rows.append([band1_img])
        rows.append([band2_img])
    else:
        rows.append([band1_img, band2_img])

# Row heights from image aspect ratios, in units of figure width
fig_width = 11
height_ratios = []
for imgs in rows:
    col_frac = 1 / len(imgs)
    height_ratios.append(max(img.shape[0] / img.shape[1] * col_frac for img in imgs))

fig = plt.figure(figsize=(fig_width, fig_width * sum(height_ratios)))
gs = GridSpec(len(rows), 2, figure=fig, height_ratios=height_ratios,
              wspace=0.02, hspace=0.05, left=0, right=1, bottom=0, top=1)

for r, imgs in enumerate(rows):
    if len(imgs) == 1:
        ax = fig.add_subplot(gs[r, :])
        ax.imshow(imgs[0])
        ax.set_anchor('C')
        ax.axis('off')
    else:
        for c, img in enumerate(imgs):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            ax.set_anchor('E' if c == 0 else 'W')  # push paired images toward center
            ax.axis('off')

plt.show()