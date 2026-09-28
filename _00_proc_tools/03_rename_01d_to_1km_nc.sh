#!/bin/bash

base_dir="/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_CloudFree"

subdirs=(
h07v06 h08v04 h08v05 h08v06 h08v07 h09v02 h09v03 h09v04 h09v05 h09v06 h09v07
h10v02 h10v03 h10v04 h10v05 h10v06 h11v02 h11v03 h11v04 h11v05 h12v01 h12v02
h12v03 h12v04 h12v05 h13v01 h13v02 h13v03 h13v04 h14v01 h14v02 h14v03 h14v04
h15v01 h15v02
)

for subdir in "${subdirs[@]}"; do
    dir_path="${base_dir}/${subdir}"
    
    if [ ! -d "$dir_path" ]; then
        echo "Warning: Directory $dir_path does not exist, skipping"
        continue
    fi
    
    for file in "$dir_path"/MCD19A1.*.061_01d_16day*.nc; do
        if [ -f "$file" ]; then
            new_file="${file/_01d_/_1km_}"
            mv "$file" "$new_file"
            echo "Renamed: $file -> $new_file"
        fi
    done
done

echo "Renaming complete"