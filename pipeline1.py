import rasterio
from scipy.ndimage import grey_opening
import numpy as np

with rasterio.open("tramo3.tiff") as src:
    dsm = src.read(1)
    profile = src.profile

# Estimate terrain using morphological opening (smooths out canopy)
dtm = grey_opening(dsm, size=(15,15))  # adjust size based on tree spacing (in pixels)

# Create CHM
chm = dsm - dtm
chm[chm < 0] = 0  # remove negatives

profile.update(dtype='float32')

with rasterio.open("CHM.tif", 'w', **profile) as dst:
    dst.write(chm.astype('float32'), 1)
