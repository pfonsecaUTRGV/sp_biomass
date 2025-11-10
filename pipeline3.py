import rasterio
from scipy.ndimage import grey_opening
from skimage.filters import gaussian
import numpy as np

# --- Load DSM ---
with rasterio.open("cerro.tif") as src:
    dsm = src.read(1)
    profile = src.profile

# --- Step 1: Estimate DTM using morphological opening ---
# The filter size should roughly match average crown size in pixels
dtm = grey_opening(dsm, size=(15, 15))  # adjust if trees are more/less spaced

# --- Step 2: Compute raw CHM ---
chm = dsm - dtm
chm[chm < 0] = 0  # remove any negative values

# --- Step 3: Clean CHM ---
# Define realistic canopy range for your region (Ebano, Mesquite, Cypress)
MIN_HEIGHT = 0
MAX_HEIGHT = 40  # meters

# Clip unrealistic heights
num_too_high = np.sum(chm > MAX_HEIGHT)
if num_too_high > 0:
    print(f"Clipping {num_too_high:,} pixels above {MAX_HEIGHT} m")
chm = np.clip(chm, MIN_HEIGHT, MAX_HEIGHT)

# Smooth with a light Gaussian filter to remove small spikes
chm_smooth = gaussian(chm, sigma=1)

# --- Step 4: Save cleaned CHM ---
profile.update(dtype="float32")

with rasterio.open("CHM.tif", "w", **profile) as dst:
    dst.write(chm_smooth.astype("float32"), 1)

print("Cleaned and filtered CHM saved as 'CHM.tif'")
