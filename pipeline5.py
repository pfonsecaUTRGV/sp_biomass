from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.transform import Affine
import geopandas as gpd
import rasterio.features
from shapely.geometry import shape
from shapely.validation import make_valid
from PIL import Image

# ---------------------------------------------------------
# STEP 1: Load CHM
# ---------------------------------------------------------
with rasterio.open("CHM.tif") as src:
    chm = src.read(1)
    chm_transform = src.transform
    crs = src.crs

print("CHM loaded.")
print(f"CHM range (min/max): {np.nanmin(chm):.2f} / {np.nanmax(chm):.2f}")

# Rescale if needed (auto-fix for small units)
if np.nanmax(chm) < 3:
    print("CHM values seem too small — scaling by 100 (cm→m)")
    chm *= 100
elif np.nanmax(chm) < 0.1:
    print("CHM appears normalized — scaling by 25 (expected canopy height)")
    chm *= 25

# ---------------------------------------------------------
# STEP 2: Load orthophoto and build vegetation mask
# ---------------------------------------------------------
print("Loading orthophoto for vegetation masking...")
Image.MAX_IMAGE_PIXELS = None
jpg_path = "colonia_orto.jpg"
tfw_path = "colonia_tfw.tfw"

# Read world file (TFW)
with open(tfw_path, "r") as f:
    a = float(f.readline())
    d = float(f.readline())
    b = float(f.readline())
    e = float(f.readline())
    c = float(f.readline())
    f_line = float(f.readline())
ortho_transform = Affine(a, b, c, d, e, f_line)

# Load JPG
img = np.array(Image.open(jpg_path)).astype(float)
r, g, b = img[..., 0], img[..., 1], img[..., 2]

# Vegetation mask (simple greenness index)
veg_index = (g - r) / (g + r + 1e-6)
veg_mask = veg_index > 0.05  # adjust threshold if needed

# Resize vegetation mask to match CHM dimensions
if veg_mask.shape != chm.shape:
    from skimage.transform import resize
    veg_mask = resize(veg_mask, chm.shape, order=0, preserve_range=True) > 0.5

# Apply mask — set non-vegetation areas to 0
chm_masked = np.where(veg_mask, chm, 0)

print("Vegetation mask applied (non-green areas removed).")

# ---------------------------------------------------------
# STEP 3: Tree detection (adaptive)
# ---------------------------------------------------------
chm_smooth = gaussian(chm_masked, sigma=2)

chm_min, chm_max, chm_mean = np.nanmin(chm_masked), np.nanmax(chm_masked), np.nanmean(chm_masked)
print(f"CHM stats → min: {chm_min:.2f}, max: {chm_max:.2f}, mean: {chm_mean:.2f}")

threshold_abs = max(0.03, 0.09 * chm_max)
min_distance = 3 if chm_max < 10 else 3 if chm_max < 20 else 5
print(f"Adaptive thresholds → min_distance={min_distance}, threshold_abs={threshold_abs:.2f}")

coords = peak_local_max(chm_smooth, min_distance=min_distance, threshold_abs=threshold_abs)
mask = chm_smooth > (0.4 * threshold_abs)
print(f"Detected {len(coords)} treetop peaks")

# ---------------------------------------------------------
# STEP 4: Watershed segmentation
# ---------------------------------------------------------
markers = np.zeros_like(chm_smooth, dtype=np.int32)
for i, (r, c) in enumerate(coords, start=1):
    markers[r, c] = i

labels = watershed(-chm_smooth, markers, mask=mask)
pixel_size = src.res[0]
pixel_area = pixel_size**2

# ---------------------------------------------------------
# STEP 5: Extract tree metrics
# ---------------------------------------------------------
tree_data = []
for i in range(1, labels.max() + 1):
    m = labels == i
    if np.sum(m) < 2:
        continue
    h_vals = chm_masked[m]
    h_max = np.nanmax(h_vals)
    h_mean = np.nanmean(h_vals)
    crown_area = np.sum(m) * pixel_area
    crown_volume = np.nansum(h_vals) * pixel_area
    if h_max > 1.0 and crown_area < 500:  # filter out low or unrealistic objects
        tree_data.append({
            "Tree_ID": i,
            "Height_m": h_max,
            "Mean_Height_m": h_mean,
            "Crown_Area_m2": crown_area,
            "Crown_Volume_m3": crown_volume
        })

df = pd.DataFrame(tree_data)
print(f"Trees detected: {len(df)}")

# ---------------------------------------------------------
# STEP 6: Biomass and Carbon
# ---------------------------------------------------------
if len(df) > 0:
    df["AGB_kg"] = 0.05 * (df["Crown_Volume_m3"] ** 1.0)
    df.to_csv("tree_biomass_estimates.csv", index=False)
    print("Biomass results saved to 'tree_biomass_estimates.csv'")
else:
    print("No valid trees detected — check thresholds or CHM quality.")

# ---------------------------------------------------------
# STEP 7: Visualization
# ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 7), facecolor="#f5f5f5")

# --- CHM Plot (Left) ---
from matplotlib.colors import Normalize
norm = Normalize(vmin=np.nanpercentile(chm_masked, 5), vmax=np.nanpercentile(chm_masked, 95))
im1 = ax[0].imshow(chm_masked, cmap='viridis', norm=norm)
ax[0].set_facecolor("#eaeaea")
ax[0].set_title("Filtered CHM (Vegetation Only)", color="black")
fig.colorbar(im1, ax=ax[0], label='Height (m)')

# Overlay detected tree tops on CHM
if len(coords) > 0:
    ax[0].scatter(coords[:, 1], coords[:, 0], c='red', s=20, alpha=0.8, marker='o', label='Treetops')
    ax[0].legend(loc='lower right')

# --- Watershed Plot (Right) ---
im2 = ax[1].imshow(labels, cmap='nipy_spectral')
ax[1].set_facecolor("#eaeaea")
ax[1].set_title("Tree Segmentation (Watershed Labels)", color="black")
fig.colorbar(im2, ax=ax[1], label='Tree ID')

# Overlay points with larger size for visibility
if len(coords) > 0:
    ax[1].scatter(coords[:, 1], coords[:, 0], c='white', s=35, alpha=0.9, edgecolors='black', linewidth=0.4, marker='o', label='Detected Peaks')
    ax[1].legend(loc='lower right')

plt.tight_layout()
plt.show()
