from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.validation import make_valid
from PIL import Image
from matplotlib.colors import Normalize

# ---------------------------------------------------------
# STEP 1: Load and Normalize CHM
# ---------------------------------------------------------
with rasterio.open("CHM.tif") as src:
    chm = src.read(1).astype(float)
    chm_transform = src.transform
    crs = src.crs
    pixel_size = src.res[0]

print("CHM loaded.")
print(f"Raw CHM range → min: {np.nanmin(chm):.2f}, max: {np.nanmax(chm):.2f}")

# --- Height normalization and rescaling ---
chm_min, chm_max, chm_mean = np.nanmin(chm), np.nanmax(chm), np.nanmean(chm)

if chm_max > 50:
    print("⚠️ CHM likely in decimeters → dividing by 10")
    chm /= 10.0
elif 15 < chm_max <= 50:
    print("⚠️ CHM likely in meters (no scaling)")
elif chm_max < 2:
    print("⚠️ CHM seems normalized 0–1 → scaling ×10 for meters")
    chm *= 10.0
else:
    print("✅ CHM already close to meters")

# Clip outliers
chm = np.clip(chm, 0, 12)

print(f"Adjusted CHM stats → min: {np.nanmin(chm):.2f}, max: {np.nanmax(chm):.2f}, mean: {np.nanmean(chm):.2f}\n")

# ---------------------------------------------------------
# STEP 2: Vegetation Mask using Orthophoto
# ---------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None
jpg_path = "colonia_orto.jpg"
tfw_path = "colonia_tfw.tfw"

with open(tfw_path, "r") as f:
    a = float(f.readline())
    d = float(f.readline())
    b = float(f.readline())
    e = float(f.readline())
    c = float(f.readline())
    f_line = float(f.readline())
ortho_transform = Affine(a, b, c, d, e, f_line)

img = np.array(Image.open(jpg_path)).astype(float)
r, g, b = img[..., 0], img[..., 1], img[..., 2]

veg_index = (g - r) / (g + r + 1e-6)
veg_mask = veg_index > 0.07  # slightly looser threshold for small urban trees

# Resize mask if needed
if veg_mask.shape != chm.shape:
    from skimage.transform import resize
    veg_mask = resize(veg_mask, chm.shape, order=0, preserve_range=True) > 0.5

chm_masked = np.where(veg_mask, chm, 0)
print("Vegetation mask applied.\n")

# ---------------------------------------------------------
# STEP 3: Tree Detection (Tuned for Urban Scale)
# ---------------------------------------------------------
chm_smooth = gaussian(chm_masked, sigma=1.8)

chm_min, chm_max, chm_mean = np.nanmin(chm_masked), np.nanmax(chm_masked), np.nanmean(chm_masked)
print(f"CHM stats → min: {chm_min:.2f}, max: {chm_max:.2f}, mean: {chm_mean:.2f}")

#threshold_abs = max(0.1, 0.06 * chm_max)
threshold_abs = max(0.2, 0.10 * chm_max)   # require stronger peaksmin_distance = 4 if chm_max < 10 else 6
min_distance = 8                           # merge close trees
print(f"Adaptive thresholds → min_distance={min_distance}, threshold_abs={threshold_abs:.2f}")

coords = peak_local_max(chm_smooth, min_distance=min_distance, threshold_abs=threshold_abs)
#mask = chm_smooth > (0.35 * threshold_abs)
mask = chm_smooth > (0.6 * threshold_abs)  # only keep tall crowns
print(f"Detected {len(coords)} treetop peaks\n")

# ---------------------------------------------------------
# STEP 4: Watershed Segmentation
# ---------------------------------------------------------
markers = np.zeros_like(chm_smooth, dtype=np.int32)
for i, (r, c) in enumerate(coords, start=1):
    markers[r, c] = i

labels = watershed(-chm_smooth, markers, mask=mask)
pixel_area = pixel_size ** 2
print(f"Watershed produced {labels.max()} crown regions\n")

# ---------------------------------------------------------
# STEP 5: Extract Tree Metrics (balanced filters)
# ---------------------------------------------------------
tree_data = []
for i in range(1, labels.max() + 1):
    m = labels == i
    if np.sum(m) < 8:  # allow smaller crowns
        continue
    h_vals = chm_masked[m]
    h_max = np.nanmax(h_vals)
    h_mean = np.nanmean(h_vals)
    crown_area = np.sum(m) * pixel_area
    crown_volume = np.nansum(h_vals) * pixel_area

    if h_max > 0.8 and h_mean > 0.3 and crown_area < 1200:
        tree_data.append({
            "Tree_ID": i,
            "Height_m": h_max,
            "Mean_Height_m": h_mean,
            "Crown_Area_m2": crown_area,
            "Crown_Volume_m3": crown_volume
        })

df = pd.DataFrame(tree_data)
print(f"✅ Trees detected: {len(df)}\n")

# ---------------------------------------------------------
# STEP 6: Biomass and Carbon
# ---------------------------------------------------------
if len(df) > 0:
    df["AGB_kg"] = 0.05 * (df["Crown_Volume_m3"] ** 1.0)
    df.to_csv("tree_biomass_estimates.csv", index=False)
    print("Biomass results saved → 'tree_biomass_estimates.csv'")
else:
    print("⚠️ No valid trees detected — try lowering threshold_abs slightly.\n")

# ---------------------------------------------------------
# STEP 7: Visualization
# ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 7), facecolor="#f5f5f5")

norm = Normalize(vmin=np.nanpercentile(chm_masked, 5), vmax=np.nanpercentile(chm_masked, 95))
im1 = ax[0].imshow(chm_masked, cmap='viridis', norm=norm)
ax[0].set_title("Filtered CHM (Vegetation Only)")
fig.colorbar(im1, ax=ax[0], label='Height (m)')
if len(coords) > 0:
    ax[0].scatter(coords[:, 1], coords[:, 0], c='red', s=25, alpha=0.8, label='Treetops')
    ax[0].legend(loc='lower right')

im2 = ax[1].imshow(labels, cmap='nipy_spectral')
ax[1].set_title("Tree Segmentation (Watershed Labels)")
fig.colorbar(im2, ax=ax[1], label='Tree ID')

plt.tight_layout()
plt.show()
