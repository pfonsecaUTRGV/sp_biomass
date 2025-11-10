from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.filters import gaussian
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import shape
import rasterio.features
from PIL import Image

from shapely.geometry import shape
from shapely.validation import make_valid


with rasterio.open("CHM.tif") as src:
    chm = src.read(1)
    chm_transform = src.transform
    #transform = src.transform
    crs = src.crs

# --- Check and rescale CHM values if too small ---
print("CHM min:", np.nanmin(chm))
print("CHM max:", np.nanmax(chm))
print("CHM mean:", np.nanmean(chm))

# If CHM seems to be in centimeters or normalized 0–1, rescale to meters
if np.nanmax(chm) < 3:  # likely not in meters
    print("CHM values seem too small — scaling by 100 (cm → m)")
    chm = chm * 100
elif np.nanmax(chm) < 0.1:  # possibly normalized 0–1
    print("CHM appears normalized — scaling by 25 m expected canopy height")
    chm = chm * 25


# Smooth CHM slightly to remove noise
chm_smooth = gaussian(chm, sigma=1)

# Find tree tops (local maxima)
# --- Adaptive tree detection based on CHM statistics ---
chm_min, chm_max, chm_mean = np.nanmin(chm), np.nanmax(chm), np.nanmean(chm)
print(f"CHM stats → min: {chm_min:.2f}, max: {chm_max:.2f}, mean: {chm_mean:.2f}")

# Estimate dynamic thresholds
# Smaller trees → smaller thresholds, bigger trees → more conservative ones
threshold_abs = max(0.05, 0.05 * chm_max)      # 5% of max height or at least 0.05
min_distance = 1 if chm_max < 10 else 2 if chm_max < 20 else 3

print(f"Adaptive thresholds → min_distance={min_distance}, threshold_abs={threshold_abs:.2f}")

# Apply local maxima detection
coords = peak_local_max(chm_smooth, min_distance=min_distance, threshold_abs=threshold_abs)
mask = chm_smooth > (0.5 * threshold_abs)  # slightly lower mask to include full crowns
print(f"Detected {len(coords)} treetop peaks")


markers = np.zeros_like(chm_smooth, dtype=np.int32)
for i, (r, c) in enumerate(coords, start=1):
    markers[r, c] = i

labels = watershed(-chm_smooth, markers, mask=mask)

pixel_size = src.res[0]
pixel_area = pixel_size**2

tree_data = []

for i in range(1, labels.max() + 1):
    mask = labels == i
    if np.sum(mask) < 2:
        continue
    h_vals = chm[mask]
    h_max = np.nanmax(h_vals)
    h_mean = np.nanmean(h_vals)
    crown_area = np.sum(mask) * pixel_area
    crown_volume = np.nansum(h_vals) * pixel_area
    tree_data.append({
        "Tree_ID": i,
        "Height_m": h_max,
        "Mean_Height_m": h_mean,
        "Crown_Area_m2": crown_area,
        "Crown_Volume_m3": crown_volume
    })

df = pd.DataFrame(tree_data)
df.to_csv("tree_metrics.csv", index=False)

df["AGB_kg"] = 0.05 * (df["Crown_Volume_m3"] ** 1.0)
df.to_csv("tree_biomass_estimates.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Plot CHM
im1 = ax[0].imshow(chm, cmap='terrain')
ax[0].set_title("Canopy Height Model (CHM)")
fig.colorbar(im1, ax=ax[0], label='Height (m)')

# Plot segmented crowns
im2 = ax[1].imshow(labels, cmap='tab20')
ax[1].set_title("Tree Segmentation (Watershed Labels)")
fig.colorbar(im2, ax=ax[1], label='Tree ID')

plt.tight_layout()
plt.show()


###########################################

'''

Image.MAX_IMAGE_PIXELS = None
# --- Load the JPEG and TFW ---
jpg_path = "cerro_j.jpg"  # your file
tfw_path = "cerro_t.tfw"  # same name, .tfw extension

# Read world file (6 lines)
with open(tfw_path, "r") as f:
    a = float(f.readline())  # pixel size in x-direction
    d = float(f.readline())  # rotation term (usually 0)
    b = float(f.readline())  # rotation term (usually 0)
    e = float(f.readline())  # pixel size in y-direction (negative)
    c = float(f.readline())  # x-coordinate of upper-left pixel center
    f_line = float(f.readline())  # y-coordinate of upper-left pixel center

#transform = Affine(a, b, c, d, e, f_line)
ortho_transform = Affine(a, b, c, d, e, f_line)

# Load the JPEG as array
img = np.array(Image.open(jpg_path))
img = img[::7, ::7]  # use every 4th pixel in both directions

# Compute extent for imshow
height, width = img.shape[0], img.shape[1]
xmin, ymax = c, f_line
xmax = xmin + width * a
ymin = ymax + height * e



# --- Convert your labels to polygons ---
shapes = list(rasterio.features.shapes(labels.astype(np.int32), transform=transform))
polys = []
for val, geom in shapes:
    if val == 0:
        continue
    polys.append({'Tree_ID': int(val), 'geometry': shape(geom)})

crowns_gdf = gpd.GeoDataFrame(polys, crs=crs)


print("CHM range (min/max):", np.nanmin(chm), np.nanmax(chm))
print("Number of watershed labels:", labels.max())
print("Unique treetop peaks detected:", len(coords))


# --- Convert segmentation labels to polygons (safe version) ---

polys = []

for (val, geom) in rasterio.features.shapes(labels.astype(np.int32), transform=transform):
    if isinstance(val, dict):
        val = val.get("value", 0)

    if val == 0 or geom is None:
        continue

    try:
        geom_obj = shape(geom)
        if geom_obj.is_valid and not geom_obj.is_empty:
            polys.append({
                "Tree_ID": int(val),
                "geometry": geom_obj
            })
    except Exception as e:
        print(f" Skipping invalid geometry: {e}")
        continue

if len(polys) > 0:
    crowns_gdf = gpd.GeoDataFrame(polys, geometry="geometry", crs=crs)
    print(f"Created GeoDataFrame with {len(crowns_gdf)} crowns")
else:
    print("No valid crown geometries found — check segmentation or CHM quality")





print("Converting segmentation labels to polygons (robust fix)...")

polys = []

# Ensure labels are integer-valued
labels_int = labels.astype(np.int32)

# Verify transform; if invalid, fallback to identity
if not isinstance(chm_transform, Affine):
    print("Transform invalid — using identity transform")
    chm_transform = Affine.identity()

# Try vectorization
for (val, geom) in rasterio.features.shapes(labels.astype(np.int32), transform=chm_transform):
    #val, geom = item
    if isinstance(val, (float, int)):
        if val == 0:
            continue
        if not isinstance(geom, dict) or "type" not in geom:
            continue
        try:
            geom_obj = shape(geom)
            if not geom_obj.is_valid:
                geom_obj = make_valid(geom_obj)
            if geom_obj.is_empty:
                continue
            polys.append({"Tree_ID": int(val), "geometry": geom_obj})
        except Exception as e:
            print(f"Skipping invalid geometry: {e}")
    else:
        # skip any malformed entries
        continue

if polys:
    crowns_gdf = gpd.GeoDataFrame(polys, geometry="geometry", crs=crs)
    print(f"Created {len(crowns_gdf)} crown polygons successfully.")
else:
    print("No valid crown polygons extracted — possible issue with transform or label dtype.")



# --- Plot orthophoto with crowns ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, extent=(xmin, xmax, ymin, ymax))

if 'crowns_gdf' in locals() and not crowns_gdf.empty:
    crowns_gdf.boundary.plot(ax=ax, color='lime', linewidth=1)
    ax.set_title("Tree Crowns over Orthophoto (Georeferenced via TFW)")
else:
    ax.set_title("No valid crowns detected – check CHM or segmentation thresholds")

plt.tight_layout()
plt.show()
'''