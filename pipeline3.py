import pandas as pd

pixel_size = src.res[0]
pixel_area = pixel_size**2

tree_data = []
for i in range(1, labels.max() + 1):
    mask = labels == i
    if np.sum(mask) < 5:
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
