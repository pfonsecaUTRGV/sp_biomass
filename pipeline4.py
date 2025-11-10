import pandas as pd

# --- Load the exported tree metrics file ---
file_path = "tree_biomass_estimates.csv"  # adjust if needed

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

if len(df) == 0:
    print("CSV is empty — no data to summarize.")
    exit()

# --- Forest area in hectares ---
forest_area_ha = 87.06  # provided area
forest_area_m2 = forest_area_ha * 10_000  # convert ha → m²

# --- Compute summary statistics ---
total_trees = len(df)
mean_height = df["Height_m"].mean()
mean_crown_area = df["Crown_Area_m2"].mean()
mean_volume = df["Crown_Volume_m3"].mean()
total_biomass = df["AGB_kg"].sum()

# --- Carbon and CO₂ equivalent estimates ---
carbon_kg = total_biomass * 0.47
co2_eq = carbon_kg * 3.67

# --- Tree density (trees per hectare) ---
tree_density_ha = total_trees / forest_area_ha

# --- Per-tree averages ---
avg_biomass_per_tree = total_biomass / total_trees
avg_carbon_per_tree = carbon_kg / total_trees
avg_co2eq_per_tree = co2_eq / total_trees

# --- Print summary report ---
print("\n===== BIOMASS AND CARBON SUMMARY =====")
print(f"Total trees detected: {total_trees:,}")
print(f"Forest area: {forest_area_ha:.2f} ha")
print(f"Tree density: {tree_density_ha:.2f} trees/ha\n")

print(f"Average tree height: {mean_height:.2f} m")
print(f"Mean crown area: {mean_crown_area:.2f} m²")
print(f"Mean crown volume: {mean_volume:.2f} m³\n")

print(f"Total biomass (AGB): {total_biomass:.2f} kg")
print(f"Carbon stored: {carbon_kg:.2f} kg C")
print(f"CO₂ equivalent: {co2_eq:.2f} kg CO₂e\n")

print(f"Avg biomass per tree: {avg_biomass_per_tree:.2f} kg/tree")
print(f"Avg carbon per tree: {avg_carbon_per_tree:.2f} kg C/tree")
print(f"Avg CO₂ eq per tree: {avg_co2eq_per_tree:.2f} kg CO₂e/tree")
print("======================================\n")

# --- Save to summary CSV ---
summary_data = {
    "Total_Trees": [total_trees],
    "Forest_Area_ha": [forest_area_ha],
    "Tree_Density_ha": [tree_density_ha],
    "Avg_Height_m": [mean_height],
    "Avg_Crown_Area_m2": [mean_crown_area],
    "Avg_Crown_Volume_m3": [mean_volume],
    "Total_Biomass_kg": [total_biomass],
    "Carbon_kg": [carbon_kg],
    "CO2e_kg": [co2_eq],
    "Avg_Biomass_per_Tree_kg": [avg_biomass_per_tree],
    "Avg_Carbon_per_Tree_kg": [avg_carbon_per_tree],
    "Avg_CO2eq_per_Tree_kg": [avg_co2eq_per_tree]
}

#summary_df = pd.DataFrame(summary_data)
#summary_df.to_csv("biomass_summary.csv", index=False)
#print("✅ Summary saved as 'biomass_summary.csv'")
