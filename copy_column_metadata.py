import pandas as pd

# --- Paths ---
csv_path = "datasets/HSSD/metadata.csv"
out_path = "datasets/HSSD/metadata_updated.csv"

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Source and target column names ---
src_col = "voxelized"
dst_col = "sdf"

# --- Check source column exists ---
if src_col not in df.columns:
    raise ValueError(f"❌ Source column '{src_col}' not found in {csv_path}")

# --- Copy values ---
df[dst_col] = df[src_col]

# --- Save result ---
df.to_csv(out_path, index=False)

print(f"✅ Copied '{src_col}' → '{dst_col}' in {out_path}")
