import pandas as pd
import re
from astropy.io import fits
from astropy.table import Table


# ---------------------- CONFIG ----------------------
snr_csv_path = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_summary_gdsgdn.csv"
CATALOGUE_PATH = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"
out_csv_path = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr5plus_with_prism_and_medium.csv"

# ---------------------- LOAD DATA ----------------------
snr_df = pd.read_csv(snr_csv_path)

# Load catalogue FITS table
table = Table.read(CATALOGUE_PATH)
cat_df = table.to_pandas()

print(f"Loaded {len(cat_df)} entries from catalogue")

# --- Filter by SNR ≥ 5 ---
snr_df = snr_df[snr_df["avg_snr_uv"] >= 5].copy()
print(f"SNR ≥ 5 spectra: {len(snr_df)}")

# ---------------------- HELPERS ----------------------
def extract_suffix_id(filename):
    """Extract the trailing numeric ID pair from a filename."""
    m = re.search(r"_(\d+_\d+)\.spec\.fits$", str(filename))
    return m.group(1) if m else None


def identify_band(filename):
    """Identify band as prism, g140m, g235m, g395m, or g395h."""
    f = str(filename).lower()
    if "prism" in f:
        return "prism"
    for grating in ["g140m", "g235m", "g395m", "g395h"]:
        if grating in f:
            return grating
    return "unknown"


# ---------------------- PREPARE DATA ----------------------

# Adjust this column name if needed
CAT_FILENAME_COLUMN = "file"

cat_df[CAT_FILENAME_COLUMN] = (
    cat_df[CAT_FILENAME_COLUMN]
    .astype(str)
    .str.strip()
)

snr_df["file"] = snr_df["file"].astype(str).str.strip()

cat_df["suffix_id"] = cat_df[CAT_FILENAME_COLUMN].apply(extract_suffix_id)
cat_df["band"] = cat_df[CAT_FILENAME_COLUMN].apply(identify_band)

snr_df["suffix_id"] = snr_df["file"].apply(extract_suffix_id)

# ---------------------- BUILD OUTPUT ----------------------
rows = []

for _, snr_row in snr_df.iterrows():
    suffix_id = snr_row["suffix_id"]
    if suffix_id is None:
        continue

    snr_value = snr_row["avg_snr_uv"]
    matches = cat_df[cat_df["suffix_id"] == suffix_id]
    if matches.empty:
        continue

    # Map files by grating
    file_map = {
        band: matches.loc[matches["band"] == band, CAT_FILENAME_COLUMN].iloc[0]
        if not matches.loc[matches["band"] == band].empty
        else None
        for band in ["prism", "g140m", "g235m", "g395m", "g395h"]
    }

    # Keep only if prism + at least one medium grating
    if file_map["prism"] and any(file_map[g] for g in ["g140m", "g235m", "g395m"]):
        rows.append({
            "suffix_id": suffix_id,
            "avg_snr_uv": snr_value,
            "prism_file": file_map["prism"],
            "g140m_file": file_map["g140m"],
            "g235m_file": file_map["g235m"],
            "g395m_file": file_map["g395m"],
            "g395h_file": file_map["g395h"]
        })

# ---------------------- SAVE RESULTS ----------------------
final_df = pd.DataFrame(rows)
final_df.to_csv(out_csv_path, index=False)
print(f"Saved {len(final_df)} galaxies with prism + medium → {out_csv_path}")
