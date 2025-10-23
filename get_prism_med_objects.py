import pandas as pd
import re
import os

# ---------------------- CONFIG ----------------------
snr_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_summary_1.csv"
master_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
out_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr5plus_with_prism_and_medium.csv"

# ---------------------- LOAD DATA ----------------------
snr_df = pd.read_csv(snr_csv_path)
master_df = pd.read_csv(master_csv_path)

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
snr_df["suffix_id"] = snr_df["file"].apply(extract_suffix_id)
master_df["suffix_id"] = master_df["file"].apply(extract_suffix_id)
master_df["band"] = master_df["file"].apply(identify_band)

# ---------------------- BUILD OUTPUT ----------------------
rows = []

for _, snr_row in snr_df.iterrows():
    suffix_id = snr_row["suffix_id"]
    if suffix_id is None:
        continue

    snr_value = snr_row["avg_snr_uv"]
    matches = master_df[master_df["suffix_id"] == suffix_id]
    if matches.empty:
        continue

    # Map files by grating
    file_map = {
        band: matches.loc[matches["band"] == band, "file"].iloc[0]
        if not matches.loc[matches["band"] == band].empty
        else None
        for band in ["prism", "g140m", "g235m", "g395m", "g395h"]
    }

    # Only keep those that have both a prism and at least one medium grating
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
