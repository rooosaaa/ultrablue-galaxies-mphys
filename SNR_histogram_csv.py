import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
# csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_summary_2.csv"
# out_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/UV_SNR_histogram.png"

# # Read the CSV
# df = pd.read_csv(csv_path)

# Filter out high SNR values (> 30) for plotting
# threshold = 30
# df_filtered = df[df['avg_snr_uv'] <= threshold]

# # Count how many spectra are above the threshold and will not be plotted
# num_excluded = (df['avg_snr_uv'] > threshold).sum()
# print(f"Number of spectra with avg SNR > {threshold} (excluded from plot): {num_excluded}")

# # Count how many spectra are being plotted
# num_plotted = len(df_filtered)
# print(f"Number of spectra being plotted (≤ {threshold}): {num_plotted}")

# # Count how many spectra have SNR > 5
# num_snr_above_5 = (df['avg_snr_uv'] > 5).sum()
# print(f"Number of spectra with avg SNR > 5: {num_snr_above_5}")

# # Extract the filtered SNR values
# snr_values = df_filtered['avg_snr_uv']

# # --- Print highest and lowest values in the plotted range ---
# max_idx = df_filtered['avg_snr_uv'].idxmax()
# min_idx = df_filtered['avg_snr_uv'].idxmin()

# max_snr = df_filtered.loc[max_idx, 'avg_snr_uv']
# min_snr = df_filtered.loc[min_idx, 'avg_snr_uv']

# max_file = df_filtered.loc[max_idx, 'file']
# min_file = df_filtered.loc[min_idx, 'file']

# print(f"Highest average SNR (≤ {threshold}): {max_snr:.2f} ({max_file})")
# print(f"Lowest average SNR:                {min_snr:.2f} ({min_file})")

# # --- Plot histogram of filtered data ---
# plt.figure(figsize=(8,5))
# plt.hist(snr_values, bins=60, color='dodgerblue', edgecolor='black', alpha=0.7)

# plt.xlabel("Average SNR in UV continuum (1250–3000 Å)")
# plt.ylabel("Number of spectra")
# plt.title(f"Histogram of Average UV SNRs (SNR ≤ {threshold})")
# plt.grid(alpha=0.3)
# plt.tight_layout()

# # Save figure
# plt.savefig(out_path, dpi=200)
# plt.show()

# # --- Paths ---
# hist_out_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/UV_SNR_histogram_5-30.png"
# csv_out_path  = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_5-30.csv"

# # --- Filter SNR values between 5 and 30 ---
# snr_min = 5
# snr_max = 30
# df_snr_range = df[(df['avg_snr_uv'] >= snr_min) & (df['avg_snr_uv'] <= snr_max)]

# num_in_range = len(df_snr_range)
# print(f"Number of spectra with avg SNR between {snr_min}-{snr_max}: {num_in_range}")

# # --- Plot histogram ---
# plt.figure(figsize=(8,5))
# plt.hist(df_snr_range['avg_snr_uv'], bins=60, color='dodgerblue', edgecolor='black', alpha=0.7)
# plt.xlabel("Average SNR in UV continuum (1250–3000 Å)")
# plt.ylabel("Number of spectra")
# plt.title(f"Histogram of Average UV SNRs ({snr_min} ≤ SNR ≤ {snr_max})")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(hist_out_path, dpi=200)
# plt.show()
# print(f"Saved histogram to: {hist_out_path}")

# # --- Save filtered CSV ---

# df_snr_range.to_csv(csv_out_path, index=False)
# print(f"Saved filtered SNR data to: {csv_out_path}")

# # Filter galaxies with avg SNR >= 30
# df_high_snr = df[df['avg_snr_uv'] >= 30]

# # Number of galaxies in this category
# # Path for the new CSV
# csv_out_high_snr = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_30plus.csv"
# num_high_snr = len(df_high_snr)
# print(f"Number of spectra with avg SNR ≥ 30: {num_high_snr}")

# # Save to new CSV
# df_high_snr.to_csv(csv_out_high_snr, index=False)
# print(f"Saved high-SNR galaxies to: {csv_out_high_snr}")

import pandas as pd
import matplotlib.pyplot as plt

# --- Paths ---
csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_summary_2.csv"
hist_out_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/UV_SNR_histogram_5plus.png"
csv_out_path  = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_5plus.csv"

# --- Read the CSV ---
df = pd.read_csv(csv_path)

# --- Filter galaxies with avg SNR >= 5 ---
snr_min = 5
df_snr_5plus = df[df['avg_snr_uv'] >= snr_min]
num_snr_5plus = len(df_snr_5plus)
print(f"Number of spectra with avg SNR ≥ {snr_min}: {num_snr_5plus}")

# --- Save filtered CSV ---
df_snr_5plus.to_csv(csv_out_path, index=False)
print(f"Saved filtered SNR ≥ {snr_min} galaxies to: {csv_out_path}")

# --- Plot histogram of filtered data ---
plt.figure(figsize=(8,5))
plt.hist(df_snr_5plus['avg_snr_uv'], bins=60, color='dodgerblue', edgecolor='black', alpha=0.7)
plt.xlabel("Average SNR in UV continuum (1250–3000 Å)")
plt.ylabel("Number of spectra")
plt.title(f"Histogram of Average UV SNRs (SNR ≥ {snr_min})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(hist_out_path, dpi=200)
plt.show()
print(f"Saved histogram to: {hist_out_path}")

