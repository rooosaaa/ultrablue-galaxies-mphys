import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_summary_2.csv"
out_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/UV_SNR_histogram.png"

# Read the CSV
df = pd.read_csv(csv_path)

# Filter out high SNR values (> 30) for plotting
threshold = 30
df_filtered = df[df['avg_snr_uv'] <= threshold]

# Count how many spectra are above the threshold and will not be plotted
num_excluded = (df['avg_snr_uv'] > threshold).sum()
print(f"Number of spectra with avg SNR > {threshold} (excluded from plot): {num_excluded}")

# Count how many spectra are being plotted
num_plotted = len(df_filtered)
print(f"Number of spectra being plotted (≤ {threshold}): {num_plotted}")

# Extract the filtered SNR values
snr_values = df_filtered['avg_snr_uv']

# --- Print highest and lowest values in the plotted range ---
max_idx = df_filtered['avg_snr_uv'].idxmax()
min_idx = df_filtered['avg_snr_uv'].idxmin()

max_snr = df_filtered.loc[max_idx, 'avg_snr_uv']
min_snr = df_filtered.loc[min_idx, 'avg_snr_uv']

max_file = df_filtered.loc[max_idx, 'file']
min_file = df_filtered.loc[min_idx, 'file']

print(f"Highest average SNR (≤ {threshold}): {max_snr:.2f} ({max_file})")
print(f"Lowest average SNR:                {min_snr:.2f} ({min_file})")

# --- Plot histogram of filtered data ---
plt.figure(figsize=(8,5))
plt.hist(snr_values, bins=60, color='dodgerblue', edgecolor='black', alpha=0.7)

plt.xlabel("Average SNR in UV continuum (1250–3000 Å)")
plt.ylabel("Number of spectra")
plt.title(f"Histogram of Average UV SNRs (SNR ≤ {threshold})")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure
plt.savefig(out_path, dpi=200)
plt.show()
