import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
from astropy.stats import sigma_clip

# --- Paths ---
fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/jades-gds1-v4/jades-gds1-v4_g140m-f070lp_1286_28737.spec.fits"
csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
out_dir = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/"
os.makedirs(out_dir, exist_ok=True)

# --- Step 1: Read CSV and extract redshift ---
df = pd.read_csv(csv_path, sep=",")  # adjust sep if needed
fits_name = os.path.basename(fits_path)
match = df[df['file'] == fits_name]

if match.empty:
    raise ValueError(f"No matching file name found in CSV for: {fits_name}")

z = float(match['z'].values[0])
print(f"Found redshift z = {z:.4f} for {fits_name}")

# --- Step 2: Read the FITS spectrum ---
with fits.open(fits_path) as hdul:
    hdul.info()
    spec = hdul['SPEC1D'].data  # update if your extension differs

wave_obs = spec['wave']   # observed wavelength [μm]
flux_obs = spec['flux']   # observed flux
err_obs  = spec['err']    # 1σ uncertainty

# --- Step 3: Convert to rest frame ---
wave_rest = wave_obs / (1 + z)
flux_rest = flux_obs * (1 + z)**2
err_rest  = err_obs * (1 + z)**2

# --- Step 4: Select UV continuum range ---
lambda_uv_min = 0.1  # μm
lambda_uv_max = 0.4  # μm

uv_mask = (wave_rest >= lambda_uv_min) & (wave_rest <= lambda_uv_max)
wave_uv = wave_rest[uv_mask]
flux_uv = flux_rest[uv_mask]
err_uv  = err_rest[uv_mask]

# --- Step 5: Ignore missing / bad pixels ---
valid = (
    np.isfinite(wave_uv) &
    np.isfinite(flux_uv) &
    np.isfinite(err_uv) &
    (err_uv > 0)
)

n_total = len(wave_uv)
n_valid = np.sum(valid)

if n_valid == 0:
    sn_uv = np.nan
else:
    sn_uv = np.median(np.abs(flux_uv[valid]) / err_uv[valid])

print(f"UV pixels in range: {n_total}, valid for S/N: {n_valid}")
if np.isnan(sn_uv):
    print(f"UV continuum S/N for {fits_name} is NaN (no valid pixels in range)")
else:
    print(f"UV continuum S/N for {fits_name} = {sn_uv:.2f}")

# --- Step 5: Plot full rest-frame spectrum (no trimming) ---
plt.figure(figsize=(10, 5))
plt.plot(wave_rest, flux_rest, color='black', lw=1, label='Rest-frame flux')
plt.fill_between(wave_rest, flux_rest - err_rest, flux_rest + err_rest,
                 color='gray', alpha=0.3, label='1σ error')

# show UV window on plot
plt.axvspan(lambda_uv_min, lambda_uv_max, color='red', alpha=0.12, label='UV continuum range')

# safe title (avoid formatting NaN as float)
sn_label = f"{sn_uv:.2f}" if np.isfinite(sn_uv) else "NaN"
plt.title(f"{fits_name}  (z={z:.3f}) | UV S/N = {sn_label}")

plt.xlabel("Rest-frame Wavelength [μm]")
plt.ylabel("Flux [μJy]")
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_file = os.path.splitext(fits_name)[0] + f"_z{z:.3f}_restframe.png"
out_path = os.path.join(out_dir, out_file)
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Saved rest-frame 1D spectrum to:\n{out_path}")


