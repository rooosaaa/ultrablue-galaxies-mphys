import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
from astropy.stats import sigma_clip

# --- Paths ---
fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/gds-barrufet-s67-v4/gds-barrufet-s67-v4_prism-clear_2198_1260.spec.fits"
csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
out_dir = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/"
os.makedirs(out_dir, exist_ok=True)

# ------------------ Helper function ------------------
def compute_binned_snr(wave, flux, err, lam_min, lam_max):
    """Compute SNR in a wavelength bin [lam_min, lam_max]"""
    mask = (wave >= lam_min) & (wave <= lam_max)
    flux_bin = flux[mask]
    err_bin  = err[mask]
    
    if len(flux_bin) == 0:
        return np.nan
    
    total_flux = np.sum(flux_bin)
    total_err  = np.sqrt(np.sum(err_bin**2))
    snr_bin = total_flux / total_err
    return snr_bin

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
    # print(hdul['SPEC1D'].header)

wave_obs = spec['wave']   # observed wavelength [μm]
flux_obs = spec['flux']   # observed flux in μJy
err_obs  = spec['err']    # 1σ uncertainty

# Convert μJy → erg/s/cm^2/Hz
flux_nu = flux_obs * 1e-29
err_nu  = err_obs  * 1e-29

# Convert wavelength μm → Å
wave_A = wave_obs * 1e4

# Convert Fν → Fλ (erg/s/cm^2/Å)
c = 2.99792458e18  # speed of light in Å/s
flux_lambda = flux_nu * c / (wave_A**2)
err_lambda  = err_nu  * c / (wave_A**2)

# --- Step 3: Convert to rest frame ---
wave_rest = wave_A / (1 + z)
flux_rest = flux_lambda * (1 + z)**2
err_rest  = err_lambda * (1 + z)**2

# --- Step 4: Plot rest-frame spectrum (λ > 2000 Å only) ---
plt.figure(figsize=(10, 5))

# Sigma-clip and mask invalid values
mask = (np.isfinite(wave_rest)
    & np.isfinite(flux_rest)
    & np.isfinite(err_rest)
    & (wave_rest > 1200)        # keep only λ > 2000 Å
)

# Apply mask
wave_plot = wave_rest[mask]
flux_plot = flux_rest[mask]
err_plot  = err_rest[mask]

# Plot flux and uncertainty
plt.plot(wave_plot, flux_plot, color='dodgerblue', lw=1.0, label='Rest-frame spectrum')
plt.fill_between(wave_plot, flux_plot - err_plot, flux_plot + err_plot,
                 color='lightblue', alpha=0.4, label='1σ uncertainty')

# Labels and title
plt.xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=13)
plt.ylabel(r"Flux density $F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=13)
plt.title(f"Rest-frame Spectrum (λ > 1200 Å): {fits_name}\n(z = {z:.3f})", fontsize=14)
plt.legend(frameon=False)
plt.grid(alpha=0.3)

# --- Tighter x-axis limits ---
plt.xlim(wave_plot.min(), wave_plot.max())

# Optional y-scaling
plt.yscale('linear')  # or 'log' if flux spans orders of magnitude
plt.tight_layout()

# Save
out_plot = os.path.join(out_dir, fits_name.replace('.fits', '_restframe_gt1200.png'))
plt.savefig(out_plot, dpi=200)
plt.show()

print(f"Saved rest-frame spectrum (λ>1200 Å) to: {out_plot}")

# --- Step 5: Compute and plot S/N per pixel ---

# Compute SNR per pixel (flux / error)
snr_pixel = flux_plot / err_plot

# Mask out invalid or negative values
valid = np.isfinite(snr_pixel)

wave_snr = wave_plot[valid]
snr_pixel = snr_pixel[valid]

# --- Step 7: Average SNR in UV continuum (1250–3000 Å) ---
uv_min, uv_max = 1250, 3000
uv_mask = (wave_snr >= uv_min) & (wave_snr <= uv_max)
snr_uv = snr_pixel[uv_mask]
avg_snr_uv = np.mean(snr_uv) if len(snr_uv) > 0 else np.nan
print(f"Average per-pixel SNR (UV 1250–3000 Å): {avg_snr_uv:.2f}")

# --- Step 8: Compute binned SNR ---
bin_min, bin_max = 1250, 3000  # UV continuum
bin_snr = compute_binned_snr(wave_snr, flux_plot[valid], err_plot[valid], bin_min, bin_max)
print(f"SNR in bin {bin_min}–{bin_max} Å: {bin_snr:.2f}")

# --- Step 9: Optional plot SNR per pixel ---
plt.figure(figsize=(10,4))
plt.plot(wave_snr, snr_pixel, color='darkorange', lw=1.0, alpha=0.7, label='Per-pixel SNR')
plt.axvspan(uv_min, uv_max, color='lightgrey', alpha=0.3, label='UV continuum (1250–3000 Å)')
plt.axhline(avg_snr_uv, color='red', linestyle='--', lw=1.0, label=f'Avg per-pixel UV SNR = {avg_snr_uv:.2f}')

# Add the binned SNR as a horizontal line over the bin wavelength range
plt.hlines(bin_snr, bin_min, bin_max, colors='blue', lw=2.0, label=f'Binned SNR {bin_min}-{bin_max} Å = {bin_snr:.2f}')
plt.axvline(bin_min, color='blue', linestyle=':', lw=1)
plt.axvline(bin_max, color='blue', linestyle=':', lw=1)

plt.xlabel("Wavelength [Å]")
plt.ylabel("S/N per pixel")
plt.title("S/N per pixel with UV average and binned SNR")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
out_snr_plot = os.path.join(out_dir, fits_name.replace('.fits','_UV_SNR.png'))
plt.savefig(out_snr_plot, dpi=200)
plt.show()