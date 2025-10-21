import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os

# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s
UV_RANGE = (1250, 3000)  # Å
MIN_WAVELENGTH = 1200    # Å - cut for the noisy start of the spectra

# --- Paths ---
fits_dir = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/jades-gds-w03-v4/"
master_csv = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
target_csv = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_5plus.csv"
out_dir = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/plots/"
os.makedirs(out_dir, exist_ok=True)

# --- Load CSVs ---
df_master = pd.read_csv(master_csv)
df_targets = pd.read_csv(target_csv)

# pick one target to test
target_name = 'jades-gds-w03-v4_prism-clear_1212_372.spec.fits'
print(f"Plotting target: {target_name}")

# find matching file info
match = df_master[df_master['file'] == target_name]
if match.empty:
    raise ValueError(f"No matching file found in master CSV for target: {target_name}")

z = match['z'].values[0]  # assumes redshift column exists
fits_path = os.path.join(fits_dir, target_name)

# ---------------------- FUNCTIONS ----------------------

def read_spectrum(fits_path):
    """Extracts 1D spectrum arrays and ensures wavelength is in microns."""
    with fits.open(fits_path) as hdul:
        try:
            spec1d = hdul['SPEC1D'].data
        except KeyError:
            spec1d = hdul[1].data
        wave, flux, err = spec1d['wave'], spec1d['flux'], spec1d['err']
        if np.nanmax(wave) > 100.0:
            wave = wave / 1e4  # convert Å to μm if needed
    return wave, flux, err


def read_2d_spectrum(fits_path):
    """Extracts the 2D spectral image."""
    with fits.open(fits_path) as hdul:
        if 'SCI' in hdul:
            data_2d = hdul['SCI'].data
        elif 'SPEC2D' in hdul:
            data_2d = hdul['SPEC2D'].data
        else:
            data_2d = hdul[0].data if len(hdul) > 0 else None
    if data_2d is None:
        raise KeyError("Could not find 2D spectral data.")
    return data_2d

def convert_to_rest_frame(wave_obs, flux_obs, err_obs, z):
    flux_nu = flux_obs * 1e-29
    err_nu = err_obs * 1e-29
    wave_A = wave_obs * 1e4
    flux_lambda = flux_nu * C_LIGHT / wave_A**2
    err_lambda = err_nu * C_LIGHT / wave_A**2
    wave_rest = wave_A / (1 + z)
    flux_rest = flux_lambda * (1 + z)**2
    err_rest = err_lambda * (1 + z)**2
    return wave_rest, flux_rest, err_rest

def clean_data(wave, flux, err, data_2d):
    """Masks non-finite values and applies same mask to 2D array."""
    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave_clean = wave[mask]
    flux_clean = flux[mask]
    err_clean = err[mask]
    data_2d_clean = data_2d[:, mask]
    return wave_clean, flux_clean, err_clean, data_2d_clean


def plot_2d_spectrum(ax, data_2d, wave_x_axis):
    """Plots the 2D spectrum using pcolormesh for precise grid alignment."""
    finite_vals = data_2d[np.isfinite(data_2d)]
    vmin, vmax = np.percentile(finite_vals, [10, 99]) if len(finite_vals) > 0 else (0, 1)

    # Pixel corners
    y_corners = np.arange(data_2d.shape[0] + 1)
    wave_midpoints = (wave_x_axis[:-1] + wave_x_axis[1:]) / 2.0
    dw_start = wave_x_axis[1] - wave_x_axis[0]
    dw_end = wave_x_axis[-1] - wave_x_axis[-2]
    x_corners = np.concatenate([
        [wave_x_axis[0] - dw_start / 2.0],
        wave_midpoints,
        [wave_x_axis[-1] + dw_end / 2.0]
    ])

    ax.pcolormesh(x_corners, y_corners, data_2d, cmap='magma_r',
                  vmin=vmin, vmax=vmax, shading='auto')
    ax.set_ylabel("Spatial Axis (pix)")
    ax.set_title("2D Spectrum (Observed Wavelength)", pad=15)


def plot_1d_spectrum(ax, wave, flux, err):
    """Plots the 1D spectrum with error shading and legend."""
    # Plot flux
    ax.plot(wave, flux, color='dodgerblue', lw=1.0, label='1D Spectrum')
    
    # Plot error region with label for legend
    ax.fill_between(wave, flux - err, flux + err, color='lightblue', alpha=0.4, label='Error')

    # Auto y-limits based on percentiles
    combined_data = np.concatenate([flux - err, flux + err])
    finite_data = combined_data[np.isfinite(combined_data)]
    if len(finite_data) > 10:
        y_min, y_max = np.percentile(finite_data, [0.9, 99.5])
        y_range = y_max - y_min
        y_margin = 0.15 * y_range
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Labels and grid
    ax.set_xlabel(r"Observed Wavelength $\lambda_{\rm obs}$ [$\mu$m]")
    ax.set_ylabel(r"$F_{\nu}$ [$\mu$Jy]")
    ax.grid(alpha=0.3)
    
    # Legend with error key
    ax.legend(frameon=True)


# ---------------------- MAIN ----------------------

# Load and clean data
wave_obs, flux_obs, err_obs = read_spectrum(fits_path)
data_2d_original = read_2d_spectrum(fits_path)
wave_obs_c, flux_obs_c, err_obs_c, data_2d_c = clean_data(wave_obs, flux_obs, err_obs, data_2d_original)

# Create the figure
fig, ax = plt.subplots(2, 1, figsize=(12, 7),
                       gridspec_kw={'height_ratios': [1, 2]}, sharex=True)

# Plot 2D spectrum
plot_2d_spectrum(ax[0], data_2d_c, wave_obs_c)
ax[0].text(0.98, 0.98, f"z = {z:.3f}", transform=ax[0].transAxes,
           ha='right', va='top',
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Corrected: flux first, err second
plot_1d_spectrum(ax[1], wave_obs_c, flux_obs_c, err_obs_c)

# Final touches
ax[0].set_xlim(wave_obs_c[0], wave_obs_c[-1])
plt.setp(ax[0].get_xticklabels(), visible=False)
plt.tight_layout(pad=0.5)

out_path = os.path.join(out_dir, target_name.replace('.fits', '_2D_1D_Aligned_pcolormesh.png'))
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved final aligned plot to: {out_path}")
