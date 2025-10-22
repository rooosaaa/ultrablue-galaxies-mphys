import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
import re

# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s
fits_dir = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/"
target_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_5plus.csv"
master_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
out_dir = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/spectra_comparison"
os.makedirs(out_dir, exist_ok=True)


# ---------------------- FUNCTIONS ----------------------

def read_spectrum(fits_path):
    """Read 1D spectrum from a FITS file."""
    with fits.open(fits_path) as hdul:
        spec = hdul['SPEC1D'].data
        wave = spec['wave']   # observed wavelength [μm]
        flux = spec['flux']   # flux [μJy]
        err  = spec['err']    # error [μJy]
    return wave, flux, err


def find_all_matching_files(base_dir, prefix, suffix):
    """
    Find all files with the same prefix and suffix (different gratings).
    Returns a sorted list of full paths.
    """
    matches = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(suffix):
                matches.append(os.path.join(root, f))
    # Sort so 'prism' appears first, then 'medium', then others alphabetically
    matches.sort(key=lambda x: ('prism' not in x, x))
    return matches


def convert_to_rest_frame(wave_obs, flux_obs, err_obs, z):
    """Convert to rest-frame λ in Å and Fλ in erg/s/cm²/Å."""
    flux_nu = flux_obs * 1e-29  # μJy → erg/s/cm²/Hz
    err_nu  = err_obs  * 1e-29
    wave_A  = wave_obs * 1e4    # μm → Å
    flux_lambda = flux_nu * C_LIGHT / wave_A**2
    err_lambda  = err_nu  * C_LIGHT / wave_A**2
    wave_rest = wave_A / (1 + z)
    flux_rest = flux_lambda * (1 + z)**2
    err_rest  = err_lambda * (1 + z)**2
    return wave_rest, flux_rest, err_rest


def plot_single_spectrum(ax, wave, flux, err, title, z, color='dodgerblue'):
    """Plot a single rest-frame spectrum (adaptive y-limits within 1575–1700 Å)."""
    if getattr(wave, "size", 0) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Focus range
    x_min, x_max = 1500, 1800
    mask = (wave >= x_min) & (wave <= x_max)

    if not mask.any() or np.all(np.isnan(flux[mask])):
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    wave_zoom = wave[mask]
    flux_zoom = flux[mask]
    err_zoom  = err[mask]

    # Plot as step function
    ax.step(wave_zoom, flux_zoom, where='mid', color=color, lw=1.0, label='Rest-frame spectrum')
    ax.fill_between(wave_zoom, flux_zoom - err_zoom, flux_zoom + err_zoom,
                    step='mid', color='lightblue', alpha=0.4, label='1σ uncertainty')

    # --- He II 1640 line and Δz range ---
    lambda_rest = 1640.0
    delta_z = 0.006
    # Observed wavelength for this z and ±Δz
    lam_obs_high = lambda_rest * (1 + z + delta_z)
    lam_obs_low  = lambda_rest * (1 + z - delta_z)
    # Convert back to rest-frame limits
    lam_rest_high = lam_obs_high / (1 + z)
    lam_rest_low  = lam_obs_low  / (1 + z)

    # Vertical dashed line at 1640 Å
    ax.axvline(lambda_rest, color='crimson', ls='--', lw=1.0, alpha=0.8, label='He II 1640 Å')
    # Shaded region for Δz tolerance
    ax.axvspan(lam_rest_low, lam_rest_high, color='crimson', alpha=0.15, label=r'Δz = ±0.006 range')

    # Adaptive y-limits with padding
    finite_flux = flux_zoom[np.isfinite(flux_zoom)]
    if finite_flux.size > 0:
        ymin, ymax = np.nanmin(finite_flux), np.nanmax(finite_flux)
        yrange = ymax - ymin if ymax > ymin else np.abs(ymax)
        pad = 0.2 * yrange if yrange > 0 else 0.1 * np.abs(ymax)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_ylim(0, 1)

    # Labels, title, styling
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=12)
    ax.set_ylabel(r"Flux density $F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=12)
    ax.set_title(f"{title}\n(z = {z:.3f})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)





# ---------------------- MAIN ROUTINE ----------------------

target_df = pd.read_csv(target_csv_path)
master_df = pd.read_csv(master_csv_path)

processed_targets = set()

for idx, tro in target_df.iterrows():
    target_file = tro['file']
    if target_file in processed_targets:
        continue

    # --- Match target file to master entry ---
    master_matches = master_df[master_df['file'] == target_file]
    if master_matches.empty:
        match = re.search(r'v4_(.+?)_\d+_\d+\.spec\.fits$', str(target_file))
        if match:
            variable_part = match.group(1)
            master_matches = master_df[master_df['file'].str.contains(re.escape(variable_part), na=False)]
        else:
            base = os.path.basename(str(target_file))
            master_matches = master_df[master_df['file'].str.contains(re.escape(base), na=False)]

    if master_matches.empty:
        print(f"[WARN] No master CSV match for target '{target_file}'. Skipping.")
        processed_targets.add(target_file)
        continue

    master_row = master_matches.iloc[0]
    master_file = master_row['file']

    try:
        z = float(master_row['z'])
    except Exception:
        print(f"[WARN] No valid z for master row {master_file}. Skipping.")
        processed_targets.add(target_file)
        continue

    # --- Build file search patterns ---
    if 'v4_' in master_file:
        prefix = master_file.split('v4_')[0] + 'v4_'
    else:
        prefix = os.path.basename(master_file).split('_')[0] + '_'

    parts = master_file.split('_')
    if len(parts) >= 3:
        suffix = '_' + '_'.join(parts[-2:])
    else:
        suffix = os.path.splitext(os.path.basename(master_file))[0] + '.spec.fits'

    # --- Find corresponding FITS spectra ---
    matched_files = find_all_matching_files(fits_dir, prefix, suffix)
    if not matched_files:
        print(f"Skipping {target_file}: no matching spectra found")
        processed_targets.add(target_file)
        continue

    # --- Skip if only one matching file (no comparison possible) ---
    if len(matched_files) == 1:
        print(f"Skipping {target_file}: only one corresponding spectrum found")
        processed_targets.add(target_file)
        continue

    # --- Convert and collect spectra ---
    spectra = []
    zoom_min, zoom_max = 1500, 1800  # region of interest

    for path in matched_files:
        try:
            wave, flux, err = read_spectrum(path)
            wave_rest, flux_rest, err_rest = convert_to_rest_frame(wave, flux, err, z)

            # Check if there's any valid (non-NaN) flux in the zoom range
            mask = (wave_rest >= zoom_min) & (wave_rest <= zoom_max)
            if not mask.any() or np.all(np.isnan(flux_rest[mask])):
                continue  # skip spectra that have no valid data in range

            spectra.append((wave_rest, flux_rest, err_rest, os.path.basename(path)))
        except Exception as e:
            print(f"[WARN] Failed to process {path}: {e}")

    # --- Skip plotting if all are invalid in this range ---
    if not spectra:
        print(f"Skipping {target_file}: all spectra have no valid data in {zoom_min}-{zoom_max} Å")
        processed_targets.add(target_file)
        continue

    # --- Plot ---
    ncols = len(spectra)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    colors = ['black', 'dodgerblue', 'seagreen', 'orange', 'purple']
    for ax, (wave, flux, err, name), color in zip(axes, spectra, colors):
        plot_single_spectrum(ax, wave, flux, err, name, z, color=color)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(target_file))[0]
    out_file = f"{base_name}_comparison_z{z:.3f}.png"
    out_path = os.path.join(out_dir, out_file)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot for {target_file} with {ncols} spectra → {out_path}")

