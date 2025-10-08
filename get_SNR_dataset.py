import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import pandas as pd
import os


# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s
UV_RANGE = (1250, 3000)  # Å
MIN_WAVELENGTH = 1200    # Å - cut for the noisy start of the spectra
SPECTRA_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D"
CSV_PATH = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
OUT_DIR = "/nvme/scratch/work/rroberts/mphys_pop_III/UV_SNRs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- CORE FUNCTIONS ----------------------

def find_prism_fits(base_dir):
    """Recursively find all FITS files with 'prism' in the filename."""
    fits_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".fits") and "prism" in f:
                fits_files.append(os.path.join(root, f))
    return fits_files


def get_redshift(csv_path, fits_name):
    df = pd.read_csv(csv_path)
    match = df[df['file'] == fits_name]
    if match.empty:
        raise ValueError(f"No matching entry in CSV for: {fits_name}")
    return float(match['z'].values[0])


def read_spectrum(fits_path):
    with fits.open(fits_path) as hdul:
        spec = hdul['SPEC1D'].data
    return spec['wave'], spec['flux'], spec['err']


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


def clean_spectrum(wave, flux, err, min_wave=MIN_WAVELENGTH):
    flux_clipped = sigma_clip(flux, sigma=5, maxiters=2)
    mask = (
        (~flux_clipped.mask)
        & np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (wave > min_wave)
    )
    return wave[mask], flux[mask], err[mask]


def compute_snr(flux, err):
    snr = flux / err
    valid = np.isfinite(snr)
    return snr[valid], valid


def average_snr_in_range(wave, snr, wave_range=UV_RANGE):
    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    snr_in_range = snr[mask]
    if len(snr_in_range) == 0:
        return np.nan
    return np.mean(snr_in_range)


def plot_spectrum(wave, flux, err, fits_name, z, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(wave, flux, color='dodgerblue', lw=1.0, label='Rest-frame spectrum')
    plt.fill_between(wave, flux - err, flux + err, color='lightblue', alpha=0.4, label='1σ uncertainty')
    plt.xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=13)
    plt.ylabel(r"Flux density $F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=13)
    plt.title(f"Rest-frame Spectrum (λ > {MIN_WAVELENGTH} Å): {fits_name}\n(z = {z:.3f})", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.xlim(wave.min(), wave.max())
    plt.tight_layout()
    out_plot = os.path.join(out_dir, fits_name.replace('.fits', '_restframe.png'))
    plt.savefig(out_plot, dpi=200)
    plt.close()
    return out_plot


def plot_snr(wave, snr, avg_snr, fits_name, out_dir, wave_range=UV_RANGE):
    plt.figure(figsize=(10, 4))
    plt.plot(wave, snr, color='darkorange', lw=1.0, alpha=0.7)
    plt.axvspan(wave_range[0], wave_range[1], color='lightgrey', alpha=0.3, label='UV continuum range')
    if not np.isnan(avg_snr):
        plt.axhline(avg_snr, color='red', lw=1.0, linestyle='--', label=f'⟨SNR⟩ = {avg_snr:.2f}')
    plt.xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=13)
    plt.ylabel("S/N per pixel", fontsize=13)
    plt.title(f"Average SNR in UV range ({wave_range[0]}–{wave_range[1]} Å)\n{fits_name}", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(out_dir, fits_name.replace('.fits', '_UV_SNR.png'))
    plt.savefig(out_plot, dpi=200)
    plt.close()
    return out_plot


def passes_quality_checks(flux, snr_uv):
    """
    Check if the spectrum passes the quality criteria:
    1. No band gap in the entire spectrum (no NaNs in flux).
    2. Average SNR in UV range is positive.
    """
    if not np.all(np.isfinite(flux)):
        return False
    if snr_uv is None or np.isnan(snr_uv) or snr_uv <= 0:
        return False
    return True


def process_spectrum(fits_path, csv_path, out_dir):
    fits_name = os.path.basename(fits_path)
    z = get_redshift(csv_path, fits_name)
    print(f"Processing {fits_name}, z = {z:.4f}")

    wave_obs, flux_obs, err_obs = read_spectrum(fits_path)
    wave_rest, flux_rest, err_rest = convert_to_rest_frame(wave_obs, flux_obs, err_obs, z)
    wave, flux, err = clean_spectrum(wave_rest, flux_rest, err_rest)

    plot_spectrum(wave, flux, err, fits_name, z, out_dir)

    snr, valid_mask = compute_snr(flux, err)
    wave_valid = wave[valid_mask]
    avg_snr_uv = average_snr_in_range(wave_valid, snr)

    # Apply the quality checks before saving to results
    if passes_quality_checks(flux, avg_snr_uv):
        plot_snr(wave_valid, snr, avg_snr_uv, fits_name, out_dir)
        print(f"Saved {fits_name} with avg SNR (1250–3000 Å) = {avg_snr_uv:.2f}")
        return {"file": fits_name, "avg_snr_uv": avg_snr_uv}
    else:
        print(f"Discarded {fits_name} due to failing quality checks")
        return None

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    fits_files = find_prism_fits(SPECTRA_DIR)
    print(f"Found {len(fits_files)} prism FITS files to process.\n")

    results = []
    for f in fits_files:
        try:
            res = process_spectrum(f, CSV_PATH, OUT_DIR)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"Failed {f}: {e}")

    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUT_DIR, "uv_snr_summary_1.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved filtered SNR summary to: {results_csv}")



