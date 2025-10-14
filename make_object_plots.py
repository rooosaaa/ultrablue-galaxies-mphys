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
# target_name = df_targets['file'].iloc[0]  # or whichever column has filenames
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
    """Extract 1D spectrum arrays (wavelength, flux, error)"""
    with fits.open(fits_path) as hdul:
        spec1d = hdul['SPEC1D'].data
        wave = spec1d['wave']
        flux = spec1d['flux']
        err = spec1d['err']
    return wave, flux, err


def read_2d_spectrum(fits_path, z):
    """Extract the 2D spectral image and compute rest-frame wavelengths from header"""
    with fits.open(fits_path) as hdul:
        if 'SPEC2D' in hdul:
            hdu = hdul['SPEC2D']
        else:
            hdu = hdul['SCI']
        data_2d = hdu.data
        header_2d = hdu.header

    # Extract wavelength info from header
    wav_start = header_2d['WAVSTART'] * 1e10  # m -> Å
    wav_end = header_2d['WAVEND'] * 1e10      # m -> Å
    n_spec = data_2d.shape[1]

    # Linear mapping from pixel to observed wavelength
    wave_obs_2d = np.linspace(wav_start, wav_end, n_spec)

    # Convert to rest-frame
    wave_rest_2d = wave_obs_2d / (1 + z)

    return data_2d, wave_rest_2d


def convert_to_rest_frame(wave_obs, flux_obs, err_obs, z):
    """Convert observed spectrum to rest-frame F_lambda in erg/s/cm^2/Å"""
    flux_nu = flux_obs * 1e-29  # assuming input is Jy
    err_nu = err_obs * 1e-29
    wave_A = wave_obs * 1e4  # microns -> Å if applicable
    flux_lambda = flux_nu * C_LIGHT / wave_A**2
    err_lambda = err_nu * C_LIGHT / wave_A**2
    wave_rest = wave_A / (1 + z)
    flux_rest = flux_lambda * (1 + z)**2
    err_rest = err_lambda * (1 + z)**2
    return wave_rest, flux_rest, err_rest


def clean_spectrum(wave, flux, err, min_wave=MIN_WAVELENGTH):
    mask = wave > min_wave
    return wave[mask], flux[mask], err[mask]


def plot_2d_spectrum(ax, data_2d, wave_rest, target_name, z):
    """
    Plot the 2D spectrum with rest-frame wavelength on x-axis.
    """
    # Number of spectral pixels in 2D
    n_spec = data_2d.shape[1]

    # Map 2D pixels linearly to the rest-frame wavelength range of the 1D spectrum
    wave_2d_rest = np.linspace(wave_rest[0], wave_rest[-1], n_spec)

    # Clip for contrast
    vmin, vmax = np.percentile(data_2d, [10, 99])
    
    n_spatial = data_2d.shape[0]
    extent = [wave_2d_rest[0], wave_2d_rest[-1], 0, n_spatial-1]
    
    im = ax.imshow(
        data_2d,
        aspect='auto',
        origin='lower',
        cmap='magma',
        interpolation='none',
        vmin=vmin,
        vmax=vmax,
        extent=extent
    )
    
    ax.set_ylabel("Spatial axis (pix)")
    ax.set_xlabel("Rest-frame Wavelength [Å]")
    ax.set_title(f"2D + 1D Spectrum: {target_name}\n(z = {z:.3f})")
    
    # Add colorbar
    # cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    # cbar.set_label('Flux / counts', rotation=270, labelpad=15)
    
    return im


def plot_1d_spectrum(ax, wave, flux, err):
    """
    Plot the 1D rest-frame spectrum with uncertainty (wavelength 1200-10000 Å).
    """
    # mask = (wave >= 1200)
    # wave, flux, err = wave[mask], flux[mask], err[mask]

    ax.plot(wave, flux, color='dodgerblue', lw=1.0, label='Rest-frame spectrum')
    ax.fill_between(wave, flux - err, flux + err, color='lightblue', alpha=0.4)
    ax.set_xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]")
    ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)


def plot_spectra_2d_1d(data_2d, wave_rest, wave_rest_2d, flux, err, z, target_name, out_dir):
    """
    Combine the 2D and 1D spectra into a single stacked figure with matching x-axis.
    """
    fig, ax = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [2, 1]}, sharex=True
    )

    # Plot both
    plot_2d_spectrum(ax[0], data_2d, wave_rest_2d, target_name, z)
    plot_1d_spectrum(ax[1], wave_rest, flux, err)

    # Limit 1D wavelength axis
    # ax[1].set_xlim(left=1200)

    plt.tight_layout()
    out_path = os.path.join(out_dir, target_name.replace('.fits', '_2D_1D.png'))
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------- MAIN ----------------------

# Load both 1D + 2D spectra
wave_obs, flux_obs, err_obs = read_spectrum(fits_path)
data_2d, wave_rest_2d = read_2d_spectrum(fits_path, z)

# Convert to rest-frame
wave_rest, flux_rest, err_rest = convert_to_rest_frame(wave_obs, flux_obs, err_obs, z)
wave_rest, flux_rest, err_rest = clean_spectrum(wave_rest, flux_rest, err_rest)

# Plot
plot_spectra_2d_1d(data_2d, wave_rest, wave_rest_2d, flux_rest, err_rest, z, target_name, out_dir)

