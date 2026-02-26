import os
from astropy.io import fits

# Paths
SPECTRA_DIR = "/raid/scratch/work/austind/GALFIND_WORK/Spectra/2D"
CATALOGUE_PATH = "/nvme/scratch/work/austind/EPOCHS-v2/tabs/spectra/EPOCHS-v2.fits"

# Load catalogue
with fits.open(CATALOGUE_PATH) as hdul:
    data = hdul[1].data  # assuming table is in first extension
    catalogue_files = [f.lower() for f in data['file']]  # lower for case-insensitive comparison

# Walk spectra directory recursively
spectra_files = set()
for root, dirs, files in os.walk(SPECTRA_DIR):
    for f in files:
        spectra_files.add(f.lower())

# Find missing spectra
missing_files = [f for f in catalogue_files if f not in spectra_files]

# Report
print(f"Found {len(missing_files)} missing spectra:")
for f in missing_files:
    print(f)
