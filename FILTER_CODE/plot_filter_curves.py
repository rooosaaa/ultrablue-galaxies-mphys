import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from adjustText import adjust_text
from astropy import units as u
from astroquery.svo_fps import SvoFps

# COSMOS_WEB_PHOT = ['HST/ACS_WFC.F814W', 'JWST/NIRCam.F115W', 'JWST/NIRCam.F150W', 'JWST/NIRCam.F277W', 'JWST/NIRCam.F444W']
# COSMOS_GROUND = ['Subaru/HSC.g', 'Subaru/HSC.r', 'Subaru/HSC.i', 'Subaru/HSC.z', 'Subaru/HSC.Y', 
#                  'Paranal/VISTA.Y', 'Paranal/VISTA.J', 'Paranal/VISTA.H', 'Paranal/VISTA.Ks']

# PRIMER_COSMOS = ['JWST/NIRCam.F090W', 'JWST/NIRCam.F115W', 'JWST/NIRCam.F150W', 'JWST/NIRCam.F200W', 
#                  'JWST/NIRCam.F277W', 'JWST/NIRCam.F356W', 'JWST/NIRCam.F410M', 'JWST/NIRCam.F444W']

NIRCAM_W2 = ['JWST/NIRCam.F150W2', 'JWST/NIRCam.F322W2']
NIRCAM_W = ['JWST/NIRCam.F070W', 'JWST/NIRCam.F090W', 'JWST/NIRCam.F115W', 'JWST/NIRCam.F150W', 
            'JWST/NIRCam.F200W', 'JWST/NIRCam.F277W', 'JWST/NIRCam.F356W', 'JWST/NIRCam.F444W']

NIRCAM_M = ['JWST/NIRCam.F140M', 'JWST/NIRCam.F162M', 'JWST/NIRCam.F182M', 'JWST/NIRCam.F210M', 'JWST/NIRCam.F250M', 'JWST/NIRCam.F300M', 
            'JWST/NIRCam.F335M', 'JWST/NIRCam.F360M', 'JWST/NIRCam.F410M', 'JWST/NIRCam.F430M', 'JWST/NIRCam.F460M', 'JWST/NIRCam.F480M']
NIRCAM_N = ['JWST/NIRCam.F164N', 'JWST/NIRCam.F187N', 'JWST/NIRCam.F212N', 'JWST/NIRCam.F323N', 
            'JWST/NIRCam.F405N', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F470N']

poop = SvoFps.get_filter_list('JWST', instrument='NIRCam')

def get_filter_curve(filter_path):

    return SvoFps.get_transmission_data(filter_path)

def make_filter_plot(
    filter_array, series_names, save_file_name,
    Plot_grid=True, label_mode='Annotate', plot_height=3,
    adjust_labels=True, label_offset=0.075,
    X_LIM_L=None, X_LIM_U=None, Y_LIM_L=None, Y_LIM_U=None):

    # --- Compute global wavelength limits across all filters ---
    all_wavelengths = []
    central_wavelengths = []  # List to hold central wavelengths
    max_transmissions = []    # List to hold max transmissions

    for filters in filter_array:
        for filter_path in filters:
            data = get_filter_curve(filter_path)
            all_wavelengths.append(data['Wavelength'])
            central_wavelength = 0.5 * (data['Wavelength'].min() + data['Wavelength'].max())
            central_wavelengths.append(central_wavelength)
            max_transmissions.append(data['Transmission'].max())  # Track max transmission for global Y_LIM_U

    global_min_wave = min([w.min() for w in all_wavelengths])
    global_max_wave = max([w.max() for w in all_wavelengths])
    global_max_transmission = max(max_transmissions)

    # Override user-provided limits if not set
    if X_LIM_L is None:
        X_LIM_L = global_min_wave * 0.8
    if X_LIM_U is None:
        X_LIM_U = global_max_wave * 1.05
    if Y_LIM_L is None:
        Y_LIM_L = 0

    # --- Set up figure ---
    n = len(filter_array)
    fig, axes = plt.subplots(n, 1, figsize=(10, plot_height * n), squeeze=False)
    fig.subplots_adjust(wspace=0.3, hspace=0.6)

    # Normalize the central wavelength for the colormap
    norm = plt.Normalize(min(central_wavelengths), max(central_wavelengths))
    cmap = cm.rainbow

    for i, (filters, name) in enumerate(zip(filter_array, series_names)):
        ax = axes[i][0]
        max_transmission_this_ax = 0
        text_objects = []
        x_anno = []
        y_anno = []

        for j, filter_path in enumerate(filters):
            instrument_name, filter_name = filter_array[i][j].split('/')[1].split('.')

            data = get_filter_curve(filter_path)
            wavelength = np.array(data['Wavelength'])
            transmission = np.array(data['Transmission'])

            central_wavelength = 0.5 * (wavelength.min() + wavelength.max())
            color = cmap(norm(central_wavelength))

            ax.plot(wavelength, transmission, label=f'{instrument_name}/{filter_name}', color=color)
            ax.fill_between(wavelength, transmission, Y_LIM_L, color=color, alpha=0.5)

            peak = transmission.max()

            if label_mode == 'Annotate':
                x_annotate = 0.5 * (wavelength.min() + wavelength.max())
                y_annotate = peak * (1 + label_offset)

                text = ax.annotate(
                    filter_name,
                    xy=(x_annotate, y_annotate),
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    color='k'
                )
                text_objects.append(text)


        # Y-axis upper limit per axis
        final_Y_LIM_U = global_max_transmission * 1.2 + 0.1 * global_max_transmission

        ax.set_title(name, fontsize=20)
        ax.set_xlabel('Wavelength $(\AA)$', fontsize=20)
        ax.set_ylabel('Transmission', fontsize=20)
        ax.set_xlim(left=X_LIM_L, right=X_LIM_U)
        ax.set_ylim(bottom=Y_LIM_L, top=final_Y_LIM_U)

        if Plot_grid:
            ax.grid(True, alpha=0.5)
        else:
            ax.grid(False)

        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=14)

        if label_mode == 'Legend':
            ax.legend()

        # Adjust text to avoid overlaps
        if label_mode == 'Annotate' and adjust_labels:
            _ = adjust_text(
                    text_objects,
                    ax=ax,
                    autoalign='x',
                    only_move={'points': 'y', 'text': 'y'},
                    expand=(1.05, 1.05),
                    force_text=0.3,
                    force_points=0.1,
                    arrowprops=None,
                    avoid_self=False
                )

    #plt.tight_layout()
    plt.savefig(fname=f'/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/FILTER_CODE/{save_file_name}_filter_curves.png',
                dpi=250, bbox_inches='tight', facecolor='white')

    return

make_filter_plot([NIRCAM_W2, NIRCAM_W, NIRCAM_M, NIRCAM_N], ['NIRCam W2 Filters', 'NIRCam W Filters', 'NIRCam M Filters', 'NIRCam N Filters'], 
                 'NIRCam', adjust_labels=True, label_offset=0.1, plot_height=4)