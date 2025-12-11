import pickle

def build_reverse_lookup(filter_dict):
    reverse = {}
    for facility, instruments in filter_dict.items():
        for instrument, filters in instruments.items():
            for filt in filters:
                reverse[filt] = (facility, instrument)
    return reverse

FILTER_DICT = {
    "HST": {
        "ACS_WFC": {
            # Broad filters
            'F435W', 'F475W', 'F550M', 'F555W', 'F606W', 'F625W',
            'F658N', 'F660N', 'F775W', 'F814W', 'F850LP',

            # Narrow/medium filters
            'F502N', 'F658N', 'F660N', 'F892N',
            'FR388N',   # ramp filter (one of the few still commonly used)
        }
    },

    "JWST": {
        "NIRCam": {
            # Short-wavelength channel (SW) — wide
            'F070W', 'F090W', 'F115W', 'F150W', 'F200W',

            # SW medium
            'F140M', 'F162M', 'F182M', 'F210M',

            # SW narrow
            'F164N', 'F187N', 'F212N',

            # Long-wavelength channel (LW) — wide
            'F277W', 'F322W2', 'F356W', 'F410M', 'F444W',

            # LW medium
            'F250M', 'F300M', 'F335M', 'F360M',

            # LW narrow
            'F323N', 'F405N', 'F430M', 'F460M', 'F480M'
        }
    }
}

REVERSE_FILTER_DICT = build_reverse_lookup(FILTER_DICT)

with open("/nvme/scratch/work/jarcidia/MPHYS_Project/WORK/FILTER_CODE/Filter_look_up_dict.pkl", "wb") as f:
    pickle.dump(REVERSE_FILTER_DICT, f)