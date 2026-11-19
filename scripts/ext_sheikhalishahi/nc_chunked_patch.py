"""I/O-only patch for data_extractor/utils.py: read nurseCharting.csv in chunks.

The original ``read_nc_table`` loads all 151.6M rows and 8 columns at once
(about 40 GB in pandas 0.24), which does not fit in 30 GB of RAM. This version
reads only the five columns ``filter_nc_on_columns`` keeps, applies the same
item filter to each chunk, and concatenates in file order; every later step is
the original code. check: ``check_patch.py`` (run in the container) shows the frame is identical.
"""
import os
import pandas as pd

from data_extractor import utils

NC_COLUMNS = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel',
              'nursingchartcelltypevalname', 'nursingchartvalue']
TYPEVALLABEL = ['Glasgow coma score', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate',
                'MAP (mmHg)', 'Arterial Line MAP (mmHg)']
TYPEVALNAME = ['Non-Invasive BP Systolic', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
               'Invasive BP Diastolic', 'Temperature (C)', 'Temperature (F)']


def read_nc_table_chunked(eicu_path, chunksize=5000000):
    parts = []
    for chunk in pd.read_csv(os.path.join(eicu_path, 'nurseCharting.csv'), header=0,
                             index_col=False, usecols=NC_COLUMNS, chunksize=chunksize):
        chunk = utils.filter_nc_on_columns(chunk)
        utils.rename_nc_columns(chunk)
        parts.append(utils.item_name_selected_from_nc(chunk, TYPEVALLABEL, TYPEVALNAME))
    nc = pd.concat(parts)
    nc = utils.check_itemvalue(nc)
    nc = utils.conv_far_cel(nc)
    utils.replace_itemname_value(nc)
    del nc['itemlabel']
    return nc
