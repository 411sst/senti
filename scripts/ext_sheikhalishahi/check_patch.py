"""The chunked nurseCharting reader returns exactly what the original returns."""
import os, sys, tempfile
sys.dont_write_bytecode = True
sys.path.insert(0, "/work/code"); sys.path.insert(0, "/work/runner")
import numpy as np, pandas as pd
from data_extractor import utils
from nc_chunked_patch import read_nc_table_chunked

rng = np.random.RandomState(0)
labels = ['Glasgow coma score', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'MAP (mmHg)',
          'Arterial Line MAP (mmHg)', 'Pain Score', 'Bedside Glucose', 'Invasive BP']
names = ['Non-Invasive BP Systolic', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
         'Invasive BP Diastolic', 'Temperature (C)', 'Temperature (F)', 'Value', 'GCS Total',
         'Eyes', 'Motor', 'Verbal', 'Other']
n = 20000
values = np.where((rng.rand(n) < 0.05) & (np.arange(n) < 1500), 'abc', np.round(rng.uniform(1, 200, n), 3).astype(str))
values[3000:3100] = ''
df = pd.DataFrame({
    'nursingchartid': np.arange(n), 'patientunitstayid': rng.randint(1000, 1100, n),
    'nursingchartoffset': rng.randint(-100, 5000, n), 'nursingchartentryoffset': rng.randint(0, 5000, n),
    'nursingchartcelltypecat': 'Vital Signs',
    'nursingchartcelltypevallabel': rng.choice(labels, n), 'nursingchartcelltypevalname': rng.choice(names, n),
    'nursingchartvalue': values})
d = tempfile.mkdtemp()
df.to_csv(os.path.join(d, 'nurseCharting.csv'), index=False)
full = utils.read_nc_table(d)
for cs in (997, 5000, 100000):
    chunked = read_nc_table_chunked(d, chunksize=cs)
    pd.testing.assert_frame_equal(full, chunked)
print("identical:", full.shape, "rows kept of", n)
