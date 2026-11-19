# Sheikhalishahi et al. extraction runner (T44, T45)

Runs their public code (`github.com/mostafaalishahi/eICU_Benchmark_updated` at
cd9b83d) locally in Docker, with the package versions their readme pins. No
network inside the container; eICU CSVs mounted read-only.

* `Dockerfile`: python 3.7 with numpy 1.15.0, scipy 1.2.0, scikit-learn 0.21.2,
  pandas 0.24.1 (`docker build -t sheikhalishahi-extract:cd9b83d .`).
* `nc_chunked_patch.py`: the one change, input only. Their `read_nc_table` loads
  all of `nurseCharting.csv` (about 40 GB in pandas 0.24); this reads it in
  chunks with the same columns and item filter. Every later step is theirs.
* `check_patch.py`: run in the container on synthetic files; shows the chunked
  reader returns a frame identical to the original at three chunk sizes.
* `run_extraction.py`: entry point, their `data_extraction_root` with the patch.

Their code is used from a clean `git archive` of cd9b83d with the committed
`__pycache__` removed; it is not vendored here.
