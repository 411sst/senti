# T40 and T41 runbook: rebuild and serving latency on an ordinary laptop

This is for the two team laptops in the paper tests: the **Core i5-13500HX**
and the **Core i9-13900H**, both on Windows. Each laptop runs the same steps.

* **What it measures.** T40 rebuilds all eight agents and both fusion heads
  from the hourly data on one CPU thread, and checks whether the scores are
  bit-identical to the reference rebuild (T36). T41 measures how fast one
  patient-hour is scored and how long an explanation card takes on the CPU.
* **Time.** About 2 hours of the laptop's time, mostly unattended:
  roughly 60–90 min for step 5, 15–25 min for step 6 and 10–20 min for step 7.
* **What leaves the laptop.** One file, `t40\result.json`. It holds timings,
  memory figures, test AUPRC/AUROC and sha256 checksums of score arrays. The
  runner refuses to write it if it contains anything shaped like a stay id,
  your user name or the computer's name. Send nothing else: no logs, nothing
  under `data\`.

## Before you start

* You need your own **PhysioNet-credentialed copy** of the seven hourly input
  files below, received over a channel the data use agreement allows. The
  runner checks each file's sha256 against the reference rebuild's.

  | file |
  | --- |
  | `ecg_index.parquet` |
  | `ecg_machine_measurements.parquet` |
  | `mimic_cadence_hourly.parquet` |
  | `mimic_cohort_hourly.parquet` |
  | `mimic_fluids_hourly.parquet` |
  | `mimic_labs_hourly.parquet` |
  | `mimic_resp_hourly.parquet` |

* Install **Python 3.12 (64-bit)** from python.org, **git**, and **ollama**
  (for step 7). Windows 10 and 11 already include `curl`.
* About 15 GB of free disk: 2 GB for the environment, 2 GB for the rebuild,
  and up to 11 GB for the two language models.
* Plug in the charger. Close browsers, IDEs, chat apps, games and sync
  clients. **Don't use the laptop while steps 5–7 run**: other programs change
  the timings and memory figures.

## 1. Clone the code into a new folder

Use a new folder, never your normal Sentinel checkout. Several stages
overwrite cached model outputs, and the runner refuses to start if
`data\cache` holds anything other than the seven inputs.

```powershell
cd $HOME
git clone https://github.com/amateurmonke/sentinel.git sentinel-t40
cd sentinel-t40
git checkout paper-tests
git remote remove origin
```

## 2. Create the environment

This installs exactly the package versions of the reference rebuild
(torch 2.11.0, CPU-only build).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-t40.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 3. Copy in the inputs

```powershell
New-Item -ItemType Directory -Force data\cache\embeddings
Copy-Item <folder with your copy>\*.parquet data\cache\
```

`data\cache` should now hold the seven files above and an empty
`embeddings` folder, nothing else.

## 4. Check

```powershell
.\.venv\Scripts\python scripts\t40_rebuild.py check
```

It prints the machine facts and saves `t40\machine.json`. Look for
`"inputs_match_t36": true`. If it says a checksum differs, stop: your input
files are a different version, and the comparison would be meaningless.

## 5. T40, grid mode (about 60–90 min)

```powershell
.\.venv\Scripts\python scripts\t40_rebuild.py grid
```

The seven stages run one after another, each printing a line when it
finishes. At the end it prints whether every score array matches the
reference.

## 6. T40, reuse mode (about 15–25 min)

```powershell
.\.venv\Scripts\python scripts\t40_rebuild.py reuse
```

This reruns only the GBM stages and the heads, with the published tuning
instead of the six-model search.

## 7. T41, serving latency and explanation cards

Download the two language models once (about 3 GB and 8 GB), then close
everything except ollama again:

```powershell
ollama --version
ollama pull medgemma:4b
ollama pull gemma4:12b
.\.venv\Scripts\python scripts\bench_single_node.py --threads 1,2,4,8 --reps 100 --llm-cards 4 --llm-models medgemma:4b,gemma4:12b
```

The language model is forced onto the CPU (`num_gpu: 0`), even on the laptop
with a discrete GPU. On the 16 GB laptop, ollama may refuse `gemma4:12b` for
lack of memory. The benchmark then records it as skipped, with ollama's
reason. That is an accepted result: don't retry.

## 8. Bundle and send

```powershell
.\.venv\Scripts\python scripts\t40_rebuild.py bundle
```

Send **only** `t40\result.json`.

## 9. Clean up

After the result has been received, delete the whole `sentinel-t40` folder.
Its `data\cache` holds data-use-agreement material and model outputs.

## If something goes wrong

* *"data/cache holds files other than the inputs"* or *"embeddings is not
  empty"*: you are in the wrong folder, or grid mode already ran there.
  Start again from step 1 in a new folder.
* *"stage … failed"*: report the stage name and the last error line of
  `logs\t40_grid_<stage>.log`. Don't send the log file itself.
* A step stops part-way, for example because the laptop went to sleep:
  start again from step 1 in a new folder, so the timings stay clean.
