# Beyond the Herd — STRIDE replication package

Replication and simulation code accompanying the manuscript:

**“Beyond the Herd: When Does Exposure Redistribution Reduce Credit Rationing of First-Time Borrowers? STRIDE and Simple Baselines in a Prosper-Calibrated Lending Marketplace Simulator.”**

Authors: Nazif Tchagafo, Abderrahmane Ez-zahout, and Belaid Ahiod  
Affiliation: Mohammed V University in Rabat, Morocco  
Contact: ntchagafo@gmail.com

> **Important scope statement.** The current `sim.py` describes the marketplace simulator as a reconstruction from the paper's description and marks several model components as `[ASSUMED]`. Results from this code should therefore be described as simulation results conditional on the stated assumptions—not as causal estimates or direct observations of Prosper platform exposure or funding decisions. Do not describe this package as a fully independent reproduction of every empirical feature unless the authors have verified the code, configuration, and outputs against the final manuscript.

## 1. Repository contents

The public repository currently contains these six root files:

| File | Purpose |
|---|---|
| `calibrate.py` | Reads the Prosper CSV and creates loan-side calibration parameters and calibration-cell statistics. |
| `requirements.txt` | Python package dependencies. |
| `run_all.py` | Runs the experiment stages (`tau`, `tune`, `main`, `frontier`, `scenarios`, `global`, or `all`). |
| `sim.py` | Agent-based lending-marketplace simulation and recommendation policies. |
| `stride_analysis.py` | Statistical post-processing and calibration utilities. |
| `README.md` | Project documentation and execution instructions. |

The following folders are created/used locally and are not part of the current tracked source tree unless you add them yourself:

```text
Beyond-herd-Stride/
├── README.md
├── calibrate.py
├── requirements.txt
├── run_all.py
├── sim.py
├── stride_analysis.py
├── data/       # place the Prosper CSV here; do not commit the raw dataset
├── configs/    # generated calibration and tuning JSON files
└── outputs/    # generated CSV results
```

## 2. Data and calibration

The calibration script expects the public Prosper loan dataset as `data/prosperLoanData.csv`. The dataset is not included in this package. The script selects 36-month loans with a Prosper rating and resolved statuses (`Completed`, `Chargedoff`, or `Defaulted`) listed from 1 July 2009 up to (but not including) 1 March 2011. First-time status is derived from missing `TotalProsperLoans`.

The script writes:

- `configs/calib.json` — calibration parameters used by the simulator;
- `configs/calib_cells.csv` — borrower-type/grade calibration-cell statistics.

The script prints the SHA-256 digest of the input CSV. Record that digest in your research archive and compare it with the dataset version used for the manuscript. Do not insert a guessed checksum in this README.

### Prepare the data folder

Download the dataset from the source/mirror documented in the manuscript and place it at:

```text
data/prosperLoanData.csv
```

Do not upload the raw dataset to GitHub unless its licence and redistribution terms explicitly permit it.

## 3. Installation

Python 3.10 or newer is recommended. From the repository root:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force data, configs, outputs | Out-Null
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data configs outputs
```

## 4. Reproduce the experiment workflow

Run all commands from the repository root, with the virtual environment activated and the Prosper CSV in `data/`.

### Step 1 — calibrate the loan side

```bash
python calibrate.py data/prosperLoanData.csv configs/calib.json
```

This step is essential. If `configs/calib.json` is absent, `run_all.py` explicitly warns that it will use an invented fallback loan-side calibration, **not the Prosper calibration** used for the paper. Do not use fallback results as the Prosper-calibrated results.

### Step 2 — run the experiment stages

```bash
python run_all.py --stage tau
python run_all.py --stage tune
python run_all.py --stage main
python run_all.py --stage frontier
python run_all.py --stage scenarios
python run_all.py --stage global
```

To run the sequence through the driver:

```bash
python run_all.py --stage all
```

Optional arguments:

```bash
python run_all.py --stage main --workers 4
python run_all.py --stage all --out outputs --calib configs/calib.json
```

`--workers` controls parallel worker processes. `--quick` is intended for a reduced smoke test; it changes simulation size, seed counts/iterations, and bootstrap effort. **Do not use `--quick` to generate the manuscript's reported results.**

### Step 3 — inspect generated files

The driver creates `configs/` and the selected output directory if needed. Depending on the stage, the generated files include `configs/tau.json`, `configs/tuned.json`, tuning CSVs, and result tables/replication-level CSVs under `outputs/`. Keep the raw replication-level CSVs when preparing tables and figures; do not report only rounded summaries.

## 5. Experiment stages and random seeds

The experiment driver defines the following seed ranges and stages:

| Stage | Seeds / design | Main purpose |
|---|---|---|
| `tau` | 900–902 | Calibrate the scarcity parameter so Hybrid is near the target funding-volume level. |
| `tune` | 101–104 | Select policy intensities subject to the stated funded-volume constraint. |
| `main` | 1–30 | Main comparison across policies. |
| `frontier` | 201–216 (benchmark uses 201–206) | Funding-volume frontiers and equal-volume comparisons. |
| `scenarios` | 301–305 | Scenario analysis. |
| `global` | 24 parameter draws; draw seed 2026 and three replications per draw | Global sensitivity analysis. |

The simulation code uses separate random streams for the world and daily decisions (see `sim.py`). Preserve the seed mapping when reproducing results. The `all` stage runs the stages in the order implemented in `run_all.py`.

## 6. Policies and outcomes

The experiment driver compares the Hybrid reference policy with baseline and exposure-redistribution policies, including STRIDE and STRIDE variants. The precise policy definitions and parameter grids are implemented in `sim.py` and `run_all.py`; consult those files rather than relying on this summary when interpreting a particular policy.

The analysis code includes paired comparisons, confidence intervals, equal-volume interpolation, bootstrap intervals, and an equivalence-verdict helper. An interval that overlaps an equivalence boundary is **unresolved**; it must not be described as proof of equivalence. In particular, report any unresolved STRIDE-versus-comparator setting as unresolved, not as “no difference.”

## 7. Interpretation and limitations

The package is a simulator. The investor-side behaviour is modelled rather than identified from Prosper impression-level data; the Prosper loan file does not provide the complete exposure/impression history or unfunded listing process needed to estimate those mechanisms directly. The model also contains assumptions explicitly marked in the source code. Consequently:

1. interpret findings as conditional simulation evidence;
2. distinguish Prosper-calibrated loan-side inputs from assumed investor-side mechanisms;
3. do not claim causal effects on real-world borrower funding;
4. do not generalise automatically from consumer lending to SME lending;
5. verify that every table, figure, confidence interval, and parameter in the manuscript can be regenerated from the archived code and outputs.

## 8. Citation

Please cite the associated manuscript using its final bibliographic information once the journal assigns volume, issue, pages, and DOI. Until then, use the manuscript title and author list above without inventing a DOI or publication status.

## 9. Reproducibility checklist before public release

- [ ] Confirm that the repository code is the exact version used to generate the submitted manuscript.
- [ ] Run calibration with the exact Prosper CSV and archive the printed SHA-256 digest.
- [ ] Confirm that `configs/calib.json` is loaded for every Prosper-calibrated run.
- [ ] Run the full experiment without `--quick`.
- [ ] Archive all raw CSV outputs, configuration JSON files, Python version, and dependency versions.
- [ ] Recompute manuscript tables and figures from those archived outputs.
- [ ] Verify every reported confidence interval and equivalence classification.
- [ ] Review all `[ASSUMED]` model components and disclose them in the manuscript and repository.
- [ ] Remove temporary files and confirm that no restricted raw data, credentials, or personal information are committed.
