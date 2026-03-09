# CreditScoringModel

Classic credit scoring model built around a simple benchmark classifier and a custom, interpretable scorecard-style model.

## Requirements / Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 2) Install dependencies

This project uses common Python ML/DS libraries:

- `pandas`
- `scikit-learn`
- `joblib`
- `matplotlib`, `seaborn`
- `streamlit` (for the dashboard)
- `plotly` (for the dashboard)

Install them with:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can install the packages directly:

```bash
pip install pandas scikit-learn joblib matplotlib seaborn streamlit plotly
```

### 3) Project assets

The scripts expect the following folders/files to exist:

- `Data/clean_train.csv` (cleaned training dataset)
- `Models/` (models will be saved/loaded from here)

If you are starting from the raw dataset, check `Data/data_cleaner.py` to generate the cleaned CSV.

## Repository Structure (Modular Flow)

At a high level, the repo is designed to be run in two steps:

1. **Fit and persist models** (run `fitter.py` first)
2. **Evaluate and visualize** (run `main.py` afterwards)

### 1) `fitter.py` — fit models and save them to `Models/`

`fitter.py` trains two logistic regression models:

- A **benchmark model** trained on the “raw” feature set (after dropping ID-like fields).
- A **weights model** trained on the selected/processed features used for the custom score.

Run:

```bash
python fitter.py
```

Expected outputs:

- `Models/benchmark_model.pkl`
- `Models/weights_model.pkl`

### 2) `main.py` — run evaluation + build the custom score + plots

`main.py`:

- Loads `Data/clean_train.csv`
- Evaluates the benchmark model (`Models/benchmark_model.pkl`)
- Splits data using `utils.data_splitter`
- Loads the weights model (`Models/weights_model.pkl`) and extracts feature coefficients
- Produces a scaled custom score (0–600)
- Applies threshold-based classification
- Generates plots and confusion matrices

Run:

```bash
python main.py
```

## Key Modules

- `utils.py`
  - `get_benchmark(...)`: evaluates the benchmark model
  - `data_splitter(...)`: feature selection + train/test split + sign-negation for “directional” scoring

- `credit_model.py`
  - `get_weights(...)`: extracts interpretable coefficients for the “Good” class
  - `scores_df(...)`: generates and scales credit scores

- `visualizations.py`
  - plotting utilities for distributions, comparisons, and confusion matrices

## Dashboard (Streamlit)

The repository includes a simple interactive dashboard in `dashboard.py`.

### Run the dashboard

Make sure you have installed `streamlit` and `plotly` (see setup above), then run:

```bash
streamlit run dashboard.py
```

### What you should see

- A Streamlit web app opens in your browser (typically at `http://localhost:8501`).
- The UI lets you input / adjust the key credit features used by the model.
- The app computes a **weighted sum** and shows a **final credit score (0–600)**.
- The main visualization is a **gauge (“semaforo”)** that indicates risk bands:
  - **0–220**: red (higher risk)
  - **220–410**: yellow (medium risk)
  - **410–600**: green (lower risk)

## Notes

- Run order matters: **train first** (`python fitter.py`) so that `main.py` can load the saved models.
- Paths are currently relative (e.g., `Data/clean_train.csv`, `Models/...`), so run commands from the repository root.
