# Emulsion Gel Confocal Microstructure ML

This repository contains the analysis code used to relate confocal image-derived emulsion gel microstructure descriptors to rheological properties, including G' at 1 Hz and breaking stress.

The workflow compares three predictor sets:

1. Formulation only
2. Microstructure descriptors only
3. Formulation + microstructure descriptors

It also generates publication-style figures, including descriptor-rheology plots, PCA plots, model comparison heatmaps, parity plots, Bland-Altman plots, and SHAP interpretation figures.

## Repository Structure

```text
.
├── src/
│   └── emulsion_gel_confocal_ml_publication.py
├── examples/
│   └── unique_formulations.csv
├── docs/
│   └── emulsion_gel_25_unique_formulations_landscape.docx
├── data/
│   └── raw/
│       └── README.md
├── outputs/
├── requirements.txt
└── README.md
```

## Data Expected by the Script

By default, place the experimental files here:

```text
data/raw/emulsion gel confocal.xlsx
data/raw/real confocal images of emulsion gels/
```

The Excel file should contain image names, formulation variables, G' at 1 Hz, and breaking stress. The image folder should contain the matching confocal image files, for example `picture1.tif`.

Raw data and images are not included by default because they may be too large or journal-restricted. If journal policy allows, add the raw data or a representative example dataset.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis:

```bash
python src/emulsion_gel_confocal_ml_publication.py
```

Outputs will be saved to:

```text
outputs/run_YYYYMMDD_HHMMSS/
```

## Run With Data Outside the Repository

If the data are stored elsewhere, set environment variables before running:

```bash
export EMULSION_GEL_ROOT="/path/to/project/root"
export EMULSION_GEL_EXCEL="/path/to/emulsion gel confocal.xlsx"
export EMULSION_GEL_IMAGE_DIR="/path/to/real confocal images of emulsion gels"
export EMULSION_GEL_OUTPUT_DIR="/path/to/output_folder"
python src/emulsion_gel_confocal_ml_publication.py
```

For the original local analysis, the data were organized as:

```text
emulsion gel confocal.xlsx
real confocal images of emulsion gels/
```

## Validation Design

The model evaluation uses grouped cross-validation by formulation to avoid replicate-image leakage. Images from the same formulation are kept in the same fold.

One image, `picture16.tif`, is excluded from training, testing, and figures to keep the analysis consistent with the manuscript workflow.

## Optional Components

- `xgboost` is optional. If it is not installed, the script skips XGBoost.
- `shap` is required for SHAP interpretation figures.
- `hoomd` is optional and only used for simulation-guided structure generation.

## Recommended GitHub/Journals Notes

For journal submission, this repository is intended to provide:

- Analysis code
- Example formulation table
- Reproducible folder structure
- Dependency list
- Clear instructions for placing the raw data

If raw confocal images cannot be public, provide a data availability statement in the manuscript and include either representative images or a small anonymized example dataset.
