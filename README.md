# CFM-HES (MICCAI 2026 Submission)

This repository contains the code for our MICCAI 2026 paper submission on virtual stain generation in histopathology:

**Making HE Histopathological Images More Colorful by Conditional Flow Matching**

The project focuses on **HE -> HES translation** in the concentration domain.  
The pipeline follows three stages: (1) stain separation, (2) conditional flow matching for saffron generation, and (3) HES reconstruction using the Beer-Lambert law.

## Overview

![CFM pipeline](Figures/CFM_scheme.png)

The repository includes three FM-based models:

- `fm_indep`: unconditional Flow Matching model to generate HES-like samples from Gaussian noise.
- `fm_cond_s_he`: Conditional Flow Matching with independent coupling for HE -> HES.
- `fm_cond_ot_s_he`: Conditional Flow Matching with OT coupling for HE -> HES.

## Demo (Streamlit)

A Streamlit demo is available in [Demo/app.py](Demo/app.py).

The demo allows:

- CFM generation on **Hospital Data** and **WSSB** data.
- Organ selection for WSSB (`breast` / `lung`).
- Stain separation visualization for **H**, **E**, and **S** components.
- Display of generated images and quality metrics (MSE, SSIM, LPIPS, PieAPP when available).

Run locally:

```bash
cd "CFM HES"
python -m pip install -r requirements.txt
streamlit run Demo/app.py
```

## Docker

A Docker setup is provided in [Dockerfile](Dockerfile).

Build and run:

```bash
cd "CFM HES"
docker build -t cfm-hes-demo .
docker run --rm -it -p 8501:8501 cfm-hes-demo
```

Then open:

- `http://localhost:8501`

## Requirements

All dependencies are listed in [requirements.txt](requirements.txt) with pinned or constrained versions for reproducibility.

## Training / Model Selection

Use [main.py](main.py) and set `model` in `--opts`:

- `model fm_indep`
- `model fm_cond_s_he`
- `model fm_cond_ot_s_he`

Example:

```bash
python main.py --opts \
  root "put path of your data here" \
  data_root "put path of your data here" \
  he_root "put path of your data here" \
  dataset hes \
  train True \
  eval False \
  model fm_cond_ot_s_he
```

## Subjective Evaluation

We conducted subjective evaluation on three organs:

- **liver**
- **breast**
- **lung**

Patch examples are available in patch_examples folder.

Subjective evaluation platform:

- https://subjective-hes-image-evaluation.vercel.app/

## Notes

- Replace every placeholder path (`put path of your data here`) with your own local paths.
- To test the model, you can download the checkpoint from: https://zenodo.org/records/18792551?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJiNjZmNmYyLTY5MGItNGVlNy05YWJmLTZiZDhmOGE0Y2U3NSIsImRhdGEiOnt9LCJyYW5kb20iOiJjNGM3NDQ2ZTRiOGRhODk1YTU5Y2U0MTIzYjNlZjNiOSJ9.-qJt4vnHrlyN0dgfVDdhSROjjhGWVIFMnbGgEeE--cAcB8GuYL_oov-eHHlbm_RJ3Sp5pxzzoGveMmJ2a42Gow
- This repository is prepared for article submission and anonymized sharing.
