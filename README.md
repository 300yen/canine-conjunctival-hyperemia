# Canine conjunctival hyperemia

Code for segmenting the conjunctiva and cornea in canine anterior segment photographs, measuring conjunctival Redness Score (RS) and vessel fractal dimension (FD), and reproducing the accompanying study analyses.

## Data and trained models

The [dataset on Zenodo](https://doi.org/10.5281/zenodo.23889738) contains images from 150 dogs (300 eyes), manual annotations for 200 eyes, predicted masks for all 300 eyes, five trained models, expert grades, measurements, and analysis results. Its README describes the files and variables.

Images are provided as 640 × 480 research-source PNGs and 512 × 512 analysis inputs. The original camera JPEG files are not included. Resizing the source PNGs with OpenCV `INTER_LINEAR` reproduces the analysis inputs pixel for pixel.

## Measure RS and FD

Use a Python 3.12 environment and install the analysis dependencies. Run commands from the repository root unless indicated otherwise.

```sh
python -m pip install -r requirements-analysis.txt
python demo.py
```

The demo runs on a synthetic image without downloading study data. To measure an image with an existing segmentation mask:

```sh
python measure.py image.png mask.png
```

Both inputs must be 512 × 512 pixels. The mask must be a single-channel label image: `0` for background, `1` for conjunctiva, and `2` for cornea. The command returns RS and FD as JSON; an undefined FD is returned as `null`.

For the study images, extract `analysis_dataset_300.zip` into `data/` and use matching files from `data/images/` and `data/predicted_masks/`.

## Segmentation

Use this workflow to generate masks from the released models. The stored masks can be used directly for measurement without running inference again.

Download `analysis_dataset_300.zip`, `original_resolution_images_300.zip`, and `prepare_segmentation_inputs.py` from Zenodo into the repository root. Prepare a new input directory:

```sh
python prepare_segmentation_inputs.py analysis_dataset_300.zip original_resolution_images_300.zip segmentation/inputs
```

Extract `model_checkpoints_5fold.zip` into `segmentation/`, so that `03_retrained_ckpt/` contains one checkpoint per fold. In a separate environment with PyTorch and the segmentation dependencies installed, run:

```sh
cd segmentation
python -m pip install -r requirements.txt
python run_inference_paper.py
python calc_metrics_from_saved.py
```

Paths are configured in `segmentation/configs/config.json`. Annotated eyes use their held-out fold model; the remaining eyes use the average probabilities from all five models. Label masks are written to `segmentation/02_ROI_Segmentation/predictions/masks_merged/`. The last command calculates per-eye and per-fold segmentation metrics, including the mean across conjunctiva and cornea, under `evaluation_paper/`.

To train a model, run `python training/train.py 1` from `segmentation/`, replacing `1` with each fold number from 1 to 5. Training requires a GPU. Preserve the supplied dog-level fold assignments.

## Reproduce the study analyses

The scripts below use the CSV files in `analysis_dataset_300.zip`; model training and inference are not required. Extract the archive into `data/` and create an `outputs/` directory before running these commands from the repository root.

| Script | Purpose |
|---|---|
| `table1.py` | Cohort characteristics and diagnostic counts for Table 1 |
| `analyze.py` | Expert agreement, associations with RS and FD, disease summaries, and segmentation performance |
| `compare_disease_groups.R` | Disease-versus-control comparisons |
| `confidence_bands.py` | Pointwise confidence limits for the descriptive regression lines |

```sh
python table1.py data/demographics.csv data/features.csv outputs/Table1.csv
python analyze.py data/features.csv data/segmentation_metrics.csv outputs
Rscript compare_disease_groups.R data/features.csv outputs
python confidence_bands.py data/plotting_table.csv outputs/confidence_bands.csv
```

The R script requires **clusrank 1.0-4** (reported by R as `1.0.4`). It uses two-sided Rosner–Glynn–Lee clustered Wilcoxon tests with dogs as clusters, a normal approximation, and Holm adjustment across 13 comparisons per outcome. The intrascleral-prosthesis group is retained in the adjustment family and marked `descriptive_only`.

The dataset retains all 300 eyes. Expert-grade and RS analyses use 298 eyes from 150 dogs; FD analyses use 296 eyes from 149 dogs. Inclusion flags and exclusion reasons are supplied in `analysis_inclusion.csv`. Keep undefined FD values missing and preserve dog identifiers and row order for the seeded bootstrap and permutation analyses.

## Reproducibility

Measurement and statistical outputs have been checked against the stored study results. The analysis environment used Windows x64, Python 3.12, and OpenCV 4.12.0.88 with Intel IPP enabled; numerical results may vary with the image-processing environment. Full GPU retraining has not been repeated for the distributed package, and the segmentation dependencies are not version-pinned.

The [study code snapshot](https://github.com/300yen/canine-conjunctival-hyperemia/tree/1ac974a5184aed30449b8a8f677b752bd942c669) is retained for reproducibility. File checksums and validation details are included in the Zenodo dataset.

## Citation and licensing

When using the study data or trained models, cite [Nakagiri et al., *Automated quantification of canine ocular redness: images, segmentation masks, measurements and trained models*, version 1.0.0](https://doi.org/10.5281/zenodo.23889738). When referring to the code, include the repository URL and the commit used.

The Zenodo data and trained models are distributed under CC BY 4.0. A software license has not been specified for this repository.
