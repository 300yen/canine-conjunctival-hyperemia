# Reproducing the study analyses

All study inputs are available from [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.23889738). Run commands from the repository root unless a step explicitly changes directory. Keep the supplied CSV row order and dog identifiers for the seeded bootstrap and permutation analyses.

## Environment and reproducibility

Use separate environments for measurement/statistics and segmentation.

- **Measurement and Python statistics:** Python 3.12 and the pinned packages in `requirements-analysis.txt`. RS and FD were verified on Windows x64 with OpenCV 4.12.0.88 and Intel IPP enabled. Image-processing results can vary across OpenCV builds.
- **Disease comparisons:** R with `clusrank` version 1.0-4, reported by R as `1.0.4`. The script checks this version. The R calculations were verified using webR; native `Rscript` execution has not been separately verified.
- **Segmentation:** PyTorch, torchvision, Lightning, and the packages in `segmentation/requirements.txt`. Training requires a GPU. Inference uses CUDA when available and otherwise runs on CPU. These dependency versions are not pinned; the list is not a frozen training environment. Full GPU retraining has not been repeated for the distributed package.

Use the stored masks and measurements when reproducing the reported statistical results. CPU inference has been checked against the stored masks, with a one-pixel difference in one image and no change in its RS or FD. Retraining is not expected to reproduce identical weights.

## 1. Prepare the analysis data

Download `analysis_dataset_300.zip` and `aggregate_results_v18.zip` from Zenodo into the repository root. The latter contains the reference result tables. Extract them and create an output directory:

```sh
python -m zipfile -e analysis_dataset_300.zip data
python -m zipfile -e aggregate_results_v18.zip reference_results
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
```

The input files used by the commands below are:

| Input | Contents |
|---|---|
| `data/images/` | 300 analysis images, 512 × 512 pixels |
| `data/predicted_masks/` | 300 aligned label masks: background 0, conjunctiva 1, cornea 2 |
| `data/features.csv` | Expert grades, RS, FD, diagnosis, eye identifiers, and fold assignments |
| `data/segmentation_metrics.csv` | Per-eye segmentation metrics for the 200 annotated eyes |
| `data/analysis_inclusion.csv` | Analysis eligibility and exclusion reasons |
| `data/plotting_table.csv` | Inputs for the regression confidence limits |

`D001`–`D150` identify dogs; `_L` and `_R` identify eyes. Both eyes share the same dog identifier. Expert grades are 0–3; code 5 denotes an ungradable image. Two ungradable eyes are excluded from the expert-grade and measurement analyses, leaving 298 eyes from 150 dogs. Two additional eyes have undefined FD, leaving 296 eyes from 149 dogs for FD analyses. Keep undefined FD missing rather than replacing it with zero.

## 2. Measure an image

```sh
python measure.py data/images/D001_L.png data/predicted_masks/D001_L.png
```

Use the same filename stem for each image and mask pair. `measure.py` erodes the conjunctival region, extracts vessels, and returns `redness_score` and `fractal_dimension` as JSON. The corresponding stored values are `AndMask_LabA_Density` and `fractal_dimension` in `features.csv`. JSON `null` corresponds to missing FD in the CSV.

## 3. Run the statistical analyses

Install the Python dependencies from the repository root:

```sh
python -m pip install -r requirements-analysis.txt
python analyze.py data/features.csv data/segmentation_metrics.csv outputs
python confidence_bands.py data/plotting_table.csv outputs/confidence_bands.csv
```

For the disease comparisons, install `clusrank` in R with `install.packages("clusrank", repos="https://cloud.r-project.org")`. Confirm that `as.character(packageVersion("clusrank"))` returns `"1.0.4"` (release 1.0-4). If your package repository supplies another version, obtain 1.0-4 from the [CRAN source archive](https://cran.r-project.org/src/contrib/Archive/clusrank/). Then run:

```sh
Rscript compare_disease_groups.R data/features.csv outputs
```

| Output under `outputs/` | Contents |
|---|---|
| `expert_agreement.csv` | Weighted kappa, absolute-agreement ICC, and paired grade differences with confidence intervals |
| `primary_associations.csv` | Spearman correlations with mean expert grade, confidence intervals, and permutation p values |
| `segmentation_metrics.csv` | Segmentation metric means and dog-bootstrap confidence intervals |
| `disease_summary.csv` | Disease-group counts, medians, and quartiles |
| `control_vs_disease_rgl.csv` | Disease-versus-control comparisons and Holm-adjusted p values |
| `disease_comparison_session.txt` | R environment information |
| `confidence_bands.csv` | Fitted values and pointwise confidence limits for the descriptive regression lines |

Agreement, association, and segmentation confidence intervals use 5,000 dog-level bootstrap draws. Association p values use dog-block permutations. The regression bands use 2,000 dog-level bootstrap draws and are pointwise intervals, not simultaneous confidence bands. The confidence-band script outputs numerical values rather than a rendered figure.

Disease comparisons use two-sided Rosner–Glynn–Lee clustered Wilcoxon tests with dogs as clusters and a normal approximation. Each outcome has a Holm family of 13 disease-versus-normal comparisons. The intrascleral-prosthesis comparison is included in that family and flagged `descriptive_only`. `rgl_abs_z` is an absolute test statistic; use group medians and `median_difference` to interpret direction.

### CSV fields

`features.csv` requires `ID`, `R/L`, `Category_Mid`, `A_label`, `B_label`, `C_label`, `AndMask_LabA_Density`, and `fractal_dimension`. `ID` is the dog key, and `R/L` is laterality. Diagnosis codes are defined in the dataset's `diagnosis_dictionary.csv`.

`segmentation_metrics.csv` uses the same dog keys in `test_ID`, with `R/L`, `fold`, and metric columns prefixed `Conj_`, `Corn_`, or `Macro_`. `Macro_` is the mean of the conjunctival and corneal metrics.

`plotting_table.csv` requires `dog_code`, `expert_mean`, `redness_score`, `fractal_dimension`, `include_expert_rs`, and `include_rs_fd`. The last two columns are Boolean inclusion flags.

## 4. Generate segmentation masks

Download `original_resolution_images_300.zip` and `model_checkpoints_5fold.zip` into the repository root, alongside `analysis_dataset_300.zip`. Prepare a new input directory:

```sh
python prepare_segmentation_inputs.py analysis_dataset_300.zip original_resolution_images_300.zip segmentation/inputs
python -m zipfile -e model_checkpoints_5fold.zip segmentation
```

The helper preserves dog-level folds and maps public dog codes to numeric training keys: `D001_L` becomes `0001_L_01.png`, for example. It writes `segmentation/inputs/dataset.csv`, annotated images and masks under `inputs/labeled_image/image/` and `inputs/labeled_image/label/`, and unannotated images under `inputs/all_img/`. The destination must not already exist.

Annotated images and manual masks retain their original 640 × 480 resolution. Manual masks use blue for conjunctiva and red for cornea; the training loader also uses their alpha channel. Model inputs are resized to 512 × 512 and normalized. The extracted checkpoints must be directly under `segmentation/03_retrained_ckpt/`, with exactly one checkpoint per fold.

Activate the segmentation environment, then run:

```sh
cd segmentation
python -m pip install -r requirements.txt
python run_inference_paper.py
python calc_metrics_from_saved.py
```

Inference reads data and checkpoint paths from `configs/config.json`. Annotated eyes use their held-out fold model. Unannotated eyes use the mean class probabilities from all five models.

Outputs are under `segmentation/02_ROI_Segmentation/`:

- `predictions/masks_merged/`: label masks for measurement, named `0001_L.png`, etc.
- `predictions/masks_conjunctiva/`, `predictions/masks_cornea/`, and `predictions/overlays/`: binary masks and visual checks.
- `evaluation/`: inference diagnostics, including a combined foreground metric.
- `evaluation_paper/`: per-eye and per-fold conjunctiva, cornea, and macro-average metrics from `calc_metrics_from_saved.py`.

The combined foreground metric and the macro average are different quantities. For the study's segmentation summary, use `evaluation_paper/metrics_detailed_per_image.csv` as the second input to `analyze.py`.

To measure a newly generated mask, return to the repository root and pair the numeric mask name with its public analysis image:

```sh
cd ..
python measure.py data/images/D001_L.png segmentation/02_ROI_Segmentation/predictions/masks_merged/0001_L.png
```

## 5. Train the models

Use the prepared `segmentation/inputs/` directory and the segmentation environment. From `segmentation/`, run one command for each fold:

```sh
python training/train.py 1 --checkpoints retrained_ckpt
python training/train.py 2 --checkpoints retrained_ckpt
python training/train.py 3 --checkpoints retrained_ckpt
python training/train.py 4 --checkpoints retrained_ckpt
python training/train.py 5 --checkpoints retrained_ckpt
```

Use a new checkpoint directory for each training run. These commands keep new weights separate from the released checkpoints. To run inference with the new weights, set `paths.model_save_dir` in `configs/config.json` to that directory and ensure it contains exactly one checkpoint per fold.

Training selects AlbuNetB (`albb`) with ImageNet-initialized ResNet34 weights. The settings are defined in `training/train.py`, `training/seg.py`, and `training/dataset.py`: 512 × 512 inputs, batch size 4, Adam with learning rate 0.0001, cross-entropy loss, at most 500 epochs, early stopping after 20 epochs without validation-loss improvement, and seed 0. Augmentation uses random resized crops, horizontal flips, tone-curve changes, and grid distortion. Checkpoints are selected by validation loss. GPU execution is not deterministic.

The training command does not read `configs/config.json`; that file is used by the inference and evaluation workflow. Other architectures defined in the model modules are not selected by the study training command.
