# Canine conjunctival hyperemia research code

Research code for canine conjunctival image analysis. The code measures conjunctival redness and vessel fractal dimension (FD) in canine anterior segment images and implements the study analyses. Run commands from this directory unless otherwise indicated.

## Measure an image

The replay environment used Python 3.12.14 on Windows x64 with OpenCV 4.12.0.88 and Intel IPP enabled. Install the analysis dependencies into an isolated environment. These versions describe the replay environment. A clean installation on another platform still needs verification.

```sh
python -m pip install -r requirements-analysis.txt
python demo.py
python measure.py image.png merged_mask.png
```

The demo generates a synthetic image in memory. Expected RS is 6.72516137 and expected FD is 1.51482262 in the replay environment. For real inputs use a 512 by 512 color image and its aligned single-channel label PNG (0 background, 1 conjunctiva, 2 cornea). The program erodes the conjunctival ROI by a disk of radius 10 pixels, then runs the notebook vessel and skeleton processing steps. It prints RS and FD as JSON. An undefined FD is represented by null.

The functions and parameters in features.py come from the final feature extraction notebook. The release wrapper was checked against 300 stored study measurements. Maximum absolute RS error was 4.17e-7. Maximum absolute FD error was 2.23e-16 among 298 finite values, with both missing values reproduced.

## Segmentation

Use a separate GPU environment for segmentation. segmentation/requirements.txt lists imports and runtime dependencies. Their training versions and compatibility with all five original checkpoints still need verification. The source training notebook used Lightning, Adam and the albb network. The CLI exposes one fold per invocation and replaces the original Colab paths.

```sh
cd segmentation
python -m pip install -r requirements.txt
python training/train.py 1
python run_inference_paper.py
python calc_metrics_from_saved.py
```

Run training once for each fold 1 through 5, or place one matching checkpoint per fold in 03_retrained_ckpt before inference. The default CSV is inputs/dataset.csv with test_ID (numeric study dog key), R/L (eye side), label (1 annotated, 0 unannotated) and fold (1 through 5 for annotated eyes). Preserve the study's dog-level folds. Use public study keys when preparing shareable inputs. The filename pattern is a four-digit key, underscore, eye side and _01.png. Put annotated RGB images under inputs/labeled_image/image, annotation PNGs (blue conjunctiva, red cornea, alpha channel for training) under inputs/labeled_image/label, and the unannotated images under inputs/all_img. Paths for inference are in configs/config.json.

Annotated eyes use held-out fold predictions. Unannotated eyes use the five-model ensemble. Use the merged masks under 02_ROI_Segmentation/predictions/masks_merged for measurement. Use calc_metrics_from_saved.py for the published conjunctiva/cornea macro averages. The inference script also produces a separate diagnostic summary. We have not rerun GPU training or inference in this release environment.

## Statistical analysis

```sh
python table1.py demographics.csv features.csv Table1.csv
python analyze.py features.csv segmentation/02_ROI_Segmentation/evaluation_paper/metrics_detailed_per_image.csv outputs
python confidence_bands.py plotting_table.csv outputs/confidence_bands.csv
```

Table 1 uses the columns ID (study dog key), month (age in months), sex, cast/spay and breed in the demographics input. Diagnostic groups use all enrolled eyes in the features input.

features.csv has one row per enrolled eye. Required fields are ID (shared dog key), R/L, Category_Mid, A_label, B_label, C_label, AndMask_LabA_Density and fractal_dimension. The three expert ratings range from 0 to 3. Code 5 denotes unassessable. Preserve missing FD, eye order and dog grouping for seeded comparisons. Category_Mid values are listed in study_statistics.py. The segmentation metrics CSV contains test_ID, R/L, fold and the six per-eye metrics for each of Conj, Corn and Macro.

The aggregate outputs cover expert agreement, associations, disease-group comparisons and segmentation performance. Statistical calculations are selected from the study analysis scripts. Table 3 uses 5,000 dog-level bootstrap draws. All 18 estimates and their confidence limits were checked against the archived verification table. The optional confidence-band command uses dog_code, expert_mean, redness_score, fractal_dimension, include_expert_rs and include_rs_fd. Supply boolean inclusion flags and retain dog ordering to reproduce the seeded pointwise limits.

## Numerical verification

On 2 November 2026, the release commands regenerated all 35 rows of Table 1, the expert agreement results used for Table 2, primary associations, global disease tests, disease summaries, segmentation summaries and 39 control-versus-disease comparisons. Table 1 matched the manuscript CSV cell for cell. The six statistical outputs matched the archived aggregates to their reported precision.

## Release status

This directory contains code only. Dataset and checkpoint accession numbers, the final software license and the public release DOI are pending. The original capture-to-input preprocessing and full GPU replay have not been verified.
