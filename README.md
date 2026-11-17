# Canine conjunctival hyperemia

Research code for quantifying canine ocular redness from anterior segment photographs. The workflow segments the conjunctiva and cornea, calculates Redness Score (RS) and vessel fractal dimension (FD), and evaluates their associations with expert grades.

## Data

Images, annotations, trained models, measurements, and reference results are available in the [Zenodo dataset](https://doi.org/10.5281/zenodo.23889738).

The dataset contains 300 eyes from 150 dogs. Manual annotations are available for 200 eyes, and predicted masks are provided for all 300 eyes. The 640 × 480 research-source PNGs and 512 × 512 analysis images are included; camera-original JPEG files are not included.

## Getting started

Use Python 3.12 in a dedicated environment. From the repository root:

```sh
python -m pip install -r requirements-analysis.txt
```

Download `analysis_dataset_300.zip` from Zenodo into the repository root and extract it:

```sh
python -m zipfile -e analysis_dataset_300.zip data
python measure.py data/images/D001_L.png data/predicted_masks/D001_L.png
```

This returns RS and FD for one study image as JSON. An undefined FD is reported as `null`. The measurement command takes a 512 × 512 color image and a matching single-channel mask with labels `0` (background), `1` (conjunctiva), and `2` (cornea).

## Workflows

The [reproduction guide](docs/reproduction.md) provides commands, input formats, and output locations for:

- **Measurement and statistical analysis:** use the released images, masks, and CSV tables to reproduce RS, FD, expert agreement, associations, disease comparisons, and confidence intervals.
- **Segmentation:** use the five trained models to generate masks and calculate segmentation performance.
- **Training:** train the study architecture with the supplied dog-level folds and training settings.

| Code | Role |
|---|---|
| `measure.py`, `features.py` | Image measurements and vessel processing |
| `analyze.py`, `study_statistics.py` | Expert agreement, associations, segmentation summaries, and disease summaries |
| `compare_disease_groups.R` | Clustered disease-versus-control comparisons |
| `confidence_bands.py` | Pointwise confidence limits for descriptive regression lines |
| `prepare_segmentation_inputs.py` | Prepare the released images and annotations for training and inference |
| `segmentation/` | Model definitions, training, inference, and mask evaluation |

The study uses AlbuNetB, a U-Net architecture with a ResNet34 encoder and bilinear upsampling. Other network definitions are available in the model modules; the study commands select AlbuNetB.

## Reproducibility

Measurement and statistical outputs have been checked against the stored study results. Image measurements were verified with OpenCV 4.12.0.88 and Intel IPP enabled on Windows x64. The [guide](docs/reproduction.md#environment-and-reproducibility) describes environment requirements and the scope of verification. GPU retraining may produce different weights because training is not deterministic.

## Citation and licensing

When using the data or trained models, cite [Nakagiri et al., *Automated quantification of canine ocular redness: images, segmentation masks, measurements and trained models*, version 1.0.0](https://doi.org/10.5281/zenodo.23889738). When referring to the code, include this repository URL and the commit used.

The Zenodo data and trained models are distributed under CC BY 4.0. A software license has not been specified for this repository.
