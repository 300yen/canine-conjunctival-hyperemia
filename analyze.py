import argparse
from pathlib import Path

from posthoc import compare
import study_statistics as study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = study.read_csv_fallback(args.features)
    metrics = study.read_csv_fallback(args.metrics)
    valid = data.loc[~data[study.RATERS].eq(5).any(axis=1)].copy()
    valid["expert_mean"] = valid[study.RATERS].mean(axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, function, frame in [
        ("expert_agreement", study.summarize_experts, valid),
        ("primary_associations", study.summarize_associations, valid),
        ("disease_group_global_tests", study.disease_global_tests, valid),
        ("segmentation_metrics", study.segmentation_summary, metrics),
    ]:
        table = function(frame, study.analysis_rng(20260728, name), 5000)
        table.to_csv(args.output / (name + ".csv"), index=False)
    compare(data).to_csv(args.output / "control_vs_disease_posthoc.csv", index=False)
    study.disease_summary(valid).to_csv(args.output / "disease_summary.csv", index=False)


if __name__ == "__main__":
    main()
