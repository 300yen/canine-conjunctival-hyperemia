"""Pointwise dog-bootstrap confidence limits for the descriptive regression lines."""
import argparse

import numpy as np
import pandas as pd


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def dog_bootstrap_band(
    frame: pd.DataFrame,
    y_column: str,
    x_grid: np.ndarray,
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = {
        dog: group[["expert_mean", y_column]].to_numpy(dtype=float)
        for dog, group in frame.groupby("dog_code", sort=True)
    }
    dogs = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    predictions = np.empty((n_bootstraps, len(x_grid)), dtype=float)
    for index in range(n_bootstraps):
        sampled_dogs = rng.choice(dogs, size=len(dogs), replace=True)
        sample = np.concatenate([grouped[dog] for dog in sampled_dogs], axis=0)
        slope, intercept = linear_fit(sample[:, 0], sample[:, 1])
        predictions[index] = intercept + slope * x_grid
    lower, upper = np.percentile(predictions, [2.5, 97.5], axis=0)
    return lower, upper



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plotting_table")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.plotting_table)
    grid = np.linspace(0.0, 3.0, 181)
    records = []
    for feature, flag, seed in [("redness_score", "include_expert_rs", 20260931),
                                ("fractal_dimension", "include_rs_fd", 20260932)]:
        group = frame.loc[frame[flag].eq(True)].dropna(subset=["expert_mean", feature, "dog_code"])
        if group.empty:
            raise ValueError(f"No eligible observations for {feature}")
        low, high = dog_bootstrap_band(group, feature, grid, n_bootstraps=2000, seed=seed)
        slope, intercept = linear_fit(group["expert_mean"].to_numpy(), group[feature].to_numpy())
        records.append(pd.DataFrame({"feature": feature, "expert_mean": grid, "fit": intercept + slope * grid,
                                     "ci_low": low, "ci_high": high}))
    pd.concat(records, ignore_index=True).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
