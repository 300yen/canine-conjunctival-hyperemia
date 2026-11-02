from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import cohen_kappa_score

RATERS = ["A_label", "B_label", "C_label"]
REDNESS = "AndMask_LabA_Density"
FRACTAL = "fractal_dimension"

DISEASE_MAP = {
    "00_Normal": "Normal control",
    "13_ISP": "Intrascleral prosthesis",
    "01_Cornea_Acute_Ulcer": "Corneal ulcer",
    "02_Cornea_KCS_Exposure": "Non-ulcerative corneo-conjunctival disease",
    "03_Cornea_Chronic_Others": "Non-ulcerative corneo-conjunctival disease",
    "04_Scleritis": "Scleritis",
    "05_Lens_Ant_Luxation": "Anterior lens luxation",
    "07_Lens_Other_Luxation": "Posterior/other lens luxation",
    "06_Lens_Cataract": "Cataract",
    "08_Uveitis": "Uveitis (including LIU)",
    "09_Glaucoma": "Glaucoma",
    "10_Retina_Neuro_NonUrgent": "Retinal/neurological (non-neuritis)",
    "14_Optic_Neuritis_Urgent": "Optic neuritis",
    "11_Tumor_Intraocular": "Intraocular tumor",
    "12_Tumor_Extra_Adnexa": "Extraocular/adnexal tumor",
}

# Stable stream identifiers keep each stochastic analysis independent of call
# order. Adding or rerunning one table therefore cannot change another table's
# bootstrap or permutation samples.
RNG_STREAM_IDS = {
    "expert_agreement": 1,
    "primary_associations": 2,
    "segmentation_metrics": 3,
    "severity_windows": 4,
    "ridge_oof_performance": 5,
    "disease_group_global_tests": 6,
    "figure4_agreement": 7,
}


def analysis_rng(base_seed: int, analysis: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([base_seed, RNG_STREAM_IDS[analysis]])
    )


def read_csv_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def percentile_ci(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    return tuple(np.percentile(finite, [2.5, 97.5]))


def cluster_index_blocks(df: pd.DataFrame, cluster_col: str) -> list[np.ndarray]:
    return [g.index.to_numpy() for _, g in df.groupby(cluster_col, sort=False)]


def bootstrap_stat(
    df: pd.DataFrame,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    work = df.reset_index(drop=True)
    blocks = cluster_index_blocks(work, cluster_col)
    estimates: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in chosen])
        try:
            estimates.append(float(statistic(work.iloc[idx])))
        except (ValueError, ZeroDivisionError):
            estimates.append(np.nan)
    return percentile_ci(estimates)


def icc_absolute(data: np.ndarray) -> tuple[float, float]:
    """Return ICC(A,1) and ICC(A,k), equivalent to two-way random absolute ICC."""
    x = np.asarray(data, dtype=float)
    n, k = x.shape
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = x - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    icc_single = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    icc_average = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    return float(icc_single), float(icc_average)


def qwk(frame: pd.DataFrame, first: str, second: str) -> float:
    return float(cohen_kappa_score(frame[first], frame[second], weights="quadratic"))


def association(frame: pd.DataFrame, feature: str) -> float:
    return float(spearmanr(frame[feature], frame["expert_mean"]).statistic)


def cluster_permutation_p(
    frame: pd.DataFrame,
    feature: str,
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    """Two-sided dog-block permutation test, stratified by available eye count."""
    work = frame.sort_values(["ID", "R/L"]).reset_index(drop=True)
    blocks = cluster_index_blocks(work, "ID")
    by_size: dict[int, list[np.ndarray]] = {}
    for block in blocks:
        by_size.setdefault(len(block), []).append(block)
    observed = abs(association(work, feature))
    y = work["expert_mean"].to_numpy(dtype=float)
    x = work[feature].to_numpy(dtype=float)
    extreme = 0
    for _ in range(n_perm):
        permuted = np.empty_like(y)
        for same_size_blocks in by_size.values():
            source_order = rng.permutation(len(same_size_blocks))
            for target, source_idx in zip(same_size_blocks, source_order):
                permuted[target] = y[same_size_blocks[source_idx]]
        value = abs(float(spearmanr(x, permuted).statistic))
        extreme += value >= observed
    return (extreme + 1) / (n_perm + 1)


def summarize_experts(
    valid: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [("A_label", "B_label"), ("A_label", "C_label"), ("B_label", "C_label")]
    for first, second in pairs:
        estimate = qwk(valid, first, second)
        low, high = bootstrap_stat(
            valid, "ID", lambda x, a=first, b=second: qwk(x, a, b), rng, n_boot
        )
        rows.append(
            {
                "analysis": "Quadratic weighted kappa",
                "contrast": f"{first[0]} vs {second[0]}",
                "n_eyes": len(valid),
                "n_dogs": valid["ID"].nunique(),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    icc_single, icc_average = icc_absolute(valid[RATERS].to_numpy())
    for label, position, estimate in (
        ("ICC(A,1)", 0, icc_single),
        ("ICC(A,3)", 1, icc_average),
    ):
        low, high = bootstrap_stat(
            valid,
            "ID",
            lambda x, p=position: icc_absolute(x[RATERS].to_numpy())[p],
            rng,
            n_boot,
        )
        rows.append(
            {
                "analysis": "Absolute-agreement ICC",
                "contrast": label,
                "n_eyes": len(valid),
                "n_dogs": valid["ID"].nunique(),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    for first, second in pairs:
        estimate = float((valid[first] - valid[second]).mean())
        low, high = bootstrap_stat(
            valid,
            "ID",
            lambda x, a=first, b=second: float((x[a] - x[b]).mean()),
            rng,
            n_boot,
        )
        rows.append(
            {
                "analysis": "Mean paired grade difference",
                "contrast": f"{first[0]} minus {second[0]}",
                "n_eyes": len(valid),
                "n_dogs": valid["ID"].nunique(),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def figure4_rater_grade_distribution(valid: pd.DataFrame) -> pd.DataFrame:
    """Return plot-ready aggregate grade counts without eye-level identifiers."""
    rows: list[dict[str, object]] = []
    for rater, column in zip(("A", "B", "C"), RATERS):
        counts = valid[column].value_counts().reindex(range(4), fill_value=0)
        for grade, count in counts.items():
            rows.append(
                {
                    "rater": rater,
                    "grade": int(grade),
                    "n_eyes": int(count),
                    "proportion": float(count / len(valid)),
                }
            )
    return pd.DataFrame(rows)


def figure4_agreement_summary(
    valid: pd.DataFrame,
    expert_summary: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    """Combine agreement, rater-bias, and intuitive agreement proportions."""
    rows = expert_summary.to_dict(orient="records")
    proportion_stats = (
        (
            "All three exact agreement",
            lambda frame: float(frame[RATERS].nunique(axis=1).eq(1).mean()),
        ),
        (
            "Maximum pairwise range <= 1 grade",
            lambda frame: float(
                ((frame[RATERS].max(axis=1) - frame[RATERS].min(axis=1)) <= 1).mean()
            ),
        ),
    )
    for contrast, statistic in proportion_stats:
        estimate = statistic(valid)
        low, high = bootstrap_stat(valid, "ID", statistic, rng, n_boot)
        rows.append(
            {
                "analysis": "Agreement proportion",
                "contrast": contrast,
                "n_eyes": len(valid),
                "n_dogs": valid["ID"].nunique(),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def figure4_expert_mean_by_disease(valid: pd.DataFrame) -> pd.DataFrame:
    """Return anonymous eye-level values needed for disease-group raw points."""
    work = valid.copy()
    work["disease_group"] = work["Category_Mid"].map(DISEASE_MAP)
    if work["disease_group"].isna().any():
        missing = sorted(work.loc[work["disease_group"].isna(), "Category_Mid"].unique())
        raise ValueError(f"Unmapped disease categories: {missing}")
    return (
        work[["disease_group", "expert_mean"]]
        .sort_values(["disease_group", "expert_mean"], kind="stable")
        .reset_index(drop=True)
    )


def summarize_associations(
    valid: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    rows = []
    for label, feature in (("Redness Score", REDNESS), ("Fractal dimension", FRACTAL)):
        complete = valid.dropna(subset=[feature, "expert_mean"]).copy()
        estimate = association(complete, feature)
        low, high = bootstrap_stat(
            complete,
            "ID",
            lambda x, f=feature: association(x, f),
            rng,
            n_boot,
        )
        p_value = cluster_permutation_p(complete, feature, rng, n_boot)
        rows.append(
            {
                "feature": label,
                "analysis": "Spearman correlation with mean expert grade",
                "n_eyes": len(complete),
                "n_dogs": complete["ID"].nunique(),
                "rho": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "cluster_permutation_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def q1(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q3(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def disease_summary(valid: pd.DataFrame) -> pd.DataFrame:
    work = valid.copy()
    work["disease_group"] = work["Category_Mid"].map(DISEASE_MAP)
    if work["disease_group"].isna().any():
        missing = sorted(work.loc[work["disease_group"].isna(), "Category_Mid"].unique())
        raise ValueError(f"Unmapped disease categories: {missing}")
    rows = []
    for group, frame in work.groupby("disease_group", sort=True):
        row: dict[str, object] = {
            "disease_group": group,
            "n_eyes": len(frame),
            "n_dogs": frame["ID"].nunique(),
        }
        for label, column in (
            ("expert_mean", "expert_mean"),
            ("redness_score", REDNESS),
            ("fractal_dimension", FRACTAL),
        ):
            values = frame[column].dropna()
            row[f"{label}_n"] = len(values)
            row[f"{label}_median"] = float(values.median())
            row[f"{label}_q1"] = q1(values)
            row[f"{label}_q3"] = q3(values)
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise-error adjustment."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def disease_kruskal_stat(frame: pd.DataFrame, outcome: str) -> float:
    samples = [
        group[outcome].to_numpy(dtype=float)
        for _, group in frame.groupby("disease_group", sort=True)
        if len(group) > 0
    ]
    return float(kruskal(*samples).statistic)


def disease_cluster_permutation_p(
    frame: pd.DataFrame,
    outcome: str,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[float, float]:
    """Global disease-group test preserving one- or two-eye dog blocks."""
    work = frame.dropna(subset=[outcome]).sort_values(["ID", "R/L"]).reset_index(drop=True)
    observed = disease_kruskal_stat(work, outcome)
    blocks = cluster_index_blocks(work, "ID")
    by_size: dict[int, list[np.ndarray]] = {}
    for block in blocks:
        by_size.setdefault(len(block), []).append(block)
    values = work[outcome].to_numpy(dtype=float)
    extreme = 0
    for _ in range(n_perm):
        permuted = np.empty_like(values)
        for same_size_blocks in by_size.values():
            source_order = rng.permutation(len(same_size_blocks))
            for target, source_idx in zip(same_size_blocks, source_order):
                permuted[target] = values[same_size_blocks[source_idx]]
        permuted_frame = work.copy()
        permuted_frame[outcome] = permuted
        extreme += disease_kruskal_stat(permuted_frame, outcome) >= observed
    return observed, (extreme + 1) / (n_perm + 1)


def disease_global_tests(
    valid: pd.DataFrame, rng: np.random.Generator, n_perm: int
) -> pd.DataFrame:
    work = valid.copy()
    work["disease_group"] = work["Category_Mid"].map(DISEASE_MAP)
    rows: list[dict[str, object]] = []
    for label, outcome in (
        ("Mean expert grade", "expert_mean"),
        ("Redness Score", REDNESS),
        ("Fractal dimension", FRACTAL),
    ):
        complete = work.dropna(subset=[outcome]).copy()
        statistic, p_value = disease_cluster_permutation_p(
            complete, outcome, rng, n_perm
        )
        rows.append(
            {
                "outcome": label,
                "n_eyes": len(complete),
                "n_dogs": complete["ID"].nunique(),
                "n_disease_groups": complete["disease_group"].nunique(),
                "kruskal_wallis_h": statistic,
                "cluster_permutation_p": p_value,
            }
        )
    adjusted = holm_adjust([float(row["cluster_permutation_p"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
    return pd.DataFrame(rows)


def segmentation_summary(
    metrics: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    metric_cols = [c for c in metrics.columns if c not in {"test_ID", "R/L", "fold"}]
    dog_means = metrics.groupby("test_ID", sort=False)[metric_cols].mean()
    values = dog_means.to_numpy(dtype=float)
    n_dogs = len(dog_means)
    # Equal numbers of eyes per dog make the mean of dog means equal to the eye-level mean.
    samples = rng.integers(0, n_dogs, size=(n_boot, n_dogs))
    boot_means = values[samples].mean(axis=1)
    lows = np.percentile(boot_means, 2.5, axis=0)
    highs = np.percentile(boot_means, 97.5, axis=0)
    return pd.DataFrame(
        {
            "metric": metric_cols,
            "n_images": len(metrics),
            "n_dogs": n_dogs,
            "mean": metrics[metric_cols].mean().to_numpy(dtype=float),
            "ci95_low": lows,
            "ci95_high": highs,
        }
    )


