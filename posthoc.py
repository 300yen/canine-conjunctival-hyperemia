"""Control-versus-disease post-hoc comparisons for expert score, RS, and FD.

The analysis uses dog-block permutation of a Mann-Whitney rank statistic and
Holm adjustment within each outcome. Outputs contain aggregates only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RATERS = ["A_label", "B_label", "C_label"]
RS = "AndMask_LabA_Density"
FD = "fractal_dimension"
CONTROL = "正常対照"
N_PERM = 5000
SEED = 20260729

DISEASE_MAP = {
    "00_Normal": "正常対照",
    "13_ISP": "眼内シリコン義眼（ISP）",
    "01_Cornea_Acute_Ulcer": "角膜潰瘍",
    "02_Cornea_KCS_Exposure": "非潰瘍性角結膜疾患",
    "03_Cornea_Chronic_Others": "非潰瘍性角結膜疾患",
    "04_Scleritis": "強膜疾患",
    "05_Lens_Ant_Luxation": "前方水晶体脱臼",
    "07_Lens_Other_Luxation": "後方・その他の水晶体脱臼",
    "06_Lens_Cataract": "白内障",
    "08_Uveitis": "ぶどう膜炎（LIUを含む）",
    "09_Glaucoma": "緑内障",
    "10_Retina_Neuro_NonUrgent": "網膜・神経疾患（視神経炎を除く）",
    "14_Optic_Neuritis_Urgent": "視神経炎",
    "11_Tumor_Intraocular": "眼内腫瘍",
    "12_Tumor_Extra_Adnexa": "眼外・付属器腫瘍",
}

OUTCOMES = {
    "専門家平均スコア": "expert_mean",
    "RS": RS,
    "FD": FD,
}


def read_csv_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def quartiles(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    return float(clean.median()), float(clean.quantile(0.25)), float(clean.quantile(0.75))


def rank_biserial_from_ranks(ranks: np.ndarray, target_mask: np.ndarray) -> float:
    n_target = int(target_mask.sum())
    n_control = int((~target_mask).sum())
    u_target = float(ranks[target_mask].sum()) - n_target * (n_target + 1) / 2.0
    return 2.0 * u_target / (n_target * n_control) - 1.0


def pairwise_cluster_permutation(
    frame: pd.DataFrame,
    outcome: str,
    target_group: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    work = frame.loc[frame["disease_group"].isin([CONTROL, target_group])].dropna(subset=[outcome]).copy()
    work = work.sort_values(["ID", "R/L"]).reset_index(drop=True)
    ranks = work[outcome].astype(float).rank(method="average").to_numpy()
    target_mask = work["disease_group"].eq(target_group).to_numpy()
    observed = rank_biserial_from_ranks(ranks, target_mask)

    block_matrices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    blocks = [group.index.to_numpy() for _, group in work.groupby("ID", sort=False)]
    for size in sorted({len(block) for block in blocks}):
        index_matrix = np.vstack([block for block in blocks if len(block) == size])
        rank_matrix = ranks[index_matrix]
        block_matrices[size] = (index_matrix, rank_matrix)

    extreme = 0
    permuted = np.empty_like(ranks)
    for _ in range(N_PERM):
        for index_matrix, rank_matrix in block_matrices.values():
            permuted[index_matrix] = rank_matrix[rng.permutation(len(rank_matrix))]
        statistic = rank_biserial_from_ranks(permuted, target_mask)
        extreme += abs(statistic) >= abs(observed) - 1e-12
    p_value = (extreme + 1.0) / (N_PERM + 1.0)
    return observed, p_value


def compare(df):
    valid = df.loc[~df[RATERS].eq(5).any(axis=1)].copy()
    valid["expert_mean"] = valid[RATERS].mean(axis=1)
    valid["disease_group"] = valid["Category_Mid"].map(DISEASE_MAP)
    if valid["disease_group"].isna().any():
        raise RuntimeError("Unmapped disease category")

    targets = sorted(group for group in valid["disease_group"].unique() if group != CONTROL)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for outcome_label, outcome in OUTCOMES.items():
        outcome_rows = []
        for target in targets:
            pair = valid.loc[valid["disease_group"].isin([CONTROL, target])].dropna(subset=[outcome]).copy()
            control = pair.loc[pair["disease_group"].eq(CONTROL)]
            disease = pair.loc[pair["disease_group"].eq(target)]
            c_med, c_q1, c_q3 = quartiles(control[outcome])
            d_med, d_q1, d_q3 = quartiles(disease[outcome])
            effect, p_value = pairwise_cluster_permutation(valid, outcome, target, rng)
            outcome_rows.append(
                {
                    "outcome": outcome_label,
                    "comparison": f"{target} vs {CONTROL}",
                    "disease_group": target,
                    "control_n_eyes": len(control),
                    "control_n_dogs": control["ID"].nunique(),
                    "disease_n_eyes": len(disease),
                    "disease_n_dogs": disease["ID"].nunique(),
                    "control_median": c_med,
                    "control_q1": c_q1,
                    "control_q3": c_q3,
                    "disease_median": d_med,
                    "disease_q1": d_q1,
                    "disease_q3": d_q3,
                    "median_difference": d_med - c_med,
                    "rank_biserial": effect,
                    "cluster_permutation_p": p_value,
                }
            )
        adjusted = holm_adjust([float(row["cluster_permutation_p"]) for row in outcome_rows])
        for row, p_adjusted in zip(outcome_rows, adjusted):
            row["holm_adjusted_p"] = p_adjusted
            row["significant_0_05"] = bool(p_adjusted < 0.05)
        rows.extend(outcome_rows)

    results = pd.DataFrame(rows)
    return results
