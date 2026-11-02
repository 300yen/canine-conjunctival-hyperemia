import argparse
from pathlib import Path

import pandas as pd
from study_statistics import DISEASE_MAP, read_csv_fallback

breed_translation = {
    'トイプードル': 'Toy Poodle',
    '柴犬': 'Shiba Inu',
    'チワワ': 'Chihuahua',
    '雑種': 'Mixed Breed',
    'ﾐﾆﾁｭｱﾀﾞｯｸｽﾌﾝﾄﾞ': 'Miniature Dachshund',
    'シーズー': 'Shih Tzu',
    'フレンチブルドッグ': 'French Bulldog',
    'ﾐﾆﾁｭｱｼｭﾅｳｻﾞｰ': 'Miniature Schnauzer',
    'ポメラニアン': 'Pomeranian',
    'ヨークシャーテリア': 'Yorkshire Terrier',
    'マルチーズ': 'Maltese',
    'ｷｬﾊﾞﾘｱｷﾝｸﾞﾁｬｰﾙｽﾞｽﾊﾟﾆ': 'Cavalier King Charles Spaniel',
    'パピヨン': 'Papillon',
    'ｼﾞｬｯｸﾗｯｾﾙﾃﾘｱ': 'Jack Russell Terrier',
    'ｶﾆﾝﾍﾝﾀﾞｯｸｽﾌﾝﾄ': 'Kaninchen Dachshund'
}


def summarize(enrolled, features):
    dogs = enrolled.drop_duplicates("ID")
    age = dogs["month"] / 12
    rows = [
        ("Total dogs, n", str(len(dogs))),
        ("Total eyes, n", str(len(enrolled))),
        ("Age (years), median [range]", f"{age.median():.1f} [{age.min():.1f}–{age.max():.1f}]"),
        ("Sex, n (%)", ""),
    ]

    def count(label, n, total):
        rows.append((label, f"{n} ({100 * n / total:.1f})"))

    altered = dogs["cast/spay"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    )
    if altered.isna().any() or not dogs.sex.isin(["♂", "♀"]).all():
        raise ValueError("Unexpected sex or neuter-status value")
    for name, sex, status in [
        ("Male (Intact)", "♂", False), ("Male (Neutered)", "♂", True),
        ("Female (Intact)", "♀", False), ("Female (Spayed)", "♀", True),
    ]:
        count(name, (dogs.sex.eq(sex) & altered.eq(status)).sum(), len(dogs))
    rows.append(("Top Breeds, n (%)", ""))
    breeds = dogs.breed.replace(breed_translation).value_counts()
    # Keep the manuscript display order for breeds tied within the top ten.
    tie_order = ["Shih Tzu", "French Bulldog", "Miniature Schnauzer", "Pomeranian", "Yorkshire Terrier"]
    rank = {name: i for i, name in enumerate(tie_order)}
    top = sorted(breeds.head(10).items(), key=lambda item: (-item[1], rank.get(item[0], 10)))
    for name, n in top:
        count(name, n, len(dogs))
    count("Others", breeds.iloc[10:].sum(), len(dogs))
    rows.append(("Diagnostic categories (eyes), n (%)", ""))
    groups = features.Category_Mid.map(DISEASE_MAP)
    if groups.isna().any() or len(groups) != len(enrolled):
        raise ValueError("Incomplete diagnosis mapping for the enrolled cohort")
    order = list(dict.fromkeys(DISEASE_MAP.values()))
    order.remove("Intrascleral prosthesis")
    for group in order + ["Intrascleral prosthesis"]:
        count(group, groups.eq(group).sum(), len(groups))
    return pd.DataFrame(rows, columns=["Characteristic", "Value"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("demographics", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(read_csv_fallback(args.demographics), read_csv_fallback(args.features)).to_csv(
        args.output, index=False
    )
