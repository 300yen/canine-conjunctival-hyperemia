import json
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ==========================================
# 設定・定数
# ==========================================
CONFIG_PATH = 'configs/config.json'
PRED_MASK_DIR = "02_ROI_Segmentation/predictions/masks_merged" # 保存された予測マスク(0,1,2)のパス
SAVE_DIR = "02_ROI_Segmentation/evaluation_paper" # 結果保存先

def calculate_binary_metrics(pred_mask, gt_mask, target_class, prefix=""):
    """
    指定したクラス(target_class)に対する2値分類指標を計算する
    """
    # ターゲットクラスを1、それ以外を0にする
    p = (pred_mask == target_class).astype(np.uint8).flatten()
    g = (gt_mask == target_class).astype(np.uint8).flatten()

    # 混同行列
    # ラベルが欠落している場合のエラー回避のため labels指定
    tn, fp, fn, tp = confusion_matrix(g, p, labels=[0, 1]).ravel()

    # ゼロ除算回避 (epsilon)
    eps = 1e-6

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)      # Jaccard
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)        # Sensitivity
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    # GTがそのクラスを含まない場合の特例処理 (IoUなどは1.0とするのが一般的だが、文脈による)
    if np.sum(g) == 0:
        # 予測も0なら満点、予測があれば0点
        score = 1.0 if np.sum(p) == 0 else 0.0
        iou = score
        f1 = score
        recall = 1.0 # 対象がないので見逃しもない

    return {
        f"{prefix}Accuracy": accuracy,
        f"{prefix}Jaccard": iou,
        f"{prefix}Specificity": specificity,
        f"{prefix}Precision": precision,
        f"{prefix}Recall": recall,
        f"{prefix}F1_Score": f1
    }

def main():
    # 1. 設定読み込み
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} が見つかりません。")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    data_dir = config['paths']['data_dir']
    csv_filename = config['dataset']['csv_filename']
    
    # 保存先作成
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. データセット定義読み込み
    df = pd.read_csv(csv_filename)
    # GTがあるデータのみ対象 (label == 1)
    target_df = df[df['label'] == 1].copy()
    
    print(f"評価対象画像数: {len(target_df)}枚")

    results = []

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        id_str = str(row['test_ID']).zfill(4)
        rl = row['R/L']
        fold = row['fold']
        
        # ファイル名定義
        filename_base = f"{id_str}_{rl}"
        img_filename = f'{filename_base}_01.png'
        
        # パス定義
        pred_path = os.path.join(PRED_MASK_DIR, f"{filename_base}.png")
        lbl_path = os.path.join(data_dir, 'labeled_image', 'label', img_filename)

        # ファイル存在確認
        if not os.path.exists(pred_path):
            raise FileNotFoundError(pred_path)
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(lbl_path)

        # 3. 画像読み込み
        # 予測マスク (0:BG, 1:Conj, 2:Corn)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # GTラベル (BGR) -> マスク変換
        lbl_src = cv2.imread(lbl_path)
        # 予測マスクのサイズに合わせてGTをリサイズ (Nearest Neighbor)
        if pred_mask.shape != lbl_src.shape[:2]:
            lbl_src = cv2.resize(lbl_src, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        gt_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
        # 青(B) > 128 -> 結膜(1), 赤(R) > 128 -> 角膜(2)
        gt_mask[lbl_src[:, :, 0] > 128] = 1 
        gt_mask[lbl_src[:, :, 2] > 128] = 2 

        # 4. 指標計算
        # (1) 結膜 (Class 1)
        metrics_conj = calculate_binary_metrics(pred_mask, gt_mask, target_class=1, prefix="Conj_")
        
        # (2) 角膜 (Class 2)
        metrics_corn = calculate_binary_metrics(pred_mask, gt_mask, target_class=2, prefix="Corn_")

        # レコード作成
        record = {
            "test_ID": row['test_ID'],
            "R/L": row['R/L'],
            "fold": fold,
        }
        record.update(metrics_conj)
        record.update(metrics_corn)
        
        # (3) Macro Average (結膜と角膜の平均)
        # 論文ではこの平均値を「モデルの性能」として載せることが多いよ
        for key in metrics_conj.keys():
            metric_name = key.replace("Conj_", "")
            val_conj = metrics_conj[f"Conj_{metric_name}"]
            val_corn = metrics_corn[f"Corn_{metric_name}"]
            record[f"Macro_{metric_name}"] = (val_conj + val_corn) / 2.0

        results.append(record)

    # 5. 結果の集計と保存
    if not results:
        print("計算できるデータがありませんでした。")
        return

    # 詳細データ (1枚ごと)
    df_results = pd.DataFrame(results)
    detailed_save_path = os.path.join(SAVE_DIR, "metrics_detailed_per_image.csv")
    df_results.to_csv(detailed_save_path, index=False)
    print(f"\n[保存完了] 詳細データ: {detailed_save_path}")

    # Foldごとの平均 (Summary)
    # 数値列のみ抽出して平均
    df_summary = df_results.groupby("fold").mean(numeric_only=True).drop(columns=["test_ID"], errors='ignore')
    
    # 全体平均 (Average of Folds) も追加
    mean_row = df_summary.mean()
    mean_row.name = 'Average'
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_row])])

    summary_save_path = os.path.join(SAVE_DIR, "metrics_summary_by_fold.csv")
    df_summary.to_csv(summary_save_path)
    print(f"[保存完了] Fold平均データ: {summary_save_path}")

    print("\n" + "="*30)
    print("   Analysis Completed!   ")
    print("="*30)
    print("Foldごとの平均スコア (一部抜粋):")
    print(df_summary[["Conj_Jaccard", "Corn_Jaccard", "Macro_Jaccard", "Macro_F1_Score"]])

if __name__ == "__main__":
    main()