import json
import os
import glob
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
# dataset.pyのクラス名に合わせてインポート
from src.dataset import get_aug, ConjDataset 
from src.lightning_model import ConjunctivaSegmentationModule

# ==========================================
# ユーティリティ関数
# ==========================================
def find_best_checkpoint(checkpoint_dir, fold):
    search_pattern = os.path.join(checkpoint_dir, f"*fold{fold}*.ckpt")
    files = glob.glob(search_pattern)
    if not files:
        print(f"\n[Error] Fold {fold} のモデルが見つかりません！")
        print(f"探した場所: {os.path.abspath(search_pattern)}")
        raise FileNotFoundError(f"Fold {fold} missing.")
    if len(files) != 1:
        raise ValueError(f"Expected one checkpoint for fold {fold}, found {len(files)}")
    return files[0]

def save_prediction(pred_mask, img_id, save_dirs, img=None):
    """マスク保存用関数"""
    mask_save_dir_merged, mask_save_dir_conj, mask_save_dir_corn, overlay_save_dir = save_dirs
    
    # 1. Merged (0,1,2)
    cv2.imwrite(os.path.join(mask_save_dir_merged, f"{img_id}.png"), pred_mask.astype(np.uint8))
    # 2. Conjunctiva (Binary)
    cv2.imwrite(os.path.join(mask_save_dir_conj, f"{img_id}.png"), (pred_mask == 1).astype(np.uint8) * 255)
    # 3. Cornea (Binary)
    cv2.imwrite(os.path.join(mask_save_dir_corn, f"{img_id}.png"), (pred_mask == 2).astype(np.uint8) * 255)
    
    # 4. Overlay (Optional)
    if img is not None:
        vis_img = img.copy()
        h, w = vis_img.shape[:2]
        pred_mask_vis = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        vis_img[pred_mask_vis == 1] = vis_img[pred_mask_vis == 1] * 0.5 + np.array([0, 255, 255]) * 0.5 
        vis_img[pred_mask_vis == 2] = vis_img[pred_mask_vis == 2] * 0.5 + np.array([255, 0, 0]) * 0.5    
        cv2.imwrite(os.path.join(overlay_save_dir, f"{img_id}_vis.jpg"), cv2.cvtColor(vis_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

def calculate_binary_metrics(pred_binary, gt_binary, prefix=""):
    """
    2値マスク(0 or 1)同士の比較を行い、6つの指標を計算する
    pred_binary: 予測マスク (0 or 1)
    gt_binary:   正解マスク (0 or 1)
    prefix:      辞書のキーにつける接頭辞 (例: "Conj_")
    """
    # フラットにする
    p = pred_binary.flatten()
    g = gt_binary.flatten()

    # 混同行列の計算 (TP, TN, FP, FN)
    # confusion_matrixは [TN, FP, FN, TP] の順で返すことがあるが、
    # ラベルが片方しかない場合などでサイズが変わるため手動計算が確実
    tp = np.sum((p == 1) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))

    # 指標計算 (ゼロ除算回避のため + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)  # Jaccard
    specificity = tn / (tn + fp + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # GTが完全に真っ黒(0)の場合の例外処理
    if np.sum(g) == 0:
        iou = 1.0 if np.sum(p) == 0 else 0.0
        recall = 1.0 # 対象がないので再現率は1とする考え方もあるが、文脈による。ここでは計算通り。
        f1 = 1.0 if np.sum(p) == 0 else 0.0

    return {
        f"{prefix}Accuracy": accuracy,
        f"{prefix}Jaccard": iou,
        f"{prefix}Specificity": specificity,
        f"{prefix}Precision": precision,
        f"{prefix}Recall": recall,
        f"{prefix}F1_Score": f1
    }

def run_full_inference():
    # 1. 設定読み込み
    with open('configs/config.json', 'r') as f:
        config = json.load(f)

    checkpoint_dir = config['paths']['model_save_dir']
    data_dir = config['paths']['data_dir']
    csv_filename = config['dataset']['csv_filename']
    img_size = config['dataset']['image_size']
    
    # CSV読み込み
    df = pd.read_csv(csv_filename)
    
    labeled_df = df[df['label'] == 1]
    unlabeled_df = df[df['label'] == 0]
    
    print(f"GTありデータ: {len(labeled_df)}枚")
    print(f"GTなしデータ: {len(unlabeled_df)}枚")

    # 出力先設定
    base_output_dir = "02_ROI_Segmentation"
    mask_dirs = [
        os.path.join(base_output_dir, "predictions", "masks_merged"),
        os.path.join(base_output_dir, "predictions", "masks_conjunctiva"),
        os.path.join(base_output_dir, "predictions", "masks_cornea"),
        os.path.join(base_output_dir, "predictions", "overlays")
    ]
    for d in mask_dirs: os.makedirs(d, exist_ok=True)
    
    # 評価結果保存用ディレクトリ
    eval_dir = os.path.join(base_output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    transform = get_aug(augmentation=False, size=img_size)
    
    # 評価結果リスト
    metrics_records = []

    # ==========================================
    # Part 1: GTあり (200枚) - CV推論 & 精度評価
    # ==========================================
    print("\n=== Part 1: Labeled Data Inference & Evaluation ===")
    
    for fold in range(1, 6):
        print(f"\n--- Processing Fold {fold} ---")
        ckpt_path = find_best_checkpoint(checkpoint_dir, fold)
        model = ConjunctivaSegmentationModule.load_from_checkpoint(ckpt_path, config=config)
        model.eval()
        if torch.cuda.is_available(): model.cuda()

        val_df = labeled_df[labeled_df['fold'] == fold]

        for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
            id_str = str(row['test_ID']).zfill(4)
            rl = row['R/L']
            filename = f'{id_str}_{rl}_01.png'
            img_id = f"{id_str}_{rl}"

            # 画像パス & ラベルパス
            img_path = os.path.join(data_dir, 'labeled_image', 'image', filename)
            lbl_path = os.path.join(data_dir, 'labeled_image', 'label', filename)

            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)

            # 1. 画像読み込み & 推論
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            augmented = transform(image=img, mask=np.zeros(img.shape[:2]))
            input_tensor = augmented['image'].unsqueeze(0)
            if torch.cuda.is_available(): input_tensor = input_tensor.cuda()

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy() # (512, 512)

            # 2. 正解ラベル読み込み & マスク化
            # dataset.pyのロジックを再現: 青=結膜(1), 赤=角膜(2)
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(lbl_path)
            if os.path.exists(lbl_path):
                lbl_src = cv2.imread(lbl_path) # BGR
                # サイズ合わせ (予測は512x512になっているので、GTもリサイズ)
                lbl_src = cv2.resize(lbl_src, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                gt_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
                # BGRなので Blue=channel 0, Red=channel 2
                # 閾値を設けて抽出
                gt_mask[lbl_src[:, :, 0] > 128] = 1 # Blue -> Conjunctiva
                gt_mask[lbl_src[:, :, 2] > 128] = 2 # Red -> Cornea

                # 3. 指標計算 (3パターン)
                # パターンA: 結膜 (Class 1)
                metrics_conj = calculate_binary_metrics(
                    (pred_mask == 1).astype(np.uint8), 
                    (gt_mask == 1).astype(np.uint8), 
                    prefix="Conj_"
                )

                # パターンB: 角膜 (Class 2)
                metrics_corn = calculate_binary_metrics(
                    (pred_mask == 2).astype(np.uint8), 
                    (gt_mask == 2).astype(np.uint8), 
                    prefix="Corn_"
                )

                # パターンC: 統合 (Class 1 or 2) vs Background
                metrics_comb = calculate_binary_metrics(
                    (pred_mask > 0).astype(np.uint8), 
                    (gt_mask > 0).astype(np.uint8), 
                    prefix="Combined_"
                )

                # 結果をまとめる
                record = {
                    "test_ID": row['test_ID'],
                    "R/L": row['R/L'],
                    "fold": fold,
                    **metrics_conj,
                    **metrics_corn,
                    **metrics_comb
                }
                metrics_records.append(record)

            # 保存
            save_prediction(pred_mask, img_id, mask_dirs, img)



    # 評価結果をCSV保存
    if metrics_records:
        df_metrics = pd.DataFrame(metrics_records)
        save_csv_path = os.path.join(eval_dir, "evaluation_results_detailed.csv")
        df_metrics.to_csv(save_csv_path, index=False)
        print(f"\n評価結果を保存しました: {save_csv_path}")
        
        # 平均値を表示
        print("\n=== Evaluation Summary (Mean) ===")
        mean_metrics = df_metrics.mean(numeric_only=True)
        # 主要なものだけピックアップ表示
        print(mean_metrics[["Conj_Jaccard", "Conj_F1_Score", "Corn_Jaccard", "Corn_F1_Score", "Combined_Jaccard", "Combined_F1_Score"]])

    # ==========================================
    # Part 2: GTなし (100枚) - Ensemble 推論
    # ==========================================
    print("\n=== Part 2: Unlabeled Data Inference (Ensemble) ===")
    
    if len(unlabeled_df) > 0:
        models = []
        print("Loading all 5 models for ensemble...")
        for fold in range(1, 6):
            ckpt_path = find_best_checkpoint(checkpoint_dir, fold)
            m = ConjunctivaSegmentationModule.load_from_checkpoint(ckpt_path, config=config)
            m.eval()
            if torch.cuda.is_available(): m.cuda()
            models.append(m)

        
        print(f"Successfully loaded {len(models)} models.")
        
        for idx, row in tqdm(unlabeled_df.iterrows(), total=len(unlabeled_df)):
            id_str = str(row['test_ID']).zfill(4)
            rl = row['R/L']
            filename = f'{id_str}_{rl}_01.png'
            img_path = os.path.join(data_dir, 'all_img', filename) 
            img_id = f"{id_str}_{rl}"
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)

            img = cv2.imread(img_path)
            if img is None: raise ValueError(f"Cannot read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            augmented = transform(image=img, mask=np.zeros(img.shape[:2]))
            input_tensor = augmented['image'].unsqueeze(0)
            if torch.cuda.is_available(): input_tensor = input_tensor.cuda()
            
            avg_probs = None
            with torch.no_grad():
                for model in models:
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    if avg_probs is None: avg_probs = probs
                    else: avg_probs += probs
                
                avg_probs /= len(models)
                final_mask = torch.argmax(avg_probs, dim=1).squeeze().cpu().numpy()
            
            save_prediction(final_mask, img_id, mask_dirs, img)
                        
    print("\nDone! 全300枚の推論が完了しました！")

if __name__ == "__main__":
    run_full_inference()