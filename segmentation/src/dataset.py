import os
import pandas as pd
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# ==========================================
# データ読み込み関数
# ==========================================
def load_dataset(mode='train', fold=1, data_dir='../01_raw_data', csv_filename='dataset.csv'):
    """
    mode: 'train' or 'val'
    fold: 交差検証のフォールド番号
    data_dir: データセットのルートディレクトリ (configから受け取る)
    csv_filename: CSVファイル名
    """
    assert mode in ['train', 'val'], 'split must be "train" or "val"'
    
    # パスの構築 (os.path.joinを使うことでWindows/Mac両対応にする)
    csv_path = os.path.join(data_dir, csv_filename)
    
    # CSVの読み込み
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # ラベルデータが存在するものだけを選択
    df = df[df['label'] > 0]
    
    # fold番号ごとに、'train' または 'val' に応じてデータを分割
    if mode == 'train':
        df = df[df['fold'] != fold]
    elif mode == 'val':
        df = df[df['fold'] == fold]

    # 画像とマスクのリストを作成
    images = []
    masks = []
    ids = []  

    # 画像フォルダとラベルフォルダのパス
    img_dir = os.path.join(data_dir, 'labeled_image', 'image')
    lbl_dir = os.path.join(data_dir, 'labeled_image', 'label')

    for i, row in df.iterrows():
        # 画像ファイル名を生成 (例: 0012_R_01.png)
        id_str = str(row['test_ID']).zfill(4)
        rl = row['R/L']
        fn = f'{id_str}_{rl}_01.png'
        
        img_path = os.path.join(img_dir, fn)
        lbl_path = os.path.join(lbl_dir, fn)
        
        # 画像がない場合はスキップ（あるいはエラーにする）
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            continue

        # 画像読み込み
        img = np.array(Image.open(img_path).convert('RGB'))
        mask_src = np.array(Image.open(lbl_path))

        images.append(img)

        # -------------------------------------------------------
        # マスク画像の処理 (青=結膜, 赤=角膜)
        # -------------------------------------------------------
        # OpenCVのinRangeで色抽出
        blue_mask = cv2.inRange(mask_src, (0, 0, 255, 255), (0, 0, 255, 255)) # 結膜
        red_mask = cv2.inRange(mask_src, (255, 0, 0, 255), (255, 0, 0, 255))  # 角膜
        
        # 将来的に使うかもしれない血管 (green)
        # green_mask = cv2.inRange(mask_src, (0, 255, 0, 255), (0, 255, 0, 255))

        # マルチクラスマスクの作成 (0:背景, 1:結膜, 2:角膜)
        combined_mask = np.zeros_like(blue_mask)
        combined_mask[np.where(blue_mask == 255)] = 1
        combined_mask[np.where(red_mask == 255)] = 2

        masks.append(combined_mask)
        id_info = f"{str(row['test_ID']).zfill(4)}_{row['R/L']}"
        ids.append(id_info)

    return images, masks, ids  # ★ 追加: idsも返す！

# ==========================================
# Augmentation (データ拡張) 定義
# ==========================================
def get_aug(augmentation, size, normalization=True):
    augs_train = [
        A.RandomResizedCrop(size=(size, size),
                            scale=(0.6, 1.0),
                            ratio=(0.8, 1.25),
                            p=1.0,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST,
                            ),
        A.HorizontalFlip(p=0.5),
        A.RandomToneCurve(scale=0.8, p=0.5),
        A.GridDistortion(p=0.5),
        ]
    
    augs_val = [
        A.Resize(width=size, height=size),
        ]

    if augmentation:
        aa = augs_train
    else:
        aa = augs_val

    if normalization:
        aa += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)]
    
    aa += [ToTensorV2()]
    
    return A.Compose(aa)

# ==========================================
# PyTorch Dataset クラス
# ==========================================
class ConjDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', fold=1, size=512, augmentation=True, 
                 normalization=True, data_dir='../01_raw_data', csv_filename='dataset.csv'):
        
        # load_datasetにパス情報を渡す
        self.images, self.labels, self.ids = load_dataset(  # ★ 変更
            mode=mode, 
            fold=fold, 
            data_dir=data_dir, 
            csv_filename=csv_filename
        )

        self.transform = transforms.Compose([ transforms.ToTensor() ])
        self.albu = get_aug(augmentation=augmentation, size=size, normalization=normalization)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx] # ★ 追加

        # Albumentationsによる変換
        auged = self.albu(image=image, mask=label)
        x = auged['image']
        t = auged['mask']
        
        return x, t.to(torch.int64), img_id # ★ 追加: IDも返す

    def __len__(self):
        return len(self.images)

# ==========================================
# DataLoader 作成関数 (学習用)
# ==========================================
def train_val_dataloader(batch_size=8, fold=1, augmentation=True, normalization=True, size=512, data_dir='../01_raw_data'):
    # Datasetの作成
    train = ConjDataset(mode='train', fold=fold, augmentation=augmentation, normalization=normalization, size=size, data_dir=data_dir)
    val = ConjDataset(mode='val', fold=fold, augmentation=augmentation, normalization=normalization, size=size, data_dir=data_dir)
    
    # DataLoaderの定義
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=False) # Valはshuffle不要
    
    return train_loader, val_loader