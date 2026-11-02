import pandas as pd
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# 300眼のデータセットから、imageと結膜、角膜のラベルのndarrayのリストを返すもの
def load_dataset(mode='train', fold=1, base_path='inputs'):
    assert mode in['train', 'val'], 'split must be "train" or "val"'
    df = pd.read_csv(f'{base_path}/dataset.csv')
    #　ラベルデータが選ばれているやつを選択
    df = df[df['label']>0]
    #　fold番号ごとに、'train' または 'val' に応じてデータを分割する
    if mode == 'train':
        df = df[df['fold'] != fold]
    if mode == 'val':
        df = df[df['fold'] == fold]

    #　それぞれの画像（nparray）が入ったリストを作成する
    images = []
    #conj_masks = []
    #cornea_masks = []
    masks = []

    for i, row in df.iterrows():
        #　画像ファイル名を生成
        id = row['test_ID']
        rl = row['R/L']
        fn = f'{str(id).zfill(4)}_{rl}_01.png'
         #　元画像と、マスクデータの読み込み
        img = np.array(Image.open(f'{base_path}/labeled_image/image/{fn}').convert('RGB'))
        mask = np.array(Image.open(f'{base_path}/labeled_image/label/{fn}'))

        images.append(img)

        blue_mask = cv2.inRange(mask,(0, 0, 255,255),(0, 0, 255,255)) #結膜
        red_mask = cv2.inRange(mask,(255, 0, 0,255),(255, 0, 0,255)) #角膜
        #green_mask = cv2.inRange(mask,(0, 255, 0,255),(0, 255, 0,255))    #将来的に使うかもしれない血管
        arr1 = np.where(blue_mask == 255)
        arr2 = np.where(red_mask == 255)
        combined_mask = np.zeros_like(blue_mask)
        combined_mask[arr1] = 1
        combined_mask[arr2] = 2

        #conj_masks.append(blue_mask)
        #cornea_masks.append(red_mask)
        masks.append(combined_mask)


    return images, masks#, conj_masks, cornea_masks


#　データ拡張（テンソル形式に変換）
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

#　pytorchに組み込むためのクラス
class ConjDataset(torch.utils.data.Dataset):

    def __init__(self, mode='train', fold=1, size=512, augmentation=True, 
                 normalization=True, base_path='inputs'):
        self.images, self.labels = load_dataset(mode=mode, fold=fold, base_path=base_path)
        self.transform = transforms.Compose([ transforms.ToTensor() ])
        self.albu = get_aug(augmentation=augmentation, size=size, normalization=normalization)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        auged = self.albu(image=image, mask=label)
        x = auged['image']
        t = auged['mask']
        return x, t.to(torch.int64)


    def __len__(self):
        return len(self.images)

#　データセットのクラスのインスタンス化するやつ
def train_val_dataset( fold=1, augmentation=True, normalization=True, size=512, base_path='inputs'):
    # データセットの取得
    train = ConjDataset(mode='train', fold=fold, augmentation=augmentation, normalization=normalization, size=size, base_path=base_path)
    val = ConjDataset(mode='val', fold=fold, augmentation=augmentation, normalization=normalization, size=size, base_path=base_path)

    return train, val 

#　データセットのデータローダーを作成
def train_val_dataloader(batch_size = 8, fold=1, augmentation=True, normalization=True, size=512, base_path='inputs'):
    train, val = train_val_dataset(fold=fold, augmentation=augmentation, normalization=normalization, size=size, base_path=base_path)
    
    # Data Loader を定義
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size)
    
    return train_loader, val_loader