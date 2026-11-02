import cv2
import numpy as np
from scipy import stats
from skimage import morphology


analysis_params = {
    'dcs_clip_percent': 2,
    'bilateral_d': 3, 'bilateral_sigma': 30,
    'blackhat_ksize': 5,
    'adaptive_blocksize': 5,
    'adaptive_c': 1,
    'green_min_size': 10,
    'a_norm_thresh': 1,
    'close_ksize': 3,
    'skel_min_size': 15
}


def extract_intermediate_steps(img_rgb, roi_mask, params):
    """画像処理パイプラインの全段階の生データを辞書として抽出する"""
    steps = {}
    steps['01_Original_Image'] = img_rgb

    img_float = img_rgb.astype(np.float32)
    R, G, B = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_normalized = img_lab[:,:,1].astype(np.float32) - 128.0

    img_green = G.copy()
    img_green[roi_mask == 0] = 0
    steps['02_Green_Channel'] = img_green

    mask = img_green != 0
    masked_image = img_green[mask]
    img_f = masked_image.astype(np.float32)
    min_val, max_val = np.min(img_f), np.max(img_f)
    hist, bins = np.histogram(img_f.flatten(), bins=256, range=(min_val, max_val))
    total_pixels = masked_image.size
    clip_pixels = int(total_pixels * (params['dcs_clip_percent'] / 100.0))

    lower_bound = bins[0]; cumulative = 0
    for i, count in enumerate(hist):
        cumulative += count
        if cumulative > clip_pixels: lower_bound = bins[i]; break

    upper_bound = bins[-1]; cumulative = 0
    for i, count in enumerate(hist[::-1]):
        cumulative += count
        if cumulative >= clip_pixels: upper_bound = bins[-(i+1)]; break

    stretched = np.clip(img_f, lower_bound, upper_bound)
    stretched = (stretched - lower_bound) / (upper_bound - lower_bound) * 255.0
    img_dcs = np.zeros_like(img_green, dtype=np.uint8)
    img_dcs[mask] = stretched.astype(np.uint8)
    steps['03_Dynamic_Contrast_Stretch'] = img_dcs

    img_blur = cv2.bilateralFilter(img_dcs, d=params['bilateral_d'],
                                   sigmaColor=params['bilateral_sigma'], sigmaSpace=params['bilateral_sigma'])
    steps['04_Bilateral_Filter'] = img_blur

    ksize_bh = params['blackhat_ksize']
    kernel_hat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_bh, ksize_bh))
    img_blackhat = cv2.morphologyEx(img_blur, cv2.MORPH_BLACKHAT, kernel_hat)
    steps['05_Blackhat_Transform'] = img_blackhat

    if np.any(img_blackhat[roi_mask == 1]):
        inv_blackhat = cv2.bitwise_not(img_blackhat)
        binary_raw = cv2.adaptiveThreshold(inv_blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize=params['adaptive_blocksize'], C=params['adaptive_c'])
        binary_raw = cv2.bitwise_and(binary_raw, binary_raw, mask=roi_mask)
    else:
        binary_raw = np.zeros_like(img_blackhat)
    steps['06_Adaptive_Threshold'] = binary_raw

    mask_green = morphology.remove_small_objects(binary_raw > 0, min_size=params['green_min_size']) & (roi_mask == 1)
    mask_vessel = mask_green & (a_normalized > params['a_norm_thresh'])

    ksize_cl = params['close_ksize']
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_cl, ksize_cl))
    mask_bridged = cv2.morphologyEx(mask_vessel.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    steps['07_Final_Vessel_Mask'] = mask_bridged * 255

    steps['08_LabA_Density_Reds'] = a_normalized # ここは元の数値を保持し、描画時にカラーマップを適用

    skel_and = morphology.skeletonize(mask_bridged > 0) & (roi_mask == 1)
    skel_clean = morphology.remove_small_objects(skel_and > 0, min_size=params['skel_min_size'], connectivity=2)
    steps['09_Skeleton'] = skel_clean

    return steps, a_normalized, mask_bridged, skel_clean


def calculate_fractal_dimension(skel_img):
    """
    【フラクタル次元の理論】
    ボックスカウント法により、血管網が平面をどの程度「充填」しているかを評価する。
    スケールを変えながら占有ボックス数を計数し、その両対数プロットの傾きから複雑性を導出する。
    """
    if skel_img.ndim != 2 or not np.any(skel_img): return np.nan
    Z = (skel_img > 0)
    p = min(Z.shape)
    n = int(np.floor(np.log2(p)))
    sizes = 2**np.arange(n, 1, -1)

    valid_sizes, valid_counts = [], []
    for size in sizes:
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
                            np.arange(0, Z.shape[1], size), axis=1)
        count = len(np.where(S > 0)[0])
        if count > 0:
            valid_sizes.append(size); valid_counts.append(count)

    if len(valid_counts) < 2: return np.nan

    slope, _, _, _, _ = stats.linregress(np.log(valid_sizes), np.log(valid_counts))
    return np.nan if np.isnan(slope) else -slope
