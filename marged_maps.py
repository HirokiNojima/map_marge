import numpy as np
from map_data import map_data
from scipy.ndimage import gaussian_filter

def _flatten_image(data, method='plane', sigma=50):
    """
    Args:
        method (str): 
            'robust' -> 外れ値を無視した平面補正 (1次)
            'poly'   -> 2次曲面補正 (スキャナーの湾曲や試料のたわみを除去) ★今回のおすすめ
            'gaussian' -> 背景除去 (局所強調)
    """
    Z = data.copy()
    mask = ~np.isnan(Z)
    if not np.any(mask):
        return Z

    h, w = Z.shape
    y_idx, x_idx = np.indices((h, w))
    
    X_valid = x_idx[mask]
    Y_valid = y_idx[mask]
    Z_valid = Z[mask]

    if method == 'offset':
        Z -= np.median(Z_valid)

    elif method == 'plane' or method == 'robust':
        # 1次平面 (Z = aX + bY + c)
        # robustの場合は外れ値を除外して計算
        if method == 'robust':
            v_min, v_max = np.percentile(Z_valid, [10, 90])
            fit_mask = (Z_valid >= v_min) & (Z_valid <= v_max)
            X_use, Y_use, Z_use = X_valid[fit_mask], Y_valid[fit_mask], Z_valid[fit_mask]
        else:
            X_use, Y_use, Z_use = X_valid, Y_valid, Z_valid

        A = np.c_[X_use, Y_use, np.ones(X_use.shape)]
        C, _, _, _ = np.linalg.lstsq(A, Z_use, rcond=None)
        background = C[0] * x_idx + C[1] * y_idx + C[2]
        Z = Z - background
        print(f"Flatten: {method.capitalize()} Plane removed")

    elif method == 'poly':
        # ★追加: 2次曲面 (Z = aX^2 + bY^2 + cXY + dX + eY + f)
        # これが最も「自然」に全体の歪みを取ります
        
        # 計算量削減のため、データ点を間引いてフィッティングする (1/100程度)
        # ※全点使うと重いため
        skip = 10 
        X_use = X_valid[::skip]
        Y_use = Y_valid[::skip]
        Z_use = Z_valid[::skip]

        # 2次の項を作成
        A = np.c_[X_use**2, Y_use**2, X_use*Y_use, X_use, Y_use, np.ones(X_use.shape)]
        
        # フィッティング
        C, _, _, _ = np.linalg.lstsq(A, Z_use, rcond=None)
        
        # 全体に対する背景曲面を作成
        background = (C[0]*x_idx**2 + C[1]*y_idx**2 + C[2]*x_idx*y_idx + 
                      C[3]*x_idx + C[4]*y_idx + C[5])
        
        Z = Z - background
        print("Flatten: 2nd Order Polynomial removed (Natural curvature correction)")

    elif method == 'gaussian':
        Z_filled = Z.copy()
        Z_filled[~mask] = np.nanmedian(Z_valid)
        background = gaussian_filter(Z_filled, sigma=sigma)
        Z = Z - background
        print(f"Flatten: Gaussian (sigma={sigma})")

    return Z


def marged_maps(dataset):
    """
    AFMマップを連結する。
    
    Args:
        dataset: map_data辞書
    Returns:
        merged_md: 連結されたmap_dataオブジェクト
    """
    
    # 1. データの整理
    sorted_keys = sorted(dataset.keys(), key=lambda k: (dataset[k].y_motor, dataset[k].x_motor))
    if not sorted_keys:
        return map_data()

    # 解像度取得
    first_data = dataset[sorted_keys[0]]
    pixel_size_x = first_data.x_range / first_data.map_array.shape[1]
    pixel_size_y = first_data.y_range / first_data.map_array.shape[0]

    # 2. キャンバスサイズ計算
    min_x_phys, min_y_phys = float('inf'), float('inf')
    max_x_phys, max_y_phys = float('-inf'), float('-inf')

    for key in sorted_keys:
        md = dataset[key]
        min_x_phys = min(min_x_phys, md.x_motor)
        min_y_phys = min(min_y_phys, md.y_motor)
        max_x_phys = max(max_x_phys, md.x_motor + md.x_range)
        max_y_phys = max(max_y_phys, md.y_motor + md.y_range)

    # 余白
    margin = 50
    total_width = int(np.ceil((max_x_phys - min_x_phys) / pixel_size_x)) + margin
    total_height = int(np.ceil((max_y_phys - min_y_phys) / pixel_size_y)) + margin

    canvas = np.full((total_height, total_width), np.nan, dtype=np.float32)
    weight_map = np.zeros((total_height, total_width), dtype=np.float32)

    origin_x = min_x_phys
    origin_y = min_y_phys

    print(f"Canvas Size: {total_width} x {total_height} pixels")

    # 3. 逐次合成
    for i, key in enumerate(sorted_keys):
        current_md = dataset[key]
        current_img = current_md.map_array.copy()
        h, w = current_img.shape
        
        # モーター座標配置
        final_x = int((current_md.x_motor - origin_x) / pixel_size_x)
        final_y = int((current_md.y_motor - origin_y) / pixel_size_y)

        # --- Z オフセット (高さ) 補正 ---
        # トポグラフィデータの場合のみ実施
        if current_md.target_name == "topography":
            sy_start = max(0, final_y)
            sy_end = min(total_height, final_y + h)
            sx_start = max(0, final_x)
            sx_end = min(total_width, final_x + w)

            iy_start = max(0, -final_y)
            iy_end = iy_start + (sy_end - sy_start)
            ix_start = max(0, -final_x)
            ix_end = ix_start + (sx_end - sx_start)

            if (sy_end > sy_start) and (sx_end > sx_start):
                canvas_overlap = canvas[sy_start:sy_end, sx_start:sx_end]
                new_img_overlap = current_img[iy_start:iy_end, ix_start:ix_end]
                
                common_mask = (~np.isnan(canvas_overlap)) & (~np.isnan(new_img_overlap))
                
                if np.sum(common_mask) > 100:
                    z_canvas = canvas_overlap[common_mask]
                    z_new = new_img_overlap[common_mask]
                    z_offset = np.median(z_canvas - z_new)
                    
                    current_img += z_offset
                    print(f"[{key}] Leveling applied: {z_offset:.4f}")

        # --- 合成 (Blending) ---
        if final_y < 0 or final_x < 0 or final_y + h > total_height or final_x + w > total_width:
             print(f"[{key}] Out of bounds")
             continue

        canvas_region = canvas[final_y:final_y+h, final_x:final_x+w]
        weight_region = weight_map[final_y:final_y+h, final_x:final_x+w]
        
        new_valid_mask = ~np.isnan(current_img)
        
        weight_region[new_valid_mask] += 1
        
        nan_in_canvas = np.isnan(canvas_region)
        canvas_region[nan_in_canvas] = 0
        
        canvas_region[new_valid_mask] += current_img[new_valid_mask]
        
        canvas[final_y:final_y+h, final_x:final_x+w] = canvas_region
        weight_map[final_y:final_y+h, final_x:final_x+w] = weight_region

    # 平均化
    with np.errstate(invalid='ignore', divide='ignore'):
        merged_array = canvas / weight_map

    # 5. 自動クロップ & 座標再計算
    # 有効データが存在する範囲を探す
    valid_mask = ~np.isnan(merged_array)
    
    if np.any(valid_mask):
        rows = np.any(valid_mask, axis=1)
        cols = np.any(valid_mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # 配列をクロップ
        cropped_array = merged_array[y_min:y_max+1, x_min:x_max+1]
        
        # 物理座標の更新 (クロップした分だけ原点が移動する)
        new_x_motor = origin_x + (x_min * pixel_size_x)
        new_y_motor = origin_y + (y_min * pixel_size_y)
        
        # 範囲の更新
        new_x_range = cropped_array.shape[1] * pixel_size_x
        new_y_range = cropped_array.shape[0] * pixel_size_y
        
        print(f"Auto-cropped: {merged_array.shape} -> {cropped_array.shape}")
    else:
        # データがない場合（全てNaN）
        cropped_array = merged_array
        new_x_motor = origin_x
        new_y_motor = origin_y
        new_x_range = (max_x_phys - min_x_phys)
        new_y_range = (max_y_phys - min_y_phys)

    # topographyの場合、最後にフラットニングを実行
    if dataset[sorted_keys[0]].target_name == "topography":
        cropped_array = _flatten_image(cropped_array, method='poly')

    # 6. 結果の格納
    merged_md = map_data()
    merged_md.file_name = "merged_result"
    merged_md.map_array = cropped_array
    merged_md.x_range = new_x_range
    merged_md.y_range = new_y_range
    merged_md.x_motor = new_x_motor
    merged_md.y_motor = new_y_motor

    return merged_md