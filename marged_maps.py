import numpy as np
from map_data import map_data
from scipy.ndimage import gaussian_filter

def create_weight_mask(shape, edge_falloff=0.1):
    """
    画像の中心が1、端が0になるような重みマスクを作成する。
    
    Args:
        shape: (height, width)
        edge_falloff: 端から何割の領域を使って減衰させるか (0.0~0.5)
                      0.1 なら 幅の10%を使って 0->1 にフェードインする
    """
    h, w = shape
    
    # X方向のウェイト (0 -> 1 -> 0)
    # 台形のような形を作る
    x_ramp = np.ones(w, dtype=np.float32)
    ramp_w = int(w * edge_falloff)
    if ramp_w > 0:
        # 左端の立ち上がり (0 -> 1)
        x_ramp[:ramp_w] = np.linspace(0, 1, ramp_w)
        # 右端の立ち下がり (1 -> 0)
        x_ramp[-ramp_w:] = np.linspace(1, 0, ramp_w)

    # Y方向のウェイト
    y_ramp = np.ones(h, dtype=np.float32)
    ramp_h = int(h * edge_falloff)
    if ramp_h > 0:
        y_ramp[:ramp_h] = np.linspace(0, 1, ramp_h)
        y_ramp[-ramp_h:] = np.linspace(1, 0, ramp_h)
    
    # 2次元マスクにする (外積)
    # w(x, y) = wx(x) * wy(y)
    weight_mask = np.outer(y_ramp, x_ramp)
    
    return weight_mask

def flatten_image(data, method='plane', sigma=50):
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
        #  2次曲面 (Z = aX^2 + bY^2 + cXY + dX + eY + f)
        
        # 計算量削減のため、データ点を間引いてフィッティングする (1/100程度)
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


def merged_maps(dataset):
    """
    AFMマップを連結する
    """
    sorted_keys = sorted(dataset.keys(), key=lambda k: (dataset[k].y_motor, dataset[k].x_motor))
    if not sorted_keys:
        return map_data()

    first_data = dataset[sorted_keys[0]]
    pixel_size_x = first_data.x_range / first_data.map_array.shape[1]
    pixel_size_y = first_data.y_range / first_data.map_array.shape[0]

    # キャンバスサイズ計算
    min_x_phys, min_y_phys = float('inf'), float('inf')
    max_x_phys, max_y_phys = float('-inf'), float('-inf')

    for key in sorted_keys:
        md = dataset[key]
        min_x_phys = min(min_x_phys, md.x_motor)
        min_y_phys = min(min_y_phys, md.y_motor)
        max_x_phys = max(max_x_phys, md.x_motor + md.x_range)
        max_y_phys = max(max_y_phys, md.y_motor + md.y_range)

    margin = 50
    total_width = int(np.ceil((max_x_phys - min_x_phys) / pixel_size_x)) + margin
    total_height = int(np.ceil((max_y_phys - min_y_phys) / pixel_size_y)) + margin

    # Canvasは 0 で初期化(累積加算のため)
    canvas = np.zeros((total_height, total_width), dtype=np.float32)
    weight_map = np.zeros((total_height, total_width), dtype=np.float32)

    origin_x = min_x_phys
    origin_y = min_y_phys

    print(f"Canvas Size: {total_width} x {total_height} pixels")

    for i, key in enumerate(sorted_keys):
        current_md = dataset[key]
        current_img = current_md.map_array.copy()
        h, w = current_img.shape
        
        # NaNマスク
        nan_mask = np.isnan(current_img)
        # 加算用にNaNを0にしておく
        current_img = np.nan_to_num(current_img, nan=0.0)

        final_x = int((current_md.x_motor - origin_x) / pixel_size_x)
        final_y = int((current_md.y_motor - origin_y) / pixel_size_y)

        # 範囲外チェック
        if final_y < 0 or final_x < 0 or final_y + h > total_height or final_x + w > total_width:
             continue

        # --- Z オフセット補正 (Topographyのみ) ---
        if current_md.target_name == "topography":
            # オーバーラップ領域の算出
            sy_start, sy_end = final_y, final_y + h
            sx_start, sx_end = final_x, final_x + w
            
            # 既存のキャンバス上の重み
            current_weight_slice = weight_map[sy_start:sy_end, sx_start:sx_end]
            
            # 「既にデータがある場所(weight > 0.1)」かつ「今貼るデータもNaNじゃない場所」
            # ※ weight_mapが0の場所で割り算しないよう注意
            valid_mask_canvas = current_weight_slice > 0.1
            valid_overlap = valid_mask_canvas & (~nan_mask)

            if np.sum(valid_overlap) > 100:
                # キャンバスの既存値（平均）を取得
                # canvasには重み付き和が入っているので、重みで割って元の高さに戻す
                existing_height = canvas[sy_start:sy_end, sx_start:sx_end][valid_overlap] / current_weight_slice[valid_overlap]
                new_height = current_img[valid_overlap]
                
                # 中央値でオフセットを合わせる
                z_offset = np.median(existing_height - new_height)
                
                # NaN以外の部分にオフセット加算
                current_img[~nan_mask] += z_offset
                print(f"[{key}] Leveling applied: offset={z_offset:.4f}")

        # 重み付き合成
        # 重みマスク作成 (端を減衰)
        blending_mask = create_weight_mask((h, w), edge_falloff=0.2)
        blending_mask[nan_mask] = 0  # データがない場所は重み0

        # 加算
        canvas[final_y:final_y+h, final_x:final_x+w] += current_img * blending_mask
        weight_map[final_y:final_y+h, final_x:final_x+w] += blending_mask

    # 正規化 (合計値 / 重みの合計)
    merged_array = np.full_like(canvas, np.nan)
    
    # 重みがある場所だけ計算（0除算回避）
    valid_pixels = weight_map > 0
    merged_array[valid_pixels] = canvas[valid_pixels] / weight_map[valid_pixels]


    # 自動クロップ
    rows = np.any(valid_pixels, axis=1)
    cols = np.any(valid_pixels, axis=0)
    
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        cropped_array = merged_array[y_min:y_max+1, x_min:x_max+1]
        
        new_x_motor = origin_x + (x_min * pixel_size_x)
        new_y_motor = origin_y + (y_min * pixel_size_y)
        new_x_range = cropped_array.shape[1] * pixel_size_x
        new_y_range = cropped_array.shape[0] * pixel_size_y
        print(f"Auto-cropped: {merged_array.shape} -> {cropped_array.shape}")
    else:
        cropped_array = merged_array
        new_x_motor, new_y_motor = origin_x, origin_y
        new_x_range, new_y_range = (max_x_phys - min_x_phys), (max_y_phys - min_y_phys)

    # 最終的な2次曲面補正
    if dataset[sorted_keys[0]].target_name == "topography":
        cropped_array = flatten_image(cropped_array, method='poly')

    # 結果格納
    merged_md = map_data()
    merged_md.file_name = "merged_result"
    merged_md.map_array = cropped_array
    merged_md.x_range = new_x_range
    merged_md.y_range = new_y_range
    merged_md.x_motor = new_x_motor
    merged_md.y_motor = new_y_motor

    return merged_md