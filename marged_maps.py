import numpy as np

def merged_maps(dataset):
    """
    ブレンディングなし、端5%カット、重なり部分の中央値でオフセット補正して合成
    """
    sorted_keys = sorted(dataset.keys(), key=lambda k: (dataset[k].y_motor, dataset[k].x_motor))
    if not sorted_keys:
        return None

    first_data = dataset[sorted_keys[0]]
    pixel_size_x = first_data.x_range / first_data.map_array.shape[1]
    pixel_size_y = first_data.y_range / first_data.map_array.shape[0]

    # --- 1. キャンバスサイズの計算 ---
    min_x_phys, min_y_phys = float('inf'), float('inf')
    max_x_phys, max_y_phys = float('-inf'), float('-inf')

    for key in sorted_keys:
        md = dataset[key]
        min_x_phys, min_y_phys = min(min_x_phys, md.x_motor), min(min_y_phys, md.y_motor)
        max_x_phys, max_y_phys = max(max_x_phys, md.x_motor + md.x_range), max(max_y_phys, md.y_motor + md.y_range)

    margin = 50
    total_width = int(np.ceil((max_x_phys - min_x_phys) / pixel_size_x)) + margin
    total_height = int(np.ceil((max_y_phys - min_y_phys) / pixel_size_y)) + margin

    # キャンバスは NaN で初期化
    canvas = np.full((total_height, total_width), np.nan, dtype=np.float32)
    origin_x, origin_y = min_x_phys, min_y_phys

    print(f"Canvas Size: {total_width} x {total_height}")

    # --- 2. 各タイルの処理 ---
    for i, key in enumerate(sorted_keys):
        current_md = dataset[key]
        img = current_md.map_array.copy().astype(np.float32)
        h, w = img.shape
        
        # 端5%をNaNで埋めて完全に無視する
        margin_h = int(h * 0.05)
        margin_w = int(w * 0.05)
        if margin_h > 0:
            img[:margin_h, :] = np.nan
            img[-margin_h:, :] = np.nan
        if margin_w > 0:
            img[:, :margin_w] = np.nan
            img[:, -margin_w:] = np.nan

        # 配置座標の計算
        final_x = int((current_md.x_motor - origin_x) / pixel_size_x)
        final_y = int((current_md.y_motor - origin_y) / pixel_size_y)

        # 範囲内チェック
        if final_y < 0 or final_x < 0 or final_y + h > total_height or final_x + w > total_width:
             continue

        sy, sx = slice(final_y, final_y + h), slice(final_x, final_x + w)
        canvas_slice = canvas[sy, sx]

        # --- 3. 重なり部分の中央値合わせ（オフセット補正） ---
        # キャンバスに既に値があり、かつ今貼るタイルもデータがある場所を特定
        overlap_mask = (~np.isnan(canvas_slice)) & (~np.isnan(img))
        
        if np.any(overlap_mask):
            # 重なり領域における「キャンバスの値 - 新しいタイルの値」の差の中央値
            diff = canvas_slice[overlap_mask] - img[overlap_mask]
            z_offset = np.median(diff)
            
            # タイル全体にオフセットを適用
            img += z_offset
            print(f"[{key}] Offset applied: {z_offset:.4f}")
        else:
            # 重なりがない場合（最初の1枚目など）は何もしない
            print(f"[{key}] Placed without offset adjustment (no overlap).")

        # --- 4. キャンバスへ配置（上書き） ---
        valid_pixel_mask = ~np.isnan(img)
        canvas[sy, sx][valid_pixel_mask] = img[valid_pixel_mask]

    # --- 5. 自動クロップ ---
    valid_pixels = ~np.isnan(canvas)
    rows = np.any(valid_pixels, axis=1)
    cols = np.any(valid_pixels, axis=0)
    
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        cropped_array = canvas[y_min:y_max+1, x_min:x_max+1]
        new_x_motor = origin_x + (x_min * pixel_size_x)
        new_y_motor = origin_y + (y_min * pixel_size_y)
    else:
        cropped_array = canvas
        new_x_motor, new_y_motor = origin_x, origin_y

    # 結果格納
    from map_data import map_data
    merged_md = map_data()
    merged_md.file_name = "median_matched_result"
    merged_md.map_array = cropped_array
    merged_md.x_range = cropped_array.shape[1] * pixel_size_x
    merged_md.y_range = cropped_array.shape[0] * pixel_size_y
    merged_md.x_motor = new_x_motor
    merged_md.y_motor = new_y_motor

    return merged_md