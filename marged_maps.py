import numpy as np
from map_data import map_data

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
    
    merged_md = map_data()
    merged_md.file_name = "merged_result"
    merged_md.map_array = merged_array
    merged_md.x_range = (max_x_phys - min_x_phys)
    merged_md.y_range = (max_y_phys - min_y_phys)
    merged_md.x_motor = min_x_phys
    merged_md.y_motor = min_y_phys

    return merged_md