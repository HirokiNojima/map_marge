import numpy as np
from map_data import map_data
from skimage.registration import phase_cross_correlation
import math

def marged_maps(dataset):
    """
    AFMマップデータを連結し、位置ズレ補正を行った単一のマップを生成する。
    
    Args:
        dataset (dict): import_filesから出力された辞書 {filename: map_dataオブジェクト}
    
    Returns:
        merged_md (map_data): 連結された単一のmap_dataオブジェクト
    """
    
    # 1. データの整理とソート (処理順序を安定させるため)
    # ファイル名またはモーター位置でソートすることをお勧めします。ここではy, x座標順に並べ替えます。
    sorted_keys = sorted(dataset.keys(), key=lambda k: (dataset[k].y_motor, dataset[k].x_motor))
    print("Processing order:", sorted_keys)
    if not sorted_keys:
        return map_data()

    # ベースとなる解像度（um/pixel）を計算 (最初のデータを使用)
    first_data = dataset[sorted_keys[0]]
    pixel_size_x = first_data.x_range / first_data.map_array.shape[1]
    pixel_size_y = first_data.y_range / first_data.map_array.shape[0]

    print(f"Base pixel size: {pixel_size_x:.4f} um/pixel (X), {pixel_size_y:.4f} um/pixel (Y)")

    # 2. 全体のキャンバスサイズを決定するための座標計算
    # 全データのモーター座標から、全体の物理的な範囲(Bounding Box)を推定
    min_x_phys, min_y_phys = float('inf'), float('inf')
    max_x_phys, max_y_phys = float('-inf'), float('-inf')

    for key in sorted_keys:
        md = dataset[key]
        # モーター位置は左上として計算
        min_x_phys = min(min_x_phys, md.x_motor)
        min_y_phys = min(min_y_phys, md.y_motor)
        max_x_phys = max(max_x_phys, md.x_motor + md.x_range)
        max_y_phys = max(max_y_phys, md.y_motor + md.y_range)

    print(f"Overall physical range: X({min_x_phys}, {max_x_phys}), Y({min_y_phys}, {max_y_phys})")

    # キャンバスのピクセルサイズ計算（少し余裕を持たせる）
    total_width = int(np.ceil(((max_x_phys - min_x_phys) / pixel_size_x))  + 100)
    total_height = int(np.ceil(((max_y_phys - min_y_phys) / pixel_size_y)) + 100)

    # 巨大なキャンバスと、重なり回数を記録するウェイトマップを作成
    # 背景はNaNで埋めておき、データがない場所を区別する
    canvas = np.full((total_height, total_width), np.nan, dtype=np.float32)
    weight_map = np.zeros((total_height, total_width), dtype=np.float32)

    # 基準位置（キャンバスの原点に対する物理座標オフセット）
    origin_x = min_x_phys
    origin_y = min_y_phys

    print(f"Canvas Size: {total_width} x {total_height} pixels")

    # 3. 逐次配置と位置補正
    for i, key in enumerate(sorted_keys):
        current_md = dataset[key]
        current_img = current_md.map_array
        
        # モーター座標に基づく「理想的な」ピクセル位置（左上）
        rough_x = int((current_md.x_motor - origin_x) / pixel_size_x)
        rough_y = int((current_md.y_motor - origin_y) / pixel_size_y)
        
        h, w = current_img.shape
        
        # 配置予定位置
        target_slice_y = slice(rough_y, rough_y + h)
        target_slice_x = slice(rough_x, rough_x + w)

        # 既にキャンバスにデータが存在するか確認（重なり領域の抽出）
        existing_patch = canvas[target_slice_y, target_slice_x]
        
        # NaNではない（既にデータがある）領域のマスクを作成
        valid_mask = ~np.isnan(existing_patch)
        
        drift_y, drift_x = 0.0, 0.0

        # 重なりが十分ある場合のみ、位置補正を実行
        # (ピクセル数の5%以上が重なっている場合など、閾値は調整可能)
        overlap_ratio = np.sum(valid_mask) / (w * h)
        
        #if overlap_ratio > 0.05: 
        if False:
            # 既存の画像データ（比較対象）と新しい画像データの重なり部分を抽出
            # phase_cross_correlation はNaNを含められないため、NaNを0または平均値で置換して比較
            ref_image = np.nan_to_num(existing_patch)
            moving_image = np.nan_to_num(current_img)
            
            # マスクを適用して比較（簡易的な方法として、単純な画像全体比較を行う）
            # より厳密にはvalid_mask内の領域だけで切り出す処理が必要だが、
            # masked registrationは計算コストが高いため、ここではupsample_factorを用いたサブピクセル推定を行う
            
            try:
                # 精度向上のため upsample_factor=10 (0.1ピクセル精度) を指定
                shift, error, diffphase = phase_cross_correlation(
                    ref_image, 
                    moving_image, 
                    upsample_factor=10,
                    reference_mask=valid_mask # 重なっている部分のみを比較に使用
                )
                
                # 検出されたズレ (shiftは (y, x))
                # 注意: shiftは「refに対してmovingをどれだけ動かせば合うか」
                drift_y, drift_x = shift[0], shift[1]
                
                # 異常な大きさの補正（誤検知）は無視するリミッター (例: 画像サイズの1/4以上のズレは採用しない)
                if abs(drift_y) > h/4 or abs(drift_x) > w/4:
                    print(f"Warning: Large drift detected in {key}. Ignoring correction.")
                    drift_y, drift_x = 0, 0
                else:
                    print(f"Correcting {key}: dy={drift_y:.2f}, dx={drift_x:.2f}")

            except Exception as e:
                print(f"Registration failed for {key}: {e}")

        # 4. 補正後の位置に配置
        final_y = int(round(rough_y - drift_y))
        final_x = int(round(rough_x - drift_x))

        # キャンバス範囲外チェック
        if final_y < 0 or final_x < 0 or final_y + h > total_height or final_x + w > total_width:
             print(f"Skipping {key}: Out of canvas bounds after correction.")
             continue

        # データの合成（加重平均の準備）
        # キャンバスの該当領域を取得
        canvas_region = canvas[final_y:final_y+h, final_x:final_x+w]
        weight_region = weight_map[final_y:final_y+h, final_x:final_x+w]

        # 新しいデータを配置
        # NaNの部分（まだデータがない部分）にはそのまま値を入れ、
        # 既にデータがある部分は値を加算して後で割る
        
        # 新しいデータの配置用マスク
        new_data_mask = ~np.isnan(current_img)
        
        # キャンバス上のNaNを0として扱い、加算できるようにする
        temp_canvas_region = np.nan_to_num(canvas_region)
        temp_new_img = np.nan_to_num(current_img)

        # 加算処理
        # (データがある場所 + 新しいデータ)
        # NaNの場所は更新しないように注意が必要だが、ここではシンプルに
        # 「新しい画像が有効な場所」について加算を行う
        
        # ロジック:
        # 1. canvasがNaN かつ newが値あり -> newの値をセット、weight=1
        # 2. canvasが値あり かつ newが値あり -> 値を加算、weight+=1
        
        # 重み更新
        weight_region[new_data_mask] += 1
        
        # 値更新 (現在NaNの場所には0を入れてから加算)
        is_nan_canvas = np.isnan(canvas_region)
        canvas_region[is_nan_canvas] = 0 # 一時的に0にする
        
        # 値を加算 (新しい画像のNaN部分は加算しない)
        canvas_region[new_data_mask] += current_img[new_data_mask]
        
        # 以前NaNだった場所で、今回データが来なかった場所をNaNに戻す（必要であれば）
        # ただし今回は初期化でNaN、加算時に0にしているので、weightで割る時に処理する
        
        canvas[final_y:final_y+h, final_x:final_x+w] = canvas_region
        weight_map[final_y:final_y+h, final_x:final_x+w] = weight_region

    # 5. 平均化処理 (合計値 / 重み)
    # 重みが0の場所（データなし）はNaNに戻す
    with np.errstate(invalid='ignore', divide='ignore'):
        merged_array = canvas / weight_map
    
    # map_dataオブジェクトに格納して返す
    merged_md = map_data()
    merged_md.file_name = "merged_result"
    merged_md.map_array = merged_array
    merged_md.x_range = (max_x_phys - min_x_phys)
    merged_md.y_range = (max_y_phys - min_y_phys)
    merged_md.x_motor = min_x_phys
    merged_md.y_motor = min_y_phys

    return merged_md