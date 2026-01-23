import matplotlib.pyplot as plt
import numpy as np
import os

def save_results(merged_md, target_name, output_folder="results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 元データをコピー（破壊しないため）
    data = merged_md.map_array.copy()
    if target_name =="youngs_modulus":
        data = np.log10(data)

    filename_base = os.path.join(output_folder, "merged_map")

    # --- 画像保存の前処理 ---
    
    # 1. NaN（背景）の処理
    # データがある部分だけのマスクを作成
    valid_mask = ~np.isnan(data)
    
    if not np.any(valid_mask):
        print("エラー: 有効なデータがありません。保存をスキップします。")
        return

    # 2. コントラスト調整 (重要!)
    # 最大値・最小値をそのまま使うと、スパイクノイズで真っ暗になるため、
    # 上位・下位 5% をカットした範囲を色の基準にします（AFM画像の定石）。
    vmin = np.percentile(data[valid_mask], 5)   # 下位5%
    vmax = np.percentile(data[valid_mask], 95)  # 上位95%

    # NaNの部分を「vmin（一番暗い色）」で埋める
    # ※ これをしておかないとimsaveが混乱します
    data_filled = data.copy()
    data_filled[~valid_mask] = vmin

    # --- 保存実行 ---

    # A. 画像のみ (imsave)
    try:
        plt.imsave(
            f"{filename_base}_high_res.png", 
            data_filled, 
            cmap='afmhot', 
            origin='lower',
            vmin=vmin, 
            vmax=vmax
        )
        print(f"Saved: {filename_base}_high_res.png")
    except Exception as e:
        print(f"imsave failed: {e}")

    # B. カラーバー付き (savefig)
    plt.figure(figsize=(10, 10))
    # ここでは set_bad を使ってNaNを特定の色（黒など）にすることも可能
    current_cmap = plt.get_cmap('afmhot').copy()
    current_cmap.set_bad(color='black') # NaNを黒にする
    
    plt.imshow(data, cmap=current_cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Merged " + target_name + " Map")
    plt.savefig(f"{filename_base}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename_base}_plot.png")

    # C. 数値データ (npz)
    np.savez_compressed(f"{filename_base}.npz", map_data=merged_md.map_array)
    print(f"Saved: {filename_base}.npz")