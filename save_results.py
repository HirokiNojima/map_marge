import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

def _draw_scalebar(ax, x_range, img_width_px, unit="um", color='white'):
    """
    Axesにスケールバーを自動計算して追加する内部関数
    """
    # 1ピクセルあたりの物理サイズ
    pixel_size = x_range / img_width_px
    
    # バーの目標長さを画像の幅の約1/5に設定
    target_length = x_range / 5
    
    # キリの良い数字（1, 2, 5の倍数）を探す
    order = 10 ** np.floor(np.log10(target_length))
    multipliers = [1, 2, 5, 10]
    bar_length_phys = order
    for m in multipliers:
        if order * m <= target_length:
            bar_length_phys = order * m
            
    # ピクセル単位の長さに変換
    bar_length_px = bar_length_phys / pixel_size
    
    # ラベル作成 (整数なら小数点なしで表示)
    label_num = int(bar_length_phys) if bar_length_phys % 1 == 0 else bar_length_phys
    label = f"{label_num} {unit}"
    
    # フォント設定
    fontprops = fm.FontProperties(size=12, weight='bold')
    
    # スケールバー作成
    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length_px, 
        label,
        'lower right', 
        pad=0.5,
        color=color,
        frameon=False,
        size_vertical=img_width_px/150, # バーの太さ
        fontproperties=fontprops
    )
    ax.add_artist(scalebar)

def save_results(merged_md, target_name, output_folder="results", add_scalebar=True, unit="um"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 元データをコピー（破壊しないため）
    data = merged_md.map_array.copy()
    if target_name =="youngs_modulus":
        data = np.log10(data)

    filename_base = os.path.join(output_folder, f"{target_name}_merged_map")

    # --- 画像保存の前処理 ---
    
    # 1. NaN（背景）の処理
    # データがある部分だけのマスクを作成
    valid_mask = ~np.isnan(data)
    
    if not np.any(valid_mask):
        print("エラー: 有効なデータがありません。保存をスキップします。")
        return

    # 2. コントラスト調整 (重要!)
    # 最大値・最小値をそのまま使うと、スパイクノイズで真っ暗になるため、
    # 上位・下位 5% をカットした範囲を色の基準にする。
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
            vmin=vmin, 
            vmax=vmax
        )
        print(f"Saved: {filename_base}_high_res.png")
    except Exception as e:
        print(f"imsave failed: {e}")

    # --- B. プロット画像 (閲覧・資料用) ---
    # こちらにスケールバーを追加
    fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_cmap = plt.get_cmap('afmhot').copy()
    plot_cmap.set_bad(color='white') # 背景色
    
    im = ax.imshow(data, cmap=plot_cmap, vmin=vmin, vmax=vmax)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Value [{unit}]" if unit else "Value")

    # タイトル
    ax.set_title("Merged AFM Map")

    # スケールバー追加処理
    if add_scalebar:
        # 文字が見えやすいように色を自動調整（背景が白なら黒、黒なら白）
        # afmhotの下の方は黒っぽいので、白文字が見やすい
        _draw_scalebar(ax, merged_md.x_range, data.shape[1], unit=unit, color='white')

    # 余白削除して保存
    plt.savefig(f"{filename_base}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename_base}_plot.png (Scalebar: {add_scalebar})")

    # --- C. 数値データ ---
    np.savez_compressed(f"{filename_base}.npz", map_data=data)