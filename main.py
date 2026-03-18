from import_files import import_files
from marged_maps import merged_maps
from save_results import save_results
import os 

from datetime import datetime


def process_afm_folder(folder_path, target_name, date_str=None):
    # 連結するファイルとモーター移動履歴を取得
    dataset = import_files(folder_path, target_name)

    # 連結する
    merged_data = merged_maps(dataset)

    # 連結結果を保存する
    # 保存フォルダ名を日付などから自動生成
    parent_folder = os.path.dirname(folder_path)
    save_folder = os.path.join(parent_folder, f"result_{target_name}_{date_str}")
    os.makedirs(save_folder, exist_ok=True)
    save_results(merged_data, target_name, output_folder=save_folder)

if __name__ == "__main__":
    # --- 設定項目 ---
    base_folder = r"C:\nojima\AFM6measurement\260226\マッピング計測＋α"
    targets = ["youngs_modulus", "topography"]  # "topography", "youngs_modulus"など
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    for target in targets:
        process_afm_folder(base_folder, target, date_str)