from import_files import import_files
from marged_maps import merged_maps
from save_results import save_results
import os 

def main():
    folder_path = r"C:\nojima\AFM6measurement\260222\画像連結"
    target_name = "youngs_modulus"  # "topography", "youngs_modulus"など

    # 連結するファイルとモーター移動履歴を取得
    dataset = import_files(folder_path, target_name)

    # 連結する
    merged_data = merged_maps(dataset)

    # 連結結果を保存する
    save_folder = os.path.join(folder_path, "results")
    os.makedirs(save_folder, exist_ok=True)
    save_results(merged_data, target_name, output_folder=save_folder)

if __name__ == "__main__":
    main()