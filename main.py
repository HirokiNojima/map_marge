from import_files import import_files
from marged_maps import merged_maps
from save_results import save_results

def main():
    folder_path = r"C:\nojima\AFM6measurement\260115_骨広範囲計測\計測フォルダ"
    target_name = "topography"  # "topography", "youngs_modulus"など

    # 連結するファイルとモーター移動履歴を取得
    dataset = import_files(folder_path, target_name)

    # 連結する
    merged_data = merged_maps(dataset)

    # 連結結果を保存する
    save_results(merged_data, target_name, output_folder="results")

if __name__ == "__main__":
    main()