from import_files import import_files
from marged_maps import marged_maps
from save_results import save_results
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    folder_path = r"C:\nojima\AFM6measurement\260115_骨広範囲計測\計測フォルダ"
    target_name = "topography"  # "topography", "youngs_modulus"など
    vmin, vmax = 5, 95

    # 連結するファイルとモーター移動履歴を取得
    dataset = import_files(folder_path, target_name)

    # 連結する
    marged_data = marged_maps(dataset)

    # 連結結果を保存する
    save_results(marged_data, target_name, output_folder="results")
if __name__ == "__main__":
    main()