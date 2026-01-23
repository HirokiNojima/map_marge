from import_files import import_files
from marged_maps import marged_maps
import matplotlib.pyplot as plt

def main(folder_path, target_name):
    # 連結するファイルとモーター移動履歴を取得
    dataset = import_files(folder_path, target_name)
    # 連結する
    marged_data = marged_maps(dataset)
    # 連結結果を保存する
    plt.imshow(marged_data.map_array, cmap='gray')
    plt.colorbar()
    plt.title("Merged AFM Map")
    plt.show()
if __name__ == "__main__":
    folder_path = r"C:\nojima\AFM6measurement\260115_骨広範囲計測\計測フォルダ"
    target_name = "topography"
    main(folder_path, target_name)