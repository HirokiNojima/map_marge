from map_data import map_data
import os
import numpy as np

def import_files(folder_path, target_name):
    """
    指定された親フォルダから、指定した種類のマップデータを抜き出し、一つの大きなデータセットに連結する。
    また、モーター位置も取得し、各map_dataオブジェクトに格納する。
    Args:
        folder_path (str): 親フォルダのパス
        target_name (str): 連結するファイルの名前(例：youngs_modulus, topography)

    Returns:
        dataset (dict): map_dataオブジェクトを格納した辞書。キーはサブフォルダ名。
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")
    
    # データを格納する辞書を初期化
    dataset = {}

    # モーター移動履歴ファイルを読み込み
    motor_history = np.loadtxt(os.path.join(folder_path, "motor_position.csv"), delimiter=",", skiprows=1)
    motor_pos = np.zeros((1, 2))  # モーター位置の初期化(x, y)

    # フォルダ内の各サブフォルダを走査
    data_count = 0
    if len(os.listdir(folder_path))-1 != motor_history.shape[0]: # -1はmotor_position.csv分
        print(len(os.listdir(folder_path)))
        print(motor_history.shape[0])
        raise ValueError("The number of subfolders does not match the number of motor history entries.")

    for item in os.listdir(folder_path):
        child_folder_path = os.path.join(folder_path, item)

        # フォルダでない場合はスキップ
        if not os.path.isdir(child_folder_path):
            continue

        # 対象ファイルのパス
        source_file = os.path.join(child_folder_path, "AFM_Analysis_Results", target_name + "_map.npz")

        # ファイルを読み込み
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"The file {source_file} does not exist.")
        data = np.load(source_file)

        # モーター位置更新
        motor_pos = motor_pos + motor_history[data_count, :]
        data_count += 1

        # map_dataオブジェクトを作成し、データを格納
        md = map_data()
        md.file_name = item
        md.map_array = data["map_data"]
        md.x_range =  data["x_max"] - data["x_min"]
        md.y_range = data["y_max"] - data["y_min"]
        md.x_motor = motor_pos[0, 0]
        md.y_motor = motor_pos[0, 1]
        md.target_name = target_name
        # 辞書に登録
        dataset[item] = md
    return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    forlder_path = r"C:\nojima\AFM6measurement\260115_骨広範囲計測\計測フォルダ"
    target_name = "topography"

    dataset = import_files(forlder_path, target_name)

    print("データセットのキー一覧:", dataset.keys())

    data = dataset["1738_0"].map_array
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.show()


