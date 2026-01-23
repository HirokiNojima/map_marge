import numpy as np

class map_data:
    """
    連結に用いるそれぞれのAFMマップのデータを格納するクラス。

    """
    def __init__(self):
        self.file_name = ""          # ファイル名(計測結果が保存されているファイルの名前)
        self.target_name = ""        # データの種類(例: youngs_modulus, topography)
        self.map_array = None       # マップデータの配列
        self.x_range = np.nan         # x方向の範囲
        self.y_range = np.nan         # y方向の範囲
        self.x_motor = np.nan         # x方向のモーター位置
        self.y_motor = np.nan         # y方向のモーター位置

                 