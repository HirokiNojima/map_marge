# AFMマップ連結プログラム

AFM（原子間力顕微鏡）で測定した複数のマップデータを、モーター移動履歴を基に連結し、広範囲のマップを生成するプログラムです。

## 概要

このプログラムは、以下の機能を提供します：

1. **データインポート**: 複数のサブフォルダに保存されたAFMマップデータを読み込み
2. **モーター位置管理**: CSV形式のモーター移動履歴を読み込み、各マップの絶対位置を計算
3. **マップ連結**: 重なり合う部分をスムーズにブレンドしながら、複数のマップを一つの大きなマップに統合
4. **データ保存**: 連結結果を画像およびNumPy形式で保存

## ファイル構成

- **import_files.py**: データインポートとモーター位置管理
- **map_data.py**: マップデータを格納するクラス定義
- **marged_maps.py**: マップ連結処理（重み付きブレンディング、平坦化処理）
- **save_results.py**: 結果の保存（画像出力、スケールバー付与）
- **main.py**: メインプログラム
- **results/**: 結果出力フォルダ

## 使用方法

### データ準備

以下の構造でデータを配置してください：

```
親フォルダ/
├── motor_position.csv          # モーター移動履歴（x, y の2列）
├── サブフォルダ1/
│   └── AFM_Analysis_Results/
│       └── {target_name}_map.npz
├── サブフォルダ2/
│   └── AFM_Analysis_Results/
│       └── {target_name}_map.npz
...
```

**motor_position.csv** の形式：
```csv
x,y
100.5,200.3
-50.2,100.1
...
```

**npzファイル**には以下のキーが必要です：
- `map_data`: マップデータ配列
- `x_min`, `x_max`: X軸の範囲
- `y_min`, `y_max`: Y軸の範囲


### パラメータ設定

#### `import_files(folder_path, target_name)`
- `folder_path`: データが保存されている親フォルダのパス
- `target_name`: 連結するマップの種類（例: "topography", "youngs_modulus"）

#### `merge_maps(dataset, edge_falloff, flatten_method)`
- `edge_falloff`: マップの端のブレンド幅（0.0～0.5、推奨: 0.1）
- `flatten_method`: 平坦化方法
  - `'poly'`: 2次曲面補正（推奨）
  - `'robust'`: 外れ値を無視した平面補正
  - `'gaussian'`: ガウシアンフィルタによる背景除去

#### `save_results(merged_md, target_name, output_folder, add_scalebar, unit)`
- `output_folder`: 結果を保存するフォルダ
- `add_scalebar`: スケールバーを追加するか（True/False）
- `unit`: スケールバーの単位（デフォルト: "um"）

## データクラス: map_data

各AFMマップのデータを格納するクラスです。

### 属性
- `file_name`: ファイル名
- `map_array`: マップデータの配列（2D NumPy配列）
- `x_range`: X方向の範囲（物理単位）
- `y_range`: Y方向の範囲（物理単位）
- `x_motor`: X方向のモーター絶対位置
- `y_motor`: Y方向のモーター絶対位置
- `target_name`: マップの種類（topography, youngs_modulusなど）

## 出力結果

結果は `results/` フォルダに保存されます：

- `{target_name}_merged_map.png`: 連結されたマップの画像（スケールバー付き）
- `{target_name}_merged_map.npz`: NumPy形式のデータ
  - `map_data`: 連結されたマップ配列
  - `x_range`, `y_range`: マップの物理サイズ
  - `x_motor`, `y_motor`: 基準モーター位置

## 注意事項

- サブフォルダの数とmotor_position.csvの行数は一致している必要があります
- モーター位置は累積加算されて絶対位置が計算されます
- マップの重なり部分は重み付きブレンディングでスムーズに接続されます
- 平坦化処理により、スキャナーの歪みや試料の傾きを補正できます

## ライセンス

研究用途に限定してご使用ください。
