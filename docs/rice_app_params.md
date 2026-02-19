# 稲群落バッチ推論＆分離ツール パラメータ解説（推奨値付き）

本ドキュメントは `python run_rice_app.py` で起動する Gradio GUI の各パラメータの意味と推奨値、出力ファイルの場所を説明します。

## 1) 入力とWSLパス正規化

- 画像フォルダパス
  - Windowsパスでも可（例: `C:\Users\...\20250621`）。WSL上では自動的に `/mnt/...` に正規化します。
  - 画像ファイル（例: `...\20250621_Flight1_7m_0201_b.png`）を貼り付けた場合は自動的に親フォルダに切り替えてスキャンします。
  - フォルダ内の拡張子 `.jpg/.jpeg/.png` を一括処理します。

- スキャン結果（入力欄からフォーカスが外れると自動表示）
  - 正規化後パス、画像枚数、先頭5件のサンプルファイル名をログに表示します。

- 注意（WSLのドライブ）
  - `H:\...` などがWSLに未マウントだと `/mnt/h/...` が存在せず読み込めません。
  - 必要なら手動マウント例（ローカルドライブ）:
    ```
    sudo mkdir -p /mnt/h
    sudo mount -t drvfs H: /mnt/h
    ```
  - ネットワークドライブなら UNC を指定:
    ```
    sudo mount -t drvfs '\\\\SERVER\\Share' /mnt/h
    ```

## 2) モデル設定

- モデル（model_key）
  - `da3-large`（相対深度, 既定）/ 将来用に `da3metric-large` も選択可（現状は相対深度で処理）。
- Model Repo 上書き（model_repo）
  - 既定: `depth-anything/DA3-LARGE`。ネットワーク/ミラー/ローカルパスに合わせて変更可。
- デバイス（device_choice）
  - `auto`（推奨）/`cuda`/`cpu`。`auto`はCUDA利用可ならGPUを用います。

## 3) 推論設定

- process_res（処理解像度）
  - 既定: 504。精細さを上げたい場合は 720〜1024 なども可。VRAM/速度に応じて調整。
- process_res_method（処理解像度メソッド）
  - `upper_bound_resize`（既定）/`low_res`/`high_res`。
  - 高精細優先なら `high_res`、速度/VRAM節約なら `low_res` も検討。
- バッチサイズ（batch_size）
  - 既定: 8。GPUでは2〜8程度、CPUでは1〜2程度が目安。
  - OOM（メモリ不足）が出る場合は小さくしてください。

## 4) 分離（2値化）設定

- 分離手法（method）
  - `otsu`（既定）: 画像毎に自動しきい値で2値化。
  - `manual`: スライダー（0..1）で手動しきい値を指定。
- 手動しきい値（manual_thresh）
  - 既定: 0.5。例として 0.3 / 0.7 を試し、結果が過小/過大なら調整。
- 反転（invert）
  - 既定: OFF（浅い=イネ）。撮影・深度の向きによっては ON（深い=イネ）が適切な場合あり。
- 最小領域ピクセル（min_area）
  - 既定: 200。小さなノイズ領域の除去に有効。ノイズが多いときは 500〜2000 などへ増加。
- クロージングカーネル（close_kernel）
  - 既定: 3。穴埋めや小さなギャップの閉鎖。より滑らかにしたいときは 5〜9 を検討。0で無効。
- 深度カラーマップ（cmap）
  - `viridis` / `turbo`。可視化用であり2値化結果には影響しません。
- オーバーレイ透過率（alpha_overlay）
  - 既定: 0.5。イネ領域を緑で重ねる強さ。

## 5) CSV 出力（植被率）

- 本ツールは画像ごとの植被率を CSV に保存します（ユーザー要望に合わせ per-image を主とします）。
- per-image CSV（既定で保存）
  - パス: `base_dir/reports/shooting_date/coverage_per_image.csv`
  - 列: `filename,width,height,plant_px,valid_px,coverage_percent`
- overall 行付き CSV（オプション）
  - チェックON時に `coverage.csv` を保存し、上記 per-image の下に `OVERALL` 行を追記。
  - パス: `base_dir/reports/shooting_date/coverage.csv`
- overall テキスト（参考）
  - `base_dir/reports/shooting_date/overall.txt` に overall 値と合計画素数を保存。

## 6) ログ出力

- 実行ログ（GUIログ）に各画像の推定と植被率が `[i/N] filename cov=xx.xxxx% ...` 形式で出ます。
- 同内容をファイルにも保存:
  - パス: `base_dir/reports/shooting_date/log.txt`
- 「デバッグログ（例外詳細）」をONにすると例外のトレースバックもログに追記されます。

## 7) 保存先の規約

入力フォルダが `base_dir/images/shooting_date/...` なら、出力は以下に保存します（同様の構造を自動生成）:

- 深度可視化PNG: `base_dir/depth_images/shooting_date/xxx_depth.png`
- 2値マスクPNG: `base_dir/seg_images/shooting_date/xxx_plant.png`
- オーバーレイPNG: `base_dir/overlay_images/shooting_date/xxx_overlay.png`
- CSV/ログ: `base_dir/reports/shooting_date/...`

上記パターンに合致しない場合でも、入力フォルダの親ディレクトリ配下に兄弟ディレクトリとして同様の構造を作成します。

## 8) チューニングのコツ

- まずは `otsu` + `invert=OFF`（浅い=イネ）で実行。
- イネ領域が少なすぎる/多すぎると感じたら:
  - `invert` を切り替えて再実行、または
  - `manual` に切り替えて しきい値 0.3 / 0.7 を試す。
- ノイズが多い場合は `min_area` を増やし、境界がギザギザなら `close_kernel` を大きく。

## 9) 実行の流れ（再掲）

1. GUI起動: `python run_rice_app.py`
2. 「画像フォルダパス」に Windows 形式で貼付け（ファイルでも可）。GUIが自動でWSLに正規化してスキャン。
3. 分離設定を調整（必要なら invert/manual/min_area/close_kernel）。
4. 実行後、保存先（depth_images/seg_images/overlay_images/reports）と `coverage_per_image.csv` を確認。
