import os
import math
import csv
import subprocess
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import gradio as gr

from depth_anything_3.api import DepthAnything3


# -------------------------------
# Utility functions
# -------------------------------

def find_images(input_dir: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def resolve_output_roots(input_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Derive output roots following:
    If input_dir like base_dir/images/shooting_date/... then:
      - depth:   base_dir/depth_images/shooting_date/...
      - seg:     base_dir/seg_images/shooting_date/...
      - overlay: base_dir/overlay_images/shooting_date/...
      - report:  base_dir/reports/shooting_date/
    Otherwise fallback to siblings of input parent.
    """
    parts = list(input_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        base_dir = Path(*parts[:idx])
        rel = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        depth_root = base_dir / "depth_images" / rel
        seg_root = base_dir / "seg_images" / rel
        overlay_root = base_dir / "overlay_images" / rel
        report_root = base_dir / "reports" / rel
    else:
        base_dir = input_dir.parent
        rel = input_dir.name
        depth_root = base_dir / "depth_images" / rel
        seg_root = base_dir / "seg_images" / rel
        overlay_root = base_dir / "overlay_images" / rel
        report_root = base_dir / "reports" / rel

    for d in [depth_root, seg_root, overlay_root, report_root]:
        d.mkdir(parents=True, exist_ok=True)

    return depth_root, seg_root, overlay_root, report_root


def device_from_choice(choice: str) -> str:
    if choice == "auto":
        try:
            import torch  # noqa

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return choice


def normalize_input_path(path_str: str) -> Path:
    """
    Normalize user input path for WSL/Windows interoperability.
    - Accepts Windows path like 'H:\\dir\\sub' and converts to '/mnt/h/dir/sub'
    - Uses `wslpath -u` if available, otherwise falls back to manual conversion
    - Expands ~ and environment variables
    """
    s = (path_str or "").strip().strip('"').strip("'")
    if not s:
        return Path("")
    # expand ~ and env vars
    s = os.path.expandvars(os.path.expanduser(s))
    # If already looks like Linux/WSL path
    if s.startswith("/"):
        return Path(s)
    # Try wslpath if available
    try:
        out = subprocess.check_output(["wslpath", "-u", s], stderr=subprocess.STDOUT).decode().strip()
        if out:
            return Path(out)
    except Exception:
        pass
    # Manual Windows path -> /mnt/<drive>/<rest>
    s2 = s.replace("\\", "/")
    if len(s2) >= 2 and s2[1] == ":":
        drive = s2[0].lower()
        rest = s2[2:].lstrip("/").replace(":", "")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s2)


def build_mount_hint(p: Path, original: str) -> str:
    """
    Return helpful hint text when a /mnt/<drive>/... path does not exist on WSL.
    Provides drvfs mount instructions for local and UNC (network) drives.
    """
    try:
        s = str(p)
        if s.startswith("/mnt/") and len(s) >= 6:
            drive = s[5].lower()
            root = Path(f"/mnt/{drive}")
            if not root.exists():
                return (
                    f"\nWSLから {drive.upper()}: ドライブがマウントされていない可能性があります。\n"
                    f"例（ローカル/物理ドライブ）:\n"
                    f"  sudo mkdir -p /mnt/{drive} && sudo mount -t drvfs {drive.upper()}: /mnt/{drive}\n"
                    f"例（ネットワークドライブの場合はUNCで指定）:\n"
                    f"  sudo mkdir -p /mnt/{drive} && sudo mount -t drvfs '\\\\\\\\SERVER\\\\Share' /mnt/{drive}\n"
                    f"  ※ UNCはWindows側で 'net use' や エクスプローラのプロパティで確認してください\n"
                )
    except Exception:
        pass
    return ""


def scan_input_path(input_path_str: str) -> Tuple[str, str, str]:
    """
    Given a user-entered path (file or folder), normalize it and summarize findings.
    Returns (normalized_dir_path_str_for_input, normalized_dir_path_str_display, info_markdown)
    """
    try:
        p = normalize_input_path(input_path_str)
        if str(p) == "":
            return input_path_str, "", "パスが空です。フォルダまたは画像ファイルのパスを入力してください。"
        # If user provided an image file path, switch to its parent directory
        dir_path = p
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            dir_path = p.parent

        if not dir_path.exists():
            hint = build_mount_hint(dir_path, input_path_str)
            return str(dir_path), str(dir_path), f"入力パスを正規化: {p}\nディレクトリが存在しません。{hint}"
        if not dir_path.is_dir():
            return str(dir_path), str(dir_path), f"入力パスを正規化: {p}\nディレクトリではありません。"

        imgs = find_images(dir_path)
        msg = f"入力パスを正規化: {p}\n検索ディレクトリ: {dir_path}\n画像枚数: {len(imgs)}"
        if imgs:
            show = [im.name for im in imgs[:5]]
            msg += "\nサンプル: " + ", ".join(show)
        else:
            msg += "\njpg/png画像が見つかりません。"

        return str(dir_path), str(dir_path), msg
    except Exception as e:
        return input_path_str, "", f"[ERROR] スキャン中に失敗: {e}"


def colormap_code(name: str) -> int:
    name = name.lower()
    if name == "viridis":
        return cv2.COLORMAP_VIRIDIS
    # default to TURBO if not viridis (widely available and visually good)
    return cv2.COLORMAP_TURBO


def depth_to_vis(depth01: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    """
    depth01: float32 [0,1] HxW
    returns BGR uint8 HxWx3
    """
    d8 = np.clip(depth01 * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, colormap_code(cmap))
    return vis


def normalize_depth_per_image(depth: np.ndarray) -> np.ndarray:
    """
    depth: float32 HxW (may contain inf/nan)
    returns depth normalized to [0,1] across valid pixels
    """
    d = depth.astype(np.float32)
    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros_like(d, dtype=np.float32)
    v = d[valid]
    dmin = float(v.min())
    dmax = float(v.max())
    if math.isclose(dmax, dmin):
        return np.zeros_like(d, dtype=np.float32)
    out = np.zeros_like(d, dtype=np.float32)
    out[valid] = (d[valid] - dmin) / (dmax - dmin)
    return out


def binary_mask_from_depth(
    depth01: np.ndarray,
    method: str = "otsu",
    manual_thresh: float = 0.5,
    invert: bool = False,
    min_area: int = 200,
    close_kernel: int = 3,
) -> np.ndarray:
    """
    depth01: [0,1] float32 HxW
    method: 'otsu' or 'manual'
    invert: default False means '浅い=イネ', i.e., smaller depth values become plant(255)
            Otsu path uses THRESH_BINARY_INV accordingly.
    returns uint8 mask 0/255
    """
    valid = np.isfinite(depth01)
    d = depth01.copy()
    d[~valid] = 1.0  # push invalids to far

    d8 = np.clip(d * 255.0, 0, 255).astype(np.uint8)

    if method == "manual":
        thr = int(np.clip(manual_thresh * 255.0, 0, 255))
        if invert:
            # deep=plant
            _, mask = cv2.threshold(d8, thr, 255, cv2.THRESH_BINARY)
        else:
            # shallow=plant
            _, mask = cv2.threshold(d8, thr, 255, cv2.THRESH_BINARY_INV)
    else:
        # Otsu on valid region
        if np.any(valid):
            # Otsu ignores mask, so compute on all but we set invalid to far(255) already
            flag = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY
            _, mask = cv2.threshold(d8, 0, 255, flag | cv2.THRESH_OTSU)
        else:
            mask = np.zeros_like(d8, dtype=np.uint8)

    # Morphological closing
    if close_kernel and close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Remove small components by area
    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        keep = np.zeros(num, dtype=bool)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[i] = True
        pruned = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num):
            if keep[i]:
                pruned[labels == i] = 255
        mask = pruned

    # Ensure invalid pixels are not counted as plant
    mask[~valid] = 0
    return mask


def overlay_plant_on_image(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    bgr: uint8 HxWx3 (OpenCV)
    mask: uint8 0/255 HxW
    returns BGR uint8
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Prepare green overlay
    green = np.zeros_like(bgr, dtype=np.uint8)
    green[:, :, 1] = 255
    blended = cv2.addWeighted(bgr, 1 - alpha, green, alpha, 0)
    # Use broadcast-safe replacement to avoid boolean index shape mismatch
    mask3 = (mask > 0).astype(np.uint8)[..., None]
    out = np.where(mask3 == 1, blended, bgr)
    return out


def compute_coverage(mask: np.ndarray, valid: Optional[np.ndarray] = None) -> Tuple[int, int, float]:
    """
    returns (plant_px, valid_px, coverage in [0..100])
    """
    if valid is None:
        valid = np.ones_like(mask, dtype=bool)
    else:
        valid = valid.astype(bool)
    plant = (mask > 0) & valid
    plant_px = int(plant.sum())
    valid_px = int(valid.sum())
    coverage = (plant_px / valid_px * 100.0) if valid_px > 0 else 0.0
    return plant_px, valid_px, coverage


def ensure_uint8_3ch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


# -------------------------------
# Inference and processing
# -------------------------------

MODEL_REPO_DEFAULTS = {
    "da3-large": "depth-anything/DA3-LARGE",
    "da3metric-large": "depth-anything/DA3METRIC-LARGE",
}


def load_model(model_key: str, model_repo_override: str, device_choice: str) -> DepthAnything3:
    repo = model_repo_override.strip() if model_repo_override.strip() else MODEL_REPO_DEFAULTS.get(
        model_key, MODEL_REPO_DEFAULTS["da3-large"]
    )
    device = device_from_choice(device_choice)
    model = DepthAnything3.from_pretrained(repo).to(device)
    # keep a flag for metric if needed later
    return model


def run_inference_chunked(
    model: DepthAnything3,
    paths: List[Path],
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    batch_size: int = 16,
) -> np.ndarray:
    """
    Returns depth array stacked across all images: (N, H, W), dtype float32
    Note: H, W can vary between chunks depending on processor. We'll keep per-chunk sizes.
    We will return a list of depth maps and align back to original image size later.
    """
    depths: List[np.ndarray] = []
    N = len(paths)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        sub = [str(p) for p in paths[s:e]]
        pred = model.inference(
            image=sub,
            process_res=process_res,
            process_res_method=process_res_method,
            export_dir=None,
            export_format="mini_npz",
        )
        # pred.depth: (n, h, w)
        depths.extend([d.astype(np.float32) for d in pred.depth])
    return depths  # list of (h,w) float32


def process_folder(
    input_dir: str,
    model_key: str,
    model_repo_override: str,
    device_choice: str,
    method: str,
    manual_thresh: float,
    invert: bool,
    min_area: int,
    close_kernel: int,
    cmap: str,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    alpha_overlay: float,
    # metric options
    metric_enable: bool,
    water_percentile: float,
    height_thresh_m: float,
    save_overall_csv: bool,
    debug: bool,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[str, str, List[np.ndarray], float]:
    """
    Returns:
      - log text (markdown)
      - CSV path
      - preview images list
      - overall coverage float
    """
    orig_input = input_dir
    p = normalize_input_path(orig_input)
    # If user passed a file path, use its parent directory
    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        p = p.parent

    details = [f"入力: {orig_input}", f"正規化: {p}"]
    if not p.exists():
        hint = build_mount_hint(p, orig_input)
        return "入力フォルダが見つかりません:\n" + "\n".join(details) + f"\n{hint}", "", [], 0.0
    if not p.is_dir():
        return "入力がディレクトリではありません:\n" + "\n".join(details), "", [], 0.0

    image_paths = find_images(p)
    if not image_paths:
        return "指定フォルダに jpg/png 画像が見つかりません。", "", [], 0.0

    # Resolve outputs
    depth_root, seg_root, overlay_root, report_root = resolve_output_roots(p)
    # We'll save per-image CSV as the main artifact; optional CSV with overall is controlled by flag
    # Logging
    log_lines = []
    log_lines.append(f"入力正規化: {p}")
    log_lines.append(f"画像枚数: {len(image_paths)}")
    log_lines.append(f"出力先: depth={depth_root} seg={seg_root} overlay={overlay_root} report={report_root}")

    # Load model
    log_lines.append(f"モデル読み込み中... [{model_key}]")
    try:
        model = load_model(model_key, model_repo_override, device_choice)
    except Exception as e:
        log_lines.append(f"[ERROR] モデル読み込み失敗: {e}")
        if debug:
            log_lines.append(traceback.format_exc())
        return "\n".join(log_lines), "", [], 0.0
    log_lines.append("モデル読み込み完了。推論開始。")

    # Inference
    try:
        depths = run_inference_chunked(
            model=model,
            paths=image_paths,
            process_res=process_res,
            process_res_method=process_res_method,
            batch_size=max(1, int(batch_size)),
        )
    except Exception as e:
        log_lines.append(f"[ERROR] 推論実行中に失敗: {e}")
        if debug:
            log_lines.append(traceback.format_exc())
        return "\n".join(log_lines), "", [], 0.0

    # Process each image: normalize, threshold, save
    total_plant = 0
    total_valid = 0
    previews: List[np.ndarray] = []
    rows = [("filename", "width", "height", "plant_px", "valid_px", "coverage_percent")]

    for idx, (img_path, depth) in enumerate(progress.tqdm(zip(image_paths, depths), total=len(image_paths))):
        try:
            # Read original image
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                log_lines.append(f"[WARN] 画像読込失敗: {img_path.name}")
                continue
            H, W = bgr.shape[:2]

            # Branch: metric (da3metric-large) or relative
            if metric_enable and model_key == "da3metric-large":
                # Treat network output as metric depth (m) or near-metric; percentile-based water plane
                depth_m = depth.astype(np.float32)
                depth_m_resized = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)
                valid = np.isfinite(depth_m_resized)

                # Estimate water plane by percentile across valid pixels
                if np.any(valid):
                    wp = float(np.percentile(depth_m_resized[valid], np.clip(water_percentile, 0.0, 100.0)))
                else:
                    wp = float(np.percentile(depth_m_resized, np.clip(water_percentile, 0.0, 100.0)))

                # Plant if closer (smaller depth) than (water_plane - height_thresh_m)
                plant = valid & (depth_m_resized <= (wp - float(height_thresh_m)))
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[plant] = 255

                plant_px, valid_px, cov = compute_coverage(mask, valid)
                # For visualization convert metric map to [0,1] on valid
                depth_vis01 = normalize_depth_per_image(np.where(valid, depth_m_resized, np.nan))
            else:
                # Relative depth: per-image normalize to [0,1]
                depth01 = normalize_depth_per_image(depth)
                # Resize depth map to original resolution for saving and mask alignment
                depth01_resized = cv2.resize(depth01, (W, H), interpolation=cv2.INTER_LINEAR)

                # Mask by chosen method
                mask = binary_mask_from_depth(
                    depth01_resized,
                    method=method,
                    manual_thresh=manual_thresh,
                    invert=invert,
                    min_area=min_area,
                    close_kernel=close_kernel,
                )

                valid = np.isfinite(depth01_resized)
                plant_px, valid_px, cov = compute_coverage(mask, valid)
                depth_vis01 = depth01_resized
            total_plant += plant_px
            total_valid += valid_px
            log_lines.append(f"[{idx+1}/{len(image_paths)}] {img_path.name} cov={cov:.4f}% plant={plant_px} valid={valid_px}")

            # Save depth vis
            depth_vis = depth_to_vis(depth_vis01, cmap=cmap)
            depth_out = depth_root / (img_path.stem + "_depth.png")
            cv2.imwrite(str(depth_out), depth_vis)

            # Save mask
            mask_out = seg_root / (img_path.stem + "_plant.png")
            cv2.imwrite(str(mask_out), mask)

            # Save overlay
            overlay = overlay_plant_on_image(bgr, mask, alpha=alpha_overlay)
            overlay_out = overlay_root / (img_path.stem + "_overlay.png")
            cv2.imwrite(str(overlay_out), overlay)

            rows.append((img_path.name, str(W), str(H), str(plant_px), str(valid_px), f"{cov:.4f}"))

            # Collect previews (limit to first few)
            if len(previews) < 9:
                # Stack small grid: original, depth, overlay each resized small
                tile_h = 256
                scale = tile_h / H
                tile_w = max(1, int(W * scale))
                bgr_s = cv2.resize(bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                depth_s = cv2.resize(depth_vis, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                overlay_s = cv2.resize(overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                # concat horizontally
                grid = np.concatenate([bgr_s, depth_s, overlay_s], axis=1)
                previews.append(grid)

        except Exception as e:
            log_lines.append(f"[ERROR] {img_path.name}: {e}")
            if debug:
                log_lines.append(traceback.format_exc())

    # Overall coverage
    overall = (total_plant / total_valid * 100.0) if total_valid > 0 else 0.0

    # Save CSVs
    per_csv_path = report_root / "coverage_per_image.csv"
    try:
        # Per-image only CSV (ヘッダ＋各画像の行のみ)
        with open(per_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        log_lines.append(f"Per-image CSV保存: {per_csv_path}")
    except Exception as e:
        log_lines.append(f"[WARN] Per-image CSV保存に失敗: {e}")
        if debug:
            log_lines.append(traceback.format_exc())

    # Optional: CSV with OVERALL row appended
    if save_overall_csv:
        csv_with_overall = report_root / "coverage.csv"
        try:
            with open(csv_with_overall, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                writer.writerow([])
                writer.writerow(("OVERALL", "", "", str(total_plant), str(total_valid), f"{overall:.4f}"))
            log_lines.append(f"overall行付きCSV保存: {csv_with_overall}")
        except Exception as e:
            log_lines.append(f"[WARN] overall行付きCSV保存に失敗: {e}")
            if debug:
                log_lines.append(traceback.format_exc())

    # Save OVERALL as a simple txt too
    overall_txt = report_root / "overall.txt"
    try:
        with open(overall_txt, "w", encoding="utf-8") as f:
            f.write(f"overall_coverage_percent={overall:.4f}\n")
            f.write(f"plant_px_total={total_plant}\n")
            f.write(f"valid_px_total={total_valid}\n")
        log_lines.append(f"overallテキスト保存: {overall_txt}")
    except Exception as e:
        log_lines.append(f"[WARN] overallテキスト保存に失敗: {e}")
        if debug:
            log_lines.append(traceback.format_exc())

    # Summary
    log_lines.append("")
    log_lines.append(f"処理完了: {len(image_paths)} 枚")
    log_lines.append(f"出力（深度可視化）: {depth_root}")
    log_lines.append(f"出力（分離マスク）: {seg_root}")
    log_lines.append(f"出力（オーバーレイ）: {overlay_root}")
    log_lines.append(f"出力（Per-image CSV）: {per_csv_path}")
    log_lines.append(f"overall 植被率: {overall:.4f}%")

    # Write debug/process log to file under reports
    log_text = "\n".join(log_lines)
    try:
        with open(report_root / "log.txt", "w", encoding="utf-8") as lf:
            lf.write(log_text)
    except Exception:
        if debug:
            log_lines.append(traceback.format_exc())

    return log_text, str(per_csv_path), previews, overall


# -------------------------------
# Gradio UI
# -------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        ver = gr.__version__
        gr.Markdown("## Depth-Anything-3 稲群落・鉛直画像 バッチ推論＆分離ツール")
        gr.Markdown(f"Gradio v{ver} / 状態: 待機中")

        with gr.Row():
            with gr.Column(scale=1):
                input_dir = gr.Textbox(
                    label="画像フォルダパス（jpg/pngのみ）",
                    placeholder="例: C:\\data\\base_dir\\images\\shooting_date",
                )
                normalized_out = gr.Textbox(
                    label="正規化後パス（処理対象ディレクトリ）",
                    value="",
                    interactive=False,
                )
                scan_btn = gr.Button("スキャン")

                # Model settings
                model_key = gr.Dropdown(
                    label="モデル（将来拡張対応）",
                    choices=["da3-large", "da3metric-large"],
                    value="da3-large",
                    info="現在は da3-large（相対深度）を使用。将来 metric に対応予定。",
                )
                model_repo = gr.Textbox(
                    label="Model Repo (HuggingFace) 上書き（空なら既定を使用）",
                    value="depth-anything/DA3-LARGE",
                )
                device_choice = gr.Dropdown(
                    label="デバイス",
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                )

                # Inference settings
                process_res = gr.Slider(
                    label="処理解像度（process_res）",
                    minimum=256,
                    maximum=1024,
                    step=8,
                    value=504,
                )
                process_res_method = gr.Dropdown(
                    label="処理解像度メソッド",
                    choices=["upper_bound_resize", "low_res", "high_res"],
                    value="upper_bound_resize",
                )
                batch_size = gr.Slider(
                    label="バッチサイズ（VRAMに合わせて調整）",
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=8,
                )

                # Segmentation settings
                method = gr.Radio(
                    label="分離手法",
                    choices=["otsu", "manual"],
                    value="otsu",
                )
                manual_thresh = gr.Slider(
                    label="手動しきい値（0..1）",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                )
                invert = gr.Checkbox(label="反転（深い=イネ）", value=False)
                min_area = gr.Slider(
                    label="最小領域ピクセル（小領域除去）",
                    minimum=0,
                    maximum=5000,
                    step=50,
                    value=200,
                )
                close_kernel = gr.Slider(
                    label="クロージングカーネル（0で無効）",
                    minimum=0,
                    maximum=15,
                    step=1,
                    value=3,
                )
                cmap = gr.Dropdown(
                    label="深度カラーマップ",
                    choices=["viridis", "turbo"],
                    value="viridis",
                )
                alpha_overlay = gr.Slider(
                    label="オーバーレイ透過率",
                    minimum=0.1,
                    maximum=0.9,
                    step=0.05,
                    value=0.5,
                )

                # Metric options (for da3metric-large)
                metric_enable = gr.Checkbox(label="メトリック高さしきい分離（da3metric用）", value=False)
                water_percentile = gr.Slider(
                    label="水面推定（深度パーセンタイル%）",
                    minimum=50,
                    maximum=100,
                    step=1,
                    value=90,
                )
                height_thresh_m = gr.Slider(
                    label="高さしきい[m]（水面よりどれだけ浅ければイネとみなすか）",
                    minimum=0.0,
                    maximum=0.3,
                    step=0.005,
                    value=0.05,
                )

                save_overall_csv_chk = gr.Checkbox(label="overall行付きCSVも保存する（coverage.csv）", value=False)
                debug_chk = gr.Checkbox(label="デバッグログ（例外詳細）", value=False)

                run_btn = gr.Button("実行", variant="primary")

            with gr.Column(scale=1):
                log = gr.Markdown("ここに処理ログが表示されます。")
                overall = gr.Number(label="overall 植被率 [%]", value=0.0, interactive=False)
                csv_out = gr.Textbox(label="CSV出力パス", value="", interactive=False)
                gallery = gr.Gallery(label="プレビュー（元/深度/オーバーレイ）", columns=1, height=700)

        # UI dynamics
        def toggle_manual(method_choice):
            return gr.update(visible=(method_choice == "manual"))

        method.change(toggle_manual, inputs=method, outputs=manual_thresh)

        # When input path changes or submitted, normalize for WSL/Windows and preview scan results
        input_dir.change(
            fn=scan_input_path,
            inputs=input_dir,
            outputs=[input_dir, normalized_out, log],
        )
        input_dir.submit(
            fn=scan_input_path,
            inputs=input_dir,
            outputs=[input_dir, normalized_out, log],
        )
        scan_btn.click(
            fn=scan_input_path,
            inputs=input_dir,
            outputs=[input_dir, normalized_out, log],
        )

        # Bind run
        run_btn.click(
            fn=process_folder,
            inputs=[
                input_dir,
                model_key,
                model_repo,
                device_choice,
                method,
                manual_thresh,
                invert,
                min_area,
                close_kernel,
                cmap,
                process_res,
                process_res_method,
                batch_size,
                alpha_overlay,
                metric_enable,
                water_percentile,
                height_thresh_m,
                save_overall_csv_chk,
                debug_chk,
            ],
            outputs=[log, csv_out, gallery, overall],
        )

    return demo


def main():
    # Allow environment override for Gradio server if needed
    host = os.environ.get("DA3_RICE_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("DA3_RICE_APP_PORT", "7861"))
    demo = build_ui()
    demo.queue(max_size=20).launch(server_name=host, server_port=port, show_error=True, ssr_mode=False)


if __name__ == "__main__":
    main()
