"""
=============================================================
03_evaluate.py - Đánh giá mô hình YOLOv8
=============================================================
Script này sẽ:
  1. Load best model đã train
  2. Chạy validation trên tập test/val
  3. Tính Precision, Recall, F1, mAP
  4. Vẽ Confusion Matrix
  5. Vẽ Precision-Recall Curve
  6. Xuất báo cáo tổng hợp
=============================================================
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
from tqdm import tqdm


# ============================================================
# CẤU HÌNH
# ============================================================
DEFAULT_MODEL = "./runs/train/fall_detection_yolov8/weights/best.pt"
DATASET_YAML  = "../data/fall_dataset.yaml"
OUTPUT_DIR    = "../outputs/evaluation"

CLASS_NAMES   = ["fall", "walking", "sitting", "standing", "lying", "bending"]
CLASS_COLORS  = ["#DC3232", "#32C832", "#3264DC", "#C8C832", "#C832C8", "#32C8C8"]

# Ngưỡng confidence để tính metrics
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45


# ============================================================
# HÀM ĐÁNH GIÁ
# ============================================================
def load_model(model_path: str) -> YOLO:
    """Load mô hình YOLOv8 đã train."""
    print("=" * 60)
    print("📦 LOAD MODEL")
    print("=" * 60)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    model = YOLO(model_path)
    print(f"  ✅ Đã load: {model_path}")
    return model


def run_official_validation(model: YOLO, data_yaml: str) -> dict:
    """Chạy validation chính thức bằng Ultralytics."""
    print("\n" + "=" * 60)
    print("🔍 CHẠY VALIDATION (OFFICIAL)")
    print("=" * 60)

    metrics = model.val(
        data   = data_yaml,
        imgsz  = 640,
        conf   = CONF_THRESHOLD,
        iou    = IOU_THRESHOLD,
        plots  = True,
        save_json = False,
        verbose   = True,
    )

    print("\n📊 KẾT QUẢ:")
    print(f"  mAP@0.5          : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95     : {metrics.box.map:.4f}")
    print(f"  Precision (mean) : {metrics.box.mp:.4f}")
    print(f"  Recall    (mean) : {metrics.box.mr:.4f}")

    # Per-class metrics
    print("\n📋 Per-class metrics:")
    print(f"  {'Class':<20} {'P':>8} {'R':>8} {'AP50':>8}")
    print(f"  {'-'*44}")
    for i, name in enumerate(CLASS_NAMES):
        try:
            p   = metrics.box.p[i] if i < len(metrics.box.p) else 0
            r   = metrics.box.r[i] if i < len(metrics.box.r) else 0
            ap  = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            print(f"  {name:<20} {p:>8.4f} {r:>8.4f} {ap:>8.4f}")
        except Exception:
            pass

    return metrics


def collect_predictions(model: YOLO, dataset_root: str, split: str = "val") -> tuple:
    """
    Thu thập tất cả prediction và ground truth từ tập val.
    Dùng để vẽ confusion matrix chi tiết.

    Returns:
        (y_true, y_pred) - lists of class indices (per bounding box)
    """
    print("\n" + "=" * 60)
    print(f"📷 THU THẬP PREDICTION ({split.upper()})")
    print("=" * 60)

    img_dir = Path(dataset_root) / "images" / split
    lbl_dir = Path(dataset_root) / "labels" / split

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in img_exts])

    y_true, y_pred = [], []
    n_no_pred = 0

    for img_path in tqdm(img_files, desc="  Predict", ncols=70):
        # Ground truth
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_classes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))

        if not gt_classes:
            continue

        # Prediction
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        pred_classes = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()

        # Match GT ↔ Pred (đơn giản: so sánh số lượng & thứ tự)
        # Dùng majority class matching cho mỗi ảnh
        for gt_cls in gt_classes:
            y_true.append(gt_cls)
            # Tìm pred nào gần nhất với gt_cls
            if pred_classes:
                # Ưu tiên pred cùng class với GT
                if gt_cls in pred_classes:
                    y_pred.append(gt_cls)
                    pred_classes.remove(gt_cls)
                else:
                    y_pred.append(pred_classes.pop(0))
            else:
                # Không có prediction nào
                y_pred.append(-1)  # Miss
                n_no_pred += 1

    print(f"  Tổng GT boxes  : {len(y_true)}")
    print(f"  Không detect   : {n_no_pred}")

    # Loại bỏ miss (-1) để tính confusion matrix sạch
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    y_true_clean = [t for t, p in valid]
    y_pred_clean = [p for t, p in valid]

    return y_true_clean, y_pred_clean, n_no_pred


def plot_confusion_matrix(y_true: list, y_pred: list, save_dir: str):
    """Vẽ Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix - Fall Detection", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Giá trị tuyệt đối", "Tỷ lệ phần trăm (Normalized)"]
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Nhãn thực tế (Ground Truth)", fontweight="bold")
        ax.set_xlabel("Nhãn dự đoán (Predicted)",    fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix → {save_path}")


def print_classification_report(y_true: list, y_pred: list, save_dir: str):
    """In và lưu Classification Report."""
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Lưu ra file
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - FALL DETECTION\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  ✅ Report → {report_path}")


def plot_per_class_metrics(y_true: list, y_pred: list, save_dir: str):
    """Vẽ biểu đồ Precision, Recall, F1 per class."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    precisions = precision_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    recalls    = recall_score   (y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    f1s        = f1_score       (y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_p = ax.bar(x - width,   precisions, width, label="Precision", color="#3498DB", alpha=0.85)
    bars_r = ax.bar(x,           recalls,    width, label="Recall",    color="#2ECC71", alpha=0.85)
    bars_f = ax.bar(x + width,   f1s,        width, label="F1-Score",  color="#E74C3C", alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Precision / Recall / F1 Per Class", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Per-class metrics → {save_path}")


def visualize_predictions(model: YOLO, dataset_root: str, n: int = 9, save_dir: str = "./outputs/evaluation"):
    """Visualize kết quả predict trên ảnh val."""
    img_dir  = Path(dataset_root) / "images" / "val"
    lbl_dir  = Path(dataset_root) / "labels" / "val"
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in img_exts])

    np.random.seed(0)
    selected = np.random.choice(img_files, min(n, len(img_files)), replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Kết quả Predict trên tập Val\n🟥 Fall  🟩 Walking  🟦 Sitting", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    COLOR_MAP = {0: (220, 50, 50), 1: (50, 200, 50), 2: (50, 100, 220)}

    for idx, img_path in enumerate(selected):
        ax = axes[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.axis("off")
            continue

        img_draw = img_bgr.copy()
        h, w = img_draw.shape[:2]

        # Vẽ GT (đường đứt nét)
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
                    color = COLOR_MAP.get(cls_id, (200, 200, 200))
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 1)  # GT: nét mỏng

        # Vẽ Prediction (nét đậm)
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id  = int(box.cls.item())
                conf    = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color   = COLOR_MAP.get(cls_id, (200, 200, 200))
                label   = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img_draw, (x1, y1 - 18), (x1 + len(label)*8, y1), color, -1)
                cv2.putText(img_draw, label, (x1+2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        ax.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
        ax.set_title(img_path.name, fontsize=7)
        ax.axis("off")

    for i in range(len(selected), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "val_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Val predictions → {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá model YOLOv8 Fall Detection")
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL, help="Đường dẫn model .pt")
    parser.add_argument("--data",    type=str, default="../data/fall_dataset",  help="Thư mục dataset root")
    parser.add_argument("--split",   type=str, default="val",          help="Split để evaluate: val hoặc test")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n🚀 ĐÁNH GIÁ MÔ HÌNH - FALL DETECTION")

    # 1. Load model
    model = load_model(args.model)

    # 2. Official validation metrics
    run_official_validation(model, DATASET_YAML)

    # 3. Thu thập predictions
    y_true, y_pred, n_miss = collect_predictions(model, args.data, split=args.split)

    print("\n" + "=" * 60)
    print("📊 TÍNH METRICS & VẼ BIỂU ĐỒ")
    print("=" * 60)

    # 4. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR)

    # 5. Classification Report
    print_classification_report(y_true, y_pred, OUTPUT_DIR)

    # 6. Per-class metrics
    plot_per_class_metrics(y_true, y_pred, OUTPUT_DIR)

    # 7. Visualize predictions
    visualize_predictions(model, args.data, n=9, save_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("✅ ĐÁNH GIÁ HOÀN TẤT!")
    print(f"   Kết quả lưu tại: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
