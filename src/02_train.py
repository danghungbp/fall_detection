"""
=============================================================
02_train.py - Train mô hình YOLOv8 phát hiện té ngã
=============================================================
Script này sẽ:
  1. Load pretrained YOLOv8 (transfer learning)
  2. Fine-tune trên Fall Detection Dataset
  3. Lưu best model & last model
  4. Visualize training curves
=============================================================
"""

import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


# ============================================================
# CẤU HÌNH TRAIN
# ============================================================
CONFIG = {
    # --- Mô hình ---
    "model"       : str(Path(__file__).resolve().parent / "runs" / "train" / "fall_detection_yolov8" / "weights" / "best.pt"),    # Fine-tune trên model cũ
                                      # n=nhẹ nhất, x=nặng nhất & chính xác nhất
    # --- Dataset ---
    "data"        : str(Path(__file__).resolve().parent.parent / "data" / "fall_dataset.yaml"),

    # --- Hyperparameters ---
    "epochs"      : 30,              # Số epoch train (chỉ cần 30 cho nhanh)
    "imgsz"       : 640,             # Kích thước ảnh input (640 chuẩn YOLO)
    "batch"       : 16,              # Batch size (giảm nếu OOM: 8 hoặc 4)
    "patience"    : 20,              # Early stopping: dừng nếu không cải thiện sau N epoch
    "lr0"         : 0.01,            # Learning rate ban đầu
    "lrf"         : 0.01,            # Learning rate cuối = lr0 * lrf
    "momentum"    : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,              # Số epoch warmup

    # --- Augmentation ---
    "hsv_h"       : 0.015,           # Augment HSV-Hue
    "hsv_s"       : 0.7,             # Augment HSV-Saturation
    "hsv_v"       : 0.4,             # Augment HSV-Value
    "degrees"     : 5.0,             # Xoay ảnh ±5 độ
    "translate"   : 0.1,             # Dịch ảnh ±10%
    "scale"       : 0.5,             # Scale ảnh ±50%
    "flipud"      : 0.0,             # Lật dọc (không dùng vì té ngã cần hướng)
    "fliplr"      : 0.5,             # Lật ngang 50%
    "mosaic"      : 1.0,             # Mosaic augmentation
    "mixup"       : 0.1,             # MixUp augmentation

    # --- Output ---
    "project"     : "./runs/train",
    "name"        : "fall_detection_yolov8",
    "save_period" : 10,              # Lưu checkpoint mỗi N epoch
    "exist_ok"    : True,            # Cho phép ghi đè kết quả cũ

    # --- Device ---
    "device"      : "0",             # "0" = GPU đầu tiên, "cpu" = CPU
    "workers"     : 8,               # Số worker DataLoader
}


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def check_gpu():
    """Kiểm tra GPU có sẵn sàng không."""
    print("=" * 60)
    print("🖥️  KIỂM TRA PHẦN CỨNG")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✅ GPU: {gpu_name}")
        print(f"  ✅ VRAM: {vram:.1f} GB")

        # Gợi ý batch size dựa trên VRAM
        if vram < 4:
            print(f"  ⚠️  VRAM thấp! Khuyến nghị batch=4")
        elif vram < 8:
            print(f"  💡 Khuyến nghị batch=8~16")
        else:
            print(f"  💡 Có thể dùng batch=16~32")
    else:
        print("  ⚠️  Không tìm thấy GPU! Chuyển sang CPU (chậm hơn)")
        CONFIG["device"] = "cpu"
        CONFIG["batch"] = 4
        CONFIG["workers"] = 2
    return torch.cuda.is_available()


def print_config(config: dict):
    """In cấu hình train ra màn hình."""
    print("\n" + "=" * 60)
    print("⚙️  CẤU HÌNH TRAIN")
    print("=" * 60)
    key_params = ["model", "data", "epochs", "imgsz", "batch", "lr0", "patience", "device"]
    for k in key_params:
        print(f"  {k:<15}: {config[k]}")


def plot_training_results(results_dir: str):
    """Đọc file results.csv và vẽ training curves."""
    results_csv = Path(results_dir) / "results.csv"
    if not results_csv.exists():
        print(f"  ⚠️  Không tìm thấy {results_csv}")
        return

    import pandas as pd
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("YOLOv8 Training Results - Fall Detection", fontsize=14, fontweight="bold")

    # Metrics cần vẽ
    plot_items = [
        ("train/box_loss",   "Train Box Loss",     axes[0, 0], "#E74C3C"),
        ("train/cls_loss",   "Train Class Loss",   axes[0, 1], "#E74C3C"),
        ("train/dfl_loss",   "Train DFL Loss",     axes[0, 2], "#E74C3C"),
        ("val/box_loss",     "Val Box Loss",       axes[1, 0], "#3498DB"),
        ("metrics/mAP50(B)", "mAP@0.5",            axes[1, 1], "#2ECC71"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95",   axes[1, 2], "#9B59B6"),
    ]

    for col, label, ax, color in plot_items:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
        else:
            ax.text(0.5, 0.5, f"Không có dữ liệu\n'{col}'",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout()
    save_path = Path(results_dir) / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Training curves → {save_path}")


# ============================================================
# MAIN TRAIN
# ============================================================
def train(args=None):
    print("\n🚀 BẮT ĐẦU TRAIN - FALL DETECTION YOLOv8")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Override config từ command line nếu có
    if args:
        if args.model   : CONFIG["model"]   = args.model
        if args.epochs  : CONFIG["epochs"]  = args.epochs
        if args.batch   : CONFIG["batch"]   = args.batch
        if args.imgsz   : CONFIG["imgsz"]   = args.imgsz
        if args.device  : CONFIG["device"]  = args.device

    # Kiểm tra GPU
    check_gpu()
    print_config(CONFIG)

    # Load model hoặc tiếp tục train từ checkpoint
    print("\n" + "=" * 60)
    if args and getattr(args, "resume", False):
        last_ckpt = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"❌ Không tìm thấy file checkpoint: {last_ckpt}")
            print("💡 Hãy chạy lại lệnh không cần cờ --resume để bắt đầu train từ đầu.")
            return
        print("🔄 TIẾP TỤC TRAIN TỪ CHECKPOINT (RESUME)")
        print("=" * 60)
        print(f"  Check point : {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        print("📦 LOAD MODEL")
        print("=" * 60)
        print(f"  Model: {CONFIG['model']} (pretrained trên COCO)")
        model = YOLO(CONFIG["model"])
        print("  ✅ Load thành công!")

        # Bắt đầu train từ đầu
        print("\n" + "=" * 60)
        print("🏋️  TRAINING...")
        print("=" * 60)
        print("  Theo dõi progress bên dưới hoặc mở TensorBoard:")
        print(f"  $ tensorboard --logdir {CONFIG['project']}")
        print()

        results = model.train(
            data          = CONFIG["data"],
            epochs        = CONFIG["epochs"],
            imgsz         = CONFIG["imgsz"],
            batch         = CONFIG["batch"],
            patience      = CONFIG["patience"],
            lr0           = CONFIG["lr0"],
            lrf           = CONFIG["lrf"],
            momentum      = CONFIG["momentum"],
            weight_decay  = CONFIG["weight_decay"],
            warmup_epochs = CONFIG["warmup_epochs"],
            hsv_h         = CONFIG["hsv_h"],
            hsv_s         = CONFIG["hsv_s"],
            hsv_v         = CONFIG["hsv_v"],
            degrees       = CONFIG["degrees"],
            translate     = CONFIG["translate"],
            scale         = CONFIG["scale"],
            flipud        = CONFIG["flipud"],
            fliplr        = CONFIG["fliplr"],
            mosaic        = CONFIG["mosaic"],
            mixup         = CONFIG["mixup"],
            project       = CONFIG["project"],
            name          = CONFIG["name"],
            save_period   = CONFIG["save_period"],
            exist_ok      = CONFIG["exist_ok"],
            device        = CONFIG["device"],
            workers       = CONFIG["workers"],
            verbose       = True,
            plots         = True,   # Tự động lưu confusion matrix, PR curve
        )

    # Lưu đường dẫn best model
    run_dir = Path(CONFIG["project"]) / CONFIG["name"]
    best_model = run_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("✅ TRAIN HOÀN THÀNH!")
    print("=" * 60)
    print(f"  Best model : {best_model}")
    print(f"  Results    : {run_dir}")

    # Vẽ training curves
    print("\n📈 Vẽ training curves...")
    plot_training_results(str(run_dir))

    # In metrics cuối
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        print("\n📊 Metrics cuối cùng:")
        for k, v in rd.items():
            if any(x in k for x in ["mAP", "precision", "recall"]):
                print(f"  {k:<30}: {v:.4f}")

    print(f"\n→ Tiếp theo: chạy 03_evaluate.py --model {best_model}")

    return str(best_model)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Fall Detection")
    parser.add_argument("--model",  type=str,   default=None, help="Model size (yolov8n.pt, yolov8s.pt, ...)")
    parser.add_argument("--epochs", type=int,   default=None, help="Số epoch")
    parser.add_argument("--batch",  type=int,   default=None, help="Batch size")
    parser.add_argument("--imgsz",  type=int,   default=None, help="Image size")
    parser.add_argument("--device", type=str,   default=None, help="Device: 0, 1, cpu")
    parser.add_argument("--resume", action="store_true",      help="Tiếp tục train từ checkpoint last.pt đã bị tạm dừng")
    args = parser.parse_args()

    try:
        train(args)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 ĐÃ TẠM DỪNG HUẤN LUYỆN (KEYBOARD INTERRUPT)")
        print("=" * 60)
        print("  Trạng thái học tập của các Epoch hoàn thành trước đó đã được tự động lưu trong file last.pt!")
        print("  Để tiếp tục train từ đúng Epoch đã dừng, gõ lệnh:")
        print("     python 02_train.py --resume")
        print("=" * 60)
