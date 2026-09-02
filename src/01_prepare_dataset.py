"""
=============================================================
01_prepare_dataset.py - Chuẩn bị & Kiểm tra Dataset
=============================================================
Script này sẽ:
  1. Kiểm tra cấu trúc thư mục dataset
  2. Thống kê phân bố các class
  3. Visualize một số ảnh mẫu với bounding box
  4. Kiểm tra file label có hợp lệ không
  5. Tạo báo cáo tổng hợp
=============================================================
"""

import os
import sys
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# ============================================================
# CẤU HÌNH
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = str(PROJECT_ROOT / "data" / "fall_dataset")       # Thư mục gốc dataset
YAML_PATH    = str(PROJECT_ROOT / "data" / "fall_dataset.yaml")  # File config YAML

CLASS_NAMES  = {
    0: "fall", 
    1: "walking", 2: "sitting", 
    3: "standing", 4: "lying", 5: "bending"
}
CLASS_COLORS = {
    0: (50,  50,  220),  # Đỏ   - fall
    1: (50,  200, 50),   # Xanh lá - walking
    2: (220, 150, 50),   # Xanh dương - sitting
    3: (200, 200, 50),   # Vàng - standing
    4: (200, 50,  200),  # Tím - lying
    5: (50,  200, 200),  # Cyan - bending
}
# Màu cho matplotlib
MPL_COLORS = {
    0: "#DC3232", 1: "#32C832", 2: "#3296DC",
    3: "#C8C832", 4: "#C832C8", 5: "#32C8C8"
}


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def check_directory_structure(dataset_root: str) -> bool:
    """Kiểm tra cấu trúc thư mục dataset có đúng chuẩn YOLO không."""
    required = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]
    print("=" * 60)
    print("📁 KIỂM TRA CẤU TRÚC THƯ MỤC")
    print("=" * 60)

    all_ok = True
    for rel_path in required:
        full_path = os.path.join(dataset_root, rel_path)
        exists = os.path.isdir(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status}  {full_path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n⚠️  Một số thư mục thiếu!")
        print("   Hãy đảm bảo dataset có cấu trúc:")
        print("   fall_dataset/")
        print("   ├── images/")
        print("   │   ├── train/   ← ảnh train")
        print("   │   └── val/     ← ảnh val")
        print("   └── labels/")
        print("       ├── train/   ← nhãn train (.txt)")
        print("       └── val/     ← nhãn val (.txt)")
    else:
        print("\n✅ Cấu trúc thư mục hợp lệ!")

    return all_ok


def count_images_and_labels(dataset_root: str) -> dict:
    """Đếm số ảnh và label trong từng split."""
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ SỐ LƯỢNG FILE")
    print("=" * 60)

    stats = {}
    for split in ["train", "val"]:
        img_dir = Path(dataset_root) / "images" / split
        lbl_dir = Path(dataset_root) / "labels" / split

        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts] if img_dir.exists() else []
        lbls = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"] if lbl_dir.exists() else []

        # Kiểm tra ảnh không có label tương ứng
        img_stems = {f.stem for f in imgs}
        lbl_stems = {f.stem for f in lbls}
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems

        stats[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
        }

        print(f"\n  [{split.upper()}]")
        print(f"    Ảnh    : {len(imgs)}")
        print(f"    Label  : {len(lbls)}")
        if missing_labels:
            print(f"    ⚠️  Ảnh thiếu label: {len(missing_labels)}")
        if missing_images:
            print(f"    ⚠️  Label thiếu ảnh: {len(missing_images)}")
        else:
            print(f"    ✅ Tất cả ảnh đều có label")

    return stats


def analyze_class_distribution(dataset_root: str) -> dict:
    """Phân tích phân bố class trong toàn bộ dataset."""
    print("\n" + "=" * 60)
    print("🏷️  PHÂN BỐ CLASS (BOUNDING BOX)")
    print("=" * 60)

    distribution = {"train": Counter(), "val": Counter(), "total": Counter()}

    for split in ["train", "val"]:
        lbl_dir = Path(dataset_root) / "labels" / split
        if not lbl_dir.exists():
            continue

        for lbl_file in tqdm(list(lbl_dir.glob("*.txt")), desc=f"  Đọc label {split}", ncols=70):
            try:
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            distribution[split][cls_id] += 1
                            distribution["total"][cls_id] += 1
            except Exception as e:
                print(f"    ⚠️  Lỗi đọc {lbl_file.name}: {e}")

    # In bảng kết quả
    print(f"\n  {'Class':<20} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*44}")
    for cls_id, cls_name in CLASS_NAMES.items():
        train_cnt = distribution["train"][cls_id]
        val_cnt   = distribution["val"][cls_id]
        total_cnt = distribution["total"][cls_id]
        print(f"  {cls_name:<20} {train_cnt:>8} {val_cnt:>8} {total_cnt:>8}")
    print(f"  {'-'*44}")
    total_all = sum(distribution["total"].values())
    print(f"  {'TỔNG':<20} {sum(distribution['train'].values()):>8} {sum(distribution['val'].values()):>8} {total_all:>8}")

    return distribution


def visualize_samples(dataset_root: str, n_samples: int = 9, save_path: str = "./outputs/sample_images.png"):
    """Visualize ảnh mẫu với bounding box và nhãn."""
    print("\n" + "=" * 60)
    print("🖼️  VISUALIZE ẢNH MẪU")
    print("=" * 60)

    img_dir = Path(dataset_root) / "images" / "train"
    lbl_dir = Path(dataset_root) / "labels" / "train"

    if not img_dir.exists():
        print("  ❌ Không tìm thấy thư mục ảnh train!")
        return

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts]

    if not img_files:
        print("  ❌ Không tìm thấy ảnh!")
        return

    # Chọn ngẫu nhiên n_samples ảnh
    np.random.seed(42)
    selected = np.random.choice(img_files, min(n_samples, len(img_files)), replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Ảnh mẫu từ Fall Detection Dataset", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, img_path in enumerate(selected):
        ax = axes[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            ax.axis("off")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        ax.imshow(img_rgb)
        ax.set_title(os.path.basename(img_path), fontsize=8)
        ax.axis("off")

        # Đọc và vẽ bounding box
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Chuyển từ normalized YOLO → pixel
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h

                    color = MPL_COLORS.get(cls_id, "#FFFFFF")
                    rect = patches.Rectangle(
                        (x1, y1), box_w, box_h,
                        linewidth=2, edgecolor=color, facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1, y1 - 4,
                        CLASS_NAMES.get(cls_id, str(cls_id)),
                        color="white", fontsize=7, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
                    )

    # Ẩn subplot thừa
    for i in range(len(selected), len(axes)):
        axes[i].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Đã lưu ảnh mẫu → {save_path}")


def plot_class_distribution(distribution: dict, save_path: str = "./outputs/class_distribution.png"):
    """Vẽ biểu đồ phân bố class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phân bố Class trong Dataset", fontsize=14, fontweight="bold")

    class_labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    colors = [MPL_COLORS[i] for i in range(len(CLASS_NAMES))]

    for ax, split in zip(axes, ["train", "val"]):
        counts = [distribution[split][i] for i in range(len(CLASS_NAMES))]
        bars = ax.bar(class_labels, counts, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(f"Tập {split.upper()}", fontweight="bold")
        ax.set_ylabel("Số lượng bounding box")
        ax.set_xlabel("Class")

        # Ghi số lên đầu mỗi cột
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontweight="bold"
            )

        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Đã lưu biểu đồ phân bố → {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n🚀 CHUẨN BỊ DATASET - FALL DETECTION")
    print("=" * 60)
    print(f"   Dataset root : {DATASET_ROOT}")
    print(f"   YAML config  : {YAML_PATH}")

    # 1. Kiểm tra cấu trúc thư mục
    ok = check_directory_structure(DATASET_ROOT)
    if not ok:
        print("\n❌ Vui lòng sửa cấu trúc thư mục trước khi tiếp tục!")
        sys.exit(1)

    # 2. Đếm số file
    count_images_and_labels(DATASET_ROOT)

    # 3. Phân tích phân bố class
    distribution = analyze_class_distribution(DATASET_ROOT)

    # 4. Vẽ biểu đồ phân bố
    plot_class_distribution(distribution, "../outputs/class_distribution.png")

    # 5. Visualize ảnh mẫu
    visualize_samples(DATASET_ROOT, n_samples=9, save_path="../outputs/sample_images.png")

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT KIỂM TRA DATASET!")
    print("   → Tiếp theo: chạy 02_train.py để train mô hình")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
