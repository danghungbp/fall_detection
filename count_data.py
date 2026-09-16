import os
import sys
from collections import Counter
from pathlib import Path
import yaml

# Đảm bảo in Tiếng Việt trên Windows Terminal không bị lỗi charmap & flush ngay lập tức
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Tự động xác định đường dẫn thư mục gốc dự án & thư mục data
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "fall_dataset"

# Trường hợp thư mục fall_dataset không tồn tại thì thử thư mục data
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data"

YAML_PATH = BASE_DIR / "data" / "fall_dataset.yaml"

# Load danh sách tên class từ file yaml nếu có, ngược lại dùng mặc định
classes = ["fall", "walking", "sitting", "standing", "lying", "bending"]
if YAML_PATH.exists():
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            if "names" in yaml_data:
                if isinstance(yaml_data["names"], list):
                    classes = yaml_data["names"]
                elif isinstance(yaml_data["names"], dict):
                    classes = [yaml_data["names"][i] for i in sorted(yaml_data["names"].keys())]
    except Exception:
        pass

splits = ["train", "val", "test"]

print("==================================================")
print(f"   THỐNG KÊ DATASET THỰC TẾ (YOLO FORMAT)")
print(f"   Thư mục: {DATA_DIR}")
print("==================================================")

total_all_images = 0
total_all_labels = 0
total_all_boxes = 0
overall_class_counter = Counter()

for split in splits:
    label_dir = DATA_DIR / "labels" / split
    img_dir = DATA_DIR / "images" / split

    if not label_dir.exists() and not img_dir.exists():
        continue

    # Đếm số lượng ảnh (hỗ trợ nhiều định dạng ảnh)
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [f for f in img_dir.glob("*.*") if f.suffix.lower() in img_extensions] if img_dir.exists() else []
    img_count = len(img_files)
    
    label_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []

    class_counter = Counter()
    total_boxes = 0

    for lf in label_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counter[cls_id] += 1
                        overall_class_counter[cls_id] += 1
                        total_boxes += 1
        except Exception as e:
            print(f"Lỗi đọc file {lf.name}: {e}")

    total_all_images += img_count
    total_all_labels += len(label_files)
    total_all_boxes += total_boxes

    print(f"\n[SPLIT: {split.upper()}]")
    print(f"  • Số lượng ảnh      : {img_count:,} ảnh")
    print(f"  • Số file nhãn (.txt): {len(label_files):,} file")
    print(f"  • Bounding Boxes   : {total_boxes:,} object(s)")
    print("  • Chi tiết từng lớp (class):")

    if class_counter:
        for cls_id in sorted(set(range(len(classes))).union(class_counter.keys())):
            count = class_counter[cls_id]
            cls_name = classes[cls_id] if cls_id < len(classes) else f"Class {cls_id}"
            percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
            print(f"      - [{cls_id}] {cls_name:<10}: {count:>6,} mẫu ({percentage:>5.1f}%)")
    else:
        print("      (Không tìm thấy nhãn dữ liệu)")

print("\n" + "=" * 50)
print("   TỔNG CỘNG TOÀN BỘ DATASET")
print("=" * 50)
print(f"  • Tổng số ảnh       : {total_all_images:,} ảnh")
print(f"  • Tổng số file nhãn : {total_all_labels:,} file")
print(f"  • Tổng số BBox      : {total_all_boxes:,} object(s)")
print("  • Thống kê tổng hợp các class:")

for cls_id in sorted(set(range(len(classes))).union(overall_class_counter.keys())):
    count = overall_class_counter[cls_id]
    cls_name = classes[cls_id] if cls_id < len(classes) else f"Class {cls_id}"
    percentage = (count / total_all_boxes * 100) if total_all_boxes > 0 else 0
    print(f"      - [{cls_id}] {cls_name:<10}: {count:>6,} mẫu ({percentage:>5.1f}%)")

print("=" * 50)

