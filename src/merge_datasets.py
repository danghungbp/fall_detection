"""
=============================================================
merge_datasets.py - Gộp & Chuẩn hóa (Remap) nhiều dataset YOLOv8
=============================================================
Sử dụng cho Giai đoạn 1: Khi tải nhiều dataset từ Roboflow Universe
(ví dụ: dataset A có fall/walking, dataset B có lying/bending...).
Script này sẽ tự động:
  1. Đọc file data.yaml của từng dataset con tải về.
  2. Chuẩn hóa ID của nhãn về đúng chuẩn 6 class của đồ án:
     0: fall, 1: walking, 2: sitting, 3: standing, 4: lying, 5: bending
  3. Gộp toàn bộ ảnh và label vào thư mục chung data/fall_dataset/
=============================================================
"""

import os
import shutil
import yaml
import argparse
from pathlib import Path

# Chuẩn 6 class mục tiêu của Đồ án Tốt nghiệp
TARGET_CLASSES = {
    0: "fall",
    1: "walking",
    2: "sitting",
    3: "standing",
    4: "lying",
    5: "bending"
}

# Từ điển ánh xạ từ khóa (để nhận diện tên class từ các dataset Roboflow khác nhau)
KEYWORD_MAPPING = {
    0: ["fall", "falling", "fall_detected", "fallen"],
    1: ["walk", "walking", "person_walking", "move"],
    2: ["sit", "sitting", "person_sitting"],
    3: ["stand", "standing", "person_standing"],
    4: ["lie", "lying", "lying_down", "sleep", "sleeping"],
    5: ["bend", "bending", "crouch", "crouching", "stoop"]
}

def get_target_class_id(raw_name: str) -> int:
    """Tìm ID chuẩn (0-5) từ tên class thô trong dataset tải về."""
    clean_name = raw_name.lower().strip().replace("-", "_").replace(" ", "_")
    for target_id, keywords in KEYWORD_MAPPING.items():
        if any(kw in clean_name for kw in keywords):
            return target_id
    return -1  # Không thuộc 6 class mục tiêu

def merge_and_remap_dataset(src_dir: str, target_dir: str, prefix: str):
    """Gộp 1 dataset con vào dataset chính."""
    src_path = Path(src_dir)
    yaml_files = list(src_path.glob("*.yaml")) + list(src_path.glob("*.yml"))
    if not yaml_files:
        print(f"⚠️ Không tìm thấy file data.yaml trong {src_dir}. Bỏ qua.")
        return 0

    with open(yaml_files[0], "r", encoding="utf-8") as f:
        src_yaml = yaml.safe_load(f)

    src_names = src_yaml.get("names", {})
    if isinstance(src_names, list):
        src_names = {i: name for i, name in enumerate(src_names)}

    # Tạo bảng ánh xạ ID cũ -> ID mới
    id_map = {}
    print(f"\n📁 Đang xử lý dataset: {src_dir}")
    print("  🔗 Ánh xạ class:")
    for old_id, old_name in src_names.items():
        new_id = get_target_class_id(str(old_name))
        if new_id != -1:
            id_map[int(old_id)] = new_id
            print(f"    [{old_id}] '{old_name}'  -->  [{new_id}] '{TARGET_CLASSES[new_id]}'")
        else:
            print(f"    [{old_id}] '{old_name}'  -->  ❌ Bỏ qua (không thuộc 6 class)")

    total_added = 0
    # Duyệt qua các tập train/val/test
    for split in ["train", "valid", "val", "test"]:
        target_split = "val" if split in ["valid", "val", "test"] else "train"
        
        img_src_dir = src_path / split / "images"
        lbl_src_dir = src_path / split / "labels"
        
        # Một số dataset Roboflow để images/train thay vì train/images
        if not img_src_dir.exists():
            img_src_dir = src_path / "images" / split
            lbl_src_dir = src_path / "labels" / split

        if not img_src_dir.exists():
            continue

        target_img_dir = Path(target_dir) / "images" / target_split
        target_lbl_dir = Path(target_dir) / "labels" / target_split
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        for img_file in img_src_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            lbl_file = lbl_src_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue

            # Đọc và chuyển đổi ID trong file label
            new_lines = []
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls = int(parts[0])
                    if old_cls in id_map:
                        new_cls = id_map[old_cls]
                        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))

            # Nếu ảnh có ít nhất 1 bounding box hợp lệ mới copy
            if new_lines:
                new_img_name = f"{prefix}_{img_file.name}"
                new_lbl_name = f"{prefix}_{img_file.stem}.txt"

                shutil.copy(img_file, target_img_dir / new_img_name)
                with open(target_lbl_dir / new_lbl_name, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")

                total_added += 1

    print(f"  ✅ Đã thêm {total_added} ảnh hợp lệ vào {target_dir}")
    return total_added

def main():
    project_root = Path(__file__).resolve().parent.parent
    default_target = str(project_root / "data" / "fall_dataset")
    
    parser = argparse.ArgumentParser(description="Gộp và remap nhãn dataset Roboflow về chuẩn 6 class")
    parser.add_argument("--sources", nargs="+", help="Danh sách các thư mục dataset tải từ Roboflow")
    parser.add_argument("--target", default=default_target, help="Thư mục dataset đích")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 GỘP VÀ CHUẨN HÓA DATASET VỀ 6 CLASS YOLOV8")
    print("=" * 60)

    if not args.sources:
        print("💡 Ví dụ cách chạy:")
        print("   python merge_datasets.py --sources ../data/roboflow_dataset1 ../data/roboflow_dataset2")
        return

    total = 0
    for idx, src in enumerate(args.sources):
        total += merge_and_remap_dataset(src, args.target, prefix=f"rf{idx+1}")

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT! Tổng cộng đã gộp thêm {total} ảnh mới vào {args.target}")
    print("👉 Hãy chạy 'python 01_prepare_dataset.py' để kiểm tra lại thống kê 6 class!")
    print("=" * 60)

if __name__ == "__main__":
    main()
