"""
=============================================================
build_final_dataset.py - Xây dựng bộ Dataset hoàn chỉnh & cân bằng
=============================================================
Sử dụng cho Giai đoạn 2 (Chuẩn bị Train):
  1. Dọn sạch thư mục làm việc data/fall_dataset/ (để không bị rác/trùng lặp).
  2. Nạp ưu tiên 100% ảnh băm ra từ Video thực tế (chuỗi hành vi liên tục theo ý thầy).
  3. Nạp bộ dữ liệu gốc Baseline.
  4. Nạp bổ sung từ kho Roboflow với cơ chế LỌC CÂN BẰNG (Downsampling),
     giới hạn tối đa mỗi class chỉ nhận ~350-400 ảnh để không bị lệch lớp.
=============================================================
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

TARGET_DIR = "../data/fall_dataset"
MAX_SAMPLES_PER_CLASS = 400  # Con số vàng để cân bằng 6 class (~2.400 ảnh tổng cộng)

def reset_target_directory():
    """Xóa sạch thư mục images/ và labels/ cũ trong fall_dataset để làm từ đầu."""
    print("🧹 Đang dọn dẹp thư mục đích: ../data/fall_dataset/ ...")
    for split in ["train", "val"]:
        img_dir = Path(TARGET_DIR) / "images" / split
        lbl_dir = Path(TARGET_DIR) / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
    print("  ✅ Đã làm sạch thư mục!")

def main():
    print("=" * 60)
    print("🎯 XÂY DỰNG DATASET HUẤN LUYỆN TỐI ƯU (HYBRID VIDEO + STATIC)")
    print("=" * 60)
    print("💡 Quy trình chuẩn bị:")
    print("   1. Nhấn nút Reset làm sạch data/fall_dataset/")
    print("   2. Dùng merge_datasets.py nạp thư mục ảnh Video (video_frames) vào trước.")
    print("   3. Dùng merge_datasets.py nạp tiếp các bộ Roboflow để bổ sung class thiếu.")
    print("=" * 60)
    
    confirm = input("❓ Bạn có muốn xóa sạch ảnh trong data/fall_dataset để chuẩn bị bộ mới? (y/n): ")
    if confirm.lower().strip() == 'y':
        reset_target_directory()
        print("\n🎉 Sẵn sàng! Bây giờ bạn có thể nạp ảnh video và ảnh chọn lọc vào rồi!")
    else:
        print("\n⏹ Đã hủy thao tác.")

if __name__ == "__main__":
    main()
