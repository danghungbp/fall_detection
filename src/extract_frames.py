"""
=============================================================
extract_frames.py - Trích xuất khung hình từ Video phục vụ gán nhãn
=============================================================
Sử dụng cho Giai đoạn 1: Thu thập và chuẩn bị dữ liệu (mở rộng 6 class)
Tự động cắt video thành các ảnh .jpg cách nhau N khung hình (hoặc theo FPS)
để upload trực tiếp lên Roboflow gán nhãn (bổ sung lying, standing, bending...).
=============================================================
"""

import os
import cv2
import argparse
from pathlib import Path

def extract_frames_from_video(video_path: str, output_dir: str, interval: int = 15, prefix: str = "frame"):
    """
    Trích xuất ảnh từ video với khoảng cách `interval` khung hình.
    
    Args:
        video_path: Đường dẫn file video (hoặc thư mục chứa video)
        output_dir: Thư mục lưu ảnh đầu ra
        interval: Số khung hình nhảy cóc giữa 2 lần chụp (vd: 15 frame ~ 0.5 giây nếu video 30fps)
        prefix: Tiền tố tên file ảnh đầu ra
    """
    if not os.path.exists(video_path):
        print(f"❌ Lỗi: Không tìm thấy file/thư mục video: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    
    # Nếu là thư mục, duyệt qua tất cả video (kể cả trong các thư mục con)
    if os.path.isdir(video_path):
        exts = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = [str(f) for f in Path(video_path).rglob("*") if f.suffix.lower() in exts]
    else:
        video_files = [video_path]

    total_saved = 0
    for v_path in video_files:
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"⚠️ Không thể mở video: {v_path}")
            continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Thêm tên thư mục cha (ví dụ chute01) vào tên video để tránh ghi đè khi trùng tên cam1.avi
        parent_name = Path(v_path).parent.name
        if parent_name and parent_name != Path(video_path).name:
            v_name = f"{parent_name}_{Path(v_path).stem}"
        else:
            v_name = Path(v_path).stem
        
        print(f"\n🎬 Đang xử lý video: {v_name} ({total_frames} frames, ~{fps} FPS)")
        
        frame_idx = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                # Tạo tên file ảnh độc nhất
                img_name = f"{prefix}_{v_name}_f{frame_idx:05d}.jpg"
                save_path = os.path.join(output_dir, img_name)
                cv2.imwrite(save_path, frame)
                saved_count += 1
                total_saved += 1
                
            frame_idx += 1
            
        cap.release()
        print(f"  ✅ Đã trích xuất {saved_count} ảnh từ {v_name} vào {output_dir}")
        
    return total_saved

def main():
    parser = argparse.ArgumentParser(description="Trích xuất khung hình từ video để gán nhãn Roboflow")
    parser.add_argument("--source", type=str, default="../data/raw_videos", help="Đường dẫn file video hoặc thư mục chứa video")
    parser.add_argument("--output", type=str, default="../data/extracted_frames", help="Thư mục lưu ảnh trích xuất")
    parser.add_argument("--interval", type=int, default=15, help="Số khung hình giữa các lần trích xuất (mặc định 15 ~ 0.5s)")
    parser.add_argument("--prefix", type=str, default="data", help="Tiền tố tên file ảnh")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📹 TRÍCH XUẤT KHUNG HÌNH TỪ VIDEO PHỤC VỤ GÁN NHÃN ROBOFLOW")
    print("=" * 60)
    
    total = extract_frames_from_video(args.source, args.output, args.interval, args.prefix)
    
    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT! Tổng cộng đã trích xuất: {total} ảnh.")
    print(f"📂 Thư mục chứa ảnh: {os.path.abspath(args.output)}")
    print("👉 Bước tiếp theo: Nén thư mục này zip lại và kéo thả lên Roboflow để gán nhãn 6 class!")
    print("=" * 60)

if __name__ == "__main__":
    main()
