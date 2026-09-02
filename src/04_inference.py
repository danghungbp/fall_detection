"""
=============================================================
04_inference.py - Inference ảnh / video / thư mục
=============================================================
Cách dùng:
  # Inference một ảnh
  python 04_inference.py --source image.jpg

  # Inference thư mục ảnh
  python 04_inference.py --source ./data/test_images/

  # Inference video
  python 04_inference.py --source video.mp4

  # Tùy chỉnh ngưỡng
  python 04_inference.py --source image.jpg --conf 0.4 --iou 0.5
=============================================================
"""

import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CẤU HÌNH
# ============================================================
DEFAULT_MODEL = "./runs/train/fall_detection_yolov8/weights/best.pt"
OUTPUT_DIR    = "../outputs/inference"

CLASS_NAMES   = {
    0: "fall", 
    1: "walking", 2: "sitting", 
    3: "standing", 4: "lying", 5: "bending"
}

# Màu BGR cho OpenCV
CLASS_COLORS  = {
    0: (50,  50,  220),   # Đỏ (BGR)  - fall
    1: (50,  200, 50),    # Xanh lá   - walking
    2: (220, 150, 50),    # Xanh dương - sitting
    3: (200, 200, 50),    # Vàng      - standing
    4: (200, 50,  200),   # Tím       - lying
    5: (50,  200, 200),   # Cyan      - bending
}

CONF_THRESHOLD = 0.40
IOU_THRESHOLD  = 0.45


# ============================================================
# HÀM VẼ KẾT QUẢ
# ============================================================
def draw_predictions(frame: np.ndarray, results, show_conf: bool = True) -> tuple:
    """
    Vẽ bounding box và nhãn lên frame.
    Returns: (annotated_frame, has_fall, fall_count)
    """
    has_fall   = False
    fall_count = 0
    annotated  = frame.copy()
    h, w = annotated.shape[:2]

    if results is None or results[0].boxes is None or len(results[0].boxes) == 0:
        return annotated, False, 0

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cls_name = CLASS_NAMES.get(cls_id, "Unknown")

        if cls_id == 0:
            has_fall   = True
            fall_count += 1

        # Vẽ bounding box
        thickness = 3 if cls_id == 0 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Nhãn
        label = f"{cls_name}: {conf:.2f}" if show_conf else cls_name
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lw, lh = label_size

        # Nền nhãn
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Cảnh báo TE NGA ở góc trên
    if has_fall:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(annotated,
                    f"⚠ TE NGA PHAT HIEN! ({fall_count} nguoi)",
                    (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    return annotated, has_fall, fall_count


def add_info_overlay(frame: np.ndarray, fps: float = 0, frame_idx: int = 0) -> np.ndarray:
    """Thêm thông tin FPS và frame index lên góc dưới."""
    h, w = frame.shape[:2]
    info = f"FPS: {fps:.1f}  |  Frame: {frame_idx}"
    cv2.putText(frame, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def infer_image(model: YOLO, img_path: str, output_dir: str, conf: float, iou: float) -> dict:
    """Inference trên một ảnh đơn."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ❌ Không đọc được ảnh: {img_path}")
        return {}

    t0 = time.time()
    results = model.predict(img_path, conf=conf, iou=iou, verbose=False, agnostic_nms=True)
    elapsed = (time.time() - t0) * 1000

    annotated, has_fall, fall_count = draw_predictions(img, results)

    # Lưu kết quả
    os.makedirs(output_dir, exist_ok=True)
    fname = Path(img_path).stem + "_pred.jpg"
    save_path = os.path.join(output_dir, fname)
    cv2.imwrite(save_path, annotated)

    # In kết quả
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id  = int(box.cls.item())
            conf_v  = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "class": CLASS_NAMES[cls_id],
                "conf" : conf_v,
                "bbox" : [x1, y1, x2, y2]
            })

    print(f"\n  Ảnh    : {img_path}")
    print(f"  Thời gian : {elapsed:.1f} ms")
    print(f"  Phát hiện : {len(detections)} đối tượng")
    for d in detections:
        icon = "🔴" if d["class"] == "Fall_Detected" else ("🟢" if d["class"] == "Walking" else "🔵")
        print(f"    {icon} {d['class']:<15} conf={d['conf']:.3f}  bbox={d['bbox']}")
    if has_fall:
        print(f"  🚨 CẢNH BÁO: Phát hiện {fall_count} người té ngã!")
    print(f"  Kết quả lưu: {save_path}")

    return {"detections": detections, "has_fall": has_fall, "time_ms": elapsed, "save_path": save_path}


def infer_folder(model: YOLO, folder_path: str, output_dir: str, conf: float, iou: float):
    """Inference trên toàn bộ ảnh trong thư mục."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in Path(folder_path).iterdir() if f.suffix.lower() in img_exts])

    if not img_files:
        print(f"  ❌ Không tìm thấy ảnh trong: {folder_path}")
        return

    print(f"\n  Tổng số ảnh: {len(img_files)}")

    total_fall   = 0
    total_detect = 0

    for img_path in img_files:
        result = infer_image(model, str(img_path), output_dir, conf, iou)
        if result.get("has_fall"):
            total_fall += 1
        total_detect += len(result.get("detections", []))

    print(f"\n{'='*50}")
    print(f"  TỔNG KẾT:")
    print(f"  Số ảnh xử lý  : {len(img_files)}")
    print(f"  Tổng detections: {total_detect}")
    print(f"  Ảnh có té ngã  : {total_fall}")


def infer_video(model: YOLO, video_path: str, output_dir: str, conf: float, iou: float,
                display: bool = True, save_video: bool = True):
    """Inference trên video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Không mở được video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Video  : {video_path}")
    print(f"  FPS    : {fps:.1f}")
    print(f"  Size   : {width}x{height}")
    print(f"  Frames : {total}")

    # Setup video writer
    writer = None
    if save_video:
        os.makedirs(output_dir, exist_ok=True)
        fname = Path(video_path).stem + "_pred.mp4"
        save_path = os.path.join(output_dir, fname)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"  Output : {save_path}")

    frame_idx  = 0
    fall_frames = 0
    t_start    = time.time()

    print("\n  Bắt đầu xử lý... (nhấn 'q' để dừng)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        # Dùng model.track thay vì predict để bật Object Tracking, giúp giữ khung hình ổn định hơn
        results = model.track(frame, conf=conf, iou=iou, persist=True, tracker="botsort.yaml", verbose=False, agnostic_nms=True)
        annotated, has_fall, _ = draw_predictions(frame, results)

        # Tính FPS thực
        elapsed = time.time() - t0
        real_fps = 1.0 / max(elapsed, 1e-6)
        annotated = add_info_overlay(annotated, real_fps, frame_idx)

        if has_fall:
            fall_frames += 1

        if writer:
            writer.write(annotated)

        if display:
            cv2.imshow("Fall Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  Người dùng đã dừng!")
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed_total = time.time() - t_start
            print(f"  Đã xử lý: {frame_idx}/{total} frames | "
                  f"Speed: {frame_idx/elapsed_total:.1f} FPS | "
                  f"Fall frames: {fall_frames}")

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\n  Hoàn tất! Tổng frames: {frame_idx} | Fall frames: {fall_frames}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Inference YOLOv8 Fall Detection")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL, help="Đường dẫn model .pt")
    parser.add_argument("--source",   type=str, required=True,         help="Ảnh / Video / Thư mục")
    parser.add_argument("--conf",     type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou",      type=float, default=IOU_THRESHOLD,  help="IOU threshold")
    parser.add_argument("--output",   type=str, default=OUTPUT_DIR,    help="Thư mục lưu kết quả")
    parser.add_argument("--no-display", action="store_true",           help="Không hiện cửa sổ (headless)")
    args = parser.parse_args()

    print(f"\n🚀 INFERENCE - FALL DETECTION")
    print(f"   Model : {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Conf  : {args.conf} | IOU: {args.iou}")

    # Load model
    model = YOLO(args.model)
    print(f"   ✅ Model đã load!")

    source = Path(args.source)

    if source.is_file():
        ext = source.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            # Ảnh
            infer_image(model, str(source), args.output, args.conf, args.iou)
        elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
            # Video
            infer_video(model, str(source), args.output, args.conf, args.iou,
                       display=not args.no_display, save_video=True)
        else:
            print(f"  ❌ Định dạng file không hỗ trợ: {ext}")

    elif source.is_dir():
        # Thư mục ảnh
        infer_folder(model, str(source), args.output, args.conf, args.iou)

    else:
        print(f"  ❌ Không tìm thấy: {args.source}")


if __name__ == "__main__":
    main()
