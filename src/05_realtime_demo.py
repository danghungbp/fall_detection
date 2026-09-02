"""
=============================================================
05_realtime_demo.py - Demo Real-time Phát Hiện Té Ngã (Webcam)
=============================================================
Tính năng:
  - Detect real-time từ webcam / RTSP camera / video file
  - Hiển thị cảnh báo âm thanh khi phát hiện té ngã
  - Dashboard thống kê trực tiếp
  - Lưu ảnh khi phát hiện té ngã
  - Gửi thông báo qua Backend API (Flask) → Firebase FCM → Mobile App

Cách dùng:
  # Webcam mặc định
  python 05_realtime_demo.py

  # Kết nối Backend API
  python 05_realtime_demo.py --backend "http://localhost:5000" --camera_id "cam_phong_ngu"

  # RTSP camera (camera IP)
  python 05_realtime_demo.py --source "rtsp://192.168.1.100:554/stream" --backend "http://192.168.1.50:5000"

Phím tắt trong cửa sổ:
  q  - Thoát
  s  - Chụp screenshot
  r  - Reset thống kê
  p  - Pause/Resume
=============================================================
"""

import os

# Fix quan trọng: Ép OpenCV dùng TCP cho RTSP (tránh mất gói tin UDP từ camera Hikvision)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
import time
import threading
import argparse
import numpy as np
import requests
import socket
from pathlib import Path
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from flask import Flask, Response

# ============================================================
# CẤU HÌNH LIVE STREAM (MJPEG Server)
# ============================================================
stream_app = Flask(__name__)
current_stream_frame = None
stream_lock = threading.Lock()

def gen_frames():
    """Generator sinh ra các frame dưới dạng multipart/x-mixed-replace (MJPEG)."""
    global current_stream_frame
    while True:
        time.sleep(0.05) # ~20 FPS
        with stream_lock:
            if current_stream_frame is None:
                continue
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', current_stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@stream_app.route('/video_feed')
def video_feed():
    """Endpoint trả về luồng stream MJPEG."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Ưu tiên biến môi trường STREAM_HOST (Tailscale IP) nếu có, 
# nếu không thì tự động lấy IP Wi-Fi cục bộ
LOCAL_IP = os.environ.get("STREAM_HOST", get_local_ip())


# ============================================================
# CẤU HÌNH
# ============================================================
DEFAULT_MODEL = "./src/runs/train/fall_detection_yolov8/weights/best.pt"
SAVE_DIR       = "./outputs/realtime_captures"


CLASS_NAMES    = {
    0: "fall", 
    1: "walking", 2: "sitting", 
    3: "standing", 4: "lying", 5: "bending"
}
CLASS_COLORS   = {
    0: (50,  50,  220),   # Đỏ  - fall
    1: (50,  200, 50),    # Xanh lá - walking
    2: (220, 150, 50),    # Xanh dương - sitting
    3: (200, 200, 50),    # Vàng - standing
    4: (200, 50,  200),   # Tím - lying
    5: (50,  200, 200),   # Cyan - bending
}

CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.45

# Thông số cảnh báo
FALL_ALERT_DURATION  = 3.0   # Giây hiện cảnh báo sau khi phát hiện
FALL_SAVE_COOLDOWN   = 30.0  # Tăng cooldown lên 30s để tránh spam app


# ============================================================
# CLASS DEMO
# ============================================================
class FallDetectionDemo:
    def __init__(self, model_path: str, source, conf: float, iou: float,
                 backend_url: str = "",
                 camera_id: str = "webcam", **kwargs):
        self.model   = YOLO(model_path)
        self.source  = source
        self.conf    = conf
        self.iou     = iou
        self.backend_url = backend_url.rstrip("/") if backend_url else ""
        self.camera_id   = camera_id
        
        self.location = kwargs.get('location', 'Webcam / RTSP Camera')
        self.stream_port = kwargs.get('stream_port', 5001)
        self.headless = kwargs.get('headless', False)

        # Bộ đếm khung hình liên tiếp để tránh báo động giả
        self.consecutive_fall_frames = 10  # Số khung hình liên tiếp nhận diện ngã mới báo động
        self.fall_frame_counter = 0

        # Thống kê
        self.stats = {
            "total_frames"  : 0,
            "fall_count"    : 0,
            "walking_count" : 0,
            "sitting_count" : 0,
            "standing_count": 0,
            "lying_count"   : 0,
            "bending_count" : 0,
            "fall_events"   : 0,   # Số lần phát hiện té ngã
            "start_time"    : time.time(),
        }

        # FPS tracking
        self.fps_queue = deque(maxlen=30)
        self.last_fall_time  = 0
        self.last_save_time  = 0
        self.alert_active    = False
        self.is_paused       = False

        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"  ✅ Model loaded: {model_path}")

        
        # Khởi chạy Live Stream Server
        def run_stream_server():
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            stream_app.run(host='0.0.0.0', port=self.stream_port, debug=False, use_reloader=False)
        
        threading.Thread(target=run_stream_server, daemon=True).start()
        self.stream_url = f"http://{LOCAL_IP}:{self.stream_port}/video_feed"
        print(f"  🎥 Live Stream Server đang chạy tại: {self.stream_url}")

        # Đăng ký camera lên backend
        if self.backend_url:
            self._register_camera()

    def _register_camera(self):
        """Đăng ký camera lên backend tại thời điểm khởi chạy."""
        print(f"\n[INFO] Đăng ký camera '{self.camera_id}' ({self.location}) lên Backend...")
        try:
            payload = {
                "camera_id": self.camera_id,
                "location": self.location,
                "mjpeg_url": self.stream_url,
                "is_active": 1
            }
            resp = requests.post(f"{self.backend_url}/api/cameras", json=payload, timeout=3)
            if resp.status_code in [200, 201]:
                print(f"  ✅ Đăng ký Camera '{self.camera_id}' thành công lên Backend!")
            elif resp.status_code == 409:
                # 409 Conflict nghĩa là đã tồn tại, ta cập nhật trạng thái hoạt động và URL mới nhất
                requests.put(f"{self.backend_url}/api/cameras/{self.camera_id}", json={"is_active": 1, "mjpeg_url": self.stream_url}, timeout=3)
                print(f"  ℹ️ Camera '{self.camera_id}' đã kích hoạt hoạt động và cập nhật luồng trực tiếp.")
            else:
                print(f"  ⚠️ Đăng ký Camera thất bại: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️ Không thể kết nối tới Backend để đăng ký Camera: {e}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Xử lý 1 frame: detect + annotate."""
        t0 = time.time()
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False, agnostic_nms=True)
        inference_ms = (time.time() - t0) * 1000
        self.fps_queue.append(1.0 / max(time.time() - t0, 1e-6))

        annotated  = frame.copy()
        has_fall   = False
        fall_count = 0
        frame_h, frame_w = frame.shape[:2]

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id  = int(box.cls.item())
                conf_v  = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                frame_area = frame_w * frame_h
                
                # Bỏ qua vật thể quá nhỏ (tranh, nhiễu) hoặc quá khổng lồ (nhận nhầm cả phòng/tủ/xe máy)
                if box_area < (frame_area * 0.05) or box_area > (frame_area * 0.60):
                    continue

                color   = CLASS_COLORS.get(cls_id, (180, 180, 180))
                label   = f"{CLASS_NAMES.get(cls_id, '?')}: {conf_v:.2f}"

                if cls_id == 0:
                    # ===== BỘ LỌC THÔNG MINH CHỐNG BÁO ĐỘNG GIẢ (FALL) =====
                    aspect_ratio = box_w / max(box_h, 1)
                    bottom_ratio = y2 / frame_h

                    # Lọc nghiêm ngặt hơn để tránh giường/võng:
                    # 1. Phải thật sự nằm bẹp (tỷ lệ > 1.2)
                    is_flat_horizontal = aspect_ratio > 1.2
                    # 2. Phải nằm rất sát mép dưới camera (sàn nhà)
                    is_on_floor = bottom_ratio > 0.75

                    if is_flat_horizontal and is_on_floor and conf_v >= 0.80:
                        has_fall   = True
                        fall_count += 1
                        self.stats["fall_count"] += 1
                        self.stats["last_confidence"] = conf_v
                        label = f"FALL: {conf_v:.2f}"
                        color = (50, 50, 220)
                    else:
                        if aspect_ratio > 1.0:
                            cls_id = 4 # Lying
                            self.stats["lying_count"] += 1
                        else:
                            cls_id = 5 # Bending
                            self.stats["bending_count"] += 1
                        label = f"{CLASS_NAMES.get(cls_id)}: {conf_v:.2f}"
                        color = CLASS_COLORS.get(cls_id)
                else:
                    # ===== BỘ LỌC HÀNH VI (NON-FALL) =====
                    aspect_ratio = box_w / max(box_h, 1)
                    center_y = (y1 + y2) / 2 / frame_h  # Vị trí tâm theo trục dọc (0=trên, 1=dưới)

                    # Sửa lỗi nhận nhầm sitting thành walking/standing/bending
                    # Điều kiện: bbox vuông/tròn + trọng tâm ở nửa dưới frame
                    if cls_id in [1, 3, 5] and (0.55 <= aspect_ratio <= 1.4) and center_y > 0.45:
                        cls_id = 2
                        color = CLASS_COLORS.get(2)
                        self.stats["sitting_count"] += 1
                    else:
                        if cls_id == 1:
                            self.stats["walking_count"] += 1
                        elif cls_id == 2:
                            self.stats["sitting_count"] += 1
                        elif cls_id == 3:
                            self.stats["standing_count"] += 1
                        elif cls_id == 4:
                            self.stats["lying_count"] += 1
                        elif cls_id == 5:
                            self.stats["bending_count"] += 1

                    # Nhãn sạch: chỉ tên class + confidence
                    label = f"{CLASS_NAMES.get(cls_id)}: {conf_v:.2f}"
                
                # Vẽ box
                thickness = 3 if (cls_id == 0 and has_fall) else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Nhãn
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Cập nhật trạng thái cảnh báo với bộ đếm khung hình liên tiếp
        if has_fall:
            self.fall_frame_counter += 1
            if self.fall_frame_counter >= self.consecutive_fall_frames:
                self.last_fall_time = time.time()
                self.stats["fall_events"] += 1

                # Tự động lưu ảnh & báo về App
                if time.time() - self.last_save_time > FALL_SAVE_COOLDOWN:
                    self._save_fall_capture(frame, fall_count)
                    self.last_save_time = time.time()
        else:
            # Reset bộ đếm nếu không phát hiện ngã ở khung hình hiện tại
            self.fall_frame_counter = 0

        self.alert_active = (time.time() - self.last_fall_time) < FALL_ALERT_DURATION

        # Vẽ overlay
        self._draw_alert_overlay(annotated)
        self._draw_dashboard(annotated, inference_ms)

        self.stats["total_frames"] += 1
        return annotated

    def _draw_alert_overlay(self, frame: np.ndarray):
        """Vẽ cảnh báo đỏ khi phát hiện té ngã."""
        if not self.alert_active:
            return

        h, w = frame.shape[:2]
        # Nền đỏ mờ toàn màn hình
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), 5)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Banner cảnh báo
        cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 160), -1)
        cv2.putText(frame,
                    "!!! CANH BAO: PHAT HIEN TE NGA !!!",
                    (w // 2 - 280, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95,
                    (255, 255, 255), 2)

    def _draw_dashboard(self, frame: np.ndarray, inference_ms: float):
        """Vẽ dashboard thống kê góc phải."""
        h, w = frame.shape[:2]
        panel_w = 250
        panel_h = 250
        x0 = w - panel_w - 10
        y0 = 10

        # Nền panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (100, 100, 100), 1)

        # Tiêu đề
        cv2.putText(frame, "FALL DETECTION STATS",
                    (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.line(frame, (x0 + 8, y0 + 28), (x0 + panel_w - 8, y0 + 28), (100, 100, 100), 1)

        # Thời gian chạy
        elapsed = time.time() - self.stats["start_time"]
        mins, secs = divmod(int(elapsed), 60)
        fps_avg = np.mean(self.fps_queue) if self.fps_queue else 0

        lines = [
            (f"Time: {mins:02d}:{secs:02d}",              (200, 200, 200)),
            (f"FPS:  {fps_avg:.1f}  ({inference_ms:.0f}ms)", (200, 200, 200)),
            (f"Frames: {self.stats['total_frames']}",      (200, 200, 200)),
            ("",                                            (0, 0, 0)),
            (f"Fall Events: {self.stats['fall_events']}",  (80, 80, 220)),
            (f"Walking Det: {self.stats['walking_count']}",(50, 200, 50)),
            (f"Sitting Det: {self.stats['sitting_count']}",(100, 180, 220)),
            (f"Standing Det: {self.stats['standing_count']}",(200, 200, 50)),
            (f"Lying Det: {self.stats['lying_count']}",    (200, 50, 200)),
            (f"Bending Det: {self.stats['bending_count']}",(50, 200, 200)),
        ]

        for i, (text, color) in enumerate(lines):
            if text:
                cv2.putText(frame, text,
                            (x0 + 8, y0 + 48 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Phím tắt ở dưới
        shortcuts = "q:Quit  s:Screenshot  r:Reset  p:Pause"
        cv2.putText(frame, shortcuts,
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # Trạng thái Pause
        if self.is_paused:
            cv2.putText(frame, "[ PAUSED ]",
                        (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

    def _save_fall_capture(self, frame: np.ndarray, count: int):
        """Lưu ảnh khi phát hiện té ngã."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_DIR, f"fall_{ts}_{count}people.jpg")
        cv2.imwrite(path, frame)
        print(f"\n  📸 Đã lưu cảnh báo: {path}")

        # Gửi Backend API notification → FCM → Mobile App
        if self.backend_url:
            self._notify_backend(frame, count)

    def _notify_backend(self, frame: np.ndarray, fall_count: int):
        """Gửi sự kiện té ngã lên Backend Flask API (trong background thread)."""
        import base64
        from datetime import timezone

        def _task():
            try:
                # Encode frame → base64 JPEG
                _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_b64   = base64.b64encode(buf).decode("utf-8")
                timestamp = datetime.now(timezone.utc).astimezone().isoformat()

                payload = {
                    "camera_id":    self.camera_id,
                    "confidence":   round(self.stats.get("last_confidence", 0.0), 4),
                    "image_base64": img_b64,
                    "location":     self.camera_id.replace("_", " ").title(),
                    "timestamp":    timestamp,
                }
                resp = requests.post(
                    f"{self.backend_url}/api/events",
                    json=payload,
                    timeout=5,
                )
                if resp.status_code == 201:
                    event_id = resp.json().get("event_id", "?")
                    print(f"  ✅ Backend API: Sự kiện #{event_id} đã ghi nhận → FCM đang gửi...")
                else:
                    print(f"  ⚠️  Backend API response: {resp.status_code} - {resp.text[:100]}")
            except Exception as e:
                print(f"  ❌ Lỗi gọi Backend API: {e}")

        threading.Thread(target=_task, daemon=True).start()



    def run(self):
        """Vòng lặp chính."""
        # Mở nguồn video
        source = self.source
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cap = cv2.VideoCapture(int(source))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"  📷 Mở webcam #{source}")
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            print(f"  🎥 Mở nguồn: {source}")

        if not cap.isOpened():
            print(f"  ❌ Không mở được nguồn: {source}")
            return

        # Với camera RTSP: bỏ qua các frame đầu bị hỏng (Hikvision cần thời gian ổn định)
        if isinstance(source, str) and source.startswith("rtsp://"):
            print(f"  ⏳ Đang chờ camera ổn định (bỏ qua frame lỗi ban đầu)...")
            good_frames = 0
            for attempt in range(100):  # Thử tối đa 100 lần
                ret, _ = cap.read()
                if ret:
                    good_frames += 1
                    if good_frames >= 3:  # Cần 3 frame tốt liên tiếp mới tính ổn định
                        break
                else:
                    good_frames = 0
                    time.sleep(0.1)
            print(f"  ✅ Camera RTSP đã sẵn sàng! (sau {attempt+1} lần thử)")

        print("\n  🟢 Đang chạy... Nhấn 'q' để thoát")
        print(f"  📁 Ảnh cảnh báo sẽ lưu vào: {SAVE_DIR}/")

        screenshot_count = 0

        is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
        max_read_failures = 50 if is_rtsp else 3  # Camera RTSP cho phép thử lại nhiều lần
        consecutive_failures = 0

        while True:
            if not self.is_paused:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_read_failures:
                        print(f"\n  📹 Mất kết nối camera sau {consecutive_failures} lần thử!")
                        break
                    time.sleep(0.1)  # Đợi một chút rồi thử lại
                    continue
                consecutive_failures = 0  # Reset khi đọc thành công

                annotated = self.process_frame(frame)
                
                # Cập nhật frame cho stream
                global current_stream_frame
                with stream_lock:
                    current_stream_frame = annotated.copy()
            else:
                # Khi pause: giữ frame cuối
                time.sleep(0.033)

            if not self.headless:
                cv2.imshow("Fall Detection - Real-time", annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                # Chạy headless, chỉ nghỉ một chút để đồng bộ FPS
                time.sleep(0.03)
                key = -1

            if key == ord("q"):
                print("\n  Thoát...")
                break

            elif key == ord("s"):
                # Screenshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SAVE_DIR, f"screenshot_{ts}.jpg")
                cv2.imwrite(path, annotated)
                screenshot_count += 1
                print(f"\n  📸 Screenshot: {path}")

            elif key == ord("r"):
                # Reset stats
                self.stats.update({
                    "total_frames" : 0,
                    "fall_count"   : 0,
                    "walking_count": 0,
                    "sitting_count": 0,
                    "standing_count": 0,
                    "lying_count": 0,
                    "bending_count": 0,
                    "fall_events"  : 0,
                    "start_time"   : time.time(),
                })
                print("\n  🔄 Đã reset thống kê!")

            elif key == ord("p"):
                # Pause / Resume
                self.is_paused = not self.is_paused
                state = "Paused" if self.is_paused else "Resumed"
                print(f"\n  ⏸  {state}")

        cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self._print_summary(screenshot_count)

    def _print_summary(self, screenshot_count: int):
        """In tổng kết khi kết thúc."""
        elapsed = time.time() - self.stats["start_time"]
        mins, secs = divmod(int(elapsed), 60)

        print("\n" + "=" * 50)
        print("📊 TỔNG KẾT PHIÊN")
        print("=" * 50)
        print(f"  Thời gian      : {mins:02d}:{secs:02d}")
        print(f"  Frames xử lý  : {self.stats['total_frames']}")
        avg_fps = self.stats['total_frames'] / max(elapsed, 1)
        print(f"  FPS trung bình : {avg_fps:.1f}")
        print(f"  Sự kiện té ngã : {self.stats['fall_events']}")
        print(f"  Screenshots    : {screenshot_count}")
        print(f"  Captures lưu   : {SAVE_DIR}/")
        print("=" * 50)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Real-time Fall Detection Demo")
    parser.add_argument("--model",     type=str,   default=DEFAULT_MODEL,   help="Đường dẫn model .pt")
    parser.add_argument("--camera",    type=int,   default=0,               help="Camera index (default: 0)")
    parser.add_argument("--source",    type=str,   default=None,            help="Video file / RTSP URL (override --camera)")
    parser.add_argument("--conf",      type=float, default=CONF_THRESHOLD,  help="Confidence threshold")
    parser.add_argument("--iou",       type=float, default=IOU_THRESHOLD,   help="IOU threshold")

    parser.add_argument("--backend",   type=str,   default="",              help="Backend API URL (vd: http://localhost:5000)")
    parser.add_argument("--camera_id", type=str,   default="webcam",        help="ID định danh camera (vd: cam_phong_ngu)")
    parser.add_argument("--location",  type=str,   default="Webcam Camera", help="Vị trí camera")
    parser.add_argument("--stream_port", type=int, default=5001,            help="Cổng HTTP stream")
    parser.add_argument("--headless",  action="store_true",                 help="Chạy ẩn giao diện GUI (Headless mode)")
    args = parser.parse_args()

    source = args.source if args.source else args.camera

    print("\n🚀 REAL-TIME FALL DETECTION DEMO")
    print(f"   Model     : {args.model}")
    print(f"   Source    : {source}")
    print(f"   Conf      : {args.conf} | IOU: {args.iou}")
    if args.backend:
        print(f"   Backend   : {args.backend}  (camera_id: {args.camera_id})")
        print(f"   Location  : {args.location}")
        print(f"   StreamPort: {args.stream_port}")


    demo = FallDetectionDemo(
        model_path          = args.model,
        source              = source,
        conf                = args.conf,
        iou                 = args.iou,
        backend_url         = args.backend,
        camera_id           = args.camera_id,
        location            = args.location,
        stream_port         = args.stream_port,
        headless            = args.headless
    )
    demo.run()


if __name__ == "__main__":
    main()
