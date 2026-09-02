"""
=============================================================
camera_manager.py - Hệ thống AI Camera Auto-Scaling
=============================================================
Tính năng:
  - Tự động lấy danh sách camera từ Database (Backend).
  - Khởi tạo tiến trình AI (05_realtime_demo.py) cho mỗi camera.
  - Tự động cấp phát cổng stream (5001, 5002,...)
  - Quản lý vòng đời tiến trình: Camera bị xóa -> tắt tiến trình.
=============================================================
"""
import time
import requests
import subprocess
import os
import sys

BACKEND_URL = "http://localhost:5000"
POLL_INTERVAL = 5
START_PORT = 5001

# Dictionary lưu tiến trình: camera_id -> {"process": Popen, "port": int, "rtsp_url": str}
active_cameras = {}

def get_cameras_from_db():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/cameras", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[MANAGER] Lỗi kết nối backend: {e}")
    return []

def allocate_port():
    """Tìm cổng trống từ START_PORT."""
    used_ports = [cam_data["port"] for cam_data in active_cameras.values()]
    port = START_PORT
    while port in used_ports:
        port += 1
    return port

def start_camera_process(cam):
    cam_id = cam.get("camera_id")
    rtsp_url = cam.get("rtsp_url")
    location = cam.get("location", "Unknown")

    if not rtsp_url:
        return

    # Nếu đang chạy với đúng RTSP URL thì thôi
    if cam_id in active_cameras:
        if active_cameras[cam_id]["rtsp_url"] == rtsp_url:
            return
        else:
            print(f"[MANAGER] RTSP URL của '{cam_id}' thay đổi. Restarting...")
            stop_camera_process(cam_id)

    port = allocate_port()
    print(f"\n[MANAGER] 🟢 KHỞI ĐỘNG AI CHO CAMERA: {cam_id} | Port: {port}")
    
    # Lệnh gọi script AI
    # Dùng list arguments an toàn hơn
    cmd = [
        sys.executable, "-u", "src/05_realtime_demo.py",
        "--camera_id", str(cam_id),
        "--location", str(location),
        "--source", str(rtsp_url),
        "--backend", BACKEND_URL,
        "--stream_port", str(port),
        "--headless"
    ]
    
    # Ghi log ra file để debug xem tại sao crash
    log_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_manager.log"), "a")
    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    active_cameras[cam_id] = {
        "process": p,
        "port": port,
        "rtsp_url": rtsp_url,
        "log_file": log_file
    }

def stop_camera_process(cam_id):
    if cam_id in active_cameras:
        print(f"[MANAGER] 🔴 TẮT AI CỦA CAMERA: {cam_id}")
        p = active_cameras[cam_id]["process"]
        p.terminate()
        try:
            active_cameras[cam_id]["log_file"].close()
        except:
            pass
        del active_cameras[cam_id]
        
        # Gọi API cập nhật is_active = 0
        try:
            requests.put(f"{BACKEND_URL}/api/cameras/{cam_id}", json={"is_active": 0}, timeout=3)
        except:
            pass

def sync_cameras():
    db_cameras = get_cameras_from_db()
    db_cam_ids = [c["camera_id"] for c in db_cameras]

    # 1. Tắt các camera không còn trong DB (đã bị xóa qua app)
    for active_id in list(active_cameras.keys()):
        if active_id not in db_cam_ids:
            stop_camera_process(active_id)
            
    # 2. Khởi động các camera mới
    for c in db_cameras:
        start_camera_process(c)

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 CAMERA MANAGER CORE ĐANG CHẠY")
    print("Tự động lắng nghe thay đổi từ App...")
    print("=" * 50)
    
    try:
        while True:
            sync_cameras()
            
            # Kiểm tra xem có tiến trình nào crash tự nhiên không
            for cam_id in list(active_cameras.keys()):
                p = active_cameras[cam_id]["process"]
                if p.poll() is not None:
                    print(f"[MANAGER] ⚠️ Camera '{cam_id}' bị crash. Sẽ restart trong chu kỳ sau.")
                    try:
                        active_cameras[cam_id]["log_file"].close()
                    except:
                        pass
                    del active_cameras[cam_id]
                    try:
                        requests.put(f"{BACKEND_URL}/api/cameras/{cam_id}", json={"is_active": 0}, timeout=3)
                    except:
                        pass
                        
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[MANAGER] Đang tắt toàn bộ tiến trình AI...")
        for cam_id in list(active_cameras.keys()):
            stop_camera_process(cam_id)
        print("[MANAGER] Đã thoát an toàn.")
