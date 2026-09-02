"""
=============================================================
config.py - Cấu hình Backend Fall Detection API
=============================================================
"""
import os
from pathlib import Path

# Thư mục gốc backend
BASE_DIR = Path(__file__).resolve().parent

# ─── Server ─────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# ─── Database ────────────────────────────────────────────────
DB_PATH = str(BASE_DIR / "fall_events.db")

# ─── Lưu ảnh ────────────────────────────────────────────────
IMAGES_DIR = str(BASE_DIR / "static" / "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ─── Firebase FCM ────────────────────────────────────────────
# Đặt đường dẫn tới file Service Account JSON của bạn
# Nếu chưa có Firebase, để trống "" → hệ thống sẽ bỏ qua phần FCM
FIREBASE_KEY_PATH = str(BASE_DIR / "firebase_key.json")
FCM_ENABLED = os.path.exists(FIREBASE_KEY_PATH)

# ─── Bảo mật ─────────────────────────────────────────────────
# Secret key để ký JWT token (thay bằng chuỗi ngẫu nhiên dài trên production)
SECRET_KEY = os.environ.get("SECRET_KEY", "fall-detection-secret-key-2026")

# Mật khẩu đăng nhập app mobile (đơn giản hóa cho demo - thay bằng DB user thật nếu cần)
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

# ─── Phân trang ──────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
