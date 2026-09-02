"""
=============================================================
app.py - Flask Backend Entry Point
Fall Detection System - Thiều Đăng Hùng (22050015)
=============================================================
Cách chạy:
    cd backend
    python app.py

API sẽ chạy tại: http://0.0.0.0:5000
=============================================================
"""
import os
import sys

# Fix encoding trên Windows (hỗ trợ emoji trong terminal)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Đảm bảo import đúng module trong cùng thư mục
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from config import HOST, PORT, DEBUG, IMAGES_DIR, FCM_ENABLED
from database import init_db
from routes.events import events_bp
from routes.cameras import cameras_bp


def create_app() -> Flask:
    """Factory function tạo Flask app."""
    app = Flask(__name__, static_folder="static")
    CORS(app)  # Cho phép CORS (cần thiết khi Flutter gọi API)

    # Đăng ký blueprints
    app.register_blueprint(events_bp)
    app.register_blueprint(cameras_bp)

    # ─── Health Check ─────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status":      "ok",
            "service":     "Fall Detection Backend API",
            "version":     "1.0.0",
            "fcm_enabled": FCM_ENABLED,
        })

    # ─── Serve ảnh tĩnh ───────────────────────────────────────
    @app.route("/static/images/<filename>")
    def serve_image(filename):
        return send_from_directory(IMAGES_DIR, filename)

    # ─── Error handlers ───────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint không tồn tại"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Lỗi server nội bộ"}), 500

    return app


# ─── Startup ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 FALL DETECTION BACKEND API")
    print("=" * 60)

    # Khởi tạo database
    print("\n📦 Khởi tạo Database...")
    init_db()

    # Khởi tạo Firebase (nếu có key)
    if FCM_ENABLED:
        print("\n🔥 Khởi tạo Firebase Admin SDK...")
        from services.fcm import init_firebase
        init_firebase()
    else:
        print("\n⚠️  Firebase key không tìm thấy → FCM bị tắt")
        print("   → Đặt file firebase_key.json vào thư mục backend/ để bật FCM")

    print(f"\n🌐 API đang chạy tại: http://{HOST}:{PORT}")
    print(f"   Health check: http://localhost:{PORT}/api/health")
    print(f"   Events API:   http://localhost:{PORT}/api/events")
    print("=" * 60)
    print("   Nhấn Ctrl+C để dừng server")
    print("=" * 60)

    app = create_app()
    app.run(host=HOST, port=PORT, debug=DEBUG)
