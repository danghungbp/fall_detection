"""
=============================================================
routes/events.py - API endpoints quản lý sự kiện té ngã
=============================================================
POST   /api/events          - Nhận sự kiện từ module phát hiện
GET    /api/events          - Danh sách sự kiện (phân trang, filter)
GET    /api/events/<id>     - Chi tiết 1 sự kiện
DELETE /api/events/<id>     - Xóa sự kiện
=============================================================
"""
import os
import base64
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

# Import từ thư mục cha (khi chạy qua app.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_conn, row_to_dict
from config import IMAGES_DIR, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

events_bp = Blueprint("events", __name__, url_prefix="/api/events")


# ─── POST /api/events ────────────────────────────────────────
@events_bp.route("", methods=["POST"])
def create_event():
    """
    Nhận sự kiện té ngã từ module Python phát hiện.

    Body JSON:
        camera_id     (str, required)
        confidence    (float, optional)
        image_base64  (str, optional)  - Ảnh chụp tại thời điểm té ngã
        location      (str, optional)
        timestamp     (str, optional)  - ISO 8601, mặc định = now
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Yêu cầu body JSON"}), 400

    camera_id = data.get("camera_id", "").strip()
    if not camera_id:
        return jsonify({"error": "camera_id là bắt buộc"}), 400

    confidence = float(data.get("confidence", 0.0))
    location   = data.get("location", "")
    timestamp  = data.get("timestamp") or datetime.now().astimezone().isoformat()

    # Lưu ảnh (nếu có)
    image_path = None
    image_b64  = data.get("image_base64", "")
    if image_b64:
        try:
            # Bỏ prefix "data:image/jpeg;base64," nếu có
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            img_bytes  = base64.b64decode(image_b64)
            fname      = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(IMAGES_DIR, fname)
            with open(image_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            current_app.logger.warning(f"Lưu ảnh thất bại: {e}")
            image_path = None

    # Lưu vào DB
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO events (camera_id, timestamp, confidence, image_path, location)
               VALUES (?, ?, ?, ?, ?)""",
            (camera_id, timestamp, confidence, image_path, location),
        )
        event_id = cur.lastrowid

    # Gửi FCM push notification (async trong thread riêng)
    _send_fcm_async(event_id, camera_id, location, confidence)

    return jsonify({
        "success":  True,
        "event_id": event_id,
        "message":  "Sự kiện đã được ghi nhận",
    }), 201


# ─── GET /api/events ─────────────────────────────────────────
@events_bp.route("", methods=["GET"])
def list_events():
    """
    Lấy danh sách sự kiện té ngã.

    Query params:
        page        (int, default=1)
        page_size   (int, default=20, max=100)
        camera_id   (str, optional) - Filter theo camera
        date        (str, optional) - Filter theo ngày "YYYY-MM-DD"
    """
    page      = max(1, int(request.args.get("page", 1)))
    page_size = min(MAX_PAGE_SIZE, max(1, int(request.args.get("page_size", DEFAULT_PAGE_SIZE))))
    camera_id = request.args.get("camera_id", "").strip()
    date_str  = request.args.get("date", "").strip()
    username  = request.args.get("username", "").strip()

    # Build câu truy vấn
    conditions = []
    params     = []

    if camera_id:
        conditions.append("camera_id = ?")
        params.append(camera_id)
    if date_str:
        conditions.append("DATE(timestamp) = ?")
        params.append(date_str)
        
    with get_conn() as conn:
        if username:
            user = conn.execute("SELECT id, role FROM users WHERE username = ?", (username,)).fetchone()
            if user and user[1] != 'admin':
                conditions.append("camera_id IN (SELECT camera_id FROM user_cameras WHERE user_id = ?)")
                params.append(user[0])

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        offset = (page - 1) * page_size
        total = conn.execute(
            f"SELECT COUNT(*) FROM events {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""SELECT id, camera_id, timestamp, confidence, image_path, location, fcm_sent, is_false_alarm, created_at
                FROM events {where}
                ORDER BY id DESC
                LIMIT ? OFFSET ?""",
            params + [page_size, offset],
        ).fetchall()

    events = []
    for row in rows:
        d = row_to_dict(row)
        # Trả về URL ảnh thay vì đường dẫn tuyệt đối
        if d.get("image_path"):
            fname = os.path.basename(d["image_path"])
            d["image_url"] = f"/static/images/{fname}"
        d.pop("image_path", None)
        events.append(d)

    return jsonify({
        "events":    events,
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size,
    })


# ─── GET /api/events/<id> ─────────────────────────────────────
@events_bp.route("/<int:event_id>", methods=["GET"])
def get_event(event_id):
    """Lấy chi tiết 1 sự kiện (kèm ảnh dưới dạng base64)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()

    if not row:
        return jsonify({"error": "Không tìm thấy sự kiện"}), 404

    d = row_to_dict(row)
    # Đính kèm ảnh base64 nếu tồn tại
    img_path = d.get("image_path")
    if img_path and os.path.exists(img_path):
        with open(img_path, "rb") as f:
            d["image_base64"] = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        d["image_url"] = f"/static/images/{os.path.basename(img_path)}"
    d.pop("image_path", None)

    return jsonify(d)


# ─── DELETE /api/events/<id> ──────────────────────────────────
@events_bp.route("/<int:event_id>", methods=["DELETE"])
def delete_event(event_id):
    """Xóa 1 sự kiện và file ảnh liên quan."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT image_path FROM events WHERE id = ?", (event_id,)
        ).fetchone()

        if not row:
            return jsonify({"error": "Không tìm thấy sự kiện"}), 404

        # Xóa file ảnh nếu có
        img_path = row["image_path"]
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))

    return jsonify({"success": True, "message": f"Đã xóa sự kiện #{event_id}"})


# ─── DELETE /api/events/batch ─────────────────────────────────
@events_bp.route("/batch", methods=["DELETE"])
def delete_events_batch():
    """Xóa nhiều sự kiện cùng lúc."""
    data = request.get_json(silent=True)
    if not data or "ids" not in data:
        return jsonify({"error": "Vui lòng cung cấp danh sách ids"}), 400

    ids = data["ids"]
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "Danh sách ids không hợp lệ hoặc rỗng"}), 400

    with get_conn() as conn:
        # Lấy đường dẫn ảnh để xóa file vật lý
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT image_path FROM events WHERE id IN ({placeholders})", ids
        ).fetchall()

        for row in rows:
            img_path = row["image_path"]
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception as e:
                    current_app.logger.warning(f"Lỗi khi xóa ảnh {img_path}: {e}")

        # Xóa các record trong DB
        conn.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids)

    return jsonify({"success": True, "message": f"Đã xóa {len(ids)} sự kiện"})


# ─── PUT /api/events/<id>/false-alarm ─────────────────────────
@events_bp.route("/<int:event_id>/false-alarm", methods=["PUT"])
def toggle_false_alarm(event_id):
    """Đánh dấu/bỏ đánh dấu báo động giả."""
    data = request.get_json(silent=True) or {}
    # Mặc định là 1 (True) nếu không gửi body
    is_false_alarm = int(data.get("is_false_alarm", 1))

    with get_conn() as conn:
        row = conn.execute("SELECT id FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row:
            return jsonify({"error": "Không tìm thấy sự kiện"}), 404
        
        conn.execute("UPDATE events SET is_false_alarm = ? WHERE id = ?", (is_false_alarm, event_id))
    
    return jsonify({"success": True, "message": "Đã cập nhật trạng thái báo động giả", "is_false_alarm": bool(is_false_alarm)})


# ─── Helper: Gửi FCM trong background thread ──────────────────
def _send_fcm_async(event_id: int, camera_id: str, location: str, confidence: float):
    """Gửi FCM notification trong thread riêng để không block response."""
    import threading

    def _task():
        try:
            from services.fcm import send_fall_notification
            # Lấy tất cả FCM tokens từ DB
            with get_conn() as conn:
                rows = conn.execute("SELECT token FROM fcm_tokens").fetchall()
            tokens = [r["token"] for r in rows]

            if not tokens:
                return

            result = send_fall_notification(camera_id, location, confidence, event_id, tokens)

            # Cập nhật trạng thái fcm_sent
            if result.get("success_count", 0) > 0:
                with get_conn() as conn:
                    conn.execute(
                        "UPDATE events SET fcm_sent = 1 WHERE id = ?", (event_id,)
                    )
        except Exception as e:
            print(f"  ❌ Lỗi gửi FCM async: {e}")

    threading.Thread(target=_task, daemon=True).start()
