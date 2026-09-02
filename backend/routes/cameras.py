"""
=============================================================
routes/cameras.py - API endpoints quản lý camera
=============================================================
GET    /api/cameras          - Danh sách camera
POST   /api/cameras          - Đăng ký camera mới
PUT    /api/cameras/<id>     - Cập nhật thông tin camera
DELETE /api/cameras/<id>     - Xóa camera
POST   /api/fcm-tokens       - Đăng ký FCM token thiết bị mobile
=============================================================
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from database import get_conn, row_to_dict

cameras_bp = Blueprint("cameras", __name__, url_prefix="/api")


# ─── GET /api/cameras ─────────────────────────────────────────
@cameras_bp.route("/cameras", methods=["GET"])
def list_cameras():
    """Danh sách tất cả camera, hoặc lọc theo username."""
    username = request.args.get("username")
    
    with get_conn() as conn:
        if username:
            user = conn.execute("SELECT id, role FROM users WHERE username = ?", (username,)).fetchone()
            if user and user[1] != 'admin':
                # Nếu là user thường, chỉ lấy camera được cấp quyền
                rows = conn.execute(
                    """SELECT c.* FROM cameras c
                       JOIN user_cameras uc ON c.camera_id = uc.camera_id
                       WHERE uc.user_id = ? ORDER BY c.created_at DESC""",
                    (user[0],)
                ).fetchall()
                return jsonify([row_to_dict(r) for r in rows])
                
        # Mặc định lấy tất cả (cho Admin hoặc nếu không truyền username)
        rows = conn.execute(
            "SELECT * FROM cameras ORDER BY created_at DESC"
        ).fetchall()
    return jsonify([row_to_dict(r) for r in rows])


# ─── POST /api/cameras ────────────────────────────────────────
@cameras_bp.route("/cameras", methods=["POST"])
def create_camera():
    """
    Đăng ký camera mới.

    Body JSON:
        camera_id  (str, required)
        name       (str, optional)
        location   (str, optional)
        rtsp_url   (str, optional)
    """
    data = request.get_json(silent=True) or {}
    camera_id = data.get("camera_id", "").strip()
    if not camera_id:
        return jsonify({"error": "camera_id là bắt buộc"}), 400

    with get_conn() as conn:
        # Kiểm tra trùng
        existing = conn.execute(
            "SELECT id FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        if existing:
            return jsonify({"error": f"camera_id '{camera_id}' đã tồn tại"}), 409

        cur = conn.execute(
            """INSERT INTO cameras (camera_id, name, location, rtsp_url, mjpeg_url)
               VALUES (?, ?, ?, ?, ?)""",
            (camera_id, data.get("name", ""), data.get("location", ""), data.get("rtsp_url", ""), data.get("mjpeg_url", "")),
        )
        cam_id = cur.lastrowid
        row = conn.execute("SELECT * FROM cameras WHERE id = ?", (cam_id,)).fetchone()

    return jsonify(row_to_dict(row)), 201


# ─── PUT /api/cameras/<camera_id> ────────────────────────────
@cameras_bp.route("/cameras/<camera_id>", methods=["PUT"])
def update_camera(camera_id):
    """Cập nhật thông tin camera."""
    data = request.get_json(silent=True) or {}
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Không tìm thấy camera"}), 404

        conn.execute(
            """UPDATE cameras
               SET name=COALESCE(?,name), location=COALESCE(?,location),
                   rtsp_url=COALESCE(?,rtsp_url), mjpeg_url=COALESCE(?,mjpeg_url), is_active=COALESCE(?,is_active)
               WHERE camera_id=?""",
            (data.get("name"), data.get("location"), data.get("rtsp_url"), data.get("mjpeg_url"),
             data.get("is_active"), camera_id),
        )
        updated = conn.execute(
            "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()

    return jsonify(row_to_dict(updated))


# ─── DELETE /api/cameras/<camera_id> ─────────────────────────
@cameras_bp.route("/cameras/<camera_id>", methods=["DELETE"])
def delete_camera(camera_id):
    """Xóa camera."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Không tìm thấy camera"}), 404
        conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
    return jsonify({"success": True, "message": f"Đã xóa camera '{camera_id}'"})


# ─── POST /api/fcm-tokens ─────────────────────────────────────
@cameras_bp.route("/fcm-tokens", methods=["POST"])
def register_fcm_token():
    """
    Đăng ký FCM token của thiết bị mobile (gọi khi app khởi động).

    Body JSON:
        token        (str, required) - FCM registration token
        device_name  (str, optional) - Tên thiết bị để dễ phân biệt
    """
    data  = request.get_json(silent=True) or {}
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"error": "token là bắt buộc"}), 400

    device_name = data.get("device_name", "")
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO fcm_tokens (token, device_name)
               VALUES (?, ?)
               ON CONFLICT(token) DO UPDATE SET
                   device_name=excluded.device_name,
                   updated_at=datetime('now','localtime')""",
            (token, device_name),
        )
    return jsonify({"success": True, "message": "FCM token đã đăng ký"}), 201


# ─── GET /api/stats ───────────────────────────────────────────
@cameras_bp.route("/stats", methods=["GET"])
def get_stats():
    """Thống kê tổng quan cho màn hình Dashboard."""
    username = request.args.get("username")

    with get_conn() as conn:
        user_filter = ""
        user_params = []
        
        if username:
            user = conn.execute("SELECT id, role FROM users WHERE username = ?", (username,)).fetchone()
            if user and user[1] != 'admin':
                user_filter = "WHERE camera_id IN (SELECT camera_id FROM user_cameras WHERE user_id = ?)"
                user_params = [user[0]]

        # Query cho sự kiện
        total_events = conn.execute(f"SELECT COUNT(*) FROM events {user_filter}", user_params).fetchone()[0]
        
        today_filter = "WHERE DATE(timestamp) = DATE('now','localtime')"
        if user_filter:
            today_filter += f" AND camera_id IN (SELECT camera_id FROM user_cameras WHERE user_id = ?)"
        today_events = conn.execute(f"SELECT COUNT(*) FROM events {today_filter}", user_params).fetchone()[0]
        
        # Query cho camera
        total_cameras = conn.execute(f"SELECT COUNT(*) FROM cameras {user_filter}", user_params).fetchone()[0]
        
        active_filter = "WHERE is_active = 1"
        if user_filter:
            active_filter += f" AND camera_id IN (SELECT camera_id FROM user_cameras WHERE user_id = ?)"
        active_cameras = conn.execute(f"SELECT COUNT(*) FROM cameras {active_filter}", user_params).fetchone()[0]
        
        # 5 sự kiện gần nhất
        recent_rows = conn.execute(
            f"""SELECT id, camera_id, timestamp, confidence, location
               FROM events {user_filter} ORDER BY id DESC LIMIT 5""", user_params
        ).fetchall()

    return jsonify({
        "total_events":    total_events,
        "today_events":    today_events,
        "total_cameras":   total_cameras,
        "active_cameras":  active_cameras,
        "recent_events":   [row_to_dict(r) for r in recent_rows],
    })


# ─── POST /api/auth/login ─────────────────────────────────────
@cameras_bp.route("/auth/login", methods=["POST"])
def login():
    """
    Đăng nhập từ DB cho mobile app.
    Returns JWT-like token (dùng PyJWT).
    """
    import jwt
    import time
    from config import SECRET_KEY

    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password)
        ).fetchone()

    if not row:
        return jsonify({"error": "Sai tên đăng nhập hoặc mật khẩu"}), 401

    user = row_to_dict(row)
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400 * 30,  # Token có hiệu lực 30 ngày
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return jsonify({
        "token": token, 
        "username": username,
        "role": user.get("role", "user"),
        "fullname": user.get("fullname", ""),
        "phone": user.get("phone", ""),
        "relative_phone": user.get("relative_phone", "")
    })


# ─── POST /api/auth/register ──────────────────────────────────
@cameras_bp.route("/auth/register", methods=["POST"])
def register():
    """Đăng ký tài khoản mới."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    fullname = data.get("fullname", "").strip()
    phone = data.get("phone", "").strip()

    if not username or not password:
        return jsonify({"error": "Username và password là bắt buộc"}), 400

    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return jsonify({"error": "Tên đăng nhập đã tồn tại"}), 409

        conn.execute(
            "INSERT INTO users (username, password, fullname, phone) VALUES (?, ?, ?, ?)",
            (username, password, fullname, phone)
        )
    return jsonify({"success": True, "message": "Đăng ký tài khoản thành công"}), 201


# ─── PUT /api/auth/profile ────────────────────────────────────
@cameras_bp.route("/auth/profile", methods=["PUT"])
def update_profile():
    """Cập nhật thông tin cá nhân (JWT auth)."""
    import jwt
    from config import SECRET_KEY
    
    auth_header = request.headers.get("Authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Thiếu token xác thực"}), 401
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload["sub"]
    except Exception:
        return jsonify({"error": "Token không hợp lệ hoặc đã hết hạn"}), 401

    data = request.get_json(silent=True) or {}
    fullname = data.get("fullname", "").strip()
    phone = data.get("phone", "").strip()
    relative_phone = data.get("relative_phone", "").strip()
    password = data.get("password", "").strip()

    with get_conn() as conn:
        if password:
            conn.execute(
                "UPDATE users SET fullname=COALESCE(NULLIF(?,''), fullname), phone=COALESCE(NULLIF(?,''), phone), relative_phone=COALESCE(NULLIF(?,''), relative_phone), password=? WHERE username=?",
                (fullname, phone, relative_phone, password, username)
            )
        else:
            conn.execute(
                "UPDATE users SET fullname=COALESCE(NULLIF(?,''), fullname), phone=COALESCE(NULLIF(?,''), phone), relative_phone=COALESCE(NULLIF(?,''), relative_phone) WHERE username=?",
                (fullname, phone, relative_phone, username)
            )
            
        row = conn.execute("SELECT fullname, phone, relative_phone, username FROM users WHERE username=?", (username,)).fetchone()
        user = row_to_dict(row)

    return jsonify({"success": True, "message": "Cập nhật hồ sơ thành công", "user": user})


# ─── GET /api/users ───────────────────────────────────────────
@cameras_bp.route("/users", methods=["GET"])
def list_users():
    """Lấy danh sách tất cả người dùng (chỉ Admin mới dùng)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, fullname, phone, role, created_at FROM users ORDER BY id"
        ).fetchall()
    return jsonify([row_to_dict(r) for r in rows])


# ─── GET /api/users/<user_id>/cameras ─────────────────────────
@cameras_bp.route("/users/<int:user_id>/cameras", methods=["GET"])
def get_user_cameras(user_id):
    """Lấy danh sách camera mà user được phép xem."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.camera_id, c.name, c.location, c.mjpeg_url, c.is_active
               FROM user_cameras uc
               JOIN cameras c ON uc.camera_id = c.camera_id
               WHERE uc.user_id = ?""",
            (user_id,)
        ).fetchall()
    return jsonify([row_to_dict(r) for r in rows])


# ─── POST /api/users/<user_id>/cameras ────────────────────────
@cameras_bp.route("/users/<int:user_id>/cameras", methods=["POST"])
def grant_camera_access(user_id):
    """Admin cấp quyền xem camera cho user."""
    data = request.get_json(silent=True) or {}
    if "camera_ids" not in data:
        return jsonify({"error": "Danh sách camera_ids là bắt buộc"}), 400
    
    camera_ids = data.get("camera_ids", [])

    with get_conn() as conn:
        # Xóa toàn bộ quyền cũ rồi gán lại
        conn.execute("DELETE FROM user_cameras WHERE user_id = ?", (user_id,))
        for cam_id in camera_ids:
            conn.execute(
                "INSERT OR IGNORE INTO user_cameras (user_id, camera_id) VALUES (?, ?)",
                (user_id, cam_id)
            )
    return jsonify({"success": True, "message": f"Đã cấp quyền {len(camera_ids)} camera cho user #{user_id}"})


# ─── DELETE /api/users/<user_id>/cameras/<camera_id> ──────────
@cameras_bp.route("/users/<int:user_id>/cameras/<camera_id>", methods=["DELETE"])
def revoke_camera_access(user_id, camera_id):
    """Admin thu hồi quyền xem camera của user."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM user_cameras WHERE user_id = ? AND camera_id = ?",
            (user_id, camera_id)
        )
    return jsonify({"success": True, "message": f"Đã thu hồi quyền camera '{camera_id}' của user #{user_id}"})
