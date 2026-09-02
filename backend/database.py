"""
=============================================================
database.py - Khởi tạo SQLite và các hàm helper
=============================================================
"""
import sqlite3
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    """Lấy connection tới SQLite, trả về rows dạng dict."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tạo bảng nếu chưa tồn tại."""
    with get_conn() as conn:
        conn.executescript("""
            -- Bảng lịch sử sự kiện té ngã
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                confidence  REAL,
                image_path  TEXT,
                location    TEXT,
                fcm_sent    INTEGER DEFAULT 0,
                is_false_alarm INTEGER DEFAULT 0,
                created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
            );

            -- Bảng danh sách camera
            CREATE TABLE IF NOT EXISTS cameras (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT    UNIQUE NOT NULL,
                name        TEXT,
                location    TEXT,
                rtsp_url    TEXT,
                mjpeg_url   TEXT,
                is_active   INTEGER DEFAULT 1,
                created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
            );

            -- Bảng FCM tokens (lưu token của từng thiết bị mobile)
            CREATE TABLE IF NOT EXISTS fcm_tokens (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                token       TEXT    UNIQUE NOT NULL,
                device_name TEXT,
                created_at  TEXT    DEFAULT (datetime('now', 'localtime')),
                updated_at  TEXT    DEFAULT (datetime('now', 'localtime'))
            );

            -- Bảng người dùng (Users)
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                fullname    TEXT,
                phone       TEXT,
                relative_phone TEXT,
                role        TEXT    DEFAULT 'user',
                created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
            );

            -- Bảng phân quyền Camera cho User (Many-to-Many)
            CREATE TABLE IF NOT EXISTS user_cameras (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                camera_id   TEXT    NOT NULL,
                created_at  TEXT    DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, camera_id)
            );
        """)
        
        # Thêm cột mjpeg_url nếu chưa có (với DB cũ)
        try:
            conn.execute("ALTER TABLE cameras ADD COLUMN mjpeg_url TEXT")
        except sqlite3.OperationalError:
            pass
        
        # Thêm cột role nếu chưa có (với DB cũ)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        except sqlite3.OperationalError:
            pass
        
        # Thêm user admin mặc định nếu bảng users chưa có ai
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count == 0:
            conn.execute(
                "INSERT INTO users (username, password, fullname, phone, role) VALUES (?, ?, ?, ?, ?)",
                ("admin", "admin123", "Thiều Đăng Hùng", "0987654321", "admin")
            )
        else:
            # Đảm bảo tài khoản admin luôn có role='admin'
            conn.execute("UPDATE users SET role='admin' WHERE username='admin'")
    print(f"  ✅ Database khởi tạo tại: {DB_PATH}")


def row_to_dict(row) -> dict:
    """Chuyển sqlite3.Row thành dict thường."""
    return dict(row) if row else None


if __name__ == "__main__":
    init_db()
    print("Database ready!")
