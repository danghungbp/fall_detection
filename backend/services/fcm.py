"""
=============================================================
services/fcm.py - Gửi Firebase Cloud Messaging push notification
=============================================================
Hướng dẫn setup Firebase:
  1. Vào https://console.firebase.google.com → Tạo project mới
  2. Project Settings → Service Accounts → Generate new private key
  3. Lưu file JSON tải về thành backend/firebase_key.json
  4. Tạo Flutter app trong cùng project → tải google-services.json
=============================================================
"""
import os
import sys

# Thêm thư mục cha vào sys.path khi chạy trực tiếp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FIREBASE_KEY_PATH, FCM_ENABLED

# Chỉ import firebase_admin nếu có key file
_firebase_initialized = False
_fcm_available = False

if FCM_ENABLED:
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging
        _firebase_initialized = True
        _fcm_available = True
    except ImportError:
        print("  ⚠️  firebase-admin chưa cài. Chạy: pip install firebase-admin")
        _fcm_available = False


def init_firebase():
    """Khởi tạo Firebase Admin SDK."""
    global _firebase_initialized
    if not _fcm_available:
        return False
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
            print("  ✅ Firebase Admin SDK khởi tạo thành công!")
            return True
        except Exception as e:
            print(f"  ❌ Lỗi khởi tạo Firebase: {e}")
            return False
    return True


def send_fall_notification(camera_id: str, location: str, confidence: float,
                           event_id: int, fcm_tokens: list[str]) -> dict:
    """
    Gửi push notification khi phát hiện té ngã.

    Args:
        camera_id:    ID camera phát hiện
        location:     Vị trí mô tả (vd: "Phòng ngủ tầng 2")
        confidence:   Độ tự tin (0.0 - 1.0)
        event_id:     ID sự kiện trong DB (dùng để điều hướng trong app)
        fcm_tokens:   Danh sách FCM token của các thiết bị cần nhận

    Returns:
        dict với success_count, failure_count, errors
    """
    if not _fcm_available or not _firebase_initialized:
        print("  ⚠️  FCM không khả dụng - bỏ qua gửi notification")
        return {"success_count": 0, "failure_count": 0, "skipped": True}

    if not fcm_tokens:
        print("  ⚠️  Không có FCM token nào để gửi")
        return {"success_count": 0, "failure_count": 0, "no_tokens": True}

    # Nội dung notification
    title = "🚨 Phát hiện té ngã!"
    body  = f"Camera: {location or camera_id} | Độ tự tin: {confidence*100:.0f}%"

    success_count = 0
    failure_count = 0
    errors = []

    for token in fcm_tokens:
        try:
            msg = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data={
                    "event_id":   str(event_id),
                    "camera_id":  camera_id,
                    "location":   location or "",
                    "confidence": str(round(confidence, 4)),
                    "type":       "fall_detected",
                },
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        icon="ic_notification",
                        color="#FF0000",
                        sound="default",
                        channel_id="fall_alerts",
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(sound="default", badge=1)
                    )
                ),
                token=token,
            )
            messaging.send(msg)
            success_count += 1
        except Exception as e:
            failure_count += 1
            errors.append(str(e))
            print(f"  ❌ Gửi FCM thất bại (token={token[:20]}...): {e}")

    print(f"  📲 FCM: {success_count} thành công, {failure_count} thất bại")
    return {"success_count": success_count, "failure_count": failure_count, "errors": errors}
