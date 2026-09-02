# Fall Detection REST API Documentation

## Base URL
`http://localhost:5000/api`

## Endpoints

### 1. Health Check
Kiểm tra trạng thái server.
- **URL**: `/health`
- **Method**: `GET`
- **Response** (200 OK):
```json
{
  "status": "ok",
  "version": "1.0.0",
  "service": "Fall Detection Backend API",
  "fcm_enabled": true
}
```

### 2. Events

#### Ghi nhận sự kiện té ngã mới
Gửi cảnh báo từ camera lên server.
- **URL**: `/events`
- **Method**: `POST`
- **Body**:
```json
{
  "camera_id": "cam_01",
  "confidence": 0.95,
  "location": "Phòng khách",
  "image_base64": "base64_string_here..." (tùy chọn)
}
```
- **Response** (201 Created):
```json
{
  "success": true,
  "message": "Sự kiện đã được ghi nhận",
  "event_id": 12
}
```

#### Lấy danh sách sự kiện
Lấy lịch sử té ngã cho Mobile App.
- **URL**: `/events?page=1&page_size=20`
- **Method**: `GET`
- **Response** (200 OK):
```json
{
  "total": 12,
  "pages": 1,
  "page": 1,
  "page_size": 20,
  "events": [
    {
      "id": 12,
      "camera_id": "cam_01",
      "timestamp": "2026-08-02T15:00:00+07:00",
      "confidence": 0.95,
      "location": "Phòng khách",
      "fcm_sent": 1
    }
  ]
}
```

### 3. Authentication
Đăng nhập ứng dụng di động.
- **URL**: `/auth/login`
- **Method**: `POST`
- **Body**:
```json
{
  "username": "admin",
  "password": "admin123"
}
```
- **Response** (200 OK):
```json
{
  "token": "jwt_token_string",
  "username": "admin"
}
```

### 4. FCM Notifications
Firebase Cloud Messaging service tự động kích hoạt khi có sự kiện POST vào `/events`. Để sử dụng, cần cấu hình file `firebase_key.json` tại thư mục root backend.
