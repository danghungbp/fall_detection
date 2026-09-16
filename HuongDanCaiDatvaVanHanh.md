# HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH HỆ THỐNG (FALL DETECTION AI)

Tài liệu này hướng dẫn chi tiết các bước thiết lập môi trường, cài đặt thư viện và khởi chạy toàn bộ hệ thống bao gồm: Phân hệ nhận diện (Camera AI / Backend) và Phân hệ giám sát (Ứng dụng di động Flutter).

## 1. Yêu cầu hệ thống

**1.1. Phần cứng:**
- **CPU:** Intel Core i5 / AMD Ryzen 5 trở lên.
- **GPU (Khuyến nghị):** NVIDIA có hỗ trợ CUDA (GTX 1060 hoặc cao hơn) để đảm bảo tốc độ suy luận thời gian thực (FPS > 30).
- **RAM:** Tối thiểu 8 GB.
- **Thiết bị thu hình:** Webcam (độ phân giải tối thiểu 720p) hoặc Camera IP.
- **Thiết bị di động:** Smartphone chạy hệ điều hành Android 8.0 trở lên để cài đặt ứng dụng.

**1.2. Phần mềm:**
- **Hệ điều hành:** Windows 10/11, macOS, hoặc Ubuntu 20.04+.
- **Ngôn ngữ lập trình:** 
  - Python (Phiên bản 3.9 - 3.11).
  - Dart (Phiên bản 3.0+ thông qua Flutter SDK).
- **Công cụ phát triển:** Visual Studio Code (hoặc PyCharm, Android Studio).
- Git (để quản lý phiên bản).

---

## 2. Cài đặt môi trường Backend & AI (Python)

Phân hệ này chịu trách nhiệm chạy mô hình YOLOv8, xử lý luồng video và gửi dữ liệu cảnh báo.

**Bước 1: Tải mã nguồn**
Mở Terminal/Command Prompt và di chuyển đến thư mục làm việc, tải mã nguồn (hoặc sao chép thư mục dự án):
```bash
git clone <url_repo>
cd fall_detection
```

**Bước 2: Khởi tạo môi trường ảo (Virtual Environment)**
Việc sử dụng môi trường ảo giúp tránh xung đột thư viện giữa các dự án:
```bash
python -m venv venv

# Kích hoạt môi trường (trên Windows):
.\venv\Scripts\activate

# Kích hoạt môi trường (trên macOS/Linux):
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện cần thiết**
Tại thư mục gốc của dự án, chạy lệnh sau để cài đặt các thư viện lõi (YOLO, OpenCV, PyTorch...):
```bash
pip install -r requirements.txt
```
Di chuyển vào thư mục `backend` và cài đặt thêm các thư viện phục vụ API (Flask, Firebase Admin):
```bash
cd backend
pip install -r requirements.txt
cd ..
```

**Bước 4: Cấu hình khóa bảo mật Firebase**
Để hệ thống có thể đẩy thông báo (Push Notification) đến điện thoại, cần cấp quyền truy cập Firebase Admin SDK:
1. Đăng nhập vào [Firebase Console](https://console.firebase.google.com/).
2. Chọn dự án, đi tới **Project settings** > **Service accounts**.
3. Bấm **Generate new private key** để tải tệp JSON.
4. Đổi tên tệp vừa tải thành `firebase_key.json` và đặt vào thư mục `backend/`.

---

## 3. Cài đặt Ứng dụng Di động (Flutter)

Phân hệ này giúp người giám sát nhận cảnh báo và theo dõi camera.

**Bước 1: Cài đặt Flutter SDK**
1. Tải và cài đặt Flutter SDK từ trang chủ [flutter.dev](https://flutter.dev/docs/get-started/install).
2. Thêm đường dẫn `flutter/bin` vào biến môi trường (Environment Variables) của hệ điều hành.
3. Chạy lệnh sau để kiểm tra môi trường (đảm bảo đã cài Android Studio):
```bash
flutter doctor
```

**Bước 2: Cài đặt thư viện Dart**
Mở Terminal, di chuyển vào thư mục ứng dụng và lấy các gói phụ thuộc:
```bash
cd app_mobile
flutter pub get
```

**Bước 3: Biên dịch ứng dụng**
Để cài đặt ứng dụng lên thiết bị thật, kết nối điện thoại Android với máy tính (nhớ bật chế độ **USB Debugging**) và chạy lệnh:
```bash
flutter run
```
Hoặc để xuất ra tệp APK cài đặt:
```bash
flutter build apk --release
```
*(Tệp APK sẽ được lưu tại `app_mobile/build/app/outputs/flutter-apk/app-release.apk`)*.

---

## 4. Vận hành toàn bộ hệ thống

Sau khi đã hoàn tất cài đặt môi trường, hệ thống cần được khởi chạy theo đúng trình tự sau để đảm bảo kết nối xuyên suốt:

**Bước 1: Khởi chạy Backend Server (Flask)**
Mở Terminal thứ nhất, kích hoạt môi trường ảo (`.\venv\Scripts\activate`) và chạy máy chủ backend:
```bash
cd backend
python app.py
```
*Hệ thống Backend sẽ khởi động và lắng nghe các yêu cầu tại địa chỉ IP của máy (Ví dụ: `http://192.168.1.X:5000`).*

**Bước 2: Khởi chạy Phân hệ Nhận diện (Camera Manager)**
Mở Terminal thứ hai (vẫn trong môi trường ảo), quay lại thư mục gốc và khởi chạy hệ thống camera AI:
```bash
python camera_manager.py
```
*Lúc này, mô hình YOLOv8 sẽ được tải vào bộ nhớ (GPU/CPU) và bắt đầu phân tích luồng video. Bất cứ khi nào phát hiện hành vi vấp ngã (thông qua bộ lọc Heuristic), một tín hiệu sẽ được gửi qua Backend để đẩy thông báo FCM.*

**Bước 3: Giám sát trên Ứng dụng Di động**
1. Đảm bảo điện thoại và máy tính chạy Backend kết nối **cùng một mạng LAN/Wi-Fi**.
2. Mở ứng dụng **Fall Detection AI** trên điện thoại.
3. Trong cài đặt mạng của App, hãy nhập đúng địa chỉ IP LAN của máy tính đang chạy Backend.
4. Đảm bảo Tường lửa (Firewall) trên máy tính đã cho phép cổng `5000` (Backend) và cổng của Camera Stream (ví dụ `8080`, `8081`...).
5. Thêm camera bằng đường dẫn RTSP hoặc số `0` (cho Webcam máy tính) và bắt đầu giám sát trực tiếp.
