# Fall Detection AI - Hệ Thống Phát Hiện Té Ngã Thông Minh

## 1. Giới thiệu tổng quan (Overview)
**Fall Detection AI** là một hệ thống toàn diện kết hợp giữa Trí tuệ nhân tạo (AI/Computer Vision), Backend Server và Ứng dụng Di động (Mobile App) nhằm mục đích giám sát, nhận diện hành vi con người thông qua camera (RTSP/Webcam) và cảnh báo té ngã theo thời gian thực (Real-time).
Dự án được xây dựng đặc biệt để hỗ trợ giám sát người già, trẻ nhỏ hoặc bệnh nhân tại nhà, bệnh viện, hoặc viện dưỡng lão.

---

## 2. Kiến trúc hệ thống (System Architecture)
Hệ thống bao gồm 3 phân hệ chính hoạt động song song và giao tiếp liên tục với nhau:
1. **AI Processing Module (Python / YOLOv8):** Xử lý luồng video trực tiếp từ camera, nhận diện hành vi (Bending, Lying, Sitting, Standing, Walking) và đặc biệt là **Té ngã (Fall)**.
2. **Backend Server (Flask / SQLite):** Đóng vai trò cầu nối, quản lý dữ liệu (người dùng, camera, lịch sử báo động ngã) và đẩy thông báo (Push Notification) thông qua Firebase Cloud Messaging (FCM).
3. **Mobile App (Flutter / Dart):** Giao diện dành cho người dùng cuối (End-user) để quản lý camera, xem video trực tiếp (Live Stream), nhận cảnh báo đẩy khi có người ngã và xem lại lịch sử.

---

## 3. Chi tiết các Phân hệ (Modules Breakdown)

### 3.1. Trí tuệ nhân tạo & Xử lý hình ảnh (AI Vision Module)
- **Công nghệ:** Python, OpenCV, YOLOv8 (Ultralytics).
- **Quy trình hoạt động:**
  - `camera_manager.py`: Là trình quản lý vòng đời camera (Process Manager). Nó thường xuyên kết nối với Backend, lấy danh sách các camera đang được kích hoạt và tự động sinh ra các tiến trình con (`05_realtime_demo.py`) để chạy AI độc lập cho từng camera.
  - `05_realtime_demo.py`: Module chính xử lý luồng (RTSP/Webcam). Model YOLOv8 nhận diện 6 class (Fall, Walking, Sitting, Standing, Lying, Bending).
- **Bộ lọc thông minh (Smart Heuristics):** Để loại bỏ báo động giả (như nằm trên giường/võng, hoặc AI nhận nhầm tủ/xe máy thành người từ góc trên cao), hệ thống áp dụng thuật toán lọc tinh chỉnh:
  1. Hủy bỏ khung nhận diện (Bounding box) chiếm quá >60% hoặc <5% diện tích khung hình để tránh lỗi "ảo giác" của AI.
  2. Để được tính là "Té ngã" (Fall), tư thế phải nằm bẹp (Aspect ratio > 1.2), vị trí ở sát sàn nhà (Bottom ratio > 0.75) và độ tin cậy của AI (Confidence) phải đạt tối thiểu 80%.
  3. Cần 3 khung hình liên tiếp đạt đủ điều kiện để kích hoạt sự kiện ngã nhằm tránh nhiễu tạm thời.
- **Phát trực tiếp (Live Stream):** Mỗi camera sẽ mở một Mini-Flask Server nội bộ (chạy mjpeg) trên một port riêng lẻ để truyền luồng hình ảnh đã vẽ AI sang Backend và Mobile App với độ trễ cực thấp.

### 3.2. Backend Server (Flask)
- **Công nghệ:** Python (Flask, Flask-RESTful), SQLite (Database), Firebase Admin SDK.
- **Cấu trúc:**
  - `app.py`: Điểm khởi chạy của server (mặc định Port 5000).
  - `database.py`: Quản lý SQLite Database gồm các bảng `users`, `cameras`, `user_cameras` (Phân quyền), `events` (Lịch sử ngã) và `fcm_tokens`.
  - `routes/`: Chứa các API endpoints.
    - `/api/auth`: Đăng ký, đăng nhập (JWT Token), cập nhật hồ sơ.
    - `/api/cameras`: Thêm, xóa, sửa, lấy danh sách camera (Hỗ trợ phân quyền user).
    - `/api/events`: Lịch sử té ngã, xóa lịch sử, đánh dấu báo động giả (False alarm).
  - `services/fcm.py`: Xử lý gửi Push Notification đến Mobile App thông qua Firebase mỗi khi AI gửi tín hiệu có người ngã.

### 3.3. Mobile App (Flutter)
- **Công nghệ:** Flutter (Dart), Firebase Cloud Messaging (FCM).
- **Giao diện (UI/UX):** Chế độ tối (Dark mode) sang trọng, mang phong cách "Fall Detection AI".
- **Chức năng chính:**
  - **Auth:** Đăng nhập, Đăng ký, Quản lý tài khoản (hỗ trợ lưu JWT Token để tự động đăng nhập).
  - **Camera Management:** Thêm camera dễ dàng thông qua ID và địa chỉ luồng RTSP (VD: Tapo, Hikvision, Ezviz).
  - **Live Monitoring (Dashboard):** Xem trực tiếp tất cả các camera đang hoạt động (chạy qua luồng MJPEG), cùng với thống kê số lượng nhận diện (Người đi bộ, đang ngồi, đang đứng, v.v.).
  - **Event History:** Danh sách lịch sử các vụ té ngã (Kèm hình ảnh cắt ra từ lúc ngã, thời gian, tên camera). Tính năng cho phép đánh dấu "Báo động giả" để bỏ qua.
  - **Push Notification:** Bất cứ khi nào AI quét thấy người ngã, App (kể cả khi đang tắt ngầm) sẽ nhận được thông báo nảy lên màn hình điện thoại cảnh báo khẩn cấp.

---

## 4. Hướng dẫn vận hành hệ thống (How to Run)

### Bước 1: Khởi động Backend Server
```bash
cd backend
python app.py
```
> Server sẽ chạy tại địa chỉ `http://[IP_MAY_TINH]:5000`. Backend chịu trách nhiệm điều phối toàn bộ data.

### Bước 2: Khởi động AI Camera Manager
```bash
python camera_manager.py
```
> Script này sẽ tự động liên hệ Backend, lấy danh sách các Camera đang có và tự động chạy luồng AI (YOLO) để nhận diện và phát stream MJPEG nội bộ.

### Bước 3: Cài đặt & Chạy Mobile App
```bash
cd app_mobile
flutter clean
flutter pub get
flutter build apk --release
```
> Cài đặt file `app_mobile/build/app/outputs/flutter-apk/app-release.apk` lên điện thoại Android.
> Đăng nhập tài khoản, bấm [+] để thêm Camera bằng thông số RTSP (Ví dụ: `rtsp://admin:pass@192.168.1.100/stream2`).
> Hệ thống sẽ tự động liên kết và hình ảnh AI sẽ hiện lên màn hình điện thoại.

---

## 5. Tổng kết
Dự án đã giải quyết trọn vẹn bài toán từ khâu "Computer Vision (Nhận diện)" -> "Data Processing (Xử lý & Lưu trữ)" -> "User Interaction (Ứng dụng cảnh báo)", đem lại một hệ thống khép kín, độ trễ thấp và có tính ứng dụng cực cao trong đời sống thực tế.
