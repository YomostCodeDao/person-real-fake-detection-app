# ĐÂY LÀ HƯỚNG DẪN SỬ DỤNG SRC

# 1. Bật terminal(Chuyển sang git bash) và xem thử đang ở vị trí folder nào

- Nếu là /src thì không cần làm gì cả
- Nếu là /project thì dùng lệnh `cd src`

# 2. Gõ lệnh
**LẦN ĐẦU CHẠY THÌ DÙNG LỆNH 2.1**
- 2.1. `python -m venv venv` (Tạo môi trường ảo - Xuất hiện một folder venv trong src) 
- 2.2. `source venv/Scripts/activate ` (Kích hoạt môi trường ảo - Xuất hiện chữ .venv trên đầu mỗi dòng lệnh)
- 2.3. `pip install -r requirements.txt` (Cài các thư viện cần thiết - Nếu lỗi hãy coi lại vị trí file requirements.txt)
- 2.4. `python app.py` (Trong terminal (đã bật venv), gõ lệnh này )

**Nếu chạy thành công, sẽ hiện log Flask và bạn mở trình duyệt tại địa chỉ: ![http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

========

# Chức năng chính

## Upload Ảnh

## Chọn ảnh từ máy → hệ thống sẽ:

### Chạy YOLO để phát hiện đối tượng.

### Chạy EfficientNet‑B4 custom để phân loại REAL/FAKE.

### Hiển thị ảnh gốc, ảnh có bounding box, bảng kết quả YOLO, và kết luận REAL/FAKE.

## Webcam Realtime

## Mở webcam → hệ thống sẽ:

### Chỉ phát hiện nhãn person.

### Overlay kết quả REAL/FAKE + nhãn person trực tiếp lên video.
### Hiển thị thêm 2 dòng thông tin dưới video:
#### Kết quả REAL/FAKE
#### Thực thể phát hiện (person / không phát hiện)

========
# CẤU TRÚC THƯ MỤC
src/
│
├── app.py                          # Flask server chính
├── templates/
│ ├── index.html                    # Giao diện upload ảnh
│ └── webcam.html                   # Giao diện webcam realtime
├── static/
│ └── uploads/                      # Ảnh upload và kết quả YOLO(chưa làm)
├── requirements.txt                # Danh sách thư viện cần cài
├── best_face_fake_detector2.pth    # Checkpoint EfficientNet-B4 custom
└── yolo11n.pt                      # Model YOLO
