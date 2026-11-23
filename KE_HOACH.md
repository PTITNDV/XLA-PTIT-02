# Kế hoạch đề tài: Hệ thống tạo ảnh nghệ thuật bằng bộ lọc

## 1. Mục tiêu và phạm vi

Xây dựng ứng dụng web chạy trên máy cá nhân. Người dùng tải ảnh JPG, PNG hoặc WebP, chọn một trong bảy hiệu ứng, chỉnh cường độ, so sánh ảnh gốc với kết quả và tải ảnh kết quả dạng PNG/JPG. Ứng dụng xử lý ảnh trên máy chạy chương trình, không cần dịch vụ bên ngoài.

**Bảy hiệu ứng:** Pencil sketch, Cartoon, Oil painting, Emboss, Edge art, Sepia, Retro.

**Đầu ra:** mã nguồn Python, giao diện Streamlit, tài liệu cách chạy, kiểm thử tự động cho thuật toán và dữ liệu đầu vào.

## 2. Yêu cầu chức năng

| Mã | Yêu cầu | Tiêu chí hoàn thành |
| --- | --- | --- |
| F01 | Tải ảnh JPG/PNG/WebP | Đọc ảnh màu, ảnh xám, ảnh có alpha; chuyển về RGB. |
| F02 | Chọn bảy bộ lọc | Mỗi bộ lọc cho kết quả RGB có cùng kích thước ảnh đầu vào đã chuẩn hóa. |
| F03 | Chỉnh cường độ | Thanh trượt 0–100%; 0% trả lại ảnh gốc, 100% cho hiệu ứng đầy đủ. |
| F04 | Xem trước và so sánh | Hiển thị ảnh gốc và ảnh đã xử lý cạnh nhau. |
| F05 | Lưu kết quả | Tải ảnh dưới dạng PNG hoặc JPEG. |
| F06 | Giới hạn tài nguyên | Ảnh lớn được thu nhỏ theo cạnh dài tối đa 1600 px; ảnh lỗi được báo rõ. |

## 3. Thiết kế kỹ thuật

Luồng xử lý: `tệp tải lên → Pillow giải mã/chuẩn hóa RGB → OpenCV xử lý → pha trộn theo cường độ → Streamlit hiển thị/xuất ảnh`.

| Bộ lọc | Phương pháp chính | Mô tả |
| --- | --- | --- |
| Pencil sketch | Grayscale, Gaussian convolution, color dodge | Đảo ảnh xám, làm mờ rồi chia ảnh để tạo nét chì sáng. |
| Cartoon | Bilateral filter, color quantization, Canny | Làm phẳng màu nhưng giữ biên, giảm số mức màu và phủ đường viền đen. |
| Oil painting | Bilateral, gradient, layered brush strokes | Trong vùng lân cận, chọn nhóm cường độ sáng phổ biến và dùng màu trung bình của nhóm. |
| Emboss | Convolution 3×3 | Lọc hướng ánh sáng rồi cộng nền xám 128. |
| Edge art | Canny edge detection | Tạo nét đen trên nền trắng, sau khi làm mờ khử nhiễu. |
| Sepia | Color transform | Nhân ma trận màu nâu cổ điển. |
| Retro | Sepia, vignette, noise, blending | Làm phai màu, tối góc ảnh và thêm hạt phim xác định. |

Tất cả hiệu ứng nhận và trả về mảng RGB `uint8`. Thuật toán nằm trong `filters.py`, đọc/xuất ảnh trong `image_io.py`, còn `app.py` phụ trách giao diện.

## 4. Các bước thực hiện

1. Tạo môi trường Python và khai báo thư viện NumPy, OpenCV, Pillow, Streamlit.
2. Viết hàm đọc ảnh, chuẩn hóa màu và giới hạn kích thước.
3. Cài đặt bảy hiệu ứng; thêm tham số cường độ chung.
4. Tạo giao diện tải ảnh, chọn hiệu ứng, xem trước và tải kết quả.
5. Kiểm thử kích thước, kiểu dữ liệu, cường độ biên, tính ổn định của từng hiệu ứng và trường hợp ảnh lỗi.
6. Chạy ứng dụng và rà soát thủ công trên ảnh thực tế.

## 5. Kiểm thử và tiêu chí nghiệm thu

- `python -m unittest discover -s tests -v` chạy thành công.
- Bảy hiệu ứng không phát sinh lỗi với ảnh màu nhỏ, ảnh xám và ảnh có alpha.
- Kết quả có kích thước và kiểu dữ liệu hợp lệ; mức 0% bằng ảnh gốc.
- Người dùng có thể xuất PNG/JPEG từ giao diện.
- Ảnh lớn được thu nhỏ trước khi xử lý, tránh tăng bộ nhớ và thời gian quá mức.

## 6. Giới hạn và hướng phát triển

Chất lượng nghệ thuật phụ thuộc nội dung ảnh; các bộ lọc cổ điển không thay thế mô hình sinh ảnh. Bộ lọc tranh sơn dầu xử lý nhiều lân cận nên chậm hơn các hiệu ứng khác. Có thể bổ sung tùy chỉnh tham số riêng cho từng bộ lọc, xử lý hàng loạt và lưu preset trong phiên bản tiếp theo.
