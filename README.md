# Hệ thống tạo ảnh nghệ thuật bằng bộ lọc

Xem [đặc trưng và nguồn tham khảo của bảy bộ lọc](./THAM_KHAO_BO_LOC.md). Đặt cường độ 100% khi so sánh phong cách; các mức thấp hơn pha lại màu và chi tiết ảnh gốc.

Ứng dụng Python xử lý ảnh với bảy hiệu ứng: Pencil sketch, Cartoon, Oil painting, Emboss, Edge art, Sepia và Retro.

## Pipeline xử lý

Các hiệu ứng dùng những luồng xử lý sau. Ba hiệu ứng Pencil, Oil painting và Edge art đã được rút gọn để dễ học và trình bày:

| Hiệu ứng | Pipeline hiện tại |
| --- | --- |
| Pencil | Xám → bilateral → Gaussian tạo nền → chia ảnh → RGB |
| Cartoon | Bilateral → lượng tử hóa màu và Canny → phủ viền mực |
| Oil painting | Bilateral → gradient → hướng cọ → vẽ hai lớp cọ |
| Emboss | Xám → Gaussian → kernel định hướng → nén đáp ứng → nền xám |
| Edge art | Xám → Gaussian → chọn ngưỡng → Canny → đảo màu → RGB |
| Sepia | Xám → nội suy bảng màu nâu → điều chỉnh độ ấm |
| Retro | Giảm bão hòa → phai màu → lệch sắc → tối góc và hạt phim |

Cường độ chung pha ảnh hiệu ứng với ảnh gốc. Canny giữ đủ Sobel, làm mảnh, ngưỡng kép và liên kết biên. Code của mỗi bước, ví dụ tính toán và hình trung gian được giải thích trong [báo cáo](./BAO_CAO_BTL.docx), mục 4.6–4.14.

## Thuật toán tự cài đặt

Phần xử lý ảnh không dùng OpenCV. [numpy_ops.py](./numpy_ops.py) tự cài đặt Gaussian, bilateral, Sobel, Canny (làm mảnh và liên kết biên), tương quan với kernel, resize theo diện tích, vẽ elip và blending bằng NumPy. [filters.py](./filters.py) kết hợp chúng thành bảy hiệu ứng. Pillow đọc/lưu ảnh, xử lý EXIF và alpha; Streamlit tạo giao diện.

Kết quả có thể khác bản OpenCV do trọng số, xử lý biên và làm tròn. Sơn dầu và Canny có thể chậm trên ảnh lớn.

### Đọc phần giải thích hàm

Mỗi hàm trong [numpy_ops.py](./numpy_ops.py), [filters.py](./filters.py) và [image_io.py](./image_io.py) có chú thích `#` tiếng Việt ngắn ngay phía trước, tóm tắt chức năng của hàm. Riêng `paint_ellipse` sửa canvas tại chỗ.

Để theo dõi luồng, bắt đầu từ `apply_filter` trong [filters.py](./filters.py), xem hàm hiệu ứng tương ứng, rồi đọc các phép toán nền được gọi trong [numpy_ops.py](./numpy_ops.py). `canny` được tách thành Sobel, làm mảnh biên và liên kết biên để thuận tiện đối chiếu thuật toán.

## Cài đặt

Yêu cầu Python 3.11 trở lên. Trong thư mục dự án:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên macOS/Linux, thay lệnh kích hoạt bằng `source .venv/bin/activate`.

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Mở địa chỉ mà Streamlit in ra (thường là `http://localhost:8501`). Tải ảnh, chọn bộ lọc trong thanh bên, điều chỉnh cường độ và nhấn **Tải ảnh kết quả**. Ảnh lớn được thu nhỏ về cạnh dài tối đa 1600 px trước khi xử lý.

Có thể dùng `art_filter_smoke.png` trong thư mục dự án để thử nhanh các hiệu ứng.

## Kiểm thử

### Tinh chỉnh theo từng ảnh

Mở **Thông số riêng của bộ lọc** trên thanh bên. Mỗi ảnh có thể cần điều chỉnh:

| Hiệu ứng | Gợi ý khi ảnh chưa đẹp |
| --- | --- |
| Pencil | Điều chỉnh cỡ nét; giảm cường độ chung nếu muốn giữ thêm chi tiết nguồn. |
| Cartoon | Viền che mặt: giảm độ đậm viền xuống 0.4–0.6. Màu quá vụn: tăng mức đơn giản hóa. |
| Oil painting | Mặt mất chi tiết: cỡ cọ 0.5–0.8. Muốn mảng cọ rõ hơn: 1.2–1.6. |
| Emboss | Cạnh gắt: độ sâu 0.5–0.8; thử xoay hướng ánh sáng. |
| Edge art | Nhiều nét vụn: giảm độ nhạy xuống 0.6–0.8. Thiếu nét: tăng lên 1.2–1.5. |
| Sepia | Muốn màu nâu nhẹ, gần ảnh đen trắng: độ ấm 0.5–0.8. |
| Retro | Ảnh nhỏ: hạt phim 0–1.5; góc tối quá: tối góc 0–0.08. |

Các khoảng trên là điểm bắt đầu để thử, không đảm bảo tối ưu cho mọi ảnh. Cường độ chung vẫn dùng để pha kết quả với ảnh gốc.

Hiệu ứng Oil painting vẽ hai lớp nét cọ theo hướng đường biên, với nét nhỏ tại vùng nhiều chi tiết. Chọn cường độ 100% để thấy rõ nét cọ, hoặc 70–85% để giữ thêm chi tiết ảnh gốc.

```powershell
python -m unittest discover -s tests -v
```

## Cấu trúc

- `filters.py`: bảy thuật toán và hàm áp dụng cường độ.
- `app.py`: giao diện Streamlit.
- `image_io.py`: đọc, chuẩn hóa và xuất ảnh.
- `numpy_ops.py`: các thuật toán xử lý ảnh tự cài đặt bằng NumPy.
- `tests/`: kiểm thử đầu ra, đầu vào lỗi và giới hạn kích thước.
- `demo/`: ảnh đầu vào dùng để chạy thử.
- `THAM_KHAO_BO_LOC.md`: đặc trưng và nguồn tham khảo các bộ lọc.

## Báo cáo

[Báo cáo bài tập lớn](./BAO_CAO_BTL.docx) là bản chính duy nhất, đã đồng bộ với pipeline hiện tại. Báo cáo gồm đủ 10 mục theo yêu cầu môn học, giải thích từng bộ lọc cho người mới, 25 khối code trích từ mã nguồn, 11 hình minh họa và kết quả thực nghiệm của bản rút gọn. Chỉnh sửa trực tiếp trong Word khi cần điền thông tin trang bìa.
