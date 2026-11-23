# Hệ thống tạo ảnh nghệ thuật bằng bộ lọc

Xem [đặc trưng và nguồn tham khảo của bảy bộ lọc](./THAM_KHAO_BO_LOC.md). Đặt cường độ 100% khi so sánh phong cách; các mức thấp hơn pha lại màu và chi tiết ảnh gốc.

Ứng dụng Python xử lý ảnh với bảy hiệu ứng: Pencil sketch, Cartoon, Oil painting, Emboss, Edge art, Sepia và Retro. Xem [kế hoạch chi tiết](./KE_HOACH.md) để biết yêu cầu và thuật toán.

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

Mở **Thông số riêng của bộ lọc** trên thanh bên. Các giá trị mặc định giữ thiết lập cũ; mỗi ảnh có thể cần điều chỉnh:

| Hiệu ứng | Gợi ý khi ảnh chưa đẹp |
| --- | --- |
| Pencil | Giảm độ đậm bóng nếu nền xám; tăng cỡ nét nếu nét quá mảnh. |
| Cartoon | Viền che mặt: giảm độ đậm viền xuống 0.4–0.6. Màu quá vụn: tăng mức đơn giản hóa. |
| Oil painting | Mặt mất chi tiết: cỡ cọ 0.5–0.8. Muốn mảng cọ rõ hơn: 1.2–1.6. |
| Emboss | Cạnh gắt: độ sâu 0.5–0.8; thử xoay hướng ánh sáng. |
| Edge art | Nhiều nét vụn: giảm độ nhạy xuống 0.6–0.8. Thiếu nét: tăng lên 1.2–1.5. |
| Sepia | Muốn màu nâu nhẹ, gần ảnh đen trắng: độ ấm 0.5–0.8. |
| Retro | Ảnh nhỏ: hạt phim 0–1.5; góc tối quá: tối góc 0–0.08. |

Các khoảng trên là điểm bắt đầu để thử, không đảm bảo tối ưu cho mọi ảnh. Cường độ chung vẫn dùng để pha kết quả với ảnh gốc.

Hiệu ứng Oil painting vẽ nhiều lớp nét cọ theo hướng đường biên, với nét nhỏ tại vùng nhiều chi tiết và độ nổi màu nhẹ. Chọn cường độ 100% để thấy rõ nét cọ, hoặc 70–85% để giữ thêm chi tiết ảnh gốc.

```powershell
python -m unittest discover -s tests -v
```

## Cấu trúc

- `filters.py`: bảy thuật toán và hàm áp dụng cường độ.
- `app.py`: giao diện Streamlit.
- `image_io.py`: đọc, chuẩn hóa và xuất ảnh.
- `tests/`: kiểm thử đầu ra, đầu vào lỗi và giới hạn kích thước.
- `KE_HOACH.md`: kế hoạch và tiêu chí nghiệm thu.
