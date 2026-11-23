# Đặc trưng và nguồn tham khảo bộ lọc

## Phong cách được chọn trong dự án

Không có một ảnh đầu ra duy nhất được coi là chuẩn cho mỗi phong cách. Các bộ lọc dưới đây mô phỏng những đặc trưng quan sát được; tham số và cỡ nét là lựa chọn thiết kế của dự án, không phải thông số bắt buộc từ các nguồn.

| Hiệu ứng | Đặc trưng mục tiêu | Cách thực hiện |
| --- | --- | --- |
| Pencil sketch | Nét graphite, nền giấy sáng, bóng xám và gạch bóng nhẹ | Bilateral, Gaussian, color dodge, shading và gạch chéo theo độ tối. |
| Cartoon | Mảng màu đơn giản, nét mực đen rõ, giữ hình dáng chính | Bilateral, lượng tử hóa màu, Canny và trộn đường viền có lõi đen. |
| Oil painting | Nét cọ chồng lớp, mảng màu, chi tiết chọn lọc | Cọ elip theo tiếp tuyến biên, ba cỡ cọ và chiếu sáng nhẹ trên độ dày lớp sơn. Đây là một lựa chọn tranh có nét cọ; không phải mọi tranh sơn dầu đều có bề mặt nổi. |
| Emboss | Nền xám trung tính, cặp sáng/tối theo hướng để tạo cảm giác chạm nổi | Convolution tổng bằng 0, nền 128, nén tương phản tránh cháy sáng. |
| Edge art | Chỉ nét đen và nền trắng, không bóng xám hoặc màu | Khử nhiễu, Canny với ngưỡng theo phân bố gradient, đảo thành nét đen. |
| Sepia | Đơn sắc nâu, chuyển sáng tối liên tục, giữ chi tiết | Chuyển độ chói sang bảng màu nâu từ tối tới trắng kem. |
| Retro | Phim màu cũ: giảm bão hòa/độ tương phản, đen phai, lệch màu nhẹ, hạt đơn sắc và tối góc | Nâng mức đen, hạ mức trắng, chỉnh màu theo độ sáng, hạt xác định và vignette nhẹ. |

Ở cường độ dưới 100%, kết quả được pha với ảnh gốc; vì vậy Pencil, Edge art và Emboss có thể còn màu. Dùng 100% khi đánh giá đúng đặc trưng của từng hiệu ứng.

## Nguồn và ảnh minh họa

- [National Portrait Gallery: Drawing techniques and materials](https://www.npg.org.uk/schools-hub/drawing-techniques-and-materials): graphite, sắc độ, gạch bóng và bề mặt giấy.
- [National Gallery of Art: Explore the Basics of Drawing](https://www.nga.gov/educational-resources/explore-basics-drawing): contour, hatching, crosshatching và độ đậm nhạt.
- [GIMP Cartoon](https://docs.gimp.org/3.0/en/gimp-filter-cartoon.html), [ảnh minh họa](https://docs.gimp.org/3.0/en/images/filters/examples/artistic-taj-cartoon.jpg): hiệu ứng bút mực tô màu; quá nhiều vùng đen làm mất chi tiết.
- [GIMP Oilify](https://docs.gimp.org/3.0/en/gimp-filter-oilify.html), [ảnh minh họa](https://docs.gimp.org/3.0/en/images/filters/examples/artistic-taj-oilify.jpg): cỡ vùng cọ và số mức cường độ ảnh hưởng chi tiết.
- [GIMP Emboss](https://docs.gimp.org/3.0/en/gimp-filter-emboss.html), [ảnh minh họa](https://docs.gimp.org/3.0/en/images/filters/examples/distort-taj-emboss.jpg): độ nổi/lõm, hướng ánh sáng và phiên bản xám giống kim loại.
- [OpenCV Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html): khử nhiễu, gradient, nét mảnh và ngưỡng liên kết biên.
- [Adobe Sepia photography](https://www.adobe.com/creativecloud/photography/type/sepia-photography.html): ảnh đơn sắc nâu.
- [Adobe Vintage aesthetic](https://www.adobe.com/creativecloud/photography/discover/add-vintage-aesthetic.html): nhiều phong cách vintage; giảm tương phản, phai màu và hạt phim.

Các ảnh tham khảo trực tuyến chỉ được liên kết, không được đóng gói lại làm tài sản của dự án. Ảnh so sánh cục bộ dùng phần ảnh gốc trong ảnh chụp màn hình người dùng cung cấp.
