"""Giao diện Streamlit cho các bộ lọc ảnh nghệ thuật."""

from PIL import Image, UnidentifiedImageError
import streamlit as st

from filters import FILTERS, FILTER_PARAMETERS, apply_filter
from image_io import decode_image, encode_image


st.set_page_config(page_title="Bộ lọc ảnh nghệ thuật", page_icon="🎨", layout="wide")
st.title("🎨 Hệ thống tạo ảnh nghệ thuật")
st.caption("Tải ảnh lên, chọn hiệu ứng và điều chỉnh cường độ để tạo tác phẩm của bạn.")


uploaded = st.file_uploader("Chọn ảnh JPG, PNG hoặc WebP", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Chọn một tệp ảnh để bắt đầu.")
    st.stop()

try:
    original = decode_image(uploaded.getvalue())
except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
    st.error(f"Không thể đọc ảnh: {exc}")
    st.stop()

with st.sidebar:
    st.header("Tùy chỉnh")
    selected = st.selectbox("Bộ lọc", list(FILTERS))
    strength = st.slider("Cường độ", 0, 100, 100, 5, format="%d%%")
    descriptions = {
        "Pencil sketch": "Nét chì trên nền giấy sáng, điều chỉnh được cỡ nét.",
        "Cartoon effect": "Mảng màu đơn giản, có đường viền mực đen.",
        "Oil painting effect": "Hai lớp nét cọ lớn và nhỏ theo đường nét vật thể.",
        "Emboss": "Chạm nổi màu xám với các cạnh sáng và tối theo hướng ánh sáng.",
        "Edge art": "Nét biên đen trên nền trắng, không tô bóng.",
        "Sepia": "Ảnh đơn sắc nâu với vùng sáng màu kem.",
        "Retro effect": "Phim màu cũ: màu phai, hạt nhẹ và tối góc.",
    }
    st.caption(descriptions[selected])
    st.caption("100%: hiệu ứng đầy đủ. Mức thấp hơn: pha với ảnh gốc.")
    parameters = {}
    with st.expander("Thông số riêng của bộ lọc", expanded=True):
        for key, (label, minimum, maximum, default, step, hint) in FILTER_PARAMETERS[selected].items():
            parameters[key] = st.slider(label, minimum, maximum, default, step,
                                        key=f"{selected}_{key}", help=hint)
    output_format = st.radio("Định dạng tải về", ["PNG", "JPEG"], horizontal=True)
    st.caption(f"Kích thước xử lý: {original.shape[1]} × {original.shape[0]} px")

with st.spinner("Đang xử lý ảnh..."):
    result = apply_filter(original, selected, strength, **parameters)

left, right = st.columns(2)
with left:
    st.subheader("Ảnh gốc")
    st.image(original, use_container_width=True)
with right:
    st.subheader("Ảnh sau xử lý")
    st.image(result, use_container_width=True)

extension = "png" if output_format == "PNG" else "jpg"
st.download_button(
    "⬇️ Tải ảnh kết quả",
    data=encode_image(result, output_format),
    file_name=f"art_{selected.lower().replace(' ', '_')}.{extension}",
    mime="image/png" if output_format == "PNG" else "image/jpeg",
)
