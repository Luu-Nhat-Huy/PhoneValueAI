
import sys
from pathlib import Path

import streamlit as st


# =====================================================
# 1. CAU HINH DUONG DAN VA IMPORT
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "app"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.predict import (
    predict_phone,
    get_available_models,
    load_metadata,
)
from database import save_valuation


# Dataset lưu giá theo nghìn đồng.
# 1 triệu đồng = 1.000 đơn vị trong dataset.
PRICE_SCALE = 1000


def format_vnd(value, decimals=0):
    """Định dạng giá trị thành VND."""
    try:
        amount = float(value) * PRICE_SCALE
        return f"{amount:,.{decimals}f} ₫"
    except (TypeError, ValueError):
        return "N/A"


# =====================================================
# 2. CAU HINH TRANG
# =====================================================

st.title("📱 Định giá điện thoại")
st.caption(
    "Sử dụng trí tuệ nhân tạo để ước tính giá bán lại "
    "của điện thoại đã qua sử dụng."
)

st.info(
    "Đơn vị giá trên giao diện là đồng Việt Nam (VND). "
    "Giá mua ban đầu được nhập theo triệu đồng."
)

st.divider()


# =====================================================
# 3. DANH SACH THUONG HIEU VA MODEL DIEN THOAI
# =====================================================

MODELS_BY_BRAND = {
    "Apple": [
        "iPhone 11",
        "iPhone 12",
        "iPhone 13",
        "iPhone 14",
        "iPhone 15",
    ],
    "Google": [
        "Pixel 6",
        "Pixel 7",
        "Pixel 8",
    ],
    "OnePlus": [
        "OnePlus 9",
        "OnePlus 10",
        "OnePlus 11",
        "OnePlus Nord 3",
    ],
    "Realme": [
        "Realme GT",
        "Realme Narzo 60",
    ],
    "Samsung": [
        "Samsung Galaxy S21",
        "Samsung Galaxy S22",
        "Samsung Galaxy S23",
        "Samsung Galaxy A54",
    ],
    "Vivo": [
        "Vivo V27",
        "Vivo X90",
    ],
    "Xiaomi": [
        "Xiaomi Poco X5",
        "Xiaomi Mi 11",
        "Xiaomi Redmi Note 12",
    ],
}

CONDITIONS = [
    "Excellent",
    "Good",
    "Fair",
    "Poor",
]

CITY_TIERS = [
    "Tier1",
    "Tier2",
    "Tier3",
]

SELLER_TYPES = [
    "Individual",
    "Store",
]


# =====================================================
# 4. LUA CHON MO HINH AI
# =====================================================

st.subheader("1. Lựa chọn mô hình AI")

try:
    available_models = get_available_models()
    metadata = load_metadata()

    if not available_models:
        st.error(
            "Không tìm thấy mô hình AI. "
            "Vui lòng kiểm tra lại thư mục models."
        )
        st.stop()

    default_model = metadata.get("selected_default_model")

    if default_model not in available_models:
        default_model = available_models[0]

    default_index = available_models.index(default_model)

    selected_model = st.selectbox(
        "Chọn thuật toán định giá",
        options=available_models,
        index=default_index,
        help="Chọn một trong ba mô hình Machine Learning đã huấn luyện.",
    )

    st.info(f"**Mô hình đang chọn:** {selected_model}")

except Exception as error:
    st.error(f"Không thể tải danh sách mô hình: {error}")
    st.stop()


# =====================================================
# 5. CHON THUONG HIEU VA MODEL DIEN THOAI
# =====================================================

st.subheader("2. Thông tin cơ bản")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Thương hiệu",
        options=list(MODELS_BY_BRAND.keys()),
        key="valuation_brand",
    )

with col2:
    model = st.selectbox(
        "Model điện thoại",
        options=MODELS_BY_BRAND[brand],
        key="valuation_model",
    )

os_type = "iOS" if brand == "Apple" else "Android"

st.info(f"**Hệ điều hành được xác định tự động:** {os_type}")


# =====================================================
# 6. FORM NHAP THONG TIN DIEN THOAI
# =====================================================

with st.form("phone_valuation_form"):

    # -------------------------------------------------
    # A. THONG TIN PHAN CUNG
    # -------------------------------------------------

    st.subheader("3. Thông số phần cứng")

    col1, col2, col3 = st.columns(3)

    with col1:
        ram_gb = st.number_input(
            "RAM (GB)",
            min_value=1,
            value=8,
            step=1,
        )

    with col2:
        storage_gb = st.number_input(
            "Bộ nhớ trong (GB)",
            min_value=1,
            value=256,
            step=1,
        )

    with col3:
        screen_size_inches = st.number_input(
            "Kích thước màn hình (inch)",
            min_value=0.1,
            value=6.5,
            step=0.1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        battery_capacity = st.number_input(
            "Dung lượng pin (mAh)",
            min_value=1,
            value=5000,
            step=100,
        )

    with col2:
        processor_score = st.number_input(
            "Điểm hiệu năng CPU",
            min_value=40,
            max_value=100,
            value=70,
            step=1,
            help="Thang điểm dataset: từ 40 đến 100.",
        )

    with col3:
        camera_score = st.number_input(
            "Điểm camera",
            min_value=40,
            max_value=100,
            value=80,
            step=1,
            help="Thang điểm dataset: từ 40 đến 100.",
        )

    with st.expander("📊 Hướng dẫn đánh giá điểm CPU và Camera"):
        st.markdown(
            """
            Điểm CPU và Camera sử dụng thang điểm từ **40 đến 100**.

            | Điểm | Mức đánh giá |
            |---|---|
            | 40–54 | Cơ bản |
            | 55–69 | Trung bình |
            | 70–84 | Khá |
            | 85–94 | Cao |
            | 95–100 | Rất cao |

            **Lưu ý:** Đây là mức phân loại tham khảo để hỗ trợ
            nhập thông số, không phải bảng benchmark chính thức
            của từng dòng điện thoại.
            """
        )

    has_5g = st.selectbox(
        "Hỗ trợ mạng 5G",
        options=["Có", "Không"],
    )

    # -------------------------------------------------
    # B. GIA VA THOI GIAN
    # -------------------------------------------------

    st.divider()
    st.subheader("4. Giá bán và thời gian sử dụng")

    col1, col2 = st.columns(2)

    with col1:
        original_price_million = st.number_input(
            "Giá mua ban đầu (triệu đồng)",
            min_value=10.0,
            max_value=150.0,
            value=50.0,
            step=1.0,
            help=(
                "Ví dụ: nhập 50 tương ứng 50.000.000 VND. "
                "Ứng dụng tự chuyển đổi sang đơn vị của mô hình."
            ),
        )

    with col2:
        release_year = st.number_input(
            "Năm ra mắt",
            min_value=2019,
            max_value=2025,
            value=2023,
            step=1,
        )

    col1, col2 = st.columns(2)

    with col1:
        purchase_year = st.number_input(
            "Năm mua điện thoại",
            min_value=2019,
            max_value=2025,
            value=2024,
            step=1,
        )

    with col2:
        age_months = st.number_input(
            "Tuổi điện thoại (tháng)",
            min_value=0,
            max_value=71,
            value=24,
            step=1,
        )

    usage_hours_per_day = st.number_input(
        "Thời gian sử dụng trung bình mỗi ngày (giờ)",
        min_value=1.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
    )

    # -------------------------------------------------
    # C. TINH TRANG DIEN THOAI
    # -------------------------------------------------

    st.divider()
    st.subheader("5. Tình trạng điện thoại")

    condition = st.selectbox(
        "Tình trạng tổng thể",
        options=CONDITIONS,
        index=2,
    )

    battery_health = st.slider(
        "Độ chai pin / sức khỏe pin (%)",
        min_value=55,
        max_value=100,
        value=90,
        step=1,
    )

    col1, col2 = st.columns(2)

    with col1:
        screen_cracked = st.checkbox(
            "Màn hình bị nứt",
            value=False,
        )

        body_damage = st.checkbox(
            "Thân máy bị hư hỏng",
            value=False,
        )

        repair_history = st.checkbox(
            "Điện thoại đã từng sửa chữa",
            value=False,
        )

    with col2:
        water_damage = st.checkbox(
            "Điện thoại từng bị vào nước",
            value=False,
        )

        box_available = st.checkbox(
            "Còn hộp điện thoại",
            value=True,
        )

        charger_available = st.checkbox(
            "Còn bộ sạc",
            value=True,
        )

    warranty_remaining_months = st.number_input(
        "Thời gian bảo hành còn lại (tháng)",
        min_value=0,
        value=0,
        step=1,
    )

    # -------------------------------------------------
    # D. THONG TIN THI TRUONG
    # -------------------------------------------------

    st.divider()
    st.subheader("6. Thông tin thị trường")

    col1, col2 = st.columns(2)

    with col1:
        city_tier = st.selectbox(
            "Phân loại khu vực",
            options=CITY_TIERS,
            help=(
                "Tier1, Tier2 và Tier3 là các nhóm khu vực "
                "được sử dụng trong dataset."
            ),
        )

    with col2:
        seller_type = st.selectbox(
            "Loại người bán",
            options=SELLER_TYPES,
        )

    market_demand_score = st.number_input(
        "Điểm nhu cầu thị trường",
        min_value=0,
        value=70,
        step=1,
    )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 ĐỊNH GIÁ ĐIỆN THOẠI",
        use_container_width=True,
        type="primary",
    )


# =====================================================
# 7. XU LY DU DOAN
# =====================================================

if submitted:

    if purchase_year < release_year:
        st.error(
            "Năm mua điện thoại không được nhỏ hơn năm ra mắt."
        )

    elif original_price_million <= 0:
        st.error(
            "Giá mua ban đầu phải lớn hơn 0."
        )

    elif not 40 <= processor_score <= 100:
        st.error(
            "Điểm hiệu năng CPU phải nằm trong khoảng 40–100."
        )

    elif not 40 <= camera_score <= 100:
        st.error(
            "Điểm camera phải nằm trong khoảng 40–100."
        )

    else:

        # -------------------------------------------------
        # CHUYEN GIA TRIEN DONG SANG NGHIN DONG
        # -------------------------------------------------

        # Ví dụ:
        # 50 triệu đồng -> 50.000 nghìn đồng.
        original_price = int(
            round(original_price_million * PRICE_SCALE)
        )

        # -------------------------------------------------
        # TAO DICTIONARY 27 DAC TRUNG
        # -------------------------------------------------

        phone_data = {
            "brand": brand,
            "model": model,
            "os_type": os_type,
            "condition": condition,
            "city_tier": city_tier,
            "seller_type": seller_type,

            "release_year": int(release_year),
            "ram_gb": int(ram_gb),
            "storage_gb": int(storage_gb),
            "screen_size_inches": float(screen_size_inches),
            "battery_capacity": int(battery_capacity),
            "processor_score": int(processor_score),
            "camera_score": int(camera_score),
            "has_5g": int(has_5g == "Có"),
            "original_price": original_price,
            "purchase_year": int(purchase_year),
            "age_months": int(age_months),
            "usage_hours_per_day": float(usage_hours_per_day),
            "battery_health": int(battery_health),
            "screen_cracked": int(screen_cracked),
            "body_damage": int(body_damage),
            "repair_history": int(repair_history),
            "water_damage": int(water_damage),
            "warranty_remaining_months": int(
                warranty_remaining_months
            ),
            "box_available": int(box_available),
            "charger_available": int(charger_available),
            "market_demand_score": int(market_demand_score),
        }

        # -------------------------------------------------
        # GOI MO HINH DU DOAN
        # -------------------------------------------------

        try:

            with st.spinner(
                f"Mô hình {selected_model} đang phân tích điện thoại..."
            ):

                # Giá dự đoán vẫn là đơn vị nghìn đồng.
                predicted_price = float(
                    predict_phone(
                        phone_data,
                        model_name=selected_model,
                    )
                )

                # Lưu dữ liệu gốc theo đơn vị dataset.
                # Không nhân 1.000 trước khi lưu database.
                save_valuation(
                    phone_data,
                    predicted_price,
                )

            # -------------------------------------------------
            # CHUYEN SANG VND DE HIEN THI
            # -------------------------------------------------

            original_price_vnd = original_price * PRICE_SCALE
            predicted_price_vnd = predicted_price * PRICE_SCALE

            difference_vnd = (
                predicted_price - original_price
            ) * PRICE_SCALE

            percentage = (
                (predicted_price - original_price)
                / original_price
                * 100
            )

            remaining_percentage = (
                predicted_price / original_price * 100
            )

            # -------------------------------------------------
            # HIEN THI KET QUA
            # -------------------------------------------------

            st.success("Định giá điện thoại thành công!")

            st.divider()
            st.subheader("7. Kết quả định giá")

            st.markdown(
                """
                <style>
                .valuation-result {
                    background: linear-gradient(
                        120deg,
                        #102A43,
                        #176B87
                    );
                    padding: 28px;
                    border-radius: 18px;
                    color: white;
                    text-align: center;
                    margin: 10px 0 20px 0;
                }

                .valuation-result .label {
                    font-size: 15px;
                    opacity: 0.85;
                }

                .valuation-result .price {
                    font-size: 36px;
                    font-weight: 800;
                    margin-top: 10px;
                }

                .valuation-result .device {
                    font-size: 16px;
                    margin-top: 8px;
                    opacity: 0.9;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="valuation-result">
                    <div class="label">
                        GIÁ BÁN LẠI ƯỚC TÍNH
                    </div>
                    <div class="price">
                        {predicted_price_vnd:,.0f} ₫
                    </div>
                    <div class="device">
                        {brand} - {model}
                    </div>
                    <div class="device">
                        Mô hình: {selected_model}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Giá mua ban đầu",
                    f"{original_price_vnd:,.0f} ₫",
                )

            with col2:
                st.metric(
                    "Giá trị ước tính",
                    f"{predicted_price_vnd:,.0f} ₫",
                )

            # -------------------------------------------------
            # SO SANH GIA
            # -------------------------------------------------

            st.subheader("8. So sánh giá")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Chênh lệch so với giá mua",
                    f"{difference_vnd:+,.0f} ₫",
                    f"{percentage:+.2f}%",
                )

            with col2:
                st.metric(
                    "Tỷ lệ giá trị còn lại",
                    f"{remaining_percentage:.2f}%",
                )

            # -------------------------------------------------
            # THONG TIN CHI TIET
            # -------------------------------------------------

            st.divider()

            with st.expander(
                "Xem toàn bộ thông tin điện thoại"
            ):
                st.json(phone_data)

                st.caption(
                    "Lưu ý: original_price trong dữ liệu chi tiết "
                    "được lưu theo đơn vị nghìn đồng để tương thích "
                    "với dataset và mô hình."
                )

            st.caption(
                "Giá dự đoán là kết quả ước tính của mô hình AI, "
                "không phải giá giao dịch được bảo đảm trên thị trường."
            )

        except Exception as error:

            st.error(
                f"Không thể định giá điện thoại: {error}"
            )

            st.exception(error)
