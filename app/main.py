import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# 1. CAU HINH DUONG DAN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "phone_pages"
MODEL_DIR = ROOT_DIR / "models"

METADATA_PATH = MODEL_DIR / "model_metadata.json"

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

for path in (ROOT_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =====================================================
# 2. CAU HINH DON VI GIA
# =====================================================

# Dataset luu gia theo don vi nghin dong.
# 1 don vi dataset = 1.000 VND.
PRICE_SCALE = 1000


def format_number(value, decimals=2):
    try:
        if value is None or pd.isna(value):
            return "Chưa có dữ liệu"

        return f"{float(value):,.{decimals}f}"

    except (TypeError, ValueError):
        return "Chưa có dữ liệu"


def format_vnd(value, decimals=0):
    try:
        if value is None or pd.isna(value):
            return "N/A"

        amount = float(value) * PRICE_SCALE
        return f"{amount:,.{decimals}f} ₫"

    except (TypeError, ValueError):
        return "N/A"


# =====================================================
# 3. CAU HINH STREAMLIT
# =====================================================

st.set_page_config(
    page_title="PhoneValue AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 4. IMPORT VA KHOI TAO DATABASE
# =====================================================

from database import initialize_database, get_all_valuations

try:
    initialize_database()
    DATABASE_READY = True
    DATABASE_ERROR = None

except Exception as error:
    DATABASE_READY = False
    DATABASE_ERROR = str(error)


# =====================================================
# 5. DOC THONG TIN MO HINH
# =====================================================

EXPECTED_MODELS = [
    "Linear Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
]


@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        return None

    try:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        if not isinstance(metadata, dict):
            return None

        return metadata

    except (json.JSONDecodeError, OSError):
        return None


def get_model_information(metadata):
    """
    Lay thong tin 3 mo hinh tu metadata.
    Khong su dung mo hinh cu.
    """

    if not metadata:
        return [], None, None

    models = metadata.get("models", [])

    if not isinstance(models, list):
        return [], None, None

    valid_models = []

    for item in models:

        if not isinstance(item, dict):
            continue

        model_name = item.get("model_name")
        model_file = item.get("model_file")

        if not model_name or not model_file:
            continue

        if Path(model_file).name != model_file:
            continue

        model_path = MODEL_DIR / model_file

        valid_models.append({
            **item,
            "file_exists": model_path.is_file(),
        })

    default_name = metadata.get("selected_default_model")

    default_model = next(
        (
            item
            for item in valid_models
            if item["model_name"] == default_name
            and item["file_exists"]
        ),
        None,
    )

    return valid_models, default_name, default_model


# =====================================================
# 6. CSS GIAO DIEN
# =====================================================

st.markdown(
    """
    <style>
    :root {
        --primary: #2563EB;
        --primary-dark: #1D4ED8;
        --navy: #0F172A;
        --text: #172554;
        --muted: #64748B;
        --surface: #FFFFFF;
        --border: #E2E8F0;
    }

    .stApp {
        background: #F4F7FC;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 252, 0.92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0F1F3D 0%,
            #142D52 100%
        );
        border-right: 1px solid #263F63;
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B8C8DF;
    }

    [data-testid="stSidebar"] hr {
        border-color: #314968;
    }

    [data-testid="stSidebar"] [data-testid="stPageLink"] {
        border-radius: 10px;
    }

    .main-title {
        color: #102A56;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .main-subtitle {
        color: #64748B;
        font-size: 15px;
        margin-bottom: 24px;
    }

    .hero {
        padding: 34px;
        border-radius: 22px;
        background: linear-gradient(
            120deg,
            #102A56 0%,
            #155EAA 60%,
            #2563EB 100%
        );
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px rgba(30, 64, 175, 0.15);
    }

    .hero h1 {
        color: white !important;
        font-size: 40px;
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #DBEAFE !important;
        font-size: 16px;
        margin: 5px 0;
    }

    .section-heading {
        color: #15365C;
        font-size: 22px;
        font-weight: 750;
        margin: 12px 0 16px 0;
    }

    .feature-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        min-height: 155px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
    }

    .feature-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .feature-title {
        color: #123B68;
        font-size: 22px;
        font-weight: 800;
        margin: 10px 0;
    }

    .feature-description {
        color: #64748B;
        font-size: 14px;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #123B68;
        font-weight: 750;
    }

    h1, h2, h3, h4 {
        color: #15365C;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    .footer-note {
        color: #94A3B8;
        font-size: 12px;
        text-align: center;
        padding: 18px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 7. TRANG TONG QUAN
# =====================================================

def dashboard():

    metadata = load_metadata()

    models, default_name, default_model = get_model_information(
        metadata
    )

    model_names = [
        item["model_name"]
        for item in models
    ]

    all_models_ready = (
        len(model_names) == 3
        and all(item["file_exists"] for item in models)
        and set(model_names) == set(EXPECTED_MODELS)
    )

    st.markdown(
        """
        <div class="hero">
            <h1>📱 PhoneValue AI</h1>
            <p>
                Hệ thống ứng dụng Machine Learning để ước tính
                giá trị điện thoại đã qua sử dụng.
            </p>
            <p>
                Phân tích thông số thiết bị, dự đoán giá bán lại
                và theo dõi lịch sử định giá trên cùng một nền tảng.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------
    # TRANG THAI HE THONG
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Tình trạng hệ thống</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mô hình AI",
            "3 mô hình sẵn sàng"
            if all_models_ready
            else "Chưa đầy đủ",
        )

    with col2:
        st.metric(
            "Cơ sở dữ liệu",
            "Đã kết nối"
            if DATABASE_READY
            else "Lỗi kết nối",
        )

    with col3:
        st.metric(
            "Dataset",
            "Đã tìm thấy"
            if DATA_PATH.exists()
            else "Chưa tìm thấy",
        )

    if not DATABASE_READY:

        st.error(
            f"Không thể khởi tạo cơ sở dữ liệu: {DATABASE_ERROR}"
        )

    if not all_models_ready:

        st.warning(
            "Danh sách mô hình chưa đầy đủ hoặc có file bị thiếu. "
            "Hãy kiểm tra models/model_metadata.json và thư mục models."
        )

        for item in models:

            status = (
                "Đã có file"
                if item["file_exists"]
                else "Thiếu file"
            )

            st.write(f"- {item['model_name']}: {status}")

    # -------------------------------------------------
    # HIEU SUAT MO HINH
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Hiệu suất mô hình</div>',
        unsafe_allow_html=True,
    )

    if metadata and default_model:

        st.success(
            f"Mô hình mặc định đang sử dụng: **{default_name}**"
        )

        metrics = default_model

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Thuật toán mặc định",
                default_model["model_name"],
            )

        with col2:
            st.metric(
                "R² Score",
                format_number(metrics.get("r2"), 4),
            )

        with col3:
            st.metric(
                "MAE",
                format_vnd(metrics.get("mae")),
            )

        with col4:
            st.metric(
                "RMSE",
                format_vnd(metrics.get("rmse")),
            )

        st.markdown("#### So sánh cả 3 mô hình")

        comparison_rows = []

        for item in models:

            comparison_rows.append({
                "Mô hình": item["model_name"],
                "MAE (VNĐ)": (
                    float(item["mae"]) * PRICE_SCALE
                    if item.get("mae") is not None
                    else None
                ),
                "RMSE (VNĐ)": (
                    float(item["rmse"]) * PRICE_SCALE
                    if item.get("rmse") is not None
                    else None
                ),
                "R² Score": item.get("r2"),
                "Thời gian huấn luyện (giây)": item.get(
                    "training_time_seconds"
                ),
                "Trạng thái": (
                    "Sẵn sàng"
                    if item["file_exists"]
                    else "Thiếu file"
                ),
            })

        st.dataframe(
            comparison_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE (VNĐ)": st.column_config.NumberColumn(
                    format="%,.0f ₫"
                ),
                "RMSE (VNĐ)": st.column_config.NumberColumn(
                    format="%,.0f ₫"
                ),
                "R² Score": st.column_config.NumberColumn(
                    format="%.4f"
                ),
                "Thời gian huấn luyện (giây)": (
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
                ),
            },
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Tổng số mô hình",
                len(models),
            )

        with col2:
            st.metric(
                "Dữ liệu huấn luyện",
                f"{metadata.get('train_rows', 0):,}",
            )

        with col3:
            st.metric(
                "Dữ liệu kiểm tra",
                f"{metadata.get('test_rows', 0):,}",
            )

        trained_at = metadata.get("trained_at")

        if trained_at:
            st.caption(
                f"Thời điểm huấn luyện gần nhất: {trained_at}"
            )

    else:

        st.error(
            "Không thể đọc thông tin mô hình mặc định. "
            "Hãy kiểm tra metadata và file mô hình mới."
        )

    st.caption(
        "MAE và RMSE đã được quy đổi sang VNĐ từ đơn vị nghìn đồng "
        "trong dataset. R² không thay đổi khi quy đổi đơn vị."
    )

    st.divider()

    # -------------------------------------------------
    # THONG KE LICH SU
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Hoạt động định giá</div>',
        unsafe_allow_html=True,
    )

    if DATABASE_READY:

        try:

            history = get_all_valuations()

            total = len(history)

            if total:

                prices = [
                    float(item["predicted_price"])
                    for item in history
                    if item.get("predicted_price") is not None
                ]

            else:

                prices = []

            if prices:

                average_price = sum(prices) / len(prices)
                highest_price = max(prices)

            else:

                average_price = 0
                highest_price = 0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Tổng lượt định giá",
                    f"{total:,}",
                )

            with col2:
                st.metric(
                    "Giá dự đoán trung bình",
                    format_vnd(average_price),
                )

            with col3:
                st.metric(
                    "Giá dự đoán cao nhất",
                    format_vnd(highest_price),
                )

            if history:

                st.markdown("#### Những lần định giá gần đây")

                recent_rows = []

                for item in history[:5]:

                    raw_price = item.get("predicted_price")

                    recent_rows.append({
                        "Thời gian": item.get("created_at"),
                        "Thương hiệu": item.get("brand"),
                        "Model": item.get("model"),
                        "Tình trạng": item.get("condition"),
                        "Giá dự đoán (VNĐ)": (
                            float(raw_price) * PRICE_SCALE
                            if raw_price is not None
                            else None
                        ),
                    })

                st.dataframe(
                    recent_rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Giá dự đoán (VNĐ)": (
                            st.column_config.NumberColumn(
                                format="%,.0f ₫"
                            )
                        )
                    },
                )

            else:

                st.info(
                    "Bạn chưa thực hiện lượt định giá nào. "
                    "Hãy mở trang Định giá điện thoại để bắt đầu."
                )

        except Exception as error:

            st.error(
                f"Không thể đọc lịch sử định giá: {error}"
            )

    st.divider()

    # -------------------------------------------------
    # CAC CHUC NANG
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Các chức năng chính</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-label">CHỨC NĂNG 01</div>
                <div class="feature-title">📱 Định giá điện thoại</div>
                <div class="feature-description">
                    Nhập thương hiệu, cấu hình, tình trạng và thông tin
                    sử dụng để nhận giá bán lại dự đoán từ mô hình AI.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-label">CHỨC NĂNG 02</div>
                <div class="feature-title">🕘 Lịch sử định giá</div>
                <div class="feature-description">
                    Lưu trữ, tìm kiếm, xem chi tiết và quản lý
                    những lần định giá đã thực hiện.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-label">CHỨC NĂNG 03</div>
                <div class="feature-title">📊 Phân tích thị trường</div>
                <div class="feature-description">
                    Khám phá phân phối giá, so sánh thương hiệu,
                    tình trạng máy và các thông số cấu hình.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-label">CHỨC NĂNG 04</div>
                <div class="feature-title">🤖 Thông tin mô hình</div>
                <div class="feature-description">
                    Theo dõi thuật toán, dữ liệu huấn luyện
                    và các chỉ số đánh giá mô hình Machine Learning.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------
    # GIOI THIEU DU AN
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Thông tin dự án</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        **PhoneValue AI** là ứng dụng dự đoán giá điện thoại đã qua sử dụng
        bằng thuật toán Machine Learning.

        Hệ thống sử dụng ba thuật toán:
        Linear Regression, Decision Tree Regression và Random Forest Regression.

        Mô hình sử dụng `resale_price` làm biến mục tiêu và các thông tin
        về thương hiệu, cấu hình, tình trạng, thời gian sử dụng
        cùng yếu tố thị trường làm đặc trưng đầu vào.
        """
    )

    st.info(
        "Đơn vị giá trên giao diện là VNĐ. "
        "Dữ liệu gốc và mô hình vẫn sử dụng đơn vị nghìn đồng. "
        "Kết quả chỉ mang tính ước tính, không phải cam kết giá giao dịch."
    )

    st.markdown(
        '<div class="footer-note">'
        'PhoneValue AI · Machine Learning Project'
        '</div>',
        unsafe_allow_html=True,
    )


# =====================================================
# 8. MENU DIEU HUONG
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 8px 0 12px 0;">
            <div style="font-size: 25px; font-weight: 800;">
                📱 PhoneValue AI
            </div>
            <div style="font-size: 12px; color: #B8C8DF;">
                USED PHONE PRICE PREDICTION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("MACHINE LEARNING PROJECT")
    st.caption("Version 3.0")


pages = [
    st.Page(
        dashboard,
        title="Tổng quan",
        icon="🏠",
        default=True,
    ),
    st.Page(
        PAGES_DIR / "1_Dinh_gia.py",
        title="Định giá điện thoại",
        icon="📱",
    ),
    st.Page(
        PAGES_DIR / "2_Phan_tich.py",
        title="Phân tích thị trường",
        icon="📊",
    ),
    st.Page(
        PAGES_DIR / "3_Lich_su.py",
        title="Lịch sử định giá",
        icon="🕘",
    ),
    st.Page(
        PAGES_DIR / "4_Thong_tin_mo_hinh.py",
        title="Thông tin mô hình",
        icon="🤖",
    ),
]

navigation = st.navigation(
    pages,
    position="sidebar",
)

navigation.run()