
import json
import sys
from pathlib import Path

import streamlit as st


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "pages"

for path in (ROOT_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =====================================================
# 2. CẤU HÌNH STREAMLIT
# =====================================================

st.set_page_config(
    page_title="PhoneValue AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 3. IMPORT VÀ KHỞI TẠO DATABASE
# =====================================================

from database import initialize_database, get_all_valuations

try:
    initialize_database()
    DATABASE_READY = True
except Exception as error:
    DATABASE_READY = False
    DATABASE_ERROR = str(error)


# =====================================================
# 4. ĐƯỜNG DẪN VÀ THÔNG TIN MÔ HÌNH
# =====================================================

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def format_number(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "Chưa có dữ liệu"


# =====================================================
# 5. CSS GIAO DIỆN
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
        background: linear-gradient(180deg, #0F1F3D 0%, #142D52 100%);
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
        background: linear-gradient(120deg, #102A56 0%, #155EAA 60%, #2563EB 100%);
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
# 6. TRANG TỔNG QUAN
# =====================================================

def dashboard():

    metadata = load_metadata()

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
    # TRẠNG THÁI HỆ THỐNG
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Tình trạng hệ thống</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mô hình AI",
            "Sẵn sàng" if MODEL_PATH.exists() else "Chưa có",
        )

    with col2:
        st.metric(
            "Cơ sở dữ liệu",
            "Đã kết nối" if DATABASE_READY else "Lỗi kết nối",
        )

    with col3:
        st.metric(
            "Dataset",
            "Đã tìm thấy" if DATA_PATH.exists() else "Chưa tìm thấy",
        )

    if not DATABASE_READY:
        st.error(f"Không thể khởi tạo cơ sở dữ liệu: {DATABASE_ERROR}")

    if not MODEL_PATH.exists():
        st.warning(
            "Chưa tìm thấy mô hình. Hãy chạy lệnh "
            "`python src\\train.py` để huấn luyện."
        )

    # -------------------------------------------------
    # THÔNG TIN MÔ HÌNH
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Hiệu suất mô hình</div>',
        unsafe_allow_html=True,
    )

    if metadata:
        metrics = metadata.get("metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Thuật toán",
                metadata.get(
                    "model_name",
                    "HistGradientBoosting",
                ),
            )

        with col2:
            st.metric(
                "R² Score",
                format_number(metrics.get("r2")),
            )

        with col3:
            st.metric(
                "MAE",
                format_number(metrics.get("mae")),
            )

        with col4:
            st.metric(
                "RMSE",
                format_number(metrics.get("rmse")),
            )

        trained_at = metadata.get("trained_at")

        if trained_at:
            st.caption(f"Thời điểm huấn luyện gần nhất: {trained_at}")

    else:
        st.info(
            "Chưa có thông tin đánh giá mô hình. "
            "Hãy kiểm tra file models/model_metadata.json."
        )

    st.caption(
        "MAE và RMSE sử dụng đơn vị giá trong dataset. "
        "R² là hệ số đánh giá khả năng giải thích biến thiên của mô hình."
    )

    st.divider()

    # -------------------------------------------------
    # THỐNG KÊ LỊCH SỬ
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
                ]

                average_price = sum(prices) / total
                highest_price = max(prices)

            else:
                average_price = 0
                highest_price = 0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Tổng lượt định giá", f"{total:,}")

            with col2:
                st.metric(
                    "Giá dự đoán trung bình",
                    format_number(average_price),
                )

            with col3:
                st.metric(
                    "Giá dự đoán cao nhất",
                    format_number(highest_price),
                )

            if history:
                st.markdown("#### Những lần định giá gần đây")

                recent_rows = []

                for item in history[:5]:
                    recent_rows.append(
                        {
                            "Thời gian": item["created_at"],
                            "Thương hiệu": item["brand"],
                            "Model": item["model"],
                            "Tình trạng": item["condition"],
                            "Giá dự đoán": item["predicted_price"],
                        }
                    )

                st.dataframe(
                    recent_rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Giá dự đoán": st.column_config.NumberColumn(
                            format="%.2f"
                        )
                    },
                )
            else:
                st.info(
                    "Bạn chưa thực hiện lượt định giá nào. "
                    "Hãy mở trang Định giá điện thoại để bắt đầu."
                )

        except Exception as error:
            st.error(f"Không thể đọc lịch sử định giá: {error}")

    st.divider()

    # -------------------------------------------------
    # CÁC CHỨC NĂNG
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
    # GIỚI THIỆU DỰ ÁN
    # -------------------------------------------------

    st.markdown(
        '<div class="section-heading">Thông tin dự án</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        **PhoneValue AI** là ứng dụng dự đoán giá điện thoại đã qua sử dụng
        bằng thuật toán Machine Learning.

        Mô hình sử dụng `resale_price` làm biến mục tiêu và các thông tin
        về thương hiệu, cấu hình, tình trạng, thời gian sử dụng
        cùng yếu tố thị trường làm đặc trưng đầu vào.
        """
    )

    st.info(
        "Lưu ý: Đơn vị giá được giữ nguyên theo dataset, chưa mặc định là VNĐ. "
        "Kết quả chỉ mang tính ước tính, không phải cam kết giá giao dịch."
    )

    st.markdown(
        '<div class="footer-note">PhoneValue AI · Machine Learning Project</div>',
        unsafe_allow_html=True,
    )


# =====================================================
# 7. MENU ĐIỀU HƯỚNG
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
    st.caption("Version 2.0")


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

