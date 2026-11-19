
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. CAU HINH DUONG DAN
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

PRICE_SCALE = 1000

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


# =========================================================
# 2. CAU HINH BA MO HINH
# =========================================================

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regression": "decision_tree_regression.pkl",
    "Random Forest Regression": "random_forest_regression.pkl",
}

MODEL_DESCRIPTIONS = {
    "Linear Regression": {
        "description": (
            "Hồi quy tuyến tính dự đoán giá bán lại "
            "dựa trên mối quan hệ tuyến tính giữa các đặc trưng."
        ),
        "advantages": [
            "Dễ hiểu và dễ giải thích.",
            "Tốc độ huấn luyện nhanh.",
            "Phù hợp làm mô hình cơ sở để so sánh.",
        ],
        "limitations": [
            "Khó mô hình hóa các quan hệ phi tuyến phức tạp.",
            "Có thể chưa nắm bắt tốt các tương tác giữa đặc trưng.",
        ],
    },

    "Decision Tree Regression": {
        "description": (
            "Cây quyết định hồi quy chia dữ liệu thành các nhóm "
            "theo điều kiện của đặc trưng và dự đoán giá "
            "dựa trên các nhóm dữ liệu đó."
        ),
        "advantages": [
            "Có khả năng học các quan hệ phi tuyến.",
            "Dễ trực quan hóa và giải thích.",
            "Không bắt buộc chuẩn hóa dữ liệu đầu vào.",
        ],
        "limitations": [
            "Có thể bị quá khớp nếu cây quá sâu.",
            "Kết quả có thể nhạy với thay đổi của dữ liệu.",
        ],
    },

    "Random Forest Regression": {
        "description": (
            "Rừng ngẫu nhiên kết hợp nhiều cây quyết định "
            "và tổng hợp kết quả dự đoán để cải thiện "
            "độ ổn định và khả năng tổng quát hóa."
        ),
        "advantages": [
            "Có khả năng học các quan hệ phi tuyến.",
            "Giảm độ biến động so với một cây quyết định đơn lẻ.",
            "Có thể xử lý nhiều đặc trưng đầu vào.",
        ],
        "limitations": [
            "Thời gian huấn luyện lâu hơn.",
            "Tiêu tốn nhiều bộ nhớ.",
            "Khó giải thích hơn một cây quyết định đơn lẻ.",
        ],
    },
}


# =========================================================
# 3. DOC METADATA
# =========================================================

@st.cache_data(ttl=30)
def load_metadata():

    if not METADATA_PATH.is_file():
        return None

    try:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return None


metadata = load_metadata()


# =========================================================
# 4. HAM HO TRO
# =========================================================

def format_number(value, decimals=2):

    try:
        number = float(value)

        if pd.isna(number):
            return "N/A"

        return f"{number:,.{decimals}f}"

    except (TypeError, ValueError):
        return "N/A"


def format_vnd(value, decimals=0):

    try:
        number = float(value) * PRICE_SCALE

        if pd.isna(number):
            return "N/A"

        return f"{number:,.{decimals}f} ₫"

    except (TypeError, ValueError):
        return "N/A"


def get_model_result(model_name, models_metadata):

    if not isinstance(models_metadata, list):
        return None

    for item in models_metadata:

        if (
            isinstance(item, dict)
            and item.get("model_name") == model_name
        ):
            return item

    return None


def get_model_status(model_name, models_metadata):

    model_file = MODEL_FILES[model_name]
    model_path = MODEL_DIR / model_file

    result = get_model_result(
        model_name,
        models_metadata,
    )

    if result is None:
        return "Chưa có metadata", model_file

    if not model_path.is_file():
        return "Thiếu file mô hình", model_file

    metadata_file = result.get("model_file")

    if metadata_file and metadata_file != model_file:
        return "Tên file không khớp", model_file

    return "Đã huấn luyện", model_file


def get_valid_comparison_data(models_metadata):

    rows = []

    for model_name in MODEL_FILES:

        result = get_model_result(
            model_name,
            models_metadata,
        )

        if result is None:
            continue

        rows.append({
            "Mô hình": model_name,

            # MAE và RMSE vẫn lấy từ metadata ở đơn vị gốc.
            # Sẽ chuyển sang VND khi tạo dữ liệu hiển thị.
            "MAE": pd.to_numeric(
                result.get("mae"),
                errors="coerce",
            ),

            "RMSE": pd.to_numeric(
                result.get("rmse"),
                errors="coerce",
            ),

            "R²": pd.to_numeric(
                result.get("r2"),
                errors="coerce",
            ),

            "Thời gian (giây)": pd.to_numeric(
                result.get("training_time_seconds"),
                errors="coerce",
            ),

            "Số dòng train": result.get("train_rows"),
            "Số dòng test": result.get("test_rows"),
        })

    return pd.DataFrame(rows)


def create_display_comparison(raw_df):

    """
    Tạo bảng hiển thị.
    MAE và RMSE được chuyển từ đơn vị dataset sang VND.
    """

    display_df = raw_df.copy()

    if not display_df.empty:

        display_df["MAE"] = (
            pd.to_numeric(
                display_df["MAE"],
                errors="coerce",
            ) * PRICE_SCALE
        )

        display_df["RMSE"] = (
            pd.to_numeric(
                display_df["RMSE"],
                errors="coerce",
            ) * PRICE_SCALE
        )

    return display_df


# =========================================================
# 5. TIEU DE TRANG
# =========================================================

st.title("🤖 Thông tin mô hình")

st.caption(
    "Thông tin thuật toán, dữ liệu huấn luyện, "
    "hiệu suất và so sánh ba mô hình học máy "
    "trong hệ thống PhoneValue AI."
)

st.caption(
    "Các chỉ số MAE và RMSE được quy đổi sang VND "
    "để thống nhất với giao diện định giá."
)

st.divider()


# =========================================================
# 6. KIEM TRA METADATA
# =========================================================

if metadata is None:

    st.error(
        "Không thể đọc model_metadata.json. "
        "Hãy kiểm tra lại thư mục models."
    )

    st.stop()


models_metadata = metadata.get("models", [])

if not isinstance(models_metadata, list):
    models_metadata = []


# =========================================================
# 7. TRANG THAI HE THONG
# =========================================================

st.subheader("📡 Trạng thái hệ thống")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("Metadata đọc thành công")

with col2:

    if DATASET_PATH.is_file():
        st.success("Dataset đã sẵn sàng")
    else:
        st.error("Không tìm thấy dataset")


with col3:

    existing_count = sum(
        1
        for model_name in MODEL_FILES
        if get_model_status(
            model_name,
            models_metadata,
        )[0] == "Đã huấn luyện"
    )

    st.metric(
        "Mô hình đã huấn luyện",
        f"{existing_count}/3",
    )


# =========================================================
# 8. DANH SACH BA MO HINH
# =========================================================

st.divider()

st.subheader("📚 Danh sách mô hình")

status_rows = []

for model_name in MODEL_FILES:

    status, model_file = get_model_status(
        model_name,
        models_metadata,
    )

    model_path = MODEL_DIR / model_file

    if model_path.is_file():

        file_size = (
            model_path.stat().st_size
            / (1024 * 1024)
        )

    else:
        file_size = None

    status_rows.append({
        "Mô hình": model_name,
        "Trạng thái": status,
        "Tên file": model_file,
        "Dung lượng (MB)": (
            round(file_size, 2)
            if file_size is not None
            else None
        ),
    })


status_df = pd.DataFrame(status_rows)

st.dataframe(
    status_df,
    use_container_width=True,
    hide_index=True,
)


if existing_count != 3:

    st.warning(
        "Một hoặc nhiều mô hình chưa có đủ file hoặc metadata. "
        "Hãy kiểm tra lại thư mục models."
    )


# =========================================================
# 9. THONG TIN THUAT TOAN
# =========================================================

st.divider()

st.subheader("🧠 Thông tin thuật toán")

algorithm_tabs = st.tabs(
    list(MODEL_DESCRIPTIONS.keys())
)

for tab, model_name in zip(
    algorithm_tabs,
    MODEL_DESCRIPTIONS.keys(),
):

    info = MODEL_DESCRIPTIONS[model_name]

    with tab:

        st.markdown(f"### {model_name}")

        st.write(info["description"])

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Ưu điểm**")

            for item in info["advantages"]:
                st.write(f"✅ {item}")

        with col2:

            st.markdown("**Hạn chế**")

            for item in info["limitations"]:
                st.write(f"⚠️ {item}")


# =========================================================
# 10. THONG TIN DATASET
# =========================================================

st.divider()

st.subheader("📊 Thông tin dữ liệu huấn luyện")

dataset_rows = metadata.get("dataset_rows", 0)
train_rows = metadata.get("train_rows", 0)
test_rows = metadata.get("test_rows", 0)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Tổng số bản ghi",
        f"{dataset_rows:,}" if dataset_rows else "N/A",
    )

with col2:

    st.metric(
        "Dữ liệu huấn luyện",
        f"{train_rows:,}" if train_rows else "N/A",
    )

with col3:

    st.metric(
        "Dữ liệu kiểm tra",
        f"{test_rows:,}" if test_rows else "N/A",
    )


test_size = metadata.get("test_size")
random_state = metadata.get("random_state")

target = metadata.get(
    "target",
    "resale_price",
)

st.write(
    f"**Tỷ lệ dữ liệu kiểm tra:** "
    f"{float(test_size) * 100:.0f}%"
    if test_size is not None
    else "**Tỷ lệ dữ liệu kiểm tra:** N/A"
)

st.write(
    f"**Random State:** {random_state}"
    if random_state is not None
    else "**Random State:** N/A"
)

st.write(f"**Biến mục tiêu:** `{target}`")

st.write(
    "**Số lượng đặc trưng đầu vào:** "
    f"{len(ALL_FEATURES)}"
)

st.caption(
    "Bộ dữ liệu có 27 đặc trưng đầu vào và 1 biến mục tiêu "
    "resale_price. Đơn vị giá gốc trong dataset là nghìn VND."
)


# =========================================================
# 11. BANG SO SANH BA MO HINH
# =========================================================

st.divider()

st.subheader("🏆 So sánh hiệu suất ba mô hình")

comparison_df = get_valid_comparison_data(
    models_metadata
)

display_comparison_df = create_display_comparison(
    comparison_df
)

if comparison_df.empty:

    st.warning(
        "Chưa có kết quả đánh giá mô hình trong metadata."
    )

else:

    st.dataframe(
        display_comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "MAE": st.column_config.NumberColumn(
                "MAE (VND)",
                format="%,.0f ₫",
            ),

            "RMSE": st.column_config.NumberColumn(
                "RMSE (VND)",
                format="%,.0f ₫",
            ),

            "R²": st.column_config.NumberColumn(
                "R²",
                format="%.5f",
            ),

            "Thời gian (giây)": st.column_config.NumberColumn(
                "Thời gian (giây)",
                format="%.2f",
            ),
        },
    )

    st.caption(
        "MAE và RMSE đã được quy đổi sang VND. "
        "Hai chỉ số càng thấp càng tốt. "
        "R² càng gần 1 thường cho thấy mô hình giải thích "
        "dữ liệu kiểm tra tốt hơn."
    )


# =========================================================
# 12. MO HINH CO MAE TOT NHAT
# =========================================================

st.divider()

st.subheader("🥇 Mô hình có kết quả tốt nhất")

valid_mae_df = comparison_df.dropna(
    subset=["MAE"]
)

if not valid_mae_df.empty:

    best_row = valid_mae_df.loc[
        valid_mae_df["MAE"].idxmin()
    ]

    best_name = best_row["Mô hình"]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mô hình tốt nhất theo MAE",
            best_name,
        )

    with col2:

        st.metric(
            "MAE",
            format_vnd(best_row["MAE"]),
        )

    with col3:

        st.metric(
            "R²",
            format_number(
                best_row["R²"],
                5,
            ),
        )

    st.write(
        "**Mô hình mặc định khi định giá:** "
        f"`{metadata.get('selected_default_model', 'Chưa xác định')}`"
    )

    default_model = metadata.get(
        "selected_default_model"
    )

    if default_model == best_name:

        st.success(
            "Mô hình mặc định cũng là mô hình có MAE thấp nhất."
        )

    else:

        st.info(
            "Mô hình mặc định và mô hình có MAE thấp nhất "
            "không trùng nhau. Hãy kiểm tra lại cấu hình."
        )

else:

    st.info(
        "Chưa có đủ dữ liệu MAE để xác định mô hình tốt nhất."
    )


# =========================================================
# 13. BIEU DO SO SANH
# =========================================================

st.divider()

st.subheader("📈 Biểu đồ so sánh hiệu suất")

if not comparison_df.empty:

    metric_tabs = st.tabs([
        "MAE",
        "RMSE",
        "R²",
        "Thời gian huấn luyện",
    ])

    chart_settings = [
        (
            "MAE",
            "MAE (VND)",
            "So sánh MAE - càng thấp càng tốt",
            True,
        ),
        (
            "RMSE",
            "RMSE (VND)",
            "So sánh RMSE - càng thấp càng tốt",
            True,
        ),
        (
            "R²",
            "R²",
            "So sánh R² - càng cao càng tốt",
            False,
        ),
        (
            "Thời gian (giây)",
            "Thời gian (giây)",
            "Thời gian huấn luyện",
            True,
        ),
    ]

    for tab, settings in zip(
        metric_tabs,
        chart_settings,
    ):

        metric_name, y_axis_name, chart_title, _ = settings

        with tab:

            plot_df = comparison_df.dropna(
                subset=[metric_name]
            ).copy()

            if plot_df.empty:

                st.info(
                    f"Chưa có dữ liệu {metric_name}."
                )

            else:

                # Chỉ chuyển MAE và RMSE sang VND.
                # R² và thời gian huấn luyện giữ nguyên.

                if metric_name in ["MAE", "RMSE"]:

                    plot_df[metric_name] = (
                        plot_df[metric_name] * PRICE_SCALE
                    )

                fig = px.bar(
                    plot_df,
                    x="Mô hình",
                    y=metric_name,
                    color="Mô hình",
                    title=chart_title,
                    text_auto=",.0f" if metric_name in [
                        "MAE",
                        "RMSE",
                    ] else ".4f",
                )

                fig.update_layout(
                    height=420,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title=y_axis_name,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


# =========================================================
# 14. XUAT BANG SO SANH CSV
# =========================================================

st.divider()

st.subheader("📥 Xuất kết quả so sánh")

if not comparison_df.empty:

    # Xuất CSV theo đơn vị VND để thống nhất với giao diện.

    csv_data = display_comparison_df.to_csv(
        index=False,
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Tải bảng so sánh mô hình CSV",
        data=csv_data,
        file_name="phonevalue_model_comparison_vnd.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "File CSV đã chuyển MAE và RMSE sang VND. "
        "R² và thời gian huấn luyện được giữ nguyên."
    )


# =========================================================
# 15. Y NGHIA CAC CHI SO
# =========================================================

st.divider()

st.subheader("📖 Ý nghĩa các chỉ số đánh giá")

with st.expander("Giải thích MAE, RMSE, R² và Training Time"):

    st.markdown(
        """
        ### 1. MAE - Mean Absolute Error

        Sai số tuyệt đối trung bình giữa giá trị thực tế
        và giá trị dự đoán.

        Đơn vị hiển thị: VND.

        Ví dụ: MAE khoảng 2.755.620 ₫ nghĩa là
        dự đoán của mô hình lệch trung bình khoảng
        2,76 triệu đồng so với giá trị thực tế.

        MAE càng thấp thì sai số dự đoán trung bình càng nhỏ.

        ### 2. RMSE - Root Mean Squared Error

        Căn bậc hai của sai số bình phương trung bình.

        Đơn vị hiển thị: VND.

        RMSE nhạy hơn với những sai số dự đoán lớn.

        ### 3. R² - Coefficient of Determination

        Đánh giá mức độ mô hình giải thích sự biến thiên
        của biến mục tiêu.

        Giá trị càng gần 1 thường cho thấy mô hình giải thích
        dữ liệu kiểm tra tốt hơn.

        R² không có đơn vị tiền tệ.

        ### 4. Training Time

        Thời gian mô hình cần để hoàn thành quá trình huấn luyện.

        Đơn vị: giây.

        Thời gian thấp hơn không đồng nghĩa với độ chính xác cao hơn.
        """
    )

st.caption(
    "MAE và RMSE được hiển thị bằng VND. "
    "Dữ liệu huấn luyện và metadata gốc không bị thay đổi."
)


# =========================================================
# 16. CAC DAC TRUNG DAU VAO
# =========================================================

st.divider()

st.subheader("🧩 Các đặc trưng đầu vào")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Tổng đặc trưng",
        len(ALL_FEATURES),
    )

with col2:

    st.metric(
        "Đặc trưng phân loại",
        len(CATEGORICAL_FEATURES),
    )

with col3:

    st.metric(
        "Đặc trưng số",
        len(NUMERIC_FEATURES),
    )


feature_tab1, feature_tab2, feature_tab3 = st.tabs([
    "Tất cả đặc trưng",
    "Đặc trưng phân loại",
    "Đặc trưng số",
])


with feature_tab1:

    feature_df = pd.DataFrame({
        "STT": range(1, len(ALL_FEATURES) + 1),
        "Tên đặc trưng": ALL_FEATURES,
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )


with feature_tab2:

    categorical_df = pd.DataFrame({
        "STT": range(1, len(CATEGORICAL_FEATURES) + 1),
        "Đặc trưng phân loại": CATEGORICAL_FEATURES,
    })

    st.dataframe(
        categorical_df,
        use_container_width=True,
        hide_index=True,
    )


with feature_tab3:

    numeric_df = pd.DataFrame({
        "STT": range(1, len(NUMERIC_FEATURES) + 1),
        "Đặc trưng số": NUMERIC_FEATURES,
    })

    st.dataframe(
        numeric_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 17. THONG TIN FEATURE IMPORTANCE
# =========================================================

st.divider()

st.subheader("🔬 Phân tích mức độ quan trọng của đặc trưng")

st.info(
    "Phân tích Feature Importance chưa được đưa vào trang này "
    "vì cần tính toán riêng cho từng mô hình. "
    "Không sử dụng kết quả từ mô hình cũ để đại diện cho "
    "Linear Regression, Decision Tree Regression "
    "hoặc Random Forest Regression."
)


# =========================================================
# 18. THONG TIN HUAN LUYEN
# =========================================================

st.divider()

st.subheader("⏱️ Thông tin huấn luyện")

trained_at = metadata.get("trained_at")

if trained_at:

    st.write(
        f"**Thời điểm huấn luyện:** {trained_at}"
    )

else:

    st.info(
        "Metadata chưa có thông tin thời điểm huấn luyện."
    )


if not comparison_df.empty:

    training_times = pd.to_numeric(
        comparison_df["Thời gian (giây)"],
        errors="coerce",
    ).dropna()

    if not training_times.empty:

        total_training_time = training_times.sum()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Tổng thời gian huấn luyện",
                f"{total_training_time:,.2f} giây",
            )

        with col2:

            st.metric(
                "Tương đương",
                f"{total_training_time / 60:,.2f} phút",
            )


# =========================================================
# 19. THONG TIN FILE HE THONG
# =========================================================

st.divider()

st.subheader("📁 Thông tin file hệ thống")

if METADATA_PATH.is_file():

    metadata_size = (
        METADATA_PATH.stat().st_size / 1024
    )

    st.write(
        f"**Metadata:** `{METADATA_PATH.name}` "
        f"- {metadata_size:.2f} KB"
    )


for model_name, model_file in MODEL_FILES.items():

    model_path = MODEL_DIR / model_file

    if model_path.is_file():

        model_size = (
            model_path.stat().st_size
            / (1024 * 1024)
        )

        st.write(
            f"**{model_name}:** `{model_file}` "
            f"- {model_size:.2f} MB"
        )

    else:

        st.write(
            f"**{model_name}:** Chưa tìm thấy file."
        )


with st.expander("Xem toàn bộ metadata JSON"):

    st.json(metadata)


# =========================================================
# 20. FOOTER
# =========================================================

st.divider()

st.caption(
    "PhoneValue AI | Hệ thống học máy định giá "
    "điện thoại đã qua sử dụng."
)
