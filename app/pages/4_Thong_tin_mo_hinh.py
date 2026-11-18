
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATASET_PATH = BASE_DIR / "data" / "raw" / "used_phone_price_prediction_1M.csv.gz"
IMPORTANCE_PATH = BASE_DIR / "data" / "processed" / "feature_importance.csv"


# =========================================================
# 2. ĐỌC DỮ LIỆU
# =========================================================

@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data
def load_feature_importance():
    if not IMPORTANCE_PATH.exists():
        return None

    try:
        df = pd.read_csv(IMPORTANCE_PATH)

        required = [
            "feature",
            "importance_mean",
            "importance_std"
        ]

        if not all(column in df.columns for column in required):
            return None

        return df.sort_values(
            "importance_mean",
            ascending=False
        )

    except Exception:
        return None


metadata = load_metadata()
importance_df = load_feature_importance()


# =========================================================
# 3. TIÊU ĐỀ
# =========================================================

st.title("Thông tin mô hình")

st.caption(
    "Thông tin thuật toán, dữ liệu huấn luyện, hiệu suất "
    "và mức độ quan trọng của các đặc trưng."
)

st.divider()


# =========================================================
# 4. KIỂM TRA TRẠNG THÁI HỆ THỐNG
# =========================================================

st.subheader("Trạng thái hệ thống")

col1, col2, col3 = st.columns(3)

with col1:
    if MODEL_PATH.exists():
        st.success("Mô hình đã sẵn sàng")
    else:
        st.error("Không tìm thấy file mô hình")

with col2:
    if metadata is not None:
        st.success("Metadata hợp lệ")
    else:
        st.error("Metadata chưa sẵn sàng")

with col3:
    if DATASET_PATH.exists():
        st.success("Dataset đã sẵn sàng")
    else:
        st.warning("Không tìm thấy dataset")

if not MODEL_PATH.exists():
    st.warning(
        "Không tìm thấy models/best_model.pkl. "
        "Hãy kiểm tra lại file mô hình."
    )

if metadata is None:
    st.error(
        "Không thể đọc model_metadata.json. "
        "Vui lòng kiểm tra file metadata."
    )
    st.stop()

st.divider()


# =========================================================
# 5. THÔNG TIN THUẬT TOÁN
# =========================================================

st.subheader("Thông tin thuật toán")

model_name = metadata.get(
    "model_name",
    "HistGradientBoostingRegressor"
)

target = metadata.get("target", "resale_price")

features = metadata.get("features", [])

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tên mô hình**")
    st.info(model_name)

with col2:
    st.markdown("**Biến mục tiêu**")
    st.info(target)

st.markdown("**Mô tả thuật toán**")

st.write(
    """
    HistGradientBoostingRegressor là thuật toán học máy
    thuộc nhóm Gradient Boosting, sử dụng nhiều cây quyết định
    để xây dựng mô hình dự đoán.

    Các cây được xây dựng tuần tự nhằm cải thiện sai số dự đoán.
    Thuật toán có khả năng học các mối quan hệ phi tuyến giữa
    đặc điểm kỹ thuật, tình trạng điện thoại và giá bán lại.

    Trong PhoneValue AI, thuật toán được sử dụng để dự đoán
    giá bán lại của điện thoại đã qua sử dụng.
    """
)

st.divider()


# =========================================================
# 6. THÔNG TIN DỮ LIỆU HUẤN LUYỆN
# =========================================================

st.subheader("Thông tin dữ liệu")

total_rows = metadata.get("total_rows", 0)
train_rows = metadata.get("train_rows", 0)
test_rows = metadata.get("test_rows", 0)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Tổng số bản ghi",
        f"{total_rows:,}" if total_rows else "Chưa có dữ liệu"
    )

with col2:
    st.metric(
        "Dữ liệu huấn luyện",
        f"{train_rows:,}" if train_rows else "Chưa có dữ liệu"
    )

with col3:
    st.metric(
        "Dữ liệu kiểm tra",
        f"{test_rows:,}" if test_rows else "Chưa có dữ liệu"
    )

test_size = metadata.get("test_size")

if test_size is not None:
    st.write(f"**Tỷ lệ kiểm tra:** {test_size * 100:.0f}%")

random_state = metadata.get("random_state")

if random_state is not None:
    st.write(f"**Random State:** {random_state}")

training_time = metadata.get("training_time_seconds")

if training_time is not None:
    st.write(f"**Thời gian huấn luyện:** {training_time:.2f} giây")

st.divider()


# =========================================================
# 7. ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH
# =========================================================

st.subheader("Đánh giá hiệu suất mô hình")

metrics = metadata.get("metrics", {})

mae = metrics.get("mae", metrics.get("MAE"))
rmse = metrics.get("rmse", metrics.get("RMSE"))
r2 = metrics.get("r2", metrics.get("R2"))

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae:,.2f}" if mae is not None else "Chưa có dữ liệu"
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:,.2f}" if rmse is not None else "Chưa có dữ liệu"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2:.4f}" if r2 is not None else "Chưa có dữ liệu"
    )

with st.expander("Ý nghĩa các chỉ số đánh giá"):
    st.markdown(
        """
        - **MAE:** Sai số tuyệt đối trung bình giữa giá trị thực tế
          và giá trị dự đoán. Càng thấp càng tốt.

        - **RMSE:** Căn bậc hai của sai số bình phương trung bình.
          Chỉ số này nhạy hơn với các sai số lớn.

        - **R² Score:** Đánh giá mức độ mô hình giải thích sự biến
          thiên của dữ liệu mục tiêu. Giá trị càng gần 1 thường
          cho thấy mô hình phù hợp tốt hơn với dữ liệu kiểm tra.
        """
    )

st.caption(
    "MAE và RMSE sử dụng đơn vị giá của dataset, "
    "không mặc định là đồng Việt Nam."
)

st.divider()


# =========================================================
# 8. PHÂN TÍCH FEATURE IMPORTANCE
# =========================================================

st.subheader("Mức độ quan trọng của các đặc trưng")

st.write(
    """
    Biểu đồ sử dụng phương pháp Permutation Feature Importance.
    Phương pháp này đo mức tăng MAE khi xáo trộn từng đặc trưng
    trên tập dữ liệu kiểm tra.
    """
)

if importance_df is not None and not importance_df.empty:

    top_features = importance_df.head(15).copy()

    top_features = top_features.sort_values(
        "importance_mean",
        ascending=True
    )

    fig = px.bar(
        top_features,
        x="importance_mean",
        y="feature",
        error_x="importance_std",
        orientation="h",
        title="Top 15 đặc trưng quan trọng nhất",
        labels={
            "importance_mean": "Mức tăng MAE",
            "feature": "Đặc trưng",
            "importance_std": "Độ lệch chuẩn"
        },
        hover_data={
            "importance_mean": ":.4f",
            "importance_std": ":.4f"
        }
    )

    fig.update_layout(
        height=600,
        xaxis_title="Mức tăng MAE khi xáo trộn đặc trưng",
        yaxis_title="",
        margin=dict(l=20, r=20, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Bảng xếp hạng đặc trưng**")

    display_df = importance_df.copy()

    display_df.columns = [
        "Xếp hạng",
        "Đặc trưng",
        "Importance trung bình",
        "Độ lệch chuẩn"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = importance_df.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="Tải bảng Feature Importance (CSV)",
        data=csv_data,
        file_name="phonevalue_feature_importance.csv",
        mime="text/csv"
    )

    st.info(
        "Giá trị Importance càng lớn cho thấy mô hình càng phụ thuộc "
        "vào đặc trưng đó. Giá trị bằng 0 hoặc âm không chứng minh "
        "đặc trưng không có ý nghĩa trong thực tế."
    )

else:
    st.warning(
        "Chưa có dữ liệu Feature Importance. "
        "Hãy chạy python src\\feature_importance.py trước."
    )

st.divider()


# =========================================================
# 9. DANH SÁCH ĐẶC TRƯNG ĐẦU VÀO
# =========================================================

st.subheader("Các đặc trưng đầu vào")

categorical_features = metadata.get("categorical_features", [])
numeric_features = metadata.get("numeric_features", [])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tổng đặc trưng", len(features))

with col2:
    st.metric("Đặc trưng phân loại", len(categorical_features))

with col3:
    st.metric("Đặc trưng số", len(numeric_features))

tab1, tab2, tab3 = st.tabs(
    [
        "Tất cả đặc trưng",
        "Đặc trưng phân loại",
        "Đặc trưng số"
    ]
)

with tab1:
    for index, feature in enumerate(features, start=1):
        st.write(f"{index}. `{feature}`")

with tab2:
    for index, feature in enumerate(categorical_features, start=1):
        st.write(f"{index}. `{feature}`")

with tab3:
    for index, feature in enumerate(numeric_features, start=1):
        st.write(f"{index}. `{feature}`")

st.divider()


# =========================================================
# 10. THỜI GIAN HUẤN LUYỆN
# =========================================================

st.subheader("Thời gian huấn luyện")

trained_at = metadata.get("trained_at")

if trained_at:
    st.write(f"**Thời điểm huấn luyện gần nhất:** {trained_at}")
else:
    st.info("Metadata chưa lưu thời điểm huấn luyện.")

if training_time is not None:
    st.write(f"**Tổng thời gian:** {training_time / 60:.2f} phút")
else:
    st.info("Metadata chưa có thông tin thời gian huấn luyện.")

st.divider()


# =========================================================
# 11. THÔNG TIN FILE HỆ THỐNG
# =========================================================

st.subheader("Thông tin file hệ thống")

file_col1, file_col2 = st.columns(2)

with file_col1:
    st.markdown("**File mô hình**")

    if MODEL_PATH.exists():
        model_size = MODEL_PATH.stat().st_size / (1024 * 1024)

        st.write("`models/best_model.pkl`")
        st.write(f"Dung lượng: {model_size:.2f} MB")
    else:
        st.warning("File mô hình không tồn tại.")

with file_col2:
    st.markdown("**File metadata**")

    if METADATA_PATH.exists():
        metadata_size = METADATA_PATH.stat().st_size / 1024

        st.write("`models/model_metadata.json`")
        st.write(f"Dung lượng: {metadata_size:.2f} KB")
    else:
        st.warning("File metadata không tồn tại.")

with st.expander("Xem toàn bộ metadata JSON"):
    st.json(metadata)

st.divider()

st.caption(
    "PhoneValue AI | Hệ thống ứng dụng học máy "
    "trong định giá điện thoại đã qua sử dụng."
)

