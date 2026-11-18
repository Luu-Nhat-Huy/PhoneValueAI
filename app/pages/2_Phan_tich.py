
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =====================================================
# 1. CẤU HÌNH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

st.title("📊 Phân tích dữ liệu")
st.caption(
    "Khám phá dữ liệu điện thoại và các yếu tố ảnh hưởng "
    "đến giá bán lại."
)

st.divider()


# =====================================================
# 2. ĐỌC DỮ LIỆU
# =====================================================

@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    required_columns = [
        "brand",
        "model",
        "release_year",
        "ram_gb",
        "storage_gb",
        "original_price",
        "resale_price",
        "condition",
        "os_type",
        "age_months",
        "battery_health",
        "screen_size_inches",
        "battery_capacity",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset thiếu các cột: "
            + ", ".join(missing_columns)
        )

    return df


try:
    df = load_data()

except Exception as error:
    st.error(f"Không thể tải dữ liệu: {error}")
    st.stop()


# =====================================================
# 3. KIỂM TRA VÀ CHUẨN BỊ DỮ LIỆU
# =====================================================

total_rows = len(df)

missing_count = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

df["resale_price"] = pd.to_numeric(
    df["resale_price"],
    errors="coerce",
)

df["original_price"] = pd.to_numeric(
    df["original_price"],
    errors="coerce",
)

df["age_months"] = pd.to_numeric(
    df["age_months"],
    errors="coerce",
)

df["battery_health"] = pd.to_numeric(
    df["battery_health"],
    errors="coerce",
)


# =====================================================
# 4. BỘ LỌC
# =====================================================

st.subheader("🔎 Bộ lọc dữ liệu")

with st.expander(
    "Mở bộ lọc phân tích",
    expanded=True,
):

    col1, col2 = st.columns(2)

    with col1:

        brands = sorted(
            df["brand"].dropna().unique().tolist()
        )

        selected_brands = st.multiselect(
            "Thương hiệu",
            options=brands,
            default=brands,
        )

        conditions = sorted(
            df["condition"].dropna().unique().tolist()
        )

        selected_conditions = st.multiselect(
            "Tình trạng",
            options=conditions,
            default=conditions,
        )

    with col2:

        os_types = sorted(
            df["os_type"].dropna().unique().tolist()
        )

        selected_os = st.multiselect(
            "Hệ điều hành",
            options=os_types,
            default=os_types,
        )

        years = sorted(
            df["release_year"].dropna().unique().tolist()
        )

        if years:

            selected_years = st.slider(
                "Khoảng năm ra mắt",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(
                    int(min(years)),
                    int(max(years)),
                ),
            )

        else:
            selected_years = None


    price_min = float(
        df["resale_price"].min()
    )

    price_max = float(
        df["resale_price"].max()
    )

    if pd.isna(price_min) or pd.isna(price_max):
        st.error("Không có dữ liệu giá bán lại hợp lệ.")
        st.stop()

    selected_price = st.slider(
        "Khoảng giá bán lại",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
    )


# =====================================================
# 5. ÁP DỤNG BỘ LỌC
# =====================================================

filtered_df = df.copy()

if not selected_brands:
    st.warning("Vui lòng chọn ít nhất một thương hiệu.")
    st.stop()

if not selected_conditions:
    st.warning("Vui lòng chọn ít nhất một tình trạng.")
    st.stop()

if not selected_os:
    st.warning("Vui lòng chọn ít nhất một hệ điều hành.")
    st.stop()


filtered_df = filtered_df[
    filtered_df["brand"].isin(selected_brands)
]

filtered_df = filtered_df[
    filtered_df["condition"].isin(selected_conditions)
]

filtered_df = filtered_df[
    filtered_df["os_type"].isin(selected_os)
]

if selected_years is not None:

    filtered_df = filtered_df[
        filtered_df["release_year"].between(
            selected_years[0],
            selected_years[1],
        )
    ]

filtered_df = filtered_df[
    filtered_df["resale_price"].between(
        selected_price[0],
        selected_price[1],
    )
]

if filtered_df.empty:
    st.warning(
        "Không tìm thấy dữ liệu phù hợp với bộ lọc."
    )
    st.stop()


# =====================================================
# 6. THỐNG KÊ TỔNG QUAN
# =====================================================

st.divider()

st.subheader("📌 Tổng quan dữ liệu")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Tổng bản ghi",
        f"{len(filtered_df):,}",
    )

with col2:
    st.metric(
        "Giá bán lại trung bình",
        f"{filtered_df['resale_price'].mean():,.2f}",
    )

with col3:
    st.metric(
        "Giá bán lại cao nhất",
        f"{filtered_df['resale_price'].max():,.2f}",
    )

with col4:
    st.metric(
        "Giá bán lại thấp nhất",
        f"{filtered_df['resale_price'].min():,.2f}",
    )


# =====================================================
# 7. PHÂN PHỐI GIÁ BÁN LẠI
# =====================================================

st.divider()

st.subheader("📈 Phân phối giá bán lại")

fig_hist = px.histogram(
    filtered_df,
    x="resale_price",
    nbins=50,
    title="Phân phối giá bán lại điện thoại",
    labels={
        "resale_price": "Giá bán lại",
        "count": "Số lượng",
    },
    color_discrete_sequence=["#2878B5"],
)

fig_hist.update_layout(
    template="plotly_white",
    xaxis_title="Giá bán lại",
    yaxis_title="Số lượng điện thoại",
)

st.plotly_chart(
    fig_hist,
    use_container_width=True,
)


# =====================================================
# 8. SO SÁNH GIÁ THEO THƯƠNG HIỆU
# =====================================================

st.divider()

st.subheader("🏷️ So sánh giá theo thương hiệu")

brand_analysis = (
    filtered_df
    .groupby("brand", as_index=False)
    .agg(
        average_price=("resale_price", "mean"),
        median_price=("resale_price", "median"),
        phone_count=("resale_price", "count"),
    )
)

fig_brand = px.bar(
    brand_analysis,
    x="brand",
    y="average_price",
    color="brand",
    text_auto=".2f",
    title="Giá bán lại trung bình theo thương hiệu",
    labels={
        "brand": "Thương hiệu",
        "average_price": "Giá bán lại trung bình",
    },
)

fig_brand.update_layout(
    template="plotly_white",
    showlegend=False,
)

st.plotly_chart(
    fig_brand,
    use_container_width=True,
)

st.dataframe(
    brand_analysis,
    use_container_width=True,
    hide_index=True,
)


# =====================================================
# 9. PHÂN TÍCH TÌNH TRẠNG ĐIỆN THOẠI
# =====================================================

st.divider()

st.subheader("📱 Phân tích tình trạng điện thoại")

condition_analysis = (
    filtered_df
    .groupby("condition", as_index=False)
    .agg(
        average_price=("resale_price", "mean"),
        phone_count=("resale_price", "count"),
    )
)

col1, col2 = st.columns(2)

with col1:

    fig_condition = px.bar(
        condition_analysis,
        x="condition",
        y="phone_count",
        color="condition",
        title="Số lượng theo tình trạng",
        labels={
            "condition": "Tình trạng",
            "phone_count": "Số lượng",
        },
    )

    fig_condition.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    st.plotly_chart(
        fig_condition,
        use_container_width=True,
    )

with col2:

    fig_condition_price = px.bar(
        condition_analysis,
        x="condition",
        y="average_price",
        color="condition",
        title="Giá trung bình theo tình trạng",
        labels={
            "condition": "Tình trạng",
            "average_price": "Giá trung bình",
        },
    )

    fig_condition_price.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    st.plotly_chart(
        fig_condition_price,
        use_container_width=True,
    )


# =====================================================
# 10. GIÁ GỐC VÀ GIÁ BÁN LẠI
# =====================================================

st.divider()

st.subheader("💰 So sánh giá mua ban đầu và giá bán lại")

scatter_df = filtered_df[
    [
        "original_price",
        "resale_price",
        "brand",
        "condition",
    ]
].dropna()

if len(scatter_df) > 5000:

    scatter_df = scatter_df.sample(
        n=5000,
        random_state=42,
    )

fig_scatter = px.scatter(
    scatter_df,
    x="original_price",
    y="resale_price",
    color="brand",
    symbol="condition",
    title="Mối quan hệ giữa giá mua và giá bán lại",
    labels={
        "original_price": "Giá mua ban đầu",
        "resale_price": "Giá bán lại",
        "brand": "Thương hiệu",
        "condition": "Tình trạng",
    },
)

fig_scatter.update_layout(
    template="plotly_white",
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True,
)


# =====================================================
# 11. PHÂN TÍCH TUỔI ĐIỆN THOẠI
# =====================================================

st.divider()

st.subheader("⏳ Ảnh hưởng của tuổi điện thoại")

age_analysis = (
    filtered_df
    .groupby("age_months", as_index=False)
    .agg(
        average_price=("resale_price", "mean"),
        phone_count=("resale_price", "count"),
    )
    .sort_values("age_months")
)

fig_age = px.line(
    age_analysis,
    x="age_months",
    y="average_price",
    markers=True,
    title="Giá bán lại trung bình theo tuổi điện thoại",
    labels={
        "age_months": "Tuổi điện thoại (tháng)",
        "average_price": "Giá bán lại trung bình",
    },
)

fig_age.update_layout(
    template="plotly_white",
)

st.plotly_chart(
    fig_age,
    use_container_width=True,
)


# =====================================================
# 12. PHÂN TÍCH RAM VÀ BỘ NHỚ
# =====================================================

st.divider()

st.subheader("🧠 Phân tích RAM và bộ nhớ trong")

col1, col2 = st.columns(2)

with col1:

    ram_analysis = (
        filtered_df
        .groupby("ram_gb", as_index=False)
        .agg(
            average_price=("resale_price", "mean"),
            phone_count=("resale_price", "count"),
        )
        .sort_values("ram_gb")
    )

    fig_ram = px.bar(
        ram_analysis,
        x="ram_gb",
        y="average_price",
        title="Giá trung bình theo RAM",
        labels={
            "ram_gb": "RAM (GB)",
            "average_price": "Giá trung bình",
        },
    )

    fig_ram.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(
        fig_ram,
        use_container_width=True,
    )

with col2:

    storage_analysis = (
        filtered_df
        .groupby("storage_gb", as_index=False)
        .agg(
            average_price=("resale_price", "mean"),
            phone_count=("resale_price", "count"),
        )
        .sort_values("storage_gb")
    )

    fig_storage = px.bar(
        storage_analysis,
        x="storage_gb",
        y="average_price",
        title="Giá trung bình theo bộ nhớ",
        labels={
            "storage_gb": "Bộ nhớ trong (GB)",
            "average_price": "Giá trung bình",
        },
    )

    fig_storage.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(
        fig_storage,
        use_container_width=True,
    )


# =====================================================
# 13. PHÂN TÍCH SỨC KHỎE PIN
# =====================================================

st.divider()

st.subheader("🔋 Phân tích sức khỏe pin")

battery_df = filtered_df[
    [
        "battery_health",
        "resale_price",
        "brand",
    ]
].dropna()

if len(battery_df) > 5000:

    battery_df = battery_df.sample(
        n=5000,
        random_state=42,
    )

fig_battery = px.scatter(
    battery_df,
    x="battery_health",
    y="resale_price",
    color="brand",
    title="Sức khỏe pin và giá bán lại",
    labels={
        "battery_health": "Sức khỏe pin (%)",
        "resale_price": "Giá bán lại",
        "brand": "Thương hiệu",
    },
)

fig_battery.update_layout(
    template="plotly_white",
)

st.plotly_chart(
    fig_battery,
    use_container_width=True,
)


# =====================================================
# 14. THỐNG KÊ CHẤT LƯỢNG DATASET
# =====================================================

st.divider()

st.subheader("🧪 Chất lượng dữ liệu")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Tổng bản ghi ban đầu",
        f"{total_rows:,}",
    )

with col2:
    st.metric(
        "Giá trị thiếu ban đầu",
        f"{missing_count:,}",
    )

with col3:
    st.metric(
        "Dòng trùng lặp ban đầu",
        f"{duplicate_count:,}",
    )


# =====================================================
# 15. BẢNG DỮ LIỆU
# =====================================================

st.divider()

st.subheader("📋 Dữ liệu sau khi lọc")

st.dataframe(
    filtered_df.head(1000),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Bảng hiển thị tối đa 1.000 dòng đầu tiên. "
    "Bạn có thể xuất toàn bộ dữ liệu đã lọc bên dưới."
)


# =====================================================
# 16. XUẤT CSV
# =====================================================

st.divider()

st.subheader("📥 Xuất dữ liệu")

csv_data = filtered_df.to_csv(
    index=False,
).encode("utf-8-sig")

st.download_button(
    label="⬇️ Tải dữ liệu đã lọc (CSV)",
    data=csv_data,
    file_name="phonevalue_filtered_data.csv",
    mime="text/csv",
    use_container_width=True,
)


# =====================================================
# 17. THÔNG TIN CUỐI TRANG
# =====================================================

st.divider()

st.caption(
    "PhoneValue AI | Phân tích dữ liệu điện thoại "
    "và giá bán lại bằng Python, Pandas, Streamlit và Plotly."
)

