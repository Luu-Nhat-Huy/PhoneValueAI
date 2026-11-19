
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =====================================================
# 1. CAU HINH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

# Dataset lưu giá theo đơn vị nghìn đồng.
# 1 đơn vị dataset = 1.000 VND.
PRICE_SCALE = 1000


def format_vnd(value):
    """Định dạng giá tiền theo VND."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f} ₫"


def add_vnd_columns(data):
    """Tạo các cột giá VND, không thay đổi dữ liệu gốc."""
    result = data.copy()

    result["original_price_vnd"] = (
        result["original_price"] * PRICE_SCALE
    )

    result["resale_price_vnd"] = (
        result["resale_price"] * PRICE_SCALE
    )

    return result


# =====================================================
# 2. CSS
# =====================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #1976D2;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #777;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 3. DOC DU LIEU
# =====================================================

@st.cache_data(show_spinner="Đang tải dữ liệu 1 triệu điện thoại...")
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATA_PATH}"
        )

    required_columns = [
        "brand",
        "model",
        "release_year",
        "ram_gb",
        "storage_gb",
        "screen_size_inches",
        "battery_capacity",
        "processor_score",
        "camera_score",
        "os_type",
        "has_5g",
        "original_price",
        "purchase_year",
        "age_months",
        "usage_hours_per_day",
        "condition",
        "battery_health",
        "screen_cracked",
        "body_damage",
        "repair_history",
        "water_damage",
        "city_tier",
        "seller_type",
        "warranty_remaining_months",
        "box_available",
        "charger_available",
        "market_demand_score",
        "resale_price",
    ]

    df = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        usecols=required_columns
    )

    numeric_columns = [
        "release_year",
        "ram_gb",
        "storage_gb",
        "screen_size_inches",
        "battery_capacity",
        "processor_score",
        "camera_score",
        "original_price",
        "purchase_year",
        "age_months",
        "usage_hours_per_day",
        "battery_health",
        "warranty_remaining_months",
        "market_demand_score",
        "resale_price",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


try:
    df = load_data()

except Exception as error:
    st.error(f"Không thể tải dữ liệu: {error}")
    st.stop()


# =====================================================
# 4. TIEU DE
# =====================================================

st.markdown(
    '<div class="main-title">📊 PHÂN TÍCH DỮ LIỆU ĐIỆN THOẠI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Khám phá dữ liệu, xu hướng giá và các yếu tố ảnh hưởng '
    'đến giá bán lại điện thoại.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Đơn vị giá hiển thị: VND (₫). "
    "Dataset gốc lưu giá theo nghìn đồng, "
    "hệ thống tự động quy đổi khi hiển thị."
)


# =====================================================
# 5. TONG QUAN DU LIEU
# =====================================================

total_rows = len(df)
total_columns = len(df.columns)
missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tổng số điện thoại", f"{total_rows:,}")
col2.metric("Tổng số thuộc tính", f"{total_columns}")
col3.metric("Giá trị bị thiếu", f"{missing_count:,}")
col4.metric("Dữ liệu trùng lặp", f"{duplicate_count:,}")

st.caption(
    f"Dataset gồm {total_columns - 1} thuộc tính đầu vào "
    f"(Features) và 1 biến mục tiêu (Target: resale_price)."
)


# =====================================================
# 6. BO LOC DU LIEU
# =====================================================

st.markdown(
    '<div class="section-title">🔎 Bộ lọc dữ liệu</div>',
    unsafe_allow_html=True
)

with st.expander("Mở bộ lọc dữ liệu", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        brands = sorted(
            df["brand"].dropna().unique().tolist()
        )

        selected_brands = st.multiselect(
            "Thương hiệu",
            options=brands,
            default=brands
        )

        conditions = sorted(
            df["condition"].dropna().unique().tolist()
        )

        selected_conditions = st.multiselect(
            "Tình trạng điện thoại",
            options=conditions,
            default=conditions
        )

    with col2:

        operating_systems = sorted(
            df["os_type"].dropna().unique().tolist()
        )

        selected_os = st.multiselect(
            "Hệ điều hành",
            options=operating_systems,
            default=operating_systems
        )

        city_tiers = sorted(
            df["city_tier"].dropna().unique().tolist()
        )

        selected_city_tiers = st.multiselect(
            "Phân loại khu vực",
            options=city_tiers,
            default=city_tiers
        )

    with col3:

        min_year = int(df["release_year"].min())
        max_year = int(df["release_year"].max())

        selected_years = st.slider(
            "Năm phát hành",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        valid_prices = df["resale_price"].dropna()

        if valid_prices.empty:
            st.error("Dataset không có dữ liệu giá bán lại hợp lệ.")
            st.stop()

        min_price_vnd = float(
            valid_prices.min() * PRICE_SCALE
        )

        max_price_vnd = float(
            valid_prices.max() * PRICE_SCALE
        )

        selected_prices_vnd = st.slider(
            "Khoảng giá bán lại (VND)",
            min_value=min_price_vnd,
            max_value=max_price_vnd,
            value=(min_price_vnd, max_price_vnd),
            format="%.0f ₫"
        )


# =====================================================
# 7. AP DUNG BO LOC
# =====================================================

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["brand"].isin(selected_brands)
]

filtered_df = filtered_df[
    filtered_df["condition"].isin(selected_conditions)
]

filtered_df = filtered_df[
    filtered_df["os_type"].isin(selected_os)
]

filtered_df = filtered_df[
    filtered_df["city_tier"].isin(selected_city_tiers)
]

filtered_df = filtered_df[
    filtered_df["release_year"].between(
        selected_years[0],
        selected_years[1]
    )
]

# Chuyển khoảng giá VND về đơn vị dataset để lọc.
filtered_df = filtered_df[
    filtered_df["resale_price"].between(
        selected_prices_vnd[0] / PRICE_SCALE,
        selected_prices_vnd[1] / PRICE_SCALE
    )
]

if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc.")
    st.stop()

st.success(
    f"Tìm thấy {len(filtered_df):,} điện thoại phù hợp."
)

# Tạo dữ liệu phục vụ hiển thị và biểu đồ.
# Không ghi đè dữ liệu gốc.
display_df = add_vnd_columns(filtered_df)


# =====================================================
# 8. THONG KE GIA BAN LAI
# =====================================================

st.markdown(
    '<div class="section-title">💰 Thống kê giá bán lại</div>',
    unsafe_allow_html=True
)

price_col1, price_col2, price_col3, price_col4 = st.columns(4)

average_price = display_df["resale_price_vnd"].mean()
median_price = display_df["resale_price_vnd"].median()
min_price = display_df["resale_price_vnd"].min()
max_price = display_df["resale_price_vnd"].max()

price_col1.metric(
    "Giá trung bình",
    format_vnd(average_price)
)

price_col2.metric(
    "Giá trung vị",
    format_vnd(median_price)
)

price_col3.metric(
    "Giá thấp nhất",
    format_vnd(min_price)
)

price_col4.metric(
    "Giá cao nhất",
    format_vnd(max_price)
)


# =====================================================
# 9. PHAN PHOI GIA
# =====================================================

st.markdown(
    '<div class="section-title">📈 Phân phối giá điện thoại</div>',
    unsafe_allow_html=True
)

fig_price = px.histogram(
    display_df,
    x="resale_price_vnd",
    nbins=50,
    title="Phân phối giá bán lại điện thoại",
    labels={
        "resale_price_vnd": "Giá bán lại (VND)",
        "count": "Số lượng điện thoại"
    },
    template="plotly_white"
)

fig_price.update_layout(
    xaxis_title="Giá bán lại (VND)",
    yaxis_title="Số lượng",
    bargap=0.08
)

st.plotly_chart(
    fig_price,
    use_container_width=True
)


# =====================================================
# 10. PHAN TICH THUONG HIEU
# =====================================================

st.markdown(
    '<div class="section-title">📱 Phân tích theo thương hiệu</div>',
    unsafe_allow_html=True
)

brand_analysis = (
    display_df
    .groupby("brand", as_index=False)
    .agg(
        average_price_vnd=("resale_price_vnd", "mean"),
        median_price_vnd=("resale_price_vnd", "median"),
        phone_count=("resale_price_vnd", "count")
    )
    .sort_values(
        "average_price_vnd",
        ascending=False
    )
)

col1, col2 = st.columns(2)

with col1:

    fig_brand_price = px.bar(
        brand_analysis,
        x="brand",
        y="average_price_vnd",
        title="Giá bán lại trung bình theo thương hiệu",
        labels={
            "brand": "Thương hiệu",
            "average_price_vnd": "Giá trung bình (VND)"
        },
        text_auto=".3s",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_brand_price,
        use_container_width=True
    )

with col2:

    fig_brand_count = px.bar(
        brand_analysis,
        x="brand",
        y="phone_count",
        title="Số lượng điện thoại theo thương hiệu",
        labels={
            "brand": "Thương hiệu",
            "phone_count": "Số lượng"
        },
        text_auto=True,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_brand_count,
        use_container_width=True
    )

brand_table = brand_analysis.copy()

brand_table["Giá trung bình (VND)"] = (
    brand_table["average_price_vnd"].apply(format_vnd)
)

brand_table["Giá trung vị (VND)"] = (
    brand_table["median_price_vnd"].apply(format_vnd)
)

brand_table = brand_table[
    [
        "brand",
        "Giá trung bình (VND)",
        "Giá trung vị (VND)",
        "phone_count"
    ]
].rename(
    columns={
        "brand": "Thương hiệu",
        "phone_count": "Số lượng điện thoại"
    }
)

st.dataframe(
    brand_table,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# 11. PHAN TICH TINH TRANG
# =====================================================

st.markdown(
    '<div class="section-title">🔧 Phân tích theo tình trạng</div>',
    unsafe_allow_html=True
)

condition_analysis = (
    display_df
    .groupby("condition", as_index=False)
    .agg(
        average_price_vnd=("resale_price_vnd", "mean"),
        phone_count=("resale_price_vnd", "count")
    )
)

col1, col2 = st.columns(2)

with col1:

    fig_condition_count = px.pie(
        condition_analysis,
        names="condition",
        values="phone_count",
        title="Tỷ lệ điện thoại theo tình trạng",
        hole=0.4
    )

    st.plotly_chart(
        fig_condition_count,
        use_container_width=True
    )

with col2:

    fig_condition_price = px.bar(
        condition_analysis,
        x="condition",
        y="average_price_vnd",
        title="Giá trung bình theo tình trạng",
        labels={
            "condition": "Tình trạng",
            "average_price_vnd": "Giá trung bình (VND)"
        },
        text_auto=".3s",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_condition_price,
        use_container_width=True
    )


# =====================================================
# 12. GIA GOC VA GIA BAN LAI
# =====================================================

st.markdown(
    '<div class="section-title">💵 Giá gốc và giá bán lại</div>',
    unsafe_allow_html=True
)

price_scatter_df = display_df.dropna(
    subset=["original_price_vnd", "resale_price_vnd"]
)

if not price_scatter_df.empty:

    price_scatter_df = price_scatter_df.sample(
        n=min(5000, len(price_scatter_df)),
        random_state=42
    )

    fig_price_relation = px.scatter(
        price_scatter_df,
        x="original_price_vnd",
        y="resale_price_vnd",
        color="brand",
        title="Mối quan hệ giữa giá gốc và giá bán lại",
        labels={
            "original_price_vnd": "Giá gốc (VND)",
            "resale_price_vnd": "Giá bán lại (VND)",
            "brand": "Thương hiệu"
        },
        template="plotly_white",
        opacity=0.6
    )

    st.plotly_chart(
        fig_price_relation,
        use_container_width=True
    )


# =====================================================
# 13. TUOI DOI DIEN THOAI
# =====================================================

st.markdown(
    '<div class="section-title">📅 Tuổi đời điện thoại</div>',
    unsafe_allow_html=True
)

age_analysis = (
    display_df
    .dropna(subset=["age_months", "resale_price_vnd"])
    .groupby("age_months", as_index=False)
    .agg(
        average_price_vnd=("resale_price_vnd", "mean")
    )
    .sort_values("age_months")
)

if not age_analysis.empty:

    fig_age = px.line(
        age_analysis,
        x="age_months",
        y="average_price_vnd",
        markers=True,
        title="Mối quan hệ giữa tuổi đời và giá bán lại",
        labels={
            "age_months": "Tuổi đời (tháng)",
            "average_price_vnd": "Giá bán lại trung bình (VND)"
        },
        template="plotly_white"
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


# =====================================================
# 14. RAM VA BO NHO
# =====================================================

st.markdown(
    '<div class="section-title">🧠 Phân tích RAM và bộ nhớ</div>',
    unsafe_allow_html=True
)

ram_analysis = (
    display_df
    .dropna(subset=["ram_gb"])
    .groupby("ram_gb", as_index=False)
    .agg(
        average_price_vnd=("resale_price_vnd", "mean"),
        phone_count=("resale_price_vnd", "count")
    )
    .sort_values("ram_gb")
)

storage_analysis = (
    display_df
    .dropna(subset=["storage_gb"])
    .groupby("storage_gb", as_index=False)
    .agg(
        average_price_vnd=("resale_price_vnd", "mean"),
        phone_count=("resale_price_vnd", "count")
    )
    .sort_values("storage_gb")
)

col1, col2 = st.columns(2)

with col1:

    fig_ram = px.bar(
        ram_analysis,
        x="ram_gb",
        y="average_price_vnd",
        title="Giá trung bình theo RAM",
        labels={
            "ram_gb": "RAM (GB)",
            "average_price_vnd": "Giá trung bình (VND)"
        },
        text_auto=".3s",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_ram,
        use_container_width=True
    )

with col2:

    fig_storage = px.bar(
        storage_analysis,
        x="storage_gb",
        y="average_price_vnd",
        title="Giá trung bình theo dung lượng bộ nhớ",
        labels={
            "storage_gb": "Bộ nhớ (GB)",
            "average_price_vnd": "Giá trung bình (VND)"
        },
        text_auto=".3s",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_storage,
        use_container_width=True
    )


# =====================================================
# 15. PIN VA GIA BAN LAI
# =====================================================

st.markdown(
    '<div class="section-title">🔋 Phân tích sức khỏe pin</div>',
    unsafe_allow_html=True
)

battery_df = display_df.dropna(
    subset=["battery_health", "resale_price_vnd"]
)

if not battery_df.empty:

    battery_df = battery_df.sample(
        n=min(5000, len(battery_df)),
        random_state=42
    )

    fig_battery = px.scatter(
        battery_df,
        x="battery_health",
        y="resale_price_vnd",
        color="condition",
        title="Sức khỏe pin và giá bán lại",
        labels={
            "battery_health": "Sức khỏe pin (%)",
            "resale_price_vnd": "Giá bán lại (VND)",
            "condition": "Tình trạng"
        },
        opacity=0.6,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_battery,
        use_container_width=True
    )


# =====================================================
# 16. DIEM CPU VA CAMERA
# =====================================================

st.markdown(
    '<div class="section-title">⚙️ Phân tích điểm CPU và Camera</div>',
    unsafe_allow_html=True
)

st.caption(
    "Điểm CPU và Camera trong dataset nằm trong khoảng 40–100. "
    "Đây là thang điểm được sử dụng trong bộ dữ liệu, "
    "không phải chuẩn benchmark chính thức của các hãng."
)

cpu_camera_col1, cpu_camera_col2 = st.columns(2)

with cpu_camera_col1:

    cpu_df = display_df.dropna(
        subset=["processor_score"]
    )

    if not cpu_df.empty:

        fig_cpu = px.histogram(
            cpu_df,
            x="processor_score",
            nbins=30,
            title="Phân phối điểm CPU",
            labels={
                "processor_score": "Điểm CPU",
                "count": "Số lượng"
            },
            template="plotly_white"
        )

        st.plotly_chart(
            fig_cpu,
            use_container_width=True
        )

with cpu_camera_col2:

    camera_df = display_df.dropna(
        subset=["camera_score"]
    )

    if not camera_df.empty:

        fig_camera = px.histogram(
            camera_df,
            x="camera_score",
            nbins=30,
            title="Phân phối điểm Camera",
            labels={
                "camera_score": "Điểm Camera",
                "count": "Số lượng"
            },
            template="plotly_white"
        )

        st.plotly_chart(
            fig_camera,
            use_container_width=True
        )


# =====================================================
# 17. DIEM CPU VA GIA BAN LAI
# =====================================================

st.markdown(
    "#### Mối quan hệ giữa điểm CPU và giá bán lại"
)

cpu_price_df = display_df.dropna(
    subset=[
        "processor_score",
        "resale_price_vnd",
        "camera_score"
    ]
)

if not cpu_price_df.empty:

    cpu_price_df = cpu_price_df.sample(
        n=min(5000, len(cpu_price_df)),
        random_state=42
    )

    fig_cpu_price = px.scatter(
        cpu_price_df,
        x="processor_score",
        y="resale_price_vnd",
        color="camera_score",
        title="Điểm CPU và giá bán lại",
        labels={
            "processor_score": "Điểm CPU",
            "resale_price_vnd": "Giá bán lại (VND)",
            "camera_score": "Điểm Camera"
        },
        opacity=0.65,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_cpu_price,
        use_container_width=True
    )


# =====================================================
# 18. PHAN TICH KHU VUC
# =====================================================

st.markdown(
    '<div class="section-title">🌍 Phân tích theo khu vực</div>',
    unsafe_allow_html=True
)

city_analysis = (
    display_df
    .dropna(subset=["city_tier"])
    .groupby("city_tier", as_index=False)
    .agg(
        average_price_vnd=("resale_price_vnd", "mean"),
        median_price_vnd=("resale_price_vnd", "median"),
        phone_count=("resale_price_vnd", "count")
    )
    .sort_values("city_tier")
)

if not city_analysis.empty:

    col1, col2 = st.columns(2)

    with col1:

        fig_city_count = px.bar(
            city_analysis,
            x="city_tier",
            y="phone_count",
            title="Số lượng điện thoại theo khu vực",
            labels={
                "city_tier": "Phân loại khu vực",
                "phone_count": "Số lượng điện thoại"
            },
            text_auto=True,
            template="plotly_white"
        )

        st.plotly_chart(
            fig_city_count,
            use_container_width=True
        )

    with col2:

        fig_city_price = px.bar(
            city_analysis,
            x="city_tier",
            y="average_price_vnd",
            title="Giá bán lại trung bình theo khu vực",
            labels={
                "city_tier": "Phân loại khu vực",
                "average_price_vnd": "Giá trung bình (VND)"
            },
            text_auto=".3s",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_city_price,
            use_container_width=True
        )

    city_table = city_analysis.copy()

    city_table["Giá trung bình (VND)"] = (
        city_table["average_price_vnd"].apply(format_vnd)
    )

    city_table["Giá trung vị (VND)"] = (
        city_table["median_price_vnd"].apply(format_vnd)
    )

    city_table = city_table[
        [
            "city_tier",
            "Giá trung bình (VND)",
            "Giá trung vị (VND)",
            "phone_count"
        ]
    ].rename(
        columns={
            "city_tier": "Phân loại khu vực",
            "phone_count": "Số lượng điện thoại"
        }
    )

    st.dataframe(
        city_table,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 19. CHAT LUONG DU LIEU
# =====================================================

st.markdown(
    '<div class="section-title">🧹 Kiểm tra chất lượng dữ liệu</div>',
    unsafe_allow_html=True
)

quality_col1, quality_col2, quality_col3 = st.columns(3)

quality_col1.metric(
    "Tổng số dòng ban đầu",
    f"{total_rows:,}"
)

quality_col2.metric(
    "Tổng giá trị bị thiếu",
    f"{missing_count:,}"
)

quality_col3.metric(
    "Số dòng trùng lặp",
    f"{duplicate_count:,}"
)

st.markdown("#### Số lượng giá trị bị thiếu theo thuộc tính")

missing_by_column = (
    df.isna()
    .sum()
    .reset_index()
)

missing_by_column.columns = [
    "column",
    "missing_count"
]

missing_by_column = missing_by_column[
    missing_by_column["missing_count"] > 0
].sort_values(
    "missing_count",
    ascending=False
)

if not missing_by_column.empty:

    fig_missing = px.bar(
        missing_by_column,
        x="column",
        y="missing_count",
        title="Giá trị bị thiếu theo từng thuộc tính",
        labels={
            "column": "Thuộc tính",
            "missing_count": "Số giá trị bị thiếu"
        },
        template="plotly_white"
    )

    st.plotly_chart(
        fig_missing,
        use_container_width=True
    )

else:

    st.success("Dataset không có giá trị bị thiếu.")


# =====================================================
# 20. BANG DU LIEU
# =====================================================

st.markdown(
    '<div class="section-title">📋 Dữ liệu sau khi lọc</div>',
    unsafe_allow_html=True
)

st.caption(
    "Hiển thị tối đa 1.000 dòng đầu tiên. "
    "Các cột giá được quy đổi sang VND."
)

table_df = display_df.head(1000).copy()

# Đổi tên cột gốc để tránh nhầm với VND.
table_df = table_df.rename(
    columns={
        "original_price": "original_price_dataset_unit",
        "resale_price": "resale_price_dataset_unit",
        "original_price_vnd": "original_price_vnd",
        "resale_price_vnd": "resale_price_vnd"
    }
)

# Đưa các cột giá VND lên gần cột giá gốc.
price_columns = [
    "original_price_dataset_unit",
    "original_price_vnd",
    "resale_price_dataset_unit",
    "resale_price_vnd"
]

existing_price_columns = [
    column for column in price_columns
    if column in table_df.columns
]

other_columns = [
    column for column in table_df.columns
    if column not in existing_price_columns
]

table_df = table_df[
    other_columns + existing_price_columns
]

table_df = table_df.rename(
    columns={
        "original_price_dataset_unit": "Giá gốc (đơn vị dataset)",
        "original_price_vnd": "Giá gốc (VND)",
        "resale_price_dataset_unit": "Giá bán lại (đơn vị dataset)",
        "resale_price_vnd": "Giá bán lại (VND)"
    }
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# 21. TAI XUONG DU LIEU
# =====================================================

st.markdown(
    '<div class="section-title">📥 Xuất dữ liệu</div>',
    unsafe_allow_html=True
)

# CSV gồm cả giá gốc theo dataset và giá đã quy đổi VND.
csv_df = display_df.copy()

csv_df = csv_df.rename(
    columns={
        "original_price": "original_price_dataset_unit",
        "resale_price": "resale_price_dataset_unit"
    }
)

csv_data = csv_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="⬇️ Tải xuống dữ liệu CSV",
    data=csv_data,
    file_name="phonevalue_filtered_data_vnd.csv",
    mime="text/csv",
    use_container_width=True
)


# =====================================================
# 22. GHI CHU
# =====================================================

st.markdown("---")

st.caption(
    "PhoneValue AI | Phân tích dữ liệu điện thoại đã qua sử dụng | "
    "Đơn vị giá hiển thị: VND"
)
