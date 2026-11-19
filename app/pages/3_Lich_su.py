
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# 1. CAU HINH DUONG DAN VA IMPORT DATABASE
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "app"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from database import (
    get_all_valuations,
    delete_valuation,
    clear_history,
)


# =====================================================
# 2. CAU HINH DON VI GIA
# =====================================================

# 1 don vi dataset = 1.000 VND
PRICE_SCALE = 1000


def format_vnd(value):
    """Chuyen gia tri dataset sang VND de hien thi."""
    try:
        if pd.isna(value):
            return "N/A"

        amount = float(value) * PRICE_SCALE
        return f"{amount:,.0f} ₫"

    except (TypeError, ValueError):
        return "N/A"


def safe_float(value):
    """Chuyen gia tri sang so thuc an toan."""
    try:
        if value is None or pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


# =====================================================
# 3. CSS
# =====================================================

st.markdown(
    """
    <style>
    .history-title {
        font-size: 32px;
        font-weight: 800;
        color: #1976D2;
        margin-bottom: 5px;
    }

    .history-description {
        color: #777;
        font-size: 15px;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 4. HAM PHAN TICH PHONE_DATA
# =====================================================

def parse_phone_data(value):

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    try:
        result = json.loads(value)

        if isinstance(result, dict):
            return result

    except (json.JSONDecodeError, TypeError):
        pass

    return {}


# =====================================================
# 5. CHUAN HOA DU LIEU LICH SU
# =====================================================

def normalize_history(records):

    if records is None or len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Phan tich phone_data
    if "phone_data" in df.columns:
        df["phone_data_parsed"] = df["phone_data"].apply(
            parse_phone_data
        )
    else:
        df["phone_data_parsed"] = [
            {} for _ in range(len(df))
        ]

    # Lay thong tin tu phone_data neu chua co cot truc tiep
    fields = [
        "brand",
        "model",
        "condition",
        "processor_score",
        "camera_score",
        "city_tier",
        "ram_gb",
        "storage_gb",
    ]

    for field in fields:

        if field not in df.columns:

            df[field] = df["phone_data_parsed"].apply(
                lambda data: data.get(field, "")
                if isinstance(data, dict)
                else ""
            )

        else:

            missing_mask = (
                df[field].isna()
                | df[field].astype(str).str.strip().eq("")
            )

            df.loc[missing_mask, field] = (
                df.loc[missing_mask, "phone_data_parsed"].apply(
                    lambda data: data.get(field, "")
                    if isinstance(data, dict)
                    else ""
                )
            )

    # Dam bao cac cot quan trong luon ton tai
    required_columns = [
        "id",
        "created_at",
        "brand",
        "model",
        "condition",
        "original_price",
        "predicted_price",
        "phone_data_parsed",
    ]

    for column in required_columns:

        if column not in df.columns:
            df[column] = None

    # Chuyen doi kieu du lieu
    numeric_columns = [
        "predicted_price",
        "original_price",
        "processor_score",
        "camera_score",
        "ram_gb",
        "storage_gb",
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Chuyen ngay gio
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )

    return df


# =====================================================
# 6. DOC LICH SU
# =====================================================

@st.cache_data(ttl=5)
def load_history():

    records = get_all_valuations()

    return normalize_history(records)


try:

    history_df = load_history()

except Exception as error:

    st.error(f"Không thể tải lịch sử định giá: {error}")
    st.stop()


# =====================================================
# 7. TIEU DE TRANG
# =====================================================

st.markdown(
    '<div class="history-title">🕘 LỊCH SỬ ĐỊNH GIÁ</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="history-description">'
    'Quản lý, tìm kiếm và xem lại các lần định giá '
    'điện thoại đã thực hiện bằng PhoneValue AI.'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Tất cả giá tiền trên trang được hiển thị bằng VNĐ. "
    "Dữ liệu gốc của mô hình vẫn được giữ nguyên."
)

st.divider()


# =====================================================
# 8. KIEM TRA LICH SU
# =====================================================

if history_df.empty:

    st.info(
        "Chưa có lịch sử định giá nào. "
        "Hãy đến trang Định giá để thực hiện lần định giá đầu tiên."
    )

    st.stop()


# =====================================================
# 9. THONG KE TONG QUAN
# =====================================================

st.subheader("📊 Tổng quan lịch sử")

total_valuations = len(history_df)

average_price = history_df["predicted_price"].mean()
highest_price = history_df["predicted_price"].max()
lowest_price = history_df["predicted_price"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Tổng lần định giá",
    f"{total_valuations:,}",
)

col2.metric(
    "Giá trung bình",
    format_vnd(average_price),
)

col3.metric(
    "Giá cao nhất",
    format_vnd(highest_price),
)

col4.metric(
    "Giá thấp nhất",
    format_vnd(lowest_price),
)


# =====================================================
# 10. BO LOC VA TIM KIEM
# =====================================================

st.divider()

st.subheader("🔎 Tìm kiếm và lọc lịch sử")

search_keyword = st.text_input(
    "Tìm theo thương hiệu hoặc model",
    placeholder="Ví dụ: Samsung, iPhone, Xiaomi...",
)

col1, col2 = st.columns(2)

with col1:

    conditions = sorted(
        history_df["condition"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_conditions = st.multiselect(
        "Tình trạng điện thoại",
        options=conditions,
        default=conditions,
    )

with col2:

    brands = sorted(
        history_df["brand"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_brands = st.multiselect(
        "Thương hiệu",
        options=brands,
        default=brands,
    )


# =====================================================
# 11. LOC THEO NGAY
# =====================================================

filtered_df = history_df.copy()

valid_dates = filtered_df[
    "created_at_parsed"
].dropna()

if not valid_dates.empty:

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    date_range = st.date_input(
        "Khoảng thời gian định giá",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)):

        if len(date_range) == 2:

            start_date, end_date = date_range

            if start_date and end_date:

                date_mask = (
                    filtered_df["created_at_parsed"]
                    .dt.date
                    .between(start_date, end_date)
                )

                filtered_df = filtered_df[
                    date_mask
                    | filtered_df["created_at_parsed"].isna()
                ]


# =====================================================
# 12. LOC THEO THUONG HIEU VA TINH TRANG
# =====================================================

if selected_conditions:

    filtered_df = filtered_df[
        filtered_df["condition"]
        .astype(str)
        .isin(selected_conditions)
    ]

else:

    filtered_df = filtered_df.iloc[0:0]


if selected_brands:

    filtered_df = filtered_df[
        filtered_df["brand"]
        .astype(str)
        .isin(selected_brands)
    ]

else:

    filtered_df = filtered_df.iloc[0:0]


# =====================================================
# 13. TIM KIEM THEO TU KHOA
# =====================================================

if search_keyword.strip():

    keyword = search_keyword.strip()

    search_mask = (
        filtered_df["brand"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        )
        |
        filtered_df["model"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        )
    )

    filtered_df = filtered_df[search_mask]


# =====================================================
# 14. DANH SACH LICH SU
# =====================================================

st.divider()

st.subheader("📋 Danh sách lịch sử")

st.write(
    f"Tìm thấy **{len(filtered_df):,}** bản ghi."
)

if filtered_df.empty:

    st.warning(
        "Không tìm thấy lịch sử phù hợp với bộ lọc."
    )

else:

    display_df = filtered_df[
        [
            "id",
            "created_at",
            "brand",
            "model",
            "condition",
            "original_price",
            "predicted_price",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "id": "ID",
            "created_at": "Thời gian",
            "brand": "Thương hiệu",
            "model": "Model",
            "condition": "Tình trạng",
            "original_price": "Giá mua ban đầu",
            "predicted_price": "Giá dự đoán",
        }
    )

    display_df["Giá mua ban đầu"] = (
        display_df["Giá mua ban đầu"].apply(format_vnd)
    )

    display_df["Giá dự đoán"] = (
        display_df["Giá dự đoán"].apply(format_vnd)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# =====================================================
# 15. XUAT LICH SU CSV
# =====================================================

st.divider()

st.subheader("📥 Xuất lịch sử")

if not filtered_df.empty:

    export_df = filtered_df.drop(
        columns=[
            "phone_data_parsed",
            "created_at_parsed",
        ],
        errors="ignore",
    ).copy()

    # Gia goc theo don vi dataset
    if "original_price" in export_df.columns:

        export_df["original_price_dataset_units"] = (
            export_df["original_price"]
        )

        export_df["original_price_vnd"] = (
            export_df["original_price"] * PRICE_SCALE
        )

        export_df = export_df.drop(
            columns=["original_price"]
        )

    if "predicted_price" in export_df.columns:

        export_df["predicted_price_dataset_units"] = (
            export_df["predicted_price"]
        )

        export_df["predicted_price_vnd"] = (
            export_df["predicted_price"] * PRICE_SCALE
        )

        export_df = export_df.drop(
            columns=["predicted_price"]
        )

    # Chuyen dictionary/list thanh JSON
    for column in export_df.columns:

        if export_df[column].dtype == "object":

            export_df[column] = export_df[column].apply(
                lambda value: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
            )

    csv_data = export_df.to_csv(
        index=False,
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Tải lịch sử định giá CSV",
        data=csv_data,
        file_name="phonevalue_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "CSV chứa giá gốc theo đơn vị dataset và giá đã quy đổi sang VNĐ."
    )


# =====================================================
# 16. XEM CHI TIET BAN GHI
# =====================================================

st.divider()

st.subheader("🔍 Chi tiết lần định giá")

selected_id = None

if not filtered_df.empty:

    available_ids = (
        filtered_df["id"]
        .dropna()
        .tolist()
    )

    if available_ids:

        selected_id = st.selectbox(
            "Chọn bản ghi muốn xem",
            options=available_ids,
            format_func=lambda value: f"Mã định giá #{value}",
        )

        selected_record = filtered_df[
            filtered_df["id"] == selected_id
        ].iloc[0]

        predicted = selected_record.get("predicted_price")
        original = selected_record.get("original_price")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Giá dự đoán",
                format_vnd(predicted),
            )

        with col2:

            st.metric(
                "Giá mua ban đầu",
                format_vnd(original),
            )

        phone_data = selected_record.get(
            "phone_data_parsed",
            {},
        )

        if not isinstance(phone_data, dict):
            phone_data = {}

        with st.expander("📱 Thông số điện thoại"):

            if phone_data:

                display_phone_data = phone_data.copy()

                raw_original_price = safe_float(
                    display_phone_data.pop(
                        "original_price",
                        None,
                    )
                )

                if raw_original_price is not None:

                    display_phone_data[
                        "original_price_vnd"
                    ] = raw_original_price * PRICE_SCALE

                st.json(display_phone_data)

                st.caption(
                    "Giá trong phần thông số đã được quy đổi sang VNĐ. "
                    "Dữ liệu lưu trong database vẫn giữ nguyên."
                )

            else:

                st.info(
                    "Bản ghi này không lưu thông số điện thoại."
                )

        with st.expander("🗂️ Thông tin bản ghi"):

            original_value = safe_float(original)
            predicted_value = safe_float(predicted)

            record_info = {
                "id": selected_record.get("id"),
                "created_at": selected_record.get("created_at"),
                "brand": selected_record.get("brand"),
                "model": selected_record.get("model"),
                "condition": selected_record.get("condition"),
                "original_price_dataset_units": original_value,
                "original_price_vnd": (
                    original_value * PRICE_SCALE
                    if original_value is not None
                    else None
                ),
                "predicted_price_dataset_units": predicted_value,
                "predicted_price_vnd": (
                    predicted_value * PRICE_SCALE
                    if predicted_value is not None
                    else None
                ),
            }

            st.json(record_info)

    else:

        st.info(
            "Các bản ghi hiện tại không có ID để xem chi tiết."
        )


# =====================================================
# 17. XOA MOT BAN GHI
# =====================================================

st.divider()

st.subheader("🗑️ Quản lý lịch sử")

if not filtered_df.empty:

    available_ids = (
        filtered_df["id"]
        .dropna()
        .tolist()
    )

    if available_ids and selected_id is not None:

        st.warning(
            "Thao tác xóa bản ghi sẽ không thể tự động hoàn tác."
        )

        confirm_delete = st.checkbox(
            "Tôi xác nhận muốn xóa bản ghi đang chọn.",
            key="confirm_delete_one",
        )

        if st.button(
            "Xóa bản ghi đang chọn",
            disabled=not confirm_delete,
            use_container_width=True,
        ):

            try:

                delete_valuation(int(selected_id))

                load_history.clear()

                st.success(
                    f"Đã xóa bản ghi #{selected_id}."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Không thể xóa bản ghi: {error}"
                )


# =====================================================
# 18. XOA TOAN BO LICH SU
# =====================================================

st.divider()

with st.expander("⚠️ Xóa toàn bộ lịch sử định giá"):

    st.error(
        "Cảnh báo: Thao tác này sẽ xóa tất cả bản ghi "
        "lịch sử định giá trong database."
    )

    confirm_clear = st.checkbox(
        "Tôi hiểu và muốn xóa toàn bộ lịch sử.",
        key="confirm_clear_all",
    )

    if st.button(
        "XÓA TOÀN BỘ LỊCH SỬ",
        disabled=not confirm_clear,
        type="primary",
        use_container_width=True,
    ):

        try:

            clear_history()

            load_history.clear()

            st.success(
                "Đã xóa toàn bộ lịch sử định giá."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Không thể xóa lịch sử: {error}"
            )


# =====================================================
# 19. FOOTER
# =====================================================

st.divider()

st.caption(
    "PhoneValue AI | Quản lý lịch sử định giá điện thoại."
)
