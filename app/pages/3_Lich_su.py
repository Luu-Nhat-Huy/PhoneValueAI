
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ IMPORT
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
# 2. CẤU HÌNH TRANG
# =====================================================

st.title("🕘 Lịch sử định giá")

st.caption(
    "Quản lý, tìm kiếm và xem lại các lần định giá "
    "điện thoại đã thực hiện."
)

st.divider()


# =====================================================
# 3. HÀM ĐỌC LỊCH SỬ
# =====================================================

@st.cache_data(ttl=5)
def load_history():

    records = get_all_valuations()

    if records is None or len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "phone_data" in df.columns:

        def parse_phone_data(value):

            if isinstance(value, dict):
                return value

            if not isinstance(value, str):
                return {}

            try:
                result = json.loads(value)

                if isinstance(result, dict):
                    return result

                return {}

            except (json.JSONDecodeError, TypeError):
                return {}

        df["phone_data_parsed"] = df[
            "phone_data"
        ].apply(parse_phone_data)

        for column in [
            "brand",
            "model",
            "condition",
        ]:

            if column not in df.columns:

                df[column] = df[
                    "phone_data_parsed"
                ].apply(
                    lambda x: x.get(column, "")
                )

    if "created_at" in df.columns:

        df["created_at_parsed"] = pd.to_datetime(
            df["created_at"],
            errors="coerce",
        )

    if "predicted_price" in df.columns:

        df["predicted_price"] = pd.to_numeric(
            df["predicted_price"],
            errors="coerce",
        )

    if "original_price" in df.columns:

        df["original_price"] = pd.to_numeric(
            df["original_price"],
            errors="coerce",
        )

    return df


# =====================================================
# 4. TẢI DỮ LIỆU
# =====================================================

try:

    history_df = load_history()

except Exception as error:

    st.error(
        f"Không thể tải lịch sử định giá: {error}"
    )

    st.stop()


if history_df.empty:

    st.info(
        "Chưa có lịch sử định giá nào. "
        "Hãy đến trang Định giá điện thoại để thực hiện "
        "lần định giá đầu tiên."
    )

    st.stop()


# =====================================================
# 5. THỐNG KÊ TỔNG QUAN
# =====================================================

st.subheader("📊 Tổng quan lịch sử")

total_valuations = len(history_df)

average_price = history_df[
    "predicted_price"
].mean()

highest_price = history_df[
    "predicted_price"
].max()

lowest_price = history_df[
    "predicted_price"
].min()

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Tổng lần định giá",
        f"{total_valuations:,}",
    )

with col2:

    st.metric(
        "Giá trung bình",
        (
            f"{average_price:,.2f}"
            if pd.notna(average_price)
            else "N/A"
        ),
    )

with col3:

    st.metric(
        "Giá cao nhất",
        (
            f"{highest_price:,.2f}"
            if pd.notna(highest_price)
            else "N/A"
        ),
    )

with col4:

    st.metric(
        "Giá thấp nhất",
        (
            f"{lowest_price:,.2f}"
            if pd.notna(lowest_price)
            else "N/A"
        ),
    )


# =====================================================
# 6. BỘ LỌC LỊCH SỬ
# =====================================================

st.divider()

st.subheader("🔎 Tìm kiếm và lọc")

search_keyword = st.text_input(
    "Tìm theo thương hiệu hoặc model",
    placeholder="Ví dụ: Samsung, iPhone, Realme...",
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

    selected_condition = st.multiselect(
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

    selected_brand = st.multiselect(
        "Thương hiệu",
        options=brands,
        default=brands,
    )


# =====================================================
# 7. LỌC THEO NGÀY
# =====================================================

filtered_df = history_df.copy()

if not filtered_df["created_at_parsed"].dropna().empty:

    min_date = (
        filtered_df["created_at_parsed"]
        .min()
        .date()
    )

    max_date = (
        filtered_df["created_at_parsed"]
        .max()
        .date()
    )

    date_range = st.date_input(
        "Khoảng thời gian định giá",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)):

        if len(date_range) == 2:

            start_date, end_date = date_range

            filtered_df = filtered_df[
                filtered_df[
                    "created_at_parsed"
                ].dt.date.between(
                    start_date,
                    end_date,
                )
            ]


# =====================================================
# 8. ÁP DỤNG BỘ LỌC
# =====================================================

if not selected_condition:
    filtered_df = filtered_df.iloc[0:0]

else:
    filtered_df = filtered_df[
        filtered_df["condition"].isin(
            selected_condition
        )
    ]


if not selected_brand:
    filtered_df = filtered_df.iloc[0:0]

else:
    filtered_df = filtered_df[
        filtered_df["brand"].isin(
            selected_brand
        )
    ]


if search_keyword.strip():

    keyword = search_keyword.strip().lower()

    search_columns = [
        "brand",
        "model",
    ]

    search_mask = pd.Series(
        False,
        index=filtered_df.index,
    )

    for column in search_columns:

        if column in filtered_df.columns:

            search_mask |= (
                filtered_df[column]
                .astype(str)
                .str.lower()
                .str.contains(
                    keyword,
                    na=False,
                    regex=False,
                )
            )

    filtered_df = filtered_df[
        search_mask
    ]


# =====================================================
# 9. HIỂN THỊ BẢNG LỊCH SỬ
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

    display_columns = [
        "id",
        "created_at",
        "brand",
        "model",
        "condition",
        "original_price",
        "predicted_price",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    display_df = filtered_df[
        available_columns
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

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# =====================================================
# 10. XUẤT CSV
# =====================================================

st.divider()

st.subheader("📥 Xuất lịch sử")

if not filtered_df.empty:

    export_df = filtered_df.drop(
        columns=["phone_data_parsed"],
        errors="ignore",
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


# =====================================================
# 11. XEM CHI TIẾT BẢN GHI
# =====================================================

st.divider()

st.subheader("🔍 Chi tiết lần định giá")

if not filtered_df.empty:

    available_ids = filtered_df["id"].tolist()

    selected_id = st.selectbox(
        "Chọn bản ghi muốn xem",
        options=available_ids,
        format_func=lambda x: f"Mã định giá #{x}",
    )

    selected_record = filtered_df[
        filtered_df["id"] == selected_id
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Giá dự đoán",
            (
                f"{selected_record['predicted_price']:,.2f}"
                if pd.notna(
                    selected_record["predicted_price"]
                )
                else "N/A"
            ),
        )

    with col2:

        original = selected_record.get(
            "original_price",
            None,
        )

        if pd.notna(original):

            st.metric(
                "Giá mua ban đầu",
                f"{original:,.2f}",
            )

    phone_data = selected_record.get(
        "phone_data_parsed",
        {},
    )

    if not isinstance(phone_data, dict):
        phone_data = {}

    with st.expander(
        "Xem thông số điện thoại"
    ):

        st.json(phone_data)

    with st.expander(
        "Xem thông tin bản ghi"
    ):

        st.json(
            {
                "id": selected_record.get("id"),
                "created_at": selected_record.get(
                    "created_at"
                ),
                "brand": selected_record.get("brand"),
                "model": selected_record.get("model"),
                "condition": selected_record.get(
                    "condition"
                ),
                "original_price": selected_record.get(
                    "original_price"
                ),
                "predicted_price": selected_record.get(
                    "predicted_price"
                ),
            }
        )


# =====================================================
# 12. XÓA MỘT BẢN GHI
# =====================================================

st.divider()

st.subheader("🗑️ Quản lý lịch sử")

if not filtered_df.empty:

    st.warning(
        "Thao tác xóa bản ghi sẽ không thể tự động hoàn tác."
    )

    confirm_delete = st.checkbox(
        "Tôi xác nhận muốn xóa một bản ghi.",
        key="confirm_delete_one",
    )

    if st.button(
        "Xóa bản ghi đang chọn",
        disabled=not confirm_delete,
        use_container_width=True,
    ):

        try:

            delete_valuation(
                int(selected_id)
            )

            load_history.clear()

            st.session_state[
                "confirm_delete_one"
            ] = False

            st.success(
                f"Đã xóa bản ghi #{selected_id}."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Không thể xóa bản ghi: {error}"
            )


# =====================================================
# 13. XÓA TOÀN BỘ LỊCH SỬ
# =====================================================

st.divider()

with st.expander(
    "⚠️ Xóa toàn bộ lịch sử định giá"
):

    st.error(
        "Cảnh báo: Thao tác này sẽ xóa tất cả "
        "bản ghi lịch sử trong database."
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

            st.session_state[
                "confirm_clear_all"
            ] = False

            st.success(
                "Đã xóa toàn bộ lịch sử định giá."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Không thể xóa lịch sử: {error}"
            )


# =====================================================
# 14. KẾT THÚC
# =====================================================

st.divider()

st.caption(
    "PhoneValue AI | Quản lý lịch sử định giá điện thoại."
)
