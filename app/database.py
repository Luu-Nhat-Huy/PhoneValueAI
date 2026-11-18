
import sqlite3
import json

from pathlib import Path
from datetime import datetime


# =====================================================
# 1. CẤU HÌNH DATABASE
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = ROOT_DIR / "phonevalue_history.db"


# =====================================================
# 2. KẾT NỐI DATABASE
# =====================================================

def get_connection():
    """
    Tạo kết nối SQLite.

    row_factory giúp truy cập dữ liệu theo tên cột.
    """

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    return connection


# =====================================================
# 3. KHỞI TẠO DATABASE
# =====================================================

def initialize_database():
    """
    Tạo bảng lịch sử định giá nếu chưa tồn tại.
    Tạo index để tăng tốc truy vấn.
    """

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS valuation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,

                brand TEXT NOT NULL,

                model TEXT NOT NULL,

                condition TEXT NOT NULL,

                original_price REAL NOT NULL,

                predicted_price REAL NOT NULL,

                phone_data TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_valuation_created_at
            ON valuation_history(created_at)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_valuation_brand
            ON valuation_history(brand)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_valuation_model
            ON valuation_history(model)
            """
        )

        connection.commit()


# =====================================================
# 4. KIỂM TRA DỮ LIỆU ĐỊNH GIÁ
# =====================================================

def validate_valuation(phone_data, predicted_price):
    """
    Kiểm tra dữ liệu trước khi lưu vào database.
    """

    if not isinstance(phone_data, dict):
        raise TypeError(
            "Thông tin điện thoại phải có dạng dictionary."
        )

    required_fields = [
        "brand",
        "model",
        "condition",
        "original_price",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in phone_data
    ]

    if missing_fields:
        raise ValueError(
            "Thiếu thông tin: "
            + ", ".join(missing_fields)
        )

    if not str(phone_data["brand"]).strip():
        raise ValueError("Thương hiệu không được để trống.")

    if not str(phone_data["model"]).strip():
        raise ValueError("Model không được để trống.")

    try:
        original_price = float(
            phone_data["original_price"]
        )

        predicted_price = float(predicted_price)

    except (TypeError, ValueError):
        raise ValueError(
            "Giá điện thoại phải là dữ liệu số."
        )

    if original_price <= 0:
        raise ValueError(
            "Giá mua ban đầu phải lớn hơn 0."
        )

    if predicted_price < 0:
        raise ValueError(
            "Giá dự đoán không được nhỏ hơn 0."
        )

    return original_price, predicted_price


# =====================================================
# 5. LƯU LỊCH SỬ ĐỊNH GIÁ
# =====================================================

def save_valuation(phone_data, predicted_price):
    """
    Lưu một lần định giá vào SQLite.
    """

    original_price, predicted_price = validate_valuation(
        phone_data,
        predicted_price,
    )

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    phone_data_json = json.dumps(
        phone_data,
        ensure_ascii=False,
        default=str,
    )

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO valuation_history (
                created_at,
                brand,
                model,
                condition,
                original_price,
                predicted_price,
                phone_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(phone_data["brand"]).strip(),
                str(phone_data["model"]).strip(),
                str(phone_data["condition"]).strip(),
                original_price,
                predicted_price,
                phone_data_json,
            ),
        )

        connection.commit()

        return cursor.lastrowid


# =====================================================
# 6. LẤY TOÀN BỘ LỊCH SỬ
# =====================================================

def get_all_valuations():
    """
    Lấy toàn bộ lịch sử, mới nhất nằm ở đầu danh sách.
    """

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM valuation_history
            ORDER BY id DESC
            """
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# =====================================================
# 7. LẤY MỘT BẢN GHI THEO ID
# =====================================================

def get_valuation_by_id(valuation_id):
    """
    Lấy chi tiết một lần định giá.
    Trả về None nếu không tìm thấy.
    """

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT *
            FROM valuation_history
            WHERE id = ?
            """,
            (int(valuation_id),),
        ).fetchone()

    return dict(row) if row else None


# =====================================================
# 8. TÌM KIẾM LỊCH SỬ
# =====================================================

def search_valuations(
    keyword=None,
    brand=None,
    condition=None,
):
    """
    Tìm kiếm lịch sử theo từ khóa, thương hiệu,
    hoặc tình trạng điện thoại.
    """

    query = """
        SELECT *
        FROM valuation_history
        WHERE 1 = 1
    """

    parameters = []

    if keyword and str(keyword).strip():

        search_keyword = f"%{str(keyword).strip()}%"

        query += """
            AND (
                brand LIKE ?
                OR model LIKE ?
            )
        """

        parameters.extend(
            [
                search_keyword,
                search_keyword,
            ]
        )

    if brand and brand != "Tất cả":

        query += " AND brand = ?"

        parameters.append(brand)

    if condition and condition != "Tất cả":

        query += " AND condition = ?"

        parameters.append(condition)

    query += " ORDER BY id DESC"

    with get_connection() as connection:

        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# =====================================================
# 9. XÓA MỘT BẢN GHI
# =====================================================

def delete_valuation(valuation_id):
    """
    Xóa một lần định giá theo ID.
    """

    with get_connection() as connection:

        cursor = connection.execute(
            """
            DELETE FROM valuation_history
            WHERE id = ?
            """,
            (int(valuation_id),),
        )

        connection.commit()

        return cursor.rowcount > 0


# =====================================================
# 10. XÓA TOÀN BỘ LỊCH SỬ
# =====================================================

def clear_history():
    """
    Xóa tất cả bản ghi lịch sử.
    """

    with get_connection() as connection:

        connection.execute(
            "DELETE FROM valuation_history"
        )

        connection.commit()


# =====================================================
# 11. THỐNG KÊ LỊCH SỬ
# =====================================================

def get_history_statistics():
    """
    Trả về các chỉ số tổng quan của lịch sử định giá.
    """

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_records,
                AVG(predicted_price) AS average_price,
                MIN(predicted_price) AS minimum_price,
                MAX(predicted_price) AS maximum_price
            FROM valuation_history
            """
        ).fetchone()

    return {
        "total_records": int(row["total_records"] or 0),
        "average_price": float(row["average_price"] or 0),
        "minimum_price": float(row["minimum_price"] or 0),
        "maximum_price": float(row["maximum_price"] or 0),
    }


# =====================================================
# 12. XUẤT LỊCH SỬ DẠNG CSV
# =====================================================

def export_history_csv():
    """
    Xuất lịch sử thành nội dung CSV dạng UTF-8 BOM.
    Có thể sử dụng trực tiếp cho st.download_button.
    """

    import csv
    import io

    history = get_all_valuations()

    output = io.StringIO()

    fieldnames = [
        "id",
        "created_at",
        "brand",
        "model",
        "condition",
        "original_price",
        "predicted_price",
        "phone_data",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for row in history:
        writer.writerow(row)

    return output.getvalue().encode("utf-8-sig")
