
from pathlib import Path
import json
import pandas as pd

# ==========================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "raw" / "used_phone_price_prediction_1M.csv.gz"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "resale_price"

EXPECTED_COLUMNS = [
    "brand",
    "model",
    "ram_gb",
    "storage_gb",
    "screen_size_inches",
    "battery_capacity",
    "processor_score",
    "camera_score",
    "os_type",
    "has_5g",
    "original_price",
    "release_year",
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
    "market_demand_score",
    "warranty_remaining_months",
    "box_available",
    "charger_available",
    "resale_price"
]


def main():
    print("=" * 65)
    print("PHONEVALUE AI - PHAN TICH DU LIEU")
    print("=" * 65)

    # ==========================================
    # 2. KIỂM TRA FILE
    # ==========================================

    if not DATA_FILE.exists():
        print(f"KHONG TIM THAY FILE: {DATA_FILE}")
        return

    print("\n[1] Dang doc du lieu...")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    print("Doc du lieu thanh cong!")

    # ==========================================
    # 3. KÍCH THƯỚC DỮ LIỆU
    # ==========================================

    print("\n[2] KICH THUOC DU LIEU")
    print("-" * 65)

    rows, columns = df.shape

    print(f"So dong: {rows:,}")
    print(f"So cot: {columns}")

    # ==========================================
    # 4. DANH SÁCH CỘT
    # ==========================================

    print("\n[3] DANH SACH CAC COT")
    print("-" * 65)

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    unexpected_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    print("\nCot thieu so voi danh sach du kien:", missing_columns)
    print("Cot ngoai danh sach du kien:", unexpected_columns)

    # ==========================================
    # 5. KIỂU DỮ LIỆU
    # ==========================================

    print("\n[4] KIEU DU LIEU")
    print("-" * 65)

    print(df.dtypes.to_string())

    # ==========================================
    # 6. KIỂM TRA GIÁ TRỊ THIẾU
    # ==========================================

    print("\n[5] KIEM TRA DU LIEU THIEU")
    print("-" * 65)

    missing = df.isnull().sum()

    missing_report = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percent": (
            missing.values / len(df) * 100
            if len(df) > 0 else 0
        )
    })

    missing_report = missing_report.sort_values(
        by="missing_count",
        ascending=False
    )

    print(missing_report.to_string(index=False))

    missing_report.to_csv(
        REPORT_DIR / "missing_values.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # 7. KIỂM TRA DỮ LIỆU TRÙNG LẶP
    # ==========================================

    print("\n[6] KIEM TRA DU LIEU TRUNG LAP")
    print("-" * 65)

    duplicate_count = int(df.duplicated().sum())

    print(f"So dong trung lap: {duplicate_count:,}")

    # ==========================================
    # 8. THỐNG KÊ CÁC CỘT SỐ
    # ==========================================

    print("\n[7] THONG KE CAC COT SO")
    print("-" * 65)

    numeric_df = df.select_dtypes(include="number")

    numeric_summary = numeric_df.describe().T

    print(numeric_summary.to_string())

    numeric_summary.to_csv(
        REPORT_DIR / "numeric_summary.csv",
        encoding="utf-8-sig"
    )

    # ==========================================
    # 9. PHÂN TÍCH BIẾN MỤC TIÊU
    # ==========================================

    target_summary = None

    print("\n[8] PHAN TICH GIA DIEN THOAI CU")
    print("-" * 65)

    if TARGET in df.columns:
        target = df[TARGET]

        target_summary = {
            "count": int(target.count()),
            "missing": int(target.isnull().sum()),
            "mean": float(target.mean()),
            "median": float(target.median()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
            "q25": float(target.quantile(0.25)),
            "q75": float(target.quantile(0.75))
        }

        for key, value in target_summary.items():
            print(f"{key}: {value:,.2f}" if isinstance(value, float)
                  else f"{key}: {value:,}")

    else:
        print(f"KHONG TIM THAY COT {TARGET}")

    # ==========================================
    # 10. GIÁ BÁN LẠI CAO HƠN GIÁ GỐC
    # ==========================================

    print("\n[9] KIEM TRA GIA BAN LAI CAO HON GIA GOC")
    print("-" * 65)

    greater_count = None
    greater_percent = None

    if "original_price" in df.columns and TARGET in df.columns:

        valid_prices = df[
            ["original_price", TARGET]
        ].dropna()

        greater = valid_prices[
            valid_prices[TARGET] > valid_prices["original_price"]
        ]

        greater_count = len(greater)

        greater_percent = (
            greater_count / len(valid_prices) * 100
            if len(valid_prices) > 0 else 0
        )

        print(f"So truong hop: {greater_count:,}")
        print(f'Ty le: {greater_percent:.2f}%')

        greater.head(1000).to_csv(
            REPORT_DIR / "resale_price_greater_than_original.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # ==========================================
    # 11. KIỂM TRA CÁC CỘT PHÂN LOẠI
    # ==========================================

    print("\n[10] GIA TRI CAC COT PHAN LOAI")
    print("-" * 65)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    categorical_summary = {}

    for column in categorical_columns:

        unique_values = df[column].dropna().unique()

        categorical_summary[column] = [
            str(value) for value in unique_values
        ]

        print(f"\n{column}:")
        print(f"So gia tri khac nhau: {len(unique_values)}")

        if len(unique_values) <= 50:
            print(list(unique_values))
        else:
            print(list(unique_values[:50]))
            print("Chi hien thi 50 gia tri dau tien.")

    # ==========================================
    # 12. KIỂM TRA GIÁ TRỊ BẤT THƯỜNG CƠ BẢN
    # ==========================================

    print("\n[11] KIEM TRA GIA TRI BAT THUONG")
    print("-" * 65)

    checks = {}

    numeric_checks = {
        "battery_health": (0, 100),
        "usage_hours_per_day": (0, 24),
        "warranty_remaining_months": (0, None),
        "age_months": (0, None),
        "ram_gb": (0, None),
        "storage_gb": (0, None),
        "original_price": (0, None),
        "resale_price": (0, None)
    }

    for column, (minimum, maximum) in numeric_checks.items():

        if column not in df.columns:
            continue

        invalid = pd.Series(False, index=df.index)

        if minimum is not None:
            invalid |= df[column].notna() & (df[column] < minimum)

        if maximum is not None:
            invalid |= df[column].notna() & (df[column] > maximum)

        count = int(invalid.sum())

        checks[column] = count

        print(f"{column}: {count:,} gia tri ngoai khoang kiem tra")

    # ==========================================
    # 13. KIỂM TRA TUỔI ĐIỆN THOẠI
    # ==========================================

    print("\n[12] KIEM TRA TUOI DIEN THOAI")
    print("-" * 65)

    age_check_count = None

    required_age_columns = [
        "age_months",
        "purchase_year",
        "release_year"
    ]

    if all(col in df.columns for col in required_age_columns):

        expected_age = (
            df["purchase_year"] - df["release_year"]
        ) * 12

        age_difference = (
            df["age_months"] - expected_age
        ).abs()

        age_check_count = int(
            (age_difference > 1).sum()
        )

        print(
            "So dong co age_months khac biet tren 1 thang "
            f"so voi purchase_year - release_year: {age_check_count:,}"
        )

        print(
            "Luu y: Day la phep kiem tra so bo, "
            "khong tu dong sua du lieu."
        )

    # ==========================================
    # 14. LƯU BÁO CÁO TỔNG HỢP
    # ==========================================

    summary = {
        "dataset": DATA_FILE.name,
        "rows": int(rows),
        "columns": int(columns),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": duplicate_count,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "target": target_summary,
        "resale_price_greater_than_original_count": greater_count,
        "resale_price_greater_than_original_percent": greater_percent,
        "range_checks": checks,
        "age_inconsistency_count": age_check_count,
        "categorical_values": categorical_summary
    }

    with open(
        REPORT_DIR / "data_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    # ==========================================
    # 15. KẾT THÚC
    # ==========================================

    print("\n" + "=" * 65)
    print("HOAN THANH PHAN TICH DU LIEU!")
    print("=" * 65)

    print(f"Bao cao duoc luu tai: {REPORT_DIR}")

    print("\nCac file bao cao:")
    print("- data_summary.json")
    print("- missing_values.csv")
    print("- numeric_summary.csv")
    print("- resale_price_greater_than_original.csv")


if __name__ == "__main__":
    main()

