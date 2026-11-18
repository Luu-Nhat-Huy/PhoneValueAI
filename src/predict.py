
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"

METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)


# =====================================================
# 2. TẢI MÔ HÌNH
# =====================================================

def load_model():
    """
    Tải mô hình và metadata.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay metadata: {METADATA_PATH}"
        )

    try:
        model = joblib.load(MODEL_PATH)

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

    except Exception as error:
        raise RuntimeError(
            f"Khong the tai mo hinh: {error}"
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError(
            "Metadata khong dung dinh dang."
        )

    features = metadata.get("features")

    if not isinstance(features, list) or not features:
        raise ValueError(
            "Metadata khong co danh sach features hop le."
        )

    return model, metadata


# =====================================================
# 3. KIỂM TRA DỮ LIỆU ĐẦU VÀO
# =====================================================

def validate_phone_data(phone_data, features):
    """
    Kiểm tra dữ liệu điện thoại trước khi dự đoán.
    """

    if not isinstance(phone_data, dict):
        raise TypeError(
            "Du lieu dau vao phai la dictionary."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in phone_data
    ]

    if missing_features:
        raise ValueError(
            "Thieu cac dac trung: "
            + ", ".join(missing_features)
        )

    cleaned_data = {}

    for feature in features:

        value = phone_data[feature]

        if value is None:
            raise ValueError(
                f"Dac trung '{feature}' khong duoc de trong."
            )

        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"Dac trung '{feature}' khong duoc de trong."
            )

        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                raise ValueError(
                    f"Dac trung '{feature}' co gia tri khong hop le."
                )

        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                raise ValueError(
                    f"Dac trung '{feature}' co gia tri khong hop le."
                )

        cleaned_data[feature] = value

    return cleaned_data


# =====================================================
# 4. DỰ ĐOÁN GIÁ ĐIỆN THOẠI
# =====================================================

def predict_phone(phone_data):
    """
    Dự đoán giá bán lại của điện thoại.

    Parameters
    ----------
    phone_data : dict
        Dữ liệu đặc trưng của điện thoại.

    Returns
    -------
    float
        Giá điện thoại dự đoán.
    """

    model, metadata = load_model()

    features = metadata["features"]

    cleaned_data = validate_phone_data(
        phone_data,
        features,
    )

    input_df = pd.DataFrame(
        [cleaned_data],
        columns=features,
    )

    try:
        prediction = model.predict(input_df)

    except Exception as error:
        raise RuntimeError(
            f"Loi trong qua trinh du doan: {error}"
        ) from error

    predicted_price = float(prediction[0])

    if not np.isfinite(predicted_price):
        raise ValueError(
            "Mo hinh tra ve ket qua du doan khong hop le."
        )

    return max(0.0, predicted_price)


# =====================================================
# 5. KIỂM TRA MÔ HÌNH BẰNG DỮ LIỆU THỰC
# =====================================================

def test_prediction():

    print("=" * 60)
    print("PHONEVALUE AI - KIEM TRA CHUC NANG DU DOAN")
    print("=" * 60)

    # Tải mô hình
    model, metadata = load_model()

    print("\nTai mo hinh thanh cong!")

    print(
        f"Mo hinh: {metadata.get('model_name', 'Unknown')}"
    )

    # Kiểm tra dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    # Đọc một dòng dữ liệu
    df = pd.read_csv(
        DATA_PATH,
        nrows=1,
    )

    features = metadata["features"]

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Dataset thieu cac dac trung: "
            + ", ".join(missing_features)
        )

    if "resale_price" not in df.columns:
        raise ValueError(
            "Dataset khong co cot resale_price."
        )

    if df.empty:
        raise ValueError(
            "Khong co du lieu kiem tra."
        )

    # Lấy dữ liệu đầu vào
    phone_data = df.iloc[0][features].to_dict()

    actual_price = float(
        df.iloc[0]["resale_price"]
    )

    # Hiển thị thông tin
    print("\nTHONG TIN KIEM TRA")
    print("-" * 60)

    for key, value in phone_data.items():
        print(f"{key}: {value}")

    # Dự đoán
    predicted_price = predict_phone(
        phone_data
    )

    error = abs(
        actual_price - predicted_price
    )

    error_percent = (
        error / abs(actual_price) * 100
        if actual_price != 0
        else 0
    )

    # Kết quả
    print("\nKET QUA")
    print("-" * 60)

    print(
        f"Gia thuc te : {actual_price:,.2f}"
    )

    print(
        f"Gia du doan : {predicted_price:,.2f}"
    )

    print(
        f"Sai lech     : {error:,.2f}"
    )

    print(
        f"Ty le sai lech: {error_percent:.2f}%"
    )

    print("\nKiem tra chuc nang du doan hoan tat!")


# =====================================================
# 6. CHẠY CHƯƠNG TRÌNH
# =====================================================

if __name__ == "__main__":

    try:
        test_prediction()

    except Exception as error:

        print("\n" + "=" * 60)
        print("DA XAY RA LOI")
        print("=" * 60)

        print(error)

        sys.exit(1)

