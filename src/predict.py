
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =====================================================
# 1. CAU HINH DUONG DAN VA MO HINH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

EXPECTED_MODELS = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regression": "decision_tree_regression.pkl",
    "Random Forest Regression": "random_forest_regression.pkl",
}

DEFAULT_MODEL = "Random Forest Regression"


# =====================================================
# 2. DOC VA KIEM TRA METADATA
# =====================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay metadata: {METADATA_PATH}"
        )

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    except Exception as error:
        raise RuntimeError(
            f"Khong the doc metadata: {error}"
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError("Metadata khong dung dinh dang.")

    features = metadata.get("features")

    if not isinstance(features, list) or not features:
        raise ValueError(
            "Metadata khong co danh sach features hop le."
        )

    if len(features) != len(set(features)):
        raise ValueError(
            "Metadata co dac trung bi trung lap."
        )

    models = metadata.get("models")

    if not isinstance(models, list):
        raise ValueError(
            "Metadata khong co danh sach 3 mo hinh hop le."
        )

    model_names = {
        item.get("model_name")
        for item in models
        if isinstance(item, dict)
    }

    expected_names = set(EXPECTED_MODELS.keys())

    if model_names != expected_names:
        raise ValueError(
            "Danh sach mo hinh trong metadata khong dung.\n"
            f"Can co: {sorted(expected_names)}\n"
            f"Hien tai: {sorted(model_names)}"
        )

    for item in models:

        expected_file = EXPECTED_MODELS[item["model_name"]]

        if item.get("model_file") != expected_file:
            raise ValueError(
                f"File mo hinh khong dung cho {item['model_name']}."
            )

    return metadata


# =====================================================
# 3. LAY DANH SACH MO HINH
# =====================================================

def get_available_models():

    return list(EXPECTED_MODELS.keys())


# =====================================================
# 4. LAY MO HINH MAC DINH
# =====================================================

def get_default_model():

    metadata = load_metadata()

    selected_model = metadata.get("selected_default_model")

    if selected_model not in EXPECTED_MODELS:
        raise ValueError(
            f"Mo hinh mac dinh khong hop le: {selected_model}"
        )

    return selected_model


# =====================================================
# 5. TAI MO HINH
# =====================================================

def load_model(model_name=None):

    metadata = load_metadata()

    if model_name is None:
        model_name = metadata.get("selected_default_model")

    if model_name not in EXPECTED_MODELS:
        raise ValueError(
            f"Mo hinh khong duoc ho tro: {model_name}\n"
            f"Cac mo hinh hien co: {list(EXPECTED_MODELS.keys())}"
        )

    model_filename = EXPECTED_MODELS[model_name]

    model_path = MODEL_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file mo hinh: {model_path}"
        )

    try:
        model = joblib.load(model_path)

    except Exception as error:
        raise RuntimeError(
            f"Khong the tai mo hinh {model_name}: {error}"
        ) from error

    return model, metadata


# =====================================================
# 6. KIEM TRA DU LIEU DAU VAO
# =====================================================

def validate_phone_data(phone_data, features):

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

        if isinstance(value, (int, float, np.integer, np.floating)):

            if not np.isfinite(value):
                raise ValueError(
                    f"Dac trung '{feature}' co gia tri khong hop le."
                )

        cleaned_data[feature] = value

    return cleaned_data


# =====================================================
# 7. DU DOAN GIA DIEN THOAI
# =====================================================

def predict_phone(phone_data, model_name=None):

    model, metadata = load_model(model_name)

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
# 8. KIEM TRA CA 3 MO HINH
# =====================================================

def test_prediction():

    print("=" * 65)
    print("PHONEVALUE AI - KIEM TRA CA 3 MO HINH")
    print("=" * 65)

    metadata = load_metadata()

    print("\nTai metadata thanh cong!")

    models = get_available_models()

    print("\nDanh sach mo hinh:")

    for name in models:
        print(f"- {name}")

    selected_model = get_default_model()

    print(f"\nMo hinh mac dinh: {selected_model}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, nrows=1)

    if df.empty:
        raise ValueError(
            "Khong co du lieu kiem tra."
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

    phone_data = df.iloc[0][features].to_dict()

    actual_price = float(df.iloc[0]["resale_price"])

    if not np.isfinite(actual_price):
        raise ValueError("Gia thuc te khong hop le.")

    print("\nTHONG TIN KIEM TRA")
    print("-" * 65)

    for key, value in phone_data.items():
        print(f"{key}: {value}")

    print("\nKET QUA KIEM TRA")
    print("-" * 65)

    for index, name in enumerate(models, start=1):

        predicted_price = predict_phone(
            phone_data,
            model_name=name,
        )

        error = abs(actual_price - predicted_price)

        error_percent = (
            error / abs(actual_price) * 100
            if actual_price != 0
            else 0
        )

        print(f"\n[{index}/3] {name}")
        print(f"Gia thuc te : {actual_price:,.2f}")
        print(f"Gia du doan : {predicted_price:,.2f}")
        print(f"Sai lech     : {error:,.2f}")
        print(f"Ty le sai lech: {error_percent:.2f}%")
        print("Trang thai   : THANH CONG")

    print("\n" + "=" * 65)
    print("DA KIEM TRA THANH CONG CA 3 MO HINH!")
    print(f"Mo hinh mac dinh: {selected_model}")
    print("=" * 65)


# =====================================================
# 9. CHAY CHUONG TRINH
# =====================================================

if __name__ == "__main__":

    try:
        test_prediction()

    except Exception as error:

        print("\n" + "=" * 65)
        print("DA XAY RA LOI")
        print("=" * 65)

        print(error)

        sys.exit(1)
