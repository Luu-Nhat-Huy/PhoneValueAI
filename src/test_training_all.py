
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from preprocessing import (
    TARGET,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


# =====================================================
# 1. CAU HINH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "data" / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

MODEL_DIR = ROOT_DIR / "models"

METADATA_PATH = MODEL_DIR / "model_metadata.json"

REPORT_DIR = ROOT_DIR / "reports"

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regression": "decision_tree_regression.pkl",
    "Random Forest Regression": "random_forest_regression.pkl",
}

SAMPLE_ROWS = 50000
RANDOM_STATE = 42


# =====================================================
# 2. KIEM TRA CAC FILE CAN THIET
# =====================================================

def check_files():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay metadata: {METADATA_PATH}"
        )

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay mo hinh {model_name}: {model_path}"
            )


# =====================================================
# 3. DOC VA CHUAN BI DU LIEU
# =====================================================

def load_test_data():

    print("\nDang doc dataset...")

    required_columns = ALL_FEATURES + [TARGET]

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
    )

    print("Tong so dong ban dau:", f"{len(df):,}")

    # Chuan hoa target
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Loai bo target khong hop le
    df = df.dropna(subset=[TARGET])

    df = df[df[TARGET] >= 0].reset_index(drop=True)

    print("So dong hop le:", f"{len(df):,}")

    # Chuan hoa kieu du lieu giong qua trinh training
    for col in CATEGORICAL_FEATURES:

        df[col] = df[col].astype("category")

    for col in NUMERIC_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).astype("float32")

    df[TARGET] = df[TARGET].astype("float32")

    # Chia tap test theo cung random_state va test_size
    indices = np.arange(len(df))

    _, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("So dong tap test:", f"{len(test_indices):,}")

    # Lay toi da 50.000 dong tu tap test
    sample_size = min(
        SAMPLE_ROWS,
        len(test_indices),
    )

    rng = np.random.default_rng(RANDOM_STATE)

    selected_indices = rng.choice(
        test_indices,
        size=sample_size,
        replace=False,
    )

    test_df = df.iloc[selected_indices].copy()

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]

    print("So mau dung de kiem thu:", f"{len(X_test):,}")

    del df, test_df
    gc.collect()

    return X_test, y_test


# =====================================================
# 4. KIEM THU MOT MO HINH DA HUAN LUYEN
# =====================================================

def evaluate_model(model_name, model_path, X_test, y_test):

    print("\n" + "=" * 65)
    print("KIEM THU:", model_name)
    print("=" * 65)

    # Tai mo hinh
    print("Dang tai mo hinh...")

    load_start = time.perf_counter()

    model = joblib.load(model_path)

    load_time = time.perf_counter() - load_start

    print("Tai mo hinh thanh cong!")

    # Du doan
    print("Dang du doan...")

    predict_start = time.perf_counter()

    predictions = model.predict(X_test)

    predict_time = time.perf_counter() - predict_start

    # Tinh cac chi so
    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2: {r2:.6f}")
    print(f"Thoi gian tai: {load_time:.2f} giay")
    print(f"Thoi gian du doan: {predict_time:.2f} giay")

    result = {
        "model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "load_time_seconds": float(load_time),
        "prediction_time_seconds": float(predict_time),
        "test_samples": int(len(X_test)),
    }

    del model, predictions
    gc.collect()

    return result


# =====================================================
# 5. CHUONG TRINH CHINH
# =====================================================

def main():

    print("=" * 75)
    print("PHONEVALUE AI - KIEM THU 3 MO HINH DA HUAN LUYEN")
    print("=" * 75)

    print("\nCac mo hinh se kiem thu:")

    for index, model_name in enumerate(MODEL_FILES, start=1):

        print(f"{index}. {model_name}")

    print("\nLuu y: Chuong trinh KHONG huan luyen lai mo hinh.")

    # Kiem tra file
    check_files()

    # Doc metadata
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    default_model = metadata.get(
        "selected_default_model",
        "Random Forest Regression",
    )

    print("\nMo hinh mac dinh:", default_model)

    if default_model not in MODEL_FILES:
        raise ValueError(
            f"Mo hinh mac dinh khong hop le: {default_model}"
        )

    # Doc du lieu
    X_test, y_test = load_test_data()

    results = []

    # Kiem thu lan luot 3 mo hinh
    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        result = evaluate_model(
            model_name,
            model_path,
            X_test,
            y_test,
        )

        results.append(result)

    # Tao bang ket qua
    print("\n" + "=" * 75)
    print("BANG SO SANH 3 MO HINH")
    print("=" * 75)

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="MAE",
        ascending=True,
    ).reset_index(drop=True)

    display_df = result_df.copy()

    print(
        display_df.to_string(
            index=False,
            formatters={
                "MAE": "{:,.2f}".format,
                "RMSE": "{:,.2f}".format,
                "R2": "{:.6f}".format,
                "load_time_seconds": "{:.2f}s".format,
                "prediction_time_seconds": "{:.2f}s".format,
            },
        )
    )

    # Mo hinh tot nhat theo MAE
    best = result_df.iloc[0]

    print("\n" + "=" * 75)
    print("MO HINH CO MAE TOT NHAT")
    print("=" * 75)

    print("Ten:", best["model"])
    print(f"MAE: {best['MAE']:,.2f}")
    print(f"RMSE: {best['RMSE']:,.2f}")
    print(f"R2: {best['R2']:.6f}")

    # Luu bao cao
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORT_DIR / "test_training_all_results.csv"

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDa luu bao cao:", output_path)

    print("\n" + "=" * 75)
    print("KIEM THU HOAN TAT!")
    print("Da kiem thu du 3 mo hinh.")
    print("Khong huan luyen lai va khong ghi de mo hinh.")
    print("=" * 75)


if __name__ == "__main__":
    main()
