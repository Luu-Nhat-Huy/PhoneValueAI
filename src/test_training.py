
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    TARGET,
    ALL_FEATURES,
    MODEL_CONFIGS,
    create_model_pipeline,
)


# =====================================================
# 1. CAU HINH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

SAMPLE_ROWS = 50000
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =====================================================
# 2. KIEM THU CAC MO HINH
# =====================================================

def main():

    print("=" * 60)
    print("PHONEVALUE AI - KIEM THU 3 MO HINH")
    print("=" * 60)

    # -------------------------------------------------
    # Doc toan bo dataset va lay mau ngau nhien
    # -------------------------------------------------

    print("\nDang doc toan bo dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Tong so dong dataset:", f"{len(df):,}")

    df = df.sample(
        n=min(SAMPLE_ROWS, len(df)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(
        f"Da lay ngau nhien {len(df):,} dong."
    )

    # -------------------------------------------------
    # Kiem tra du lieu
    # -------------------------------------------------

    print("\nDang kiem tra du lieu...")

    required_columns = ALL_FEATURES + [TARGET]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thieu cac cot: {missing_columns}"
        )

    df = df[required_columns].copy()

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[TARGET]
    )

    df = df[
        df[TARGET] >= 0
    ]

    print("So dong hop le:", f"{len(df):,}")
    print("So cot dau vao:", len(ALL_FEATURES))

    if len(df) < 100:
        raise ValueError(
            "Du lieu hop le qua it de kiem thu."
        )

    # -------------------------------------------------
    # Chia tap train va test
    # -------------------------------------------------

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nSo dong huan luyen:", f"{len(X_train):,}")
    print("So dong kiem thu:", f"{len(X_test):,}")

    del df
    gc.collect()

    # -------------------------------------------------
    # Huan luyen tung mo hinh
    # -------------------------------------------------

    results = []

    for model_name in MODEL_CONFIGS:

        print("\n" + "=" * 60)
        print("DANG KIEM THU:", model_name)
        print("=" * 60)

        model = create_model_pipeline(model_name)

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter() - start_time
        )

        predictions = model.predict(X_test)

        # -------------------------------------------------
        # Danh gia mo hinh
        # -------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        result = {
            "model": model_name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "training_time": float(training_time),
        }

        results.append(result)

        # -------------------------------------------------
        # In ket qua
        # -------------------------------------------------

        print(f"MAE: {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R2: {r2:.4f}")

        print(
            f"Thoi gian huan luyen: {training_time:.2f} giay"
        )

        del model
        del predictions

        gc.collect()

    # -------------------------------------------------
    # So sanh 3 mo hinh
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("KET QUA SO SANH 3 MO HINH")
    print("=" * 60)

    result_df = pd.DataFrame(results)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "MAE": "{:,.2f}".format,
                "RMSE": "{:,.2f}".format,
                "R2": "{:.4f}".format,
                "training_time": "{:.2f}s".format,
            }
        )
    )

    # -------------------------------------------------
    # Tim mo hinh tot nhat
    # -------------------------------------------------

    best_model = result_df.loc[
        result_df["MAE"].idxmin()
    ]

    print("\n" + "=" * 60)
    print("MO HINH TOT NHAT TRONG LAN KIEM THU")
    print("=" * 60)

    print("Ten mo hinh:", best_model["model"])
    print(f"MAE: {best_model['MAE']:,.2f}")
    print(f"RMSE: {best_model['RMSE']:,.2f}")
    print(f"R2: {best_model['R2']:.4f}")

    print("\nKIEM THU HOAN TAT.")
    print("Khong co mo hinh nao bi ghi de.")


# =====================================================
# 3. CHAY CHUONG TRINH
# =====================================================

if __name__ == "__main__":
    main()
