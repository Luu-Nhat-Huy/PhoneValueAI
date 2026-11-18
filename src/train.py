
import json
import time
from datetime import datetime
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
    create_model_pipeline,
)


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =====================================================
# 2. HÀM KIỂM TRA DỮ LIỆU
# =====================================================

def validate_data(df):
    """
    Kiểm tra dữ liệu trước khi huấn luyện.
    """

    required_columns = ALL_FEATURES + [TARGET]

    # Kiểm tra các cột bắt buộc
    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Du lieu thieu cac cot: "
            + ", ".join(missing_columns)
        )

    # Kiểm tra dataset rỗng
    if df.empty:
        raise ValueError(
            "Dataset khong co du lieu."
        )

    # Kiểm tra giá trị thiếu ở biến mục tiêu
    if df[TARGET].isna().any():
        raise ValueError(
            "Cot resale_price co gia tri thieu."
        )

    # Kiểm tra giá trị vô hạn ở biến mục tiêu
    if not np.isfinite(df[TARGET]).all():
        raise ValueError(
            "Cot resale_price co gia tri khong hop le."
        )

    # Kiểm tra giá bán lại âm
    if (df[TARGET] < 0).any():
        raise ValueError(
            "Cot resale_price co gia tri am."
        )

    print("Kiem tra du lieu thanh cong!")


# =====================================================
# 3. HÀM TÍNH TOÁN CHỈ SỐ ĐÁNH GIÁ
# =====================================================

def evaluate_model(y_test, predictions):
    """
    Tính các chỉ số MAE, RMSE và R2.
    """

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# =====================================================
# 4. HÀM LƯU METADATA
# =====================================================

def save_metadata(
    metrics,
    train_rows,
    test_rows,
    training_time,
):
    """
    Lưu thông tin mô hình và kết quả đánh giá.
    """

    metadata = {
        "model_name": "HistGradientBoostingRegressor",

        "target": TARGET,

        "features": ALL_FEATURES,

        "categorical_features": CATEGORICAL_FEATURES,

        "numeric_features": NUMERIC_FEATURES,

        "train_rows": int(train_rows),

        "test_rows": int(test_rows),

        "total_rows": int(train_rows + test_rows),

        "test_size": TEST_SIZE,

        "random_state": RANDOM_STATE,

        "training_time_seconds": round(
            training_time,
            2,
        ),

        "training_time_minutes": round(
            training_time / 60,
            2,
        ),

        "metrics": {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
        },

        "trained_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"Da luu metadata: {METADATA_PATH}"
    )


# =====================================================
# 5. CHƯƠNG TRÌNH HUẤN LUYỆN
# =====================================================

def main():

    start_time = time.time()

    print("=" * 65)
    print("PHONEVALUE AI - HUAN LUYEN MO HINH")
    print("=" * 65)

    # -------------------------------------------------
    # BƯỚC 1: KIỂM TRA FILE
    # -------------------------------------------------

    print("\n[1] Kiem tra duong dan...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Dataset da duoc tim thay!")

    # -------------------------------------------------
    # BƯỚC 2: ĐỌC DỮ LIỆU
    # -------------------------------------------------

    print("\n[2] Dang doc dataset...")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print(f"So dong: {len(df):,}")
    print(f"So cot: {len(df.columns)}")

    # -------------------------------------------------
    # BƯỚC 3: KIỂM TRA DỮ LIỆU
    # -------------------------------------------------

    print("\n[3] Kiem tra du lieu...")

    validate_data(df)

    # Chỉ sử dụng những đặc trưng đã xác định
    X = df[ALL_FEATURES].copy()

    # Biến mục tiêu
    y = df[TARGET].copy()

    # -------------------------------------------------
    # BƯỚC 4: CHIA TRAIN VÀ TEST
    # -------------------------------------------------

    print("\n[4] Chia tap train va test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(X_train):,} dong")
    print(f"Test : {len(X_test):,} dong")

    # -------------------------------------------------
    # BƯỚC 5: KHỞI TẠO PIPELINE
    # -------------------------------------------------

    print("\n[5] Khoi tao mo hinh...")

    pipeline = create_model_pipeline()

    print(
        "Mo hinh: HistGradientBoostingRegressor"
    )

    print(
        "Tien xu ly: OneHotEncoder + Numeric passthrough"
    )

    # -------------------------------------------------
    # BƯỚC 6: HUẤN LUYỆN
    # -------------------------------------------------

    print("\n[6] Bat dau huan luyen...")

    print(
        "Qua trinh co the mat mot khoang thoi gian."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Huan luyen hoan tat!")

    # -------------------------------------------------
    # BƯỚC 7: DỰ ĐOÁN TRÊN TẬP TEST
    # -------------------------------------------------

    print("\n[7] Danh gia mo hinh...")

    predictions = pipeline.predict(
        X_test,
    )

    # -------------------------------------------------
    # BƯỚC 8: TÍNH CHỈ SỐ
    # -------------------------------------------------

    metrics = evaluate_model(
        y_test,
        predictions,
    )

    print("\n" + "=" * 65)
    print("KET QUA DANH GIA")
    print("=" * 65)

    print(
        f"MAE  : {metrics['mae']:,.4f}"
    )

    print(
        f"RMSE : {metrics['rmse']:,.4f}"
    )

    print(
        f"R2   : {metrics['r2']:.6f}"
    )

    # -------------------------------------------------
    # BƯỚC 9: LƯU MÔ HÌNH
    # -------------------------------------------------

    print("\n[8] Luu mo hinh...")

    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3,
    )

    print(
        f"Da luu mo hinh: {MODEL_PATH}"
    )

    # -------------------------------------------------
    # BƯỚC 10: LƯU METADATA
    # -------------------------------------------------

    print("\n[9] Luu metadata...")

    elapsed_time = time.time() - start_time

    save_metadata(
        metrics=metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
        training_time=elapsed_time,
    )

    # -------------------------------------------------
    # BƯỚC 11: HOÀN TẤT
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("HUAN LUYEN THANH CONG!")
    print("=" * 65)

    print(
        f"Thoi gian: {elapsed_time / 60:.2f} phut"
    )

    print(
        f"MAE : {metrics['mae']:,.2f}"
    )

    print(
        f"RMSE: {metrics['rmse']:,.2f}"
    )

    print(
        f"R2  : {metrics['r2']:.6f}"
    )

    print("\nCac file da tao:")

    print("- models/best_model.pkl")

    print("- models/model_metadata.json")

    print("\nPhoneValue AI - Hoan tat huan luyen!")


# =====================================================
# 6. CHẠY CHƯƠNG TRÌNH
# =====================================================

if __name__ == "__main__":
    main()

