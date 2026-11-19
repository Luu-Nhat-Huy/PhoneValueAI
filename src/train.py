
import gc
import json
import os
import shutil
import tempfile
import time
import ctypes

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

MODEL_DIR = ROOT_DIR / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Dung lai neu RAM kha dung thap.
MIN_FREE_RAM_GB = 2.5


# =====================================================
# 2. KIEM TRA RAM
# =====================================================

def get_free_ram_gb():
    """
    Lay dung luong RAM con trong tren Windows.
    """

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

        result = ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        )

        if result:
            return status.ullAvailPhys / (1024 ** 3)

    except Exception:
        pass

    # Phuong an du phong neu khong lay duoc RAM bang Windows API.
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return None


def check_memory(model_name):
    free_ram = get_free_ram_gb()

    print("\nKIEM TRA RAM:")
    print(f"Mo hinh: {model_name}")

    if free_ram is None:
        print("Khong lay duoc thong tin RAM.")
        print("Hay kiem tra RAM thu cong truoc khi tiep tuc.")
        return

    print(f"RAM kha dung: {free_ram:.2f} GB")

    if free_ram < MIN_FREE_RAM_GB:
        raise MemoryError(
            f"RAM con trong {free_ram:.2f} GB, "
            f"thap hon nguong {MIN_FREE_RAM_GB:.2f} GB. "
            "Dung huan luyen de tranh thieu bo nho."
        )


# =====================================================
# 3. DOC VA KIEM TRA DU LIEU
# =====================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    print("\nDANG DOC DATASET...")

    dtype_config = {
        column: "category"
        for column in CATEGORICAL_FEATURES
    }

    dtype_config.update({
        column: "float32"
        for column in NUMERIC_FEATURES
    })

    dtype_config[TARGET] = "float32"

    df = pd.read_csv(
        DATA_PATH,
        dtype=dtype_config,
        low_memory=False,
    )

    print(f"So dong ban dau: {len(df):,}")

    required_columns = ALL_FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset thieu cot: "
            + ", ".join(missing_columns)
        )

    # Chi giu cac cot can thiet.
    df = df[required_columns]

    # Loai bo target khong hop le.
    target = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    valid_mask = (
        target.notna()
        & np.isfinite(target)
        & (target >= 0)
    )

    df = df.loc[valid_mask].copy()

    df[TARGET] = target.loc[valid_mask].astype("float32")

    if df.empty:
        raise ValueError("Khong con du lieu hop le.")

    print(f"So dong hop le: {len(df):,}")
    print(f"So dac trung: {len(ALL_FEATURES)}")

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    print(f"Dung luong DataFrame: {memory_mb:.2f} MB")

    return df


# =====================================================
# 4. DANH GIA MO HINH
# =====================================================

def evaluate_model(y_test, predictions):

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

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# =====================================================
# 5. HUAN LUYEN MOT MO HINH
# =====================================================

def train_one_model(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    temp_model_dir,
):

    print("\n" + "=" * 70)
    print(f"BAT DAU HUAN LUYEN: {model_name}")
    print("=" * 70)

    check_memory(model_name)

    pipeline = create_model_pipeline(model_name)

    start_time = time.time()

    print("\nDang fit mo hinh...")

    pipeline.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    print("Dang du doan tren tap test...")

    predictions = pipeline.predict(X_test)

    metrics = evaluate_model(
        y_test,
        predictions,
    )

    model_filename = MODEL_CONFIGS[model_name]["file"]

    temp_model_path = temp_model_dir / model_filename

    print("Dang luu mo hinh vao thu muc tam...")

    joblib.dump(
        pipeline,
        temp_model_path,
        compress=3,
    )

    result = {
        "model_name": model_name,
        "model_file": model_filename,
        "training_time_seconds": float(training_time),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        **metrics,
    }

    print("\nKET QUA:")
    print(f"MAE : {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R2  : {metrics['r2']:.6f}")
    print(f"Thoi gian: {training_time:.2f} giay")
    print(f"File tam: {temp_model_path}")

    # Giai phong pipeline va du lieu du doan.
    del pipeline
    del predictions

    gc.collect()

    return result


# =====================================================
# 6. LUU METADATA
# =====================================================

def create_metadata(results, train_rows, test_rows):

    best_model = min(
        results,
        key=lambda item: item["mae"],
    )

    return {
        "project_name": "PhoneValue AI",
        "model_type": "Multiple Regression Models",
        "features": ALL_FEATURES,
        "target": TARGET,
        "dataset_rows": int(train_rows + test_rows),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "selected_default_model": best_model["model_name"],
        "best_model": best_model,
        "models": results,
    }


# =====================================================
# 7. CONG BO CAC MO HINH
# =====================================================

def publish_models(temp_model_dir, metadata):

    print("\n" + "=" * 70)
    print("DANG CONG BO CAC MO HINH...")
    print("=" * 70)

    temp_metadata = temp_model_dir / "model_metadata.json"

    with open(
        temp_metadata,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # Chi cong bo sau khi ca 3 mo hinh da duoc huan luyen.
    for model_name, config in MODEL_CONFIGS.items():

        filename = config["file"]

        source = temp_model_dir / filename
        destination = MODEL_DIR / filename

        if not source.exists():
            raise FileNotFoundError(
                f"Khong tim thay mo hinh tam: {source}"
            )

        os.replace(
            source,
            destination,
        )

        print(f"Da cong bo: {destination}")

    # Metadata luon duoc ghi cuoi cung.
    os.replace(
        temp_metadata,
        METADATA_PATH,
    )

    print(f"Da cap nhat metadata: {METADATA_PATH}")


# =====================================================
# 8. CHUONG TRINH CHINH
# =====================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PHONEVALUE AI - HUAN LUYEN 3 MO HINH")
    print("DATASET MUC TIEU: 1.000.000 DONG")
    print("=" * 70)

    check_memory("Doc dataset")

    df = load_dataset()

    print("\nDANG CHIA TAP TRAIN/TEST...")

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(train_indices):,} dong")
    print(f"Test : {len(test_indices):,} dong")

    # Tao cac tap du lieu dung mot lan.
    X_train = df.iloc[train_indices][ALL_FEATURES]
    X_test = df.iloc[test_indices][ALL_FEATURES]

    y_train = df[TARGET].iloc[train_indices]
    y_test = df[TARGET].iloc[test_indices]

    # Giai phong DataFrame goc va chi so.
    del df
    del indices
    del train_indices
    del test_indices

    gc.collect()

    results = []

    # Tao thu muc tam trong models.
    temp_model_dir = Path(
        tempfile.mkdtemp(
            prefix="training_temp_",
            dir=MODEL_DIR,
        )
    )

    try:

        # Huấn luyện lần lượt đúng 3 mô hình.
        for model_name in MODEL_CONFIGS:

            result = train_one_model(
                model_name,
                X_train,
                X_test,
                y_train,
                y_test,
                temp_model_dir,
            )

            results.append(result)

            gc.collect()

        # Tao metadata sau khi ca 3 mo hinh thanh cong.
        metadata = create_metadata(
            results,
            len(X_train),
            len(X_test),
        )

        # Giai phong tap train/test truoc khi cong bo.
        del X_train
        del X_test
        del y_train
        del y_test

        gc.collect()

        publish_models(
            temp_model_dir,
            metadata,
        )

        print("\n" + "=" * 75)
        print("BANG SO SANH 3 MO HINH")
        print("=" * 75)

        comparison = pd.DataFrame(results)

        print(
            comparison[
                [
                    "model_name",
                    "mae",
                    "rmse",
                    "r2",
                    "training_time_seconds",
                ]
            ].to_string(index=False)
        )

        print("\nMO HINH CO MAE THAP NHAT:")
        print(metadata["best_model"]["model_name"])
        print(f"MAE : {metadata['best_model']['mae']:,.2f}")
        print(f"RMSE: {metadata['best_model']['rmse']:,.2f}")
        print(f"R2  : {metadata['best_model']['r2']:.6f}")

        print("\nHUAN LUYEN HOAN TAT!")

        print("\nCAC MO HINH:")
        for config in MODEL_CONFIGS.values():
            print(f"- {MODEL_DIR / config['file']}")

        print(f"\nMetadata: {METADATA_PATH}")

    finally:

        # Don thu muc tam neu con file.
        if temp_model_dir.exists():
            shutil.rmtree(
                temp_model_dir,
                ignore_errors=True,
            )

        gc.collect()


# =====================================================
# 9. CHAY CHUONG TRINH
# =====================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n" + "=" * 70)
        print("DA XAY RA LOI KHI HUAN LUYEN")
        print("=" * 70)

        print(f"{type(error).__name__}: {error}")

        raise
