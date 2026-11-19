
import gc
import json
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


# =====================================================
# 1. CAU HINH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)

MODEL_DIR = BASE_DIR / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

REPORT_DIR = BASE_DIR / "reports"
OUTPUT_CSV = REPORT_DIR / "model_overfitting_evaluation.csv"

SAMPLE_SIZE = 20000
RANDOM_STATE = 42

EXPECTED_MODELS = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regression": "decision_tree_regression.pkl",
    "Random Forest Regression": "random_forest_regression.pkl",
}


# =====================================================
# 2. DOC METADATA
# =====================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay metadata: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata khong hop le.")

    features = metadata.get("features")

    if not isinstance(features, list) or not features:
        raise ValueError("Metadata khong co features hop le.")

    models = metadata.get("models")

    if not isinstance(models, list):
        raise ValueError("Metadata khong co danh sach mo hinh.")

    actual_models = {
        item.get("model_name"): item.get("model_file")
        for item in models
        if isinstance(item, dict)
    }

    if actual_models != EXPECTED_MODELS:
        raise ValueError(
            "Danh sach mo hinh trong metadata khong dung."
        )

    return metadata


# =====================================================
# 3. DOC DATASET
# =====================================================

def load_dataset(metadata):

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATASET_PATH}"
        )

    features = metadata["features"]
    target = metadata.get("target", "resale_price")

    print("\nDang doc dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        usecols=features + [target],
    )

    print(f"So dong ban dau: {len(df):,}")

    for column in features:
        if column not in df.columns:
            raise ValueError(
                f"Dataset thieu cot: {column}"
            )

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    valid_mask = (
        df[target].notna()
        & np.isfinite(df[target])
        & (df[target] >= 0)
    )

    df = df.loc[valid_mask].copy()

    df = df.dropna(subset=features + [target])

    if df.empty:
        raise ValueError("Khong co du lieu hop le.")

    print(f"So dong hop le: {len(df):,}")

    return df, features, target


# =====================================================
# 4. CHIA TRAIN / TEST
# =====================================================

def split_dataset(df, features, target, metadata):

    test_size = metadata.get("test_size", 0.2)
    random_state = metadata.get("random_state", RANDOM_STATE)

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )

    X_train = df.iloc[train_indices][features]
    X_test = df.iloc[test_indices][features]

    y_train = df.iloc[train_indices][target]
    y_test = df.iloc[test_indices][target]

    print(f"\nTrain: {len(X_train):,} dong")
    print(f"Test : {len(X_test):,} dong")

    return X_train, X_test, y_train, y_test


# =====================================================
# 5. LAY MAU DU LIEU
# =====================================================

def sample_data(X, y, sample_size):

    count = min(sample_size, len(X))

    sampled_X = X.sample(
        n=count,
        random_state=RANDOM_STATE,
    )

    sampled_y = y.loc[sampled_X.index]

    return sampled_X, sampled_y


# =====================================================
# 6. TINH CAC CHI SO
# =====================================================

def calculate_metrics(y_true, y_pred):

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        ),
    }


# =====================================================
# 7. DANH GIA MOT MO HINH
# =====================================================

def evaluate_one_model(
    model_name,
    model_filename,
    X_train,
    X_test,
    y_train,
    y_test,
):

    model_path = MODEL_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {model_path}"
        )

    print("\n" + "=" * 65)
    print(f"DANH GIA: {model_name}")
    print("=" * 65)

    print("Dang tai mo hinh...")

    model = joblib.load(model_path)

    X_train_sample, y_train_sample = sample_data(
        X_train,
        y_train,
        SAMPLE_SIZE,
    )

    X_test_sample, y_test_sample = sample_data(
        X_test,
        y_test,
        SAMPLE_SIZE,
    )

    print(
        f"So mau train: {len(X_train_sample):,}"
    )

    print(
        f"So mau test : {len(X_test_sample):,}"
    )

    print("Dang du doan tap train...")

    train_predictions = model.predict(X_train_sample)

    train_metrics = calculate_metrics(
        y_train_sample,
        train_predictions,
    )

    print("Dang du doan tap test...")

    test_predictions = model.predict(X_test_sample)

    test_metrics = calculate_metrics(
        y_test_sample,
        test_predictions,
    )

    result = {
        "model_name": model_name,
        "train_samples": len(X_train_sample),
        "test_samples": len(X_test_sample),

        "train_mae": train_metrics["MAE"],
        "test_mae": test_metrics["MAE"],

        "train_rmse": train_metrics["RMSE"],
        "test_rmse": test_metrics["RMSE"],

        "train_r2": train_metrics["R2"],
        "test_r2": test_metrics["R2"],

        "mae_gap": (
            test_metrics["MAE"]
            - train_metrics["MAE"]
        ),

        "rmse_gap": (
            test_metrics["RMSE"]
            - train_metrics["RMSE"]
        ),

        "r2_gap": (
            train_metrics["R2"]
            - test_metrics["R2"]
        ),
    }

    print("\nTAP TRAIN:")
    for name, value in train_metrics.items():
        print(f"{name}: {value:,.6f}")

    print("\nTAP TEST:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:,.6f}")

    print("\nCHENH LECH:")
    print(f"MAE gap : {result['mae_gap']:,.6f}")
    print(f"RMSE gap: {result['rmse_gap']:,.6f}")
    print(f"R2 gap  : {result['r2_gap']:,.6f}")

    del model
    del train_predictions
    del test_predictions
    del X_train_sample
    del X_test_sample
    del y_train_sample
    del y_test_sample

    gc.collect()

    return result


# =====================================================
# 8. CHUONG TRINH CHINH
# =====================================================

def main():

    print("=" * 70)
    print("PHONEVALUE AI - KIEM TRA OVERFITTING 3 MO HINH")
    print("=" * 70)

    metadata = load_metadata()

    print("\nMetadata hop le.")
    print(f"So dac trung: {len(metadata['features'])}")

    df, features, target = load_dataset(metadata)

    X_train, X_test, y_train, y_test = split_dataset(
        df,
        features,
        target,
        metadata,
    )

    del df
    gc.collect()

    results = []

    for model_name, model_filename in EXPECTED_MODELS.items():

        result = evaluate_one_model(
            model_name,
            model_filename,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("BANG SO SANH OVERFITTING")
    print("=" * 80)

    columns = [
        "model_name",
        "train_mae",
        "test_mae",
        "train_rmse",
        "test_rmse",
        "train_r2",
        "test_r2",
        "mae_gap",
        "r2_gap",
    ]

    print(
        result_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )

    print("\nBAO CAO DA DUOC LUU:")
    print(OUTPUT_CSV)

    print("\nLuu y:")
    print(
        "Ket qua nay su dung toi da 20.000 mau moi tap "
        "de kiem tra nguy co overfitting."
    )

    print(
        "MAE va RMSE cang thap cang tot; R2 cang cao cang tot. "
        "Khoang cach train-test lon co the la dau hieu overfitting."
    )

    print("\nHOAN TAT DANH GIA!")


# =====================================================
# 9. CHAY CHUONG TRINH
# =====================================================

if __name__ == "__main__":
    main()
