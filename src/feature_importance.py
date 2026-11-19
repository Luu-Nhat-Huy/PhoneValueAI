
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


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

MODEL_PATH = (
    MODEL_DIR
    / "random_forest_regression.pkl"
)

METADATA_PATH = (
    MODEL_DIR
    / "model_metadata.json"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_CSV = OUTPUT_DIR / "feature_importance.csv"
OUTPUT_IMAGE = REPORT_DIR / "feature_importance.png"

SAMPLE_SIZE = 5000
N_REPEATS = 5

RANDOM_STATE = 42


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

    expected_default = "Random Forest Regression"

    if metadata.get("selected_default_model") != expected_default:
        raise ValueError(
            "Mo hinh mac dinh trong metadata khong phai Random Forest Regression."
        )

    models = metadata.get("models", [])

    rf_model = next(
        (
            item
            for item in models
            if item.get("model_name") == expected_default
        ),
        None,
    )

    if rf_model is None:
        raise ValueError(
            "Metadata khong co Random Forest Regression."
        )

    if rf_model.get("model_file") != "random_forest_regression.pkl":
        raise ValueError(
            "Ten file Random Forest trong metadata khong hop le."
        )

    return metadata, features


# =====================================================
# 3. DOC DATASET
# =====================================================

def load_dataset(features, target):

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATASET_PATH}"
        )

    print("\nDang doc dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        usecols=features + [target],
    )

    print(f"So dong ban dau: {len(df):,}")

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

    df = df.dropna(
        subset=features + [target]
    )

    if df.empty:
        raise ValueError("Dataset khong co dong hop le.")

    print(f"So dong hop le: {len(df):,}")

    return df


# =====================================================
# 4. CHIA TAP TRAIN / TEST
# =====================================================

def prepare_test_data(df, features, target, metadata):

    test_size = metadata.get("test_size", 0.2)
    random_state = metadata.get("random_state", RANDOM_STATE)

    indices = np.arange(len(df))

    _, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )

    X_test = df.iloc[test_indices][features]
    y_test = df.iloc[test_indices][target]

    print(f"So dong tap test: {len(X_test):,}")

    return X_test, y_test


# =====================================================
# 5. LAY MAU PHAN TICH
# =====================================================

def sample_test_data(X_test, y_test):

    sample_size = min(
        SAMPLE_SIZE,
        len(X_test),
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    y_sample = y_test.loc[X_sample.index]

    print(f"So mau phan tich: {len(X_sample):,}")

    return X_sample, y_sample


# =====================================================
# 6. TAI RANDOM FOREST
# =====================================================

def load_random_forest():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay Random Forest: {MODEL_PATH}"
        )

    print("\nDang tai Random Forest Regression...")

    model = joblib.load(MODEL_PATH)

    print("Tai mo hinh thanh cong!")

    return model


# =====================================================
# 7. TINH PERMUTATION FEATURE IMPORTANCE
# =====================================================

def calculate_importance(model, X_sample, y_sample):

    print("\n" + "=" * 70)
    print("DANG TINH PERMUTATION FEATURE IMPORTANCE")
    print("=" * 70)

    print(f"So dac trung: {len(X_sample.columns)}")
    print(f"So mau: {len(X_sample):,}")
    print(f"So lan lap: {N_REPEATS}")

    print(
        "\nQua trinh co the mat vai phut "
        "do phai du doan lai nhieu lan."
    )

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return result


# =====================================================
# 8. TAO BANG KET QUA
# =====================================================

def create_importance_dataframe(features, result):

    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1),
    )

    return importance_df


# =====================================================
# 9. LUU CSV
# =====================================================

def save_csv(importance_df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nDa luu CSV: {OUTPUT_CSV}")


# =====================================================
# 10. VE BIEU DO
# =====================================================

def save_chart(importance_df):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            by="importance_mean",
            ascending=True,
        )
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        capsize=3,
    )

    plt.xlabel(
        "Muc tang MAE khi dao tron dac trung"
    )

    plt.ylabel("Dac trung")

    plt.title(
        "PhoneValue AI - Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Da luu bieu do: {OUTPUT_IMAGE}")


# =====================================================
# 11. HIEN THI TOP 15
# =====================================================

def display_results(importance_df):

    print("\n" + "=" * 75)
    print("TOP 15 DAC TRUNG ANH HUONG DEN DU DOAN")
    print("=" * 75)

    for _, row in importance_df.head(15).iterrows():

        print(
            f"{int(row['rank']):2d}. "
            f"{row['feature']:<35} "
            f"Importance: {row['importance_mean']:>12.4f} "
            f"+/- {row['importance_std']:.4f}"
        )

    print("\nGIAI THICH:")

    print(
        "Importance duong lon: khi dao tron dac trung, "
        "MAE tang nhieu hon."
    )

    print(
        "Importance gan 0: dac trung co it anh huong "
        "den kha nang du doan tren mau phan tich."
    )

    print(
        "Importance am: viec dao tron dac trung co the "
        "lam MAE giam tren mau phan tich."
    )

    print(
        "Day la muc do quan trong doi voi mo hinh, "
        "khong khang dinh quan he nhan qua."
    )


# =====================================================
# 12. CHUONG TRINH CHINH
# =====================================================

def main():

    print("=" * 75)
    print("PHONEVALUE AI - PHAN TICH FEATURE IMPORTANCE")
    print("MO HINH: RANDOM FOREST REGRESSION")
    print("=" * 75)

    metadata, features = load_metadata()

    target = metadata.get(
        "target",
        "resale_price",
    )

    print(f"\nSo dac trung: {len(features)}")
    print(f"Target: {target}")

    df = load_dataset(
        features,
        target,
    )

    X_test, y_test = prepare_test_data(
        df,
        features,
        target,
        metadata,
    )

    del df
    gc.collect()

    X_sample, y_sample = sample_test_data(
        X_test,
        y_test,
    )

    del X_test
    del y_test
    gc.collect()

    model = load_random_forest()

    result = calculate_importance(
        model,
        X_sample,
        y_sample,
    )

    importance_df = create_importance_dataframe(
        features,
        result,
    )

    save_csv(importance_df)

    save_chart(importance_df)

    display_results(importance_df)

    del model
    del X_sample
    del y_sample
    del result

    gc.collect()

    print("\n" + "=" * 75)
    print("HOAN TAT PHAN TICH FEATURE IMPORTANCE!")
    print("=" * 75)


# =====================================================
# 13. CHAY CHUONG TRINH
# =====================================================

if __name__ == "__main__":
    main()
