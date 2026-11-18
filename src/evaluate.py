
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. ĐƯỜNG DẪN
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "data" / "raw" / "used_phone_price_prediction_1M.csv.gz"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

SAMPLE_SIZE = 20000


# =====================================================
# 2. KIỂM TRA FILE
# =====================================================

for path in [DATASET_PATH, MODEL_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")


# =====================================================
# 3. ĐỌC METADATA VÀ DATASET
# =====================================================

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

features = metadata["features"]
target = metadata.get("target", "resale_price")

df = pd.read_csv(DATASET_PATH)

required_columns = features + [target]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Dataset thieu cot: {missing_columns}")

df = df[required_columns].dropna()

X = df[features]
y = df[target]


# =====================================================
# 4. CHIA TẬP DỮ LIỆU GIỐNG KHI HUẤN LUYỆN
# =====================================================

test_size = metadata.get("test_size", 0.2)
random_state = metadata.get("random_state", 42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


# =====================================================
# 5. LẤY MẪU 20.000 DÒNG MỖI TẬP
# =====================================================

train_count = min(SAMPLE_SIZE, len(X_train))
test_count = min(SAMPLE_SIZE, len(X_test))

X_train_sample = X_train.sample(
    n=train_count,
    random_state=42
)

X_test_sample = X_test.sample(
    n=test_count,
    random_state=42
)

y_train_sample = y_train.loc[X_train_sample.index]
y_test_sample = y_test.loc[X_test_sample.index]


# =====================================================
# 6. TẢI MÔ HÌNH
# =====================================================

print("=" * 60)
print("PHONEVALUE AI - KIEM TRA OVERFITTING")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("Da tai mo hinh thanh cong!")
print(f"Mo hinh: {metadata.get('model_name', type(model).__name__)}")


# =====================================================
# 7. ĐÁNH GIÁ TRÊN HAI TẬP
# =====================================================

print("\nDang danh gia tap huan luyen...")
train_predictions = model.predict(X_train_sample)

print("Dang danh gia tap kiem tra...")
test_predictions = model.predict(X_test_sample)


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


train_metrics = calculate_metrics(
    y_train_sample,
    train_predictions
)

test_metrics = calculate_metrics(
    y_test_sample,
    test_predictions
)


# =====================================================
# 8. HIỂN THỊ KẾT QUẢ
# =====================================================

print("\n" + "=" * 60)
print("KET QUA SO SANH")
print("=" * 60)

print(f"\nSo mau tap huan luyen: {train_count:,}")
print(f"So mau tap kiem tra:   {test_count:,}")

print("\n--- TAP HUAN LUYEN ---")
for name, value in train_metrics.items():
    print(f"{name}: {value:,.6f}")

print("\n--- TAP KIEM TRA ---")
for name, value in test_metrics.items():
    print(f"{name}: {value:,.6f}")


# =====================================================
# 9. ĐÁNH GIÁ CHÊNH LỆCH
# =====================================================

print("\n" + "=" * 60)
print("CHENH LECH GIUA HAI TAP")
print("=" * 60)

mae_gap = test_metrics["MAE"] - train_metrics["MAE"]
rmse_gap = test_metrics["RMSE"] - train_metrics["RMSE"]
r2_gap = train_metrics["R2"] - test_metrics["R2"]

print(f"MAE gap:  {mae_gap:,.6f}")
print(f"RMSE gap: {rmse_gap:,.6f}")
print(f"R2 gap:   {r2_gap:,.6f}")

print("\nLuu y:")
print(
    "Day la ket qua tren mau 20.000 dong moi tap, "
    "khong phai ket qua danh gia toan bo du lieu."
)

print(
    "Can xem xet dong thoi MAE, RMSE, R2 va muc chenh lech "
    "de danh gia nguy co overfitting."
)

print("\nHoan tat kiem tra!")

