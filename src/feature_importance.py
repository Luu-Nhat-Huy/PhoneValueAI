
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# =====================================================
# 1. CẤU HÌNH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "data" / "raw" / "used_phone_price_prediction_1M.csv.gz"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_CSV = OUTPUT_DIR / "feature_importance.csv"
OUTPUT_IMAGE = REPORT_DIR / "feature_importance.png"

SAMPLE_SIZE = 5000
N_REPEATS = 5


# =====================================================
# 2. KIỂM TRA FILE
# =====================================================

for path in [DATASET_PATH, MODEL_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 3. ĐỌC METADATA
# =====================================================

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

features = metadata["features"]
target = metadata.get("target", "resale_price")

print("=" * 65)
print("PHONEVALUE AI - PHAN TICH FEATURE IMPORTANCE")
print("=" * 65)

print(f"So dac trung: {len(features)}")
print(f"So mau phan tich toi da: {SAMPLE_SIZE:,}")
print(f"So lan lap moi dac trung: {N_REPEATS}")


# =====================================================
# 4. ĐỌC VÀ KIỂM TRA DATASET
# =====================================================

print("\nDang doc dataset...")

df = pd.read_csv(DATASET_PATH)

required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Dataset thieu cac cot: {missing_columns}")

df = df[required_columns].dropna()

X = df[features]
y = df[target]

print(f"Tong so dong hop le: {len(df):,}")


# =====================================================
# 5. CHIA TẬP GIỐNG KHI HUẤN LUYỆN
# =====================================================

test_size = metadata.get("test_size", 0.2)
random_state = metadata.get("random_state", 42)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


# =====================================================
# 6. LẤY MẪU TẬP KIỂM TRA
# =====================================================

sample_size = min(SAMPLE_SIZE, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

y_sample = y_test.loc[X_sample.index]

print(f"So mau dung de phan tich: {len(X_sample):,}")


# =====================================================
# 7. TẢI MÔ HÌNH
# =====================================================

print("\nDang tai mo hinh...")

model = joblib.load(MODEL_PATH)

print(f"Da tai mo hinh: {type(model).__name__}")


# =====================================================
# 8. TÍNH PERMUTATION FEATURE IMPORTANCE
# =====================================================

print("\nDang tinh Feature Importance...")
print("Qua trinh nay co the mat vai phut.")

result = permutation_importance(
    model,
    X_sample,
    y_sample,
    scoring="neg_mean_absolute_error",
    n_repeats=N_REPEATS,
    random_state=42,
    n_jobs=-1
)


# =====================================================
# 9. TẠO BẢNG KẾT QUẢ
# =====================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

importance_df = importance_df.sort_values(
    by="importance_mean",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "rank",
    range(1, len(importance_df) + 1)
)

importance_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 10. HIỂN THỊ TOP 15
# =====================================================

print("\n" + "=" * 65)
print("TOP 15 DAC TRUNG ANH HUONG DEN DU DOAN")
print("=" * 65)

for _, row in importance_df.head(15).iterrows():
    print(
        f"{int(row['rank']):2d}. "
        f"{row['feature']:<35} "
        f"Importance: {row['importance_mean']:>12.4f} "
        f"+/- {row['importance_std']:.4f}"
    )


# =====================================================
# 11. VẼ BIỂU ĐỒ
# =====================================================

top_features = importance_df.head(15).sort_values(
    by="importance_mean",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"],
    capsize=3
)

plt.xlabel("Muc tang MAE khi dao tron dac trung")
plt.ylabel("Dac trung")
plt.title("PhoneValue AI - Permutation Feature Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_IMAGE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 12. THÔNG BÁO KẾT QUẢ
# =====================================================

print("\n" + "=" * 65)
print("HOAN TAT PHAN TICH")
print("=" * 65)

print(f"File CSV: {OUTPUT_CSV}")
print(f"Bieu do:  {OUTPUT_IMAGE}")

print(
    "\nLuu y: Importance duong lon hon cho thay khi dao tron "
    "dac trung do, MAE tang nhieu hon. "
    "Day la muc do quan trong doi voi mo hinh, "
    "khong khang dinh quan he nhan qua."
)

