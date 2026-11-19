
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from predict import (
    load_metadata,
    get_available_models,
    predict_phone,
)


DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "used_phone_price_prediction_1M.csv.gz"
)


def main():
    print("=" * 65)
    print("PHONEVALUE AI - KIEM TRA CA 3 MO HINH")
    print("=" * 65)

    metadata = load_metadata()
    models = get_available_models()
    features = metadata["features"]

    print("\nSo mo hinh:", len(models))

    if len(models) != 3:
        raise RuntimeError(
            f"Can dung 3 mo hinh, nhung tim thay {len(models)}."
        )

    print("\nDang doc mot dong du lieu thuc...")

    df = pd.read_csv(DATA_PATH, nrows=1)

    if df.empty:
        raise RuntimeError("Dataset khong co du lieu.")

    phone_data = df.iloc[0][features].to_dict()
    actual_price = float(df.iloc[0]["resale_price"])

    print("Doc du lieu thanh cong!")
    print(f"Gia thuc te: {actual_price:,.0f} VND")

    print("\n" + "=" * 65)
    print("BAT DAU KIEM TRA")
    print("=" * 65)

    results = []

    for index, model_name in enumerate(models, start=1):
        print(f"\n[{index}/3] Dang kiem tra: {model_name}")

        predicted_price = predict_phone(
            phone_data,
            model_name=model_name,
        )

        error = abs(actual_price - predicted_price)

        results.append({
            "model": model_name,
            "prediction": predicted_price,
            "error": error,
        })

        print(f"Du doan: {predicted_price:,.0f} VND")
        print(f"Sai lech: {error:,.0f} VND")
        print("Trang thai: THANH CONG")

    print("\n" + "=" * 65)
    print("TONG KET KIEM TRA")
    print("=" * 65)

    for result in results:
        print(
            f"{result['model']}: "
            f"{result['prediction']:,.0f} VND"
        )

    print("\nDa kiem tra thanh cong ca 3 mo hinh!")
    print("Mo hinh mac dinh:",
          metadata.get("selected_default_model"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nKIEM TRA THAT BAI!")
        print("Chi tiet:", error)
        sys.exit(1)
