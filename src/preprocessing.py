
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


# Các cột dạng chữ
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "os_type",
    "condition",
    "city_tier",
    "seller_type",
]

# Các cột dạng số
NUMERIC_FEATURES = [
    "release_year",
    "ram_gb",
    "storage_gb",
    "screen_size_inches",
    "battery_capacity",
    "processor_score",
    "camera_score",
    "has_5g",
    "original_price",
    "purchase_year",
    "age_months",
    "usage_hours_per_day",
    "battery_health",
    "screen_cracked",
    "body_damage",
    "repair_history",
    "water_damage",
    "warranty_remaining_months",
    "box_available",
    "charger_available",
    "market_demand_score",
]

TARGET = "resale_price"

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def create_preprocessor():
    """
    Mã hóa dữ liệu phân loại.
    Các cột số được giữ nguyên.
    """

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype="float32",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_model_pipeline():
    """
    Kết hợp tiền xử lý và mô hình thành một pipeline.
    """

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", model),
        ]
    )

    return pipeline
