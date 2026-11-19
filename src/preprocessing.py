
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# =====================================================
# 1. DANH SACH DAC TRUNG
# =====================================================

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "os_type",
    "condition",
    "city_tier",
    "seller_type",
]

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


# =====================================================
# 2. DANH SACH 3 MO HINH
# =====================================================

MODEL_CONFIGS = {
    "Linear Regression": {
        "file": "linear_regression.pkl",
        "description": "Mo hinh hoi quy tuyen tinh",
    },

    "Decision Tree Regression": {
        "file": "decision_tree_regression.pkl",
        "description": "Mo hinh cay hoi quy",
    },

    "Random Forest Regression": {
        "file": "random_forest_regression.pkl",
        "description": "Mo hinh rung ngau nhien",
    },
}


# =====================================================
# 3. TAO BO TIEN XU LY
# =====================================================

def create_preprocessor(model_name):

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "to_float32",
            FunctionTransformer(
                np.asarray,
                kw_args={"dtype": np.float32},
            ),
        ),
    ]

    if model_name == "Linear Regression":

        numeric_steps.append(
            (
                "scaler",
                StandardScaler(
                    with_mean=False
                ),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor


# =====================================================
# 4. TAO MO HINH
# =====================================================

def create_model_pipeline(model_name):

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Mo hinh khong duoc ho tro: {model_name}"
        )

    preprocessor = create_preprocessor(
        model_name
    )

    # -------------------------------------------------
    # LINEAR REGRESSION
    # -------------------------------------------------

    if model_name == "Linear Regression":

        model = LinearRegression(
            n_jobs=2,
        )

    # -------------------------------------------------
    # DECISION TREE REGRESSION
    # -------------------------------------------------

    elif model_name == "Decision Tree Regression":

        model = DecisionTreeRegressor(
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
        )

    # -------------------------------------------------
    # RANDOM FOREST REGRESSION
    # -------------------------------------------------

    elif model_name == "Random Forest Regression":

        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=5,
            max_features=0.5,
            n_jobs=2,
            random_state=42,
        )

    # -------------------------------------------------
    # PIPELINE HOAN CHINH
    # -------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
