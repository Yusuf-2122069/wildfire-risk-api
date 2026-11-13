import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Initialize FastAPI
app = FastAPI(
    title="NASA Space Apps - Wildfire Next-Day Risk API",
    description="Predicts next-day elevated wildfire activity (T+1) on 0.5° grid cells.",
    version="1.0.0"
)

# Enable CORS so your friend's frontend/backend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your website's actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. API Key Security Setup
# Reads API_KEY from server environment variables, or falls back to a default key for local/initial testing
API_KEY = os.getenv("API_KEY", "nasa-wildfire-secret-key-2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API Key. Pass 'X-API-Key' in request headers."
        )
    return api_key

# 3. Load Cross-Platform Model Bundle at Startup
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

with open(ARTIFACTS_DIR / "metadata.json", "r") as f:
    meta = json.load(f)

CHAMPION_NAME: str = meta["model_name"]
THRESHOLD: float = meta["optimal_threshold"]
ENGINEERED_NUM: List[str] = meta["engineered_num_features"]
CAT_FEATURES: List[str] = meta["cat_features"]
GRID_DEG: float = meta["grid_deg"]

# Load Scikit-Learn Preprocessor & Random Forest
sk_parts = joblib.load(ARTIFACTS_DIR / "sklearn_parts.joblib")
PREPROCESSOR = sk_parts["preprocessor"]
RF_MODEL = sk_parts["rf_model"]

# Load Native Cross-Platform Boosters
XGB_MODEL = xgb.XGBClassifier()
XGB_MODEL.load_model(ARTIFACTS_DIR / "xgb_model.json")

CAT_MODEL = CatBoostClassifier()
CAT_MODEL.load_model(ARTIFACTS_DIR / "cat_model.cbm")

LGB_BOOSTER = lgb.Booster(model_file=str(ARTIFACTS_DIR / "lgb_model.txt"))


# 4. Request & Response Schemas
class FireCellInput(BaseModel):
    latitude: float = Field(..., example=37.75)
    longitude: float = Field(..., example=-122.25)
    date: str = Field(..., example="2026-08-15", description="YYYY-MM-DD")
    country: Optional[str] = Field(None, example="USA")
    
    # Today's satellite observations (T)
    detections_today: int = Field(..., ge=0, example=5)
    frp_sum_today: float = Field(..., ge=0.0, example=145.6)
    frp_max_today: float = Field(..., ge=0.0, example=62.1)
    frp_mean_today: float = Field(..., ge=0.0, example=29.12)
    brightness_mean_today: float = Field(315.0, example=324.5)
    modis_count_today: int = Field(0, ge=0, example=1)
    viirs_count_today: int = Field(0, ge=0, example=4)

    # Historical Lags (Defaults to 0 if no prior fire in cell)
    detections_lag_1: int = Field(0, ge=0, example=2)
    frp_lag_1: float = Field(0.0, ge=0.0, example=48.0)
    detections_lag_2: int = Field(0, ge=0, example=0)
    frp_lag_2: float = Field(0.0, ge=0.0, example=0.0)
    detections_rolling_3d: int = Field(0, ge=0, example=2)
    frp_rolling_3d: float = Field(0.0, ge=0.0, example=48.0)
    detections_rolling_7d: int = Field(0, ge=0, example=4)
    frp_rolling_7d: float = Field(0.0, ge=0.0, example=95.0)


class PredictionResponse(BaseModel):
    cell_lat: float
    cell_lon: float
    country: str
    champion_model: str
    risk_probability: float
    decision_threshold: float
    fire_risk_next_day: int
    risk_label: str


# 5. Feature Engineering Helper (Matches Cell 2 & Cell 3)
def build_features(data: FireCellInput) -> tuple[pd.DataFrame, float, float, str]:
    dt = pd.to_datetime(data.date)
    cell_lat = float(np.floor(data.latitude / GRID_DEG) * GRID_DEG)
    cell_lon = float(np.floor(data.longitude / GRID_DEG) * GRID_DEG)
    country = data.country if data.country else ("Canada" if data.latitude >= 49.0 else "USA")

    doy = dt.dayofyear
    month = dt.month
    season = (
        "winter" if month in (12, 1, 2)
        else "spring" if month in (3, 4, 5)
        else "summer" if month in (6, 7, 8)
        else "autumn"
    )

    total_sensors = max(data.modis_count_today + data.viirs_count_today, 1)
    viirs_ratio_today = data.viirs_count_today / total_sensors

    row = {
        "cell_lat": cell_lat,
        "cell_lon": cell_lon,
        "year": dt.year,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "brightness_mean_today": data.brightness_mean_today,
        "modis_count_today": data.modis_count_today,
        "viirs_count_today": data.viirs_count_today,
        "detections_change": data.detections_today - data.detections_lag_1,
        "frp_change": data.frp_sum_today - data.frp_lag_1,
        "viirs_ratio_today": viirs_ratio_today,
        # Log transforms
        "log_detections_today": np.log1p(data.detections_today),
        "log_frp_sum_today": np.log1p(data.frp_sum_today),
        "log_frp_max_today": np.log1p(data.frp_max_today),
        "log_frp_mean_today": np.log1p(data.frp_mean_today),
        "log_frp_lag_1": np.log1p(data.frp_lag_1),
        "log_frp_lag_2": np.log1p(data.frp_lag_2),
        "log_frp_rolling_3d": np.log1p(data.frp_rolling_3d),
        "log_frp_rolling_7d": np.log1p(data.frp_rolling_7d),
        "log_detections_lag_1": np.log1p(data.detections_lag_1),
        "log_detections_rolling_3d": np.log1p(data.detections_rolling_3d),
        "log_detections_rolling_7d": np.log1p(data.detections_rolling_7d),
        # Categoricals
        "country": country,
        "season": season,
    }

    df = pd.DataFrame([row])
    return df[ENGINEERED_NUM + CAT_FEATURES], cell_lat, cell_lon, country


# 6. Champion Ensemble Probability Resolver
def predict_champion_proba(X_transformed: np.ndarray) -> float:
    probs_map = {
        "XGBoost": float(XGB_MODEL.predict_proba(X_transformed)[:, 1][0]),
        "CatBoost": float(CAT_MODEL.predict_proba(X_transformed)[:, 1][0]),
        "LightGBM": float(LGB_BOOSTER.predict(X_transformed)[0]),
        "Random Forest": float(RF_MODEL.predict_proba(X_transformed)[:, 1][0]),
    }

    if CHAMPION_NAME == "Full Ensemble (All 4)":
        return float(np.mean(list(probs_map.values())))

    if " + " in CHAMPION_NAME:
        m1, m2 = CHAMPION_NAME.split(" + ")
        return float((probs_map[m1] + probs_map[m2]) / 2.0)

    return probs_map[CHAMPION_NAME]


# 7. Endpoints
@app.get("/")
def root():
    return {
        "message": "NASA Space Apps - Wildfire Next-Day Risk API is running",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict"
    }


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "champion_model": CHAMPION_NAME,
        "optimal_threshold": THRESHOLD,
        "test_metrics": meta.get("test_metrics", {})
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_fire_risk(payload: FireCellInput):
    try:
        X_df, cell_lat, cell_lon, country = build_features(payload)
        X_trans = PREPROCESSOR.transform(X_df)
        prob = predict_champion_proba(X_trans)
        pred_class = int(prob >= THRESHOLD)

        return PredictionResponse(
            cell_lat=cell_lat,
            cell_lon=cell_lon,
            country=country,
            champion_model=CHAMPION_NAME,
            risk_probability=round(prob, 4),
            decision_threshold=THRESHOLD,
            fire_risk_next_day=pred_class,
            risk_label="Elevated Activity Risk (>=3 detections)" if pred_class == 1 else "Lower Activity Risk (<3 detections)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))