import sys, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Graceful check for XGBoost (if installed use it, otherwise fallback gracefully)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------CONFIG------------------------------------------------
data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Prop-IQ\Delhi.csv")
model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\Prop-IQ\models")
random_state = 42
target = "Price"
core_cols = ["Price", "Area", "Location", "No. of Bedrooms", "Resale"]
srs_target_improvement = 20.0
ci_low, ci_high = 0.10, 0.90
model_dir.mkdir(parents=True, exist_ok=True)

def lakh(x):
    """Format currency in Lakhs or Crores."""
    if abs(x) >= 1e7:
        return f"Rs {x / 1e7:,.2f} Cr"
    return f"Rs {x / 1e5:,.2f} L"

def metrics(y_true, y_pred):
    """Calculate standard regression evaluation metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

# ---------------------------------------[1] LOAD + CLEANING------------------------------------
print("=" * 60)
print(" Prop-IQ: Real Estate Valuation & Price Prediction Pipeline")
print("=" * 60)
print(f"[1] Loading dataset from: {data_path}")

df = pd.read_csv(data_path)
print("Raw Shape:", df.shape)

missing_core = [c for c in core_cols if c not in df.columns]
assert not missing_core, f"Columns not found in CSV: {missing_core}"

df = df.drop_duplicates().reset_index(drop=True)
df["Location"] = df["Location"].astype(str).str.strip()

# Amenity handling: replace '9' (unreported) with 0, and compute amenity count feature
amenity_cols = [c for c in df.columns if c not in core_cols]
df[amenity_cols] = df[amenity_cols].replace(9, 0)
df["amenity_count"] = df[amenity_cols].sum(axis=1)

# Filter valid positive prices & areas (Fixed typo: 'Price' instead of 'Prices')
df = df[(df["Price"] > 0) & (df["Area"] > 0)]

# Outlier clipping: 1st to 99th percentiles
for col in ["Price", "Area"]:
    lo, hi = df[col].quantile([0.01, 0.99])
    df = df[df[col].between(lo, hi)]
df = df.reset_index(drop=True)

print("Clean Shape:", df.shape, "| Unique Localities:", df["Location"].nunique())

# -------------------------------------------[2] SPLIT-------------------------------------------
print("\n[2] Train/Test Split (80/20)")
X = df.drop(columns=[target])
y = df[target]
y_log = np.log1p(y)

X_train, X_test, ylog_train, ylog_test = train_test_split(
    X, y_log, test_size=0.2, random_state=random_state
)
y_train, y_test = np.expm1(ylog_train), np.expm1(ylog_test)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ------------------------------------------[3] BASELINE-----------------------------------------
print("\n[3] Baseline Calculations")
train_df = X_train.assign(Price=y_train)
overall_mean = train_df["Price"].mean()
overall_median = train_df["Price"].median()
loc_median = train_df.groupby("Location")["Price"].median()

pred_l0 = np.full(len(X_test), overall_mean)                                   # Level 0
pred_l1 = X_test["Location"].map(loc_median).fillna(overall_median).values     # Level 1

m_l0, m_l1 = metrics(y_test, pred_l0), metrics(y_test, pred_l1)
print(f"Level 0 (Overall Mean)     MAE: {lakh(m_l0['MAE']):>12} | RMSE: {lakh(m_l0['RMSE']):>12} | R2: {m_l0['R2']:.3f}")
print(f"Level 1 (Locality Median)  MAE: {lakh(m_l1['MAE']):>12} | RMSE: {lakh(m_l1['RMSE']):>12} | R2: {m_l1['R2']:.3f}")

# -------------------------------------[4] MODEL COMPARISON (CV)---------------------------------
print("\n[4] Cross-Validation (5-fold, evaluated on Train set)")
num_cols = [c for c in X.columns if c != "Location"]
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["Location"]),
])

models = {
    "Ridge": Ridge(alpha=5.0, random_state=random_state),
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2,
                                          n_jobs=-1, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.06,
                                                  max_depth=5, subsample=0.85, random_state=random_state)
}

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=random_state, n_jobs=-1)

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
cv_mae = {}

for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    oof = cross_val_predict(pipe, X_train, ylog_train, cv=kf)
    cv_mae[name] = mean_absolute_error(y_train, np.expm1(oof))
    print(f"  {name:18s} CV MAE: {lakh(cv_mae[name])}")

best_name = min(cv_mae, key=cv_mae.get)
print(f"\n=> Best Model by CV: {best_name}")

# -------------------------------------------[5] TUNING------------------------------------------
print(f"\n[5] Hyperparameter Tuning for {best_name}")
grids = {
    "RandomForest": {
        "model__n_estimators": [150, 250],
        "model__max_depth": [12, 16, 20],
        "model__min_samples_leaf": [1, 2, 4]
    },
    "GradientBoosting": {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.04, 0.07],
        "model__max_depth": [4, 6],
        "model__subsample": [0.8, 0.9]
    }
}
if HAS_XGB:
    grids["XGBoost"] = {
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__n_estimators": [250, 400],
        "model__subsample": [0.75, 0.9]
    }

base_pipe = Pipeline([("pre", pre), ("model", models[best_name])])
if best_name in grids:
    search = RandomizedSearchCV(
        base_pipe, grids[best_name], n_iter=8, cv=kf,
        scoring="neg_mean_absolute_error", random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, ylog_train)
    final = search.best_estimator_
    best_params = search.best_params_
    print("Tuned Best Params:", best_params)
else:
    final = base_pipe.fit(X_train, ylog_train)
    best_params = {}

# -------------------------------------[6] FINAL TEST EVALUATION----------------------------------
print("\n[6] Final Evaluation on Held-Out Test Set (Single Evaluation)")
pred_test = np.expm1(final.predict(X_test))
m_model = metrics(y_test, pred_test)
improve = (m_l1["MAE"] - m_model["MAE"]) / m_l1["MAE"] * 100
srs_status = "PASS" if improve >= srs_target_improvement else "NOT MET (reported honestly)"

print(f"Model ({best_name}) MAE : {lakh(m_model['MAE']):>12} | RMSE: {lakh(m_model['RMSE']):>12} | R2: {m_model['R2']:.3f}")
print(f"Improvement over Level 1: {improve:.1f}% (SRS Target >= {srs_target_improvement:.0f}%) -> {srs_status}")

# ----------------------------------------[7] CONFIDENCE RANGE------------------------------------
print("\n[7] Confidence Range from Out-of-Fold Residuals")
oof_final = cross_val_predict(final, X_train, ylog_train, cv=kf)
resid = ylog_train.values - oof_final
lo_q, hi_q = np.quantile(resid, [ci_low, ci_high])

# Prediction bounds on test set
pred_test_log = final.predict(X_test)
low_t = np.expm1(pred_test_log + lo_q)
high_t = np.expm1(pred_test_log + hi_q)
coverage = float(np.mean((y_test.values >= low_t) & (y_test.values <= high_t)))

print(f"80% Range Factor Multiplier : x{np.exp(lo_q):.2f} to x{np.exp(hi_q):.2f}")
print(f"Actual Test Empirical Coverage: {coverage:.1%}")

# ----------------------------------------[8] FEATURE IMPORTANCE----------------------------------
print("\n[8] Feature Importance (Permutation on Test Set)")
imp = permutation_importance(
    final, X_test, ylog_test, n_repeats=5, random_state=random_state,
    scoring="neg_mean_absolute_error", n_jobs=1
)
importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("Top 8 Influential Features:")
print(importance.head(8).round(4).to_string())

# ----------------------------------------[9] SAVE ARTIFACTS--------------------------------------
print(f"\n[9] Saving Artifacts to {model_dir} ...")

# 1. Save trained sklearn pipeline
pipeline_path = model_dir / "prop_iq_pipeline.joblib"
joblib.dump(final, pipeline_path)
print(f"  [+] Saved Model Pipeline: {pipeline_path}")

# 2. Save metadata & metrics
metadata = {
    "project_name": "Prop-IQ Real Estate Price Prediction",
    "best_model_name": best_name,
    "best_params": {str(k): str(v) for k, v in best_params.items()},
    "srs_target_improvement_pct": srs_target_improvement,
    "achieved_improvement_pct": round(improve, 2),
    "srs_status": srs_status,
    "metrics": {
        "level_0_mean_baseline": m_l0,
        "level_1_locality_median_baseline": m_l1,
        "final_model": m_model
    },
    "confidence_interval": {
        "ci_low_percentile": ci_low * 100,
        "ci_high_percentile": ci_high * 100,
        "multiplier_low": round(float(np.exp(lo_q)), 4),
        "multiplier_high": round(float(np.exp(hi_q)), 4),
        "test_coverage_pct": round(coverage * 100, 2)
    },
    "overall_median_price": float(overall_median),
    "locality_medians": {str(k): float(v) for k, v in loc_median.items()},
    "unique_locations": sorted(df["Location"].unique().tolist()),
    "feature_columns": list(X.columns),
    "amenity_columns": amenity_cols,
    "top_features": {str(k): round(float(v), 5) for k, v in importance.head(10).items()}
}

metadata_path = model_dir / "metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print(f"  [+] Saved Metadata & Metrics: {metadata_path}")

print("=" * 60)
print(" Prop-IQ Step 1 Completed Successfully!")
print("=" * 60)