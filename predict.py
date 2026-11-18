import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "prop_iq_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

# Global cache for model and metadata
_MODEL = None
_METADATA = None


def load_artifacts():
    """Load model pipeline and metadata JSON (cached)."""
    global _MODEL, _METADATA
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run train_model.py first.")
        _MODEL = joblib.load(MODEL_PATH)

    if _METADATA is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata file not found at: {METADATA_PATH}. Run train_model.py first.")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)

    return _MODEL, _METADATA


def format_currency(amount: float) -> str:
    """Format numerical price into Lakhs (L) or Crores (Cr)."""
    if abs(amount) >= 1e7:
        return f"₹ {amount / 1e7:,.2f} Cr"
    return f"₹ {amount / 1e5:,.2f} L"


def get_available_locations():
    """Returns a sorted list of all unique localities in Delhi-NCR."""
    _, meta = load_artifacts()
    return meta.get("unique_locations", [])


def predict_property_price(
    location: str,
    area: float,
    bedrooms: int = 2,
    resale: int = 1,
    amenities: list = None
) -> dict:
    """
    Predict property price with confidence intervals and locality comparison.

    Parameters:
    -----------
    location : str
        Locality name in Delhi-NCR (e.g., 'Sector 10 Dwarka', 'Uttam Nagar')
    area : float
        Super built-up area in square feet (e.g., 1200)
    bedrooms : int
        Number of bedrooms / BHK (default: 2)
    resale : int
        1 if resale property, 0 if new booking / fresh property (default: 1)
    amenities : list of str
        List of amenity names present (e.g., ['Gymnasium', 'LiftAvailable', '24X7Security'])

    Returns:
    --------
    dict containing:
        - estimated_price (raw & formatted)
        - price_range (low to high 80% confidence interval)
        - locality_median (raw & formatted)
        - price_per_sqft (raw & formatted)
        - comparison_vs_locality
    """
    model, meta = load_artifacts()
    amenities = amenities or []

    # Clean location string
    location = str(location).strip()

    # Build input feature dictionary matching training schema
    feature_columns = meta["feature_columns"]
    amenity_columns = meta["amenity_columns"]

    row = {col: 0 for col in feature_columns}
    row["Area"] = float(area)
    row["Location"] = location
    row["No. of Bedrooms"] = int(bedrooms)
    row["Resale"] = int(resale)

    # Enable amenities (case-insensitive lookup)
    amenity_map = {c.lower(): c for c in amenity_columns}
    active_amenities = []
    for item in amenities:
        item_clean = str(item).strip().lower()
        if item_clean in amenity_map:
            actual_col = amenity_map[item_clean]
            row[actual_col] = 1
            active_amenities.append(actual_col)

    # Calculate amenity count
    row["amenity_count"] = sum(row[c] for c in amenity_columns)

    # Create DataFrame for model input
    df_input = pd.DataFrame([row])

    # Model inference (reverse log1p transformation)
    pred_log = model.predict(df_input)[0]
    estimated_price = float(np.expm1(pred_log))

    # Calculate 80% Confidence Valuation Range
    ci = meta["confidence_interval"]
    multiplier_low = ci["multiplier_low"]
    multiplier_high = ci["multiplier_high"]
    price_low = estimated_price * multiplier_low
    price_high = estimated_price * multiplier_high

    # Locality comparison
    locality_medians = meta.get("locality_medians", {})
    overall_median = meta.get("overall_median_price", 7500000.0)

    is_known_locality = location in locality_medians
    locality_median = locality_medians.get(location, overall_median)

    diff_from_median = estimated_price - locality_median
    diff_pct = (diff_from_median / locality_median) * 100

    price_per_sqft = estimated_price / area if area > 0 else 0

    return {
        "status": "success",
        "inputs": {
            "location": location,
            "is_known_locality": is_known_locality,
            "area_sqft": area,
            "bedrooms": bedrooms,
            "resale": bool(resale),
            "amenities_count": len(active_amenities),
            "amenities_selected": active_amenities
        },
        "valuation": {
            "estimated_price": round(estimated_price, 2),
            "estimated_price_formatted": format_currency(estimated_price),
            "price_per_sqft": round(price_per_sqft, 2),
            "price_per_sqft_formatted": f"₹ {price_per_sqft:,.0f} / sq.ft",
            "confidence_level": "80%",
            "price_range_low": round(price_low, 2),
            "price_range_low_formatted": format_currency(price_low),
            "price_range_high": round(price_high, 2),
            "price_range_high_formatted": format_currency(price_high)
        },
        "locality_benchmark": {
            "locality": location if is_known_locality else f"{location} (Overall Delhi Average used)",
            "locality_median_price": round(locality_median, 2),
            "locality_median_formatted": format_currency(locality_median),
            "diff_from_median_pct": round(diff_pct, 2),
            "benchmark_insight": (
                f"{abs(diff_pct):.1f}% higher than locality median"
                if diff_pct > 0 else
                f"{abs(diff_pct):.1f}% lower than locality median"
            )
        }
    }


def print_valuation_report(res: dict):
    """Prints a beautiful CLI valuation summary."""
    val = res["valuation"]
    bench = res["locality_benchmark"]
    inp = res["inputs"]

    print("\n" + "=" * 55)
    print("          PROP-IQ PROPERTY VALUATION REPORT")
    print("=" * 55)
    print(f" Property Details:")
    print(f"   • Location     : {inp['location']}")
    print(f"   • Area         : {inp['area_sqft']} sq. ft.")
    print(f"   • Bedrooms     : {inp['bedrooms']} BHK")
    print(f"   • Type         : {'Resale' if inp['resale'] else 'New Booking'}")
    print(f"   • Amenities    : {inp['amenities_count']} selected")
    print("-" * 55)
    print(f" Fair Market Valuation:")
    print(f"   ► Estimated Price : {val['estimated_price_formatted']}")
    print(f"   ► Price per sq.ft : {val['price_per_sqft_formatted']}")
    print(f"   ► Valuation Range : {val['price_range_low_formatted']} — {val['price_range_high_formatted']} ({val['confidence_level']} CI)")
    print("-" * 55)
    print(f" Locality Benchmark:")
    print(f"   • Locality Median : {bench['locality_median_formatted']}")
    print(f"   • Comparison      : {bench['benchmark_insight']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    # Test example
    sample_result = predict_property_price(
        location="Mayur Vihar",
        area=1300,
        bedrooms=2,
        resale=1,
        amenities=["Gymnasium", "LiftAvailable", "24X7Security", "PowerBackup"]
    )
    print_valuation_report(sample_result)
