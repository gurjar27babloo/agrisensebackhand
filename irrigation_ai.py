"""
irrigation_ai.py
-----------------
The core AquaSense AI decision engine.

Loads the trained classifier + regressor and exposes a single function,
decide_irrigation(), that the backend (Flask API, or directly the ESP32
data-ingestion handler) calls every sensing cycle.

This is the module described throughout the project docs as the
"AI decision engine" / "ML model" sitting between sensor ingestion and
relay/pump control.
"""

import joblib
import pandas as pd
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_classifier = joblib.load(os.path.join(MODEL_DIR, "irrigation_classifier.joblib"))
_regressor = joblib.load(os.path.join(MODEL_DIR, "irrigation_power_regressor.joblib"))

CROP_NAMES = {0: "wheat", 1: "rice", 2: "vegetables", 3: "cotton", 4: "maize"}
CROP_IDS = {v: k for k, v in CROP_NAMES.items()}

FEATURES = ["soil_moisture", "temperature", "humidity", "rain_forecast", "crop_type"]

# Hard safety bounds — the AI's power suggestion is always clamped to these,
# regardless of what the model predicts. Protects hardware from nonsense
# values if inputs are ever out of the training distribution.
MIN_POWER = 0
MAX_POWER = 100


def _resolve_crop_type(crop_type):
    """Accept either a crop name (str) or an already-encoded id (int)."""
    if isinstance(crop_type, str):
        crop_type = crop_type.strip().lower()
        if crop_type not in CROP_IDS:
            raise ValueError(
                f"Unknown crop_type '{crop_type}'. Valid options: {list(CROP_IDS.keys())}"
            )
        return CROP_IDS[crop_type]
    return int(crop_type)


def decide_irrigation(soil_moisture, temperature, humidity, rain_forecast, crop_type):
    """
    Run one irrigation decision cycle.

    Parameters
    ----------
    soil_moisture : float   % volumetric water content, 0-100
    temperature   : float   air temperature in Celsius
    humidity      : float   relative humidity %, 0-100
    rain_forecast : float   forecast rainfall over next 24h, in mm
    crop_type     : str|int crop name ("wheat","rice","vegetables","cotton","maize")
                             or the equivalent encoded id (0-4)

    Returns
    -------
    dict with keys:
        irrigate        : bool
        power_level     : float (0-100), 0 if irrigate is False
        confidence      : float (0-1), classifier's probability for its decision
        reason          : short human-readable explanation
        crop            : resolved crop name
    """
    crop_id = _resolve_crop_type(crop_type)

    row = pd.DataFrame([{
        "soil_moisture": float(soil_moisture),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "rain_forecast": float(rain_forecast),
        "crop_type": crop_id,
    }])[FEATURES]

    irrigate_pred = int(_classifier.predict(row)[0])
    proba = _classifier.predict_proba(row)[0]
    # proba is [P(class=0), P(class=1)] assuming classes_ == [0, 1]
    class_index = list(_classifier.classes_).index(irrigate_pred)
    confidence = float(proba[class_index])

    if irrigate_pred == 1:
        power = float(_regressor.predict(row)[0])
        power = max(MIN_POWER, min(MAX_POWER, power))
    else:
        power = 0.0

    reason = _explain(soil_moisture, rain_forecast, irrigate_pred, crop_id)

    return {
        "irrigate": bool(irrigate_pred),
        "power_level": round(power, 1),
        "confidence": round(confidence, 3),
        "reason": reason,
        "crop": CROP_NAMES[crop_id],
    }


def _explain(soil_moisture, rain_forecast, irrigate_pred, crop_id):
    """Generate a short, farmer-readable reason string for the dashboard/logs."""
    crop = CROP_NAMES[crop_id]
    if irrigate_pred == 1:
        if rain_forecast >= 8:
            return (f"Soil critically dry ({soil_moisture:.0f}%) for {crop} — "
                     f"irrigating at reduced power despite {rain_forecast:.0f}mm rain forecast.")
        return f"Soil moisture low ({soil_moisture:.0f}%) for {crop} — irrigation needed."
    else:
        if rain_forecast >= 8:
            return f"Skipping irrigation — {rain_forecast:.0f}mm rain forecast in next 24h."
        return f"Soil moisture adequate ({soil_moisture:.0f}%) for {crop} — no irrigation needed."


if __name__ == "__main__":
    # Quick smoke test / demo when run directly: python3 irrigation_ai.py
    test_cases = [
        dict(soil_moisture=25, temperature=34, humidity=40, rain_forecast=0, crop_type="cotton"),
        dict(soil_moisture=20, temperature=32, humidity=55, rain_forecast=15, crop_type="wheat"),
        dict(soil_moisture=65, temperature=29, humidity=60, rain_forecast=2, crop_type="rice"),
        dict(soil_moisture=40, temperature=31, humidity=45, rain_forecast=0, crop_type="vegetables"),
        dict(soil_moisture=10, temperature=38, humidity=25, rain_forecast=20, crop_type="maize"),
    ]

    print("AquaSense AI — Decision Engine Smoke Test")
    print("=" * 60)
    for i, case in enumerate(test_cases, 1):
        result = decide_irrigation(**case)
        print(f"\nCase {i}: {case}")
        print(f"  -> Irrigate: {result['irrigate']}  |  Power: {result['power_level']}%  "
              f"|  Confidence: {result['confidence']}")
        print(f"     Reason: {result['reason']}")
