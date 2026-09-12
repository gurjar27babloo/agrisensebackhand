"""
app.py
-------
Minimal Flask API exposing the AquaSense AI decision engine.

This is the piece the ESP32 (or the main backend, per the architecture
diagram) calls after fetching weather data — POST sensor + weather data,
get back an irrigation decision.

Run:
    python3 app.py
Then test:
    curl -X POST http://localhost:5000/api/decide \
      -H "Content-Type: application/json" \
      -d '{"soil_moisture":22,"temperature":33,"humidity":40,"rain_forecast":0,"crop_type":"cotton"}'
"""

from flask import Flask, request, jsonify, send_from_directory
from irrigation_ai import decide_irrigation, CROP_NAMES
from weather import get_rain_forecast_mm
import traceback
import sqlite3
import os
from datetime import datetime, timezone

app = Flask(__name__, static_folder="static")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            soil_moisture REAL,
            temperature REAL,
            humidity REAL,
            rain_forecast_mm REAL,
            crop_type TEXT,
            irrigate INTEGER,
            power_level REAL,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_reading(soil, temp, hum, rain, crop, irrigate, power, reason):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO readings
           (timestamp, soil_moisture, temperature, humidity, rain_forecast_mm,
            crop_type, irrigate, power_level, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), soil, temp, hum, rain,
         crop, int(irrigate), power, reason),
    )
    conn.commit()
    conn.close()


init_db()

# rain_forecast is now OPTIONAL in the request — if the ESP32 doesn't send
# it (recommended, see weather.py), the backend fetches it automatically.
REQUIRED_FIELDS = ["soil_moisture", "temperature", "humidity", "crop_type"]


@app.route("/api/decide", methods=["POST"])
def api_decide():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing or invalid JSON body"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        # If the caller (ESP32) already provided a rain_forecast value,
        # respect it. Otherwise, fetch live weather for the device's
        # configured location.
        if "rain_forecast" in data and data["rain_forecast"] is not None:
            rain_forecast = data["rain_forecast"]
            weather_source = "provided_by_caller"
        else:
            rain_forecast = get_rain_forecast_mm()
            weather_source = "openweathermap_live"

        result = decide_irrigation(
            soil_moisture=data["soil_moisture"],
            temperature=data["temperature"],
            humidity=data["humidity"],
            rain_forecast=rain_forecast,
            crop_type=data["crop_type"],
        )
        result["rain_forecast_mm"] = rain_forecast
        result["weather_source"] = weather_source

        log_reading(
            soil=data["soil_moisture"],
            temp=data["temperature"],
            hum=data["humidity"],
            rain=rain_forecast,
            crop=data["crop_type"],
            irrigate=result["irrigate"],
            power=result["power_level"],
            reason=result["reason"],
        )

        return jsonify({
            "status": "success",
            "decision": result,
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal error processing decision"}), 500


@app.route("/api/weather", methods=["GET"])
def api_weather():
    """Standalone endpoint to check the current cached/live rain forecast."""
    rain_mm = get_rain_forecast_mm()
    return jsonify({"rain_forecast_mm": rain_mm}), 200


@app.route("/api/crops", methods=["GET"])
def api_crops():
    """Lists supported crop types — useful for a frontend dropdown."""
    return jsonify({"crops": list(CROP_NAMES.values())}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AquaSense AI decision engine"}), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns the most recent readings for the dashboard to display."""
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 500))  # sane bounds

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    readings = [dict(row) for row in rows]
    readings.reverse()  # oldest first, easier for charting

    latest = readings[-1] if readings else None

    # Simple cumulative estimate for the dashboard's "water saved" style stat —
    # counts irrigation events and roughly estimates liters based on power level.
    total_events = sum(1 for r in readings if r["irrigate"])
    estimated_liters = sum(
        (r["power_level"] / 100.0) * 5.0 for r in readings if r["irrigate"]
    )  # assumes ~5L per 100%-power 15s burst — adjust to your actual pump flow rate

    return jsonify({
        "latest": latest,
        "readings": readings,
        "summary": {
            "total_readings": len(readings),
            "irrigation_events": total_events,
            "estimated_liters_used": round(estimated_liters, 1),
        },
    }), 200


@app.route("/")
def dashboard():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
