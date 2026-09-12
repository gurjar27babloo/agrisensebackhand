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

from flask import Flask, request, jsonify
from irrigation_ai import decide_irrigation, CROP_NAMES
from weather import get_rain_forecast_mm
import traceback

app = Flask(__name__)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
