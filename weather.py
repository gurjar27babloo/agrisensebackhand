"""
weather.py
-----------
Fetches a 24-hour rain forecast from OpenWeatherMap for the farm's fixed
location, so the AI decision engine can suppress irrigation when rain is
coming — exactly the "weather-aware" behavior described throughout the
project.

Why this lives in the BACKEND, not the ESP32:
    The ESP32 could call OpenWeatherMap directly over its 4G connection,
    but parsing JSON weather responses on a microcontroller is fragile and
    wastes SIM800L's limited/paid data on a large JSON payload every cycle.
    It's simpler, cheaper, and more reliable for the backend (which already
    has the sensor data) to fetch weather itself and combine both before
    running the AI model. The ESP32 just needs to send its sensor readings;
    it no longer needs to know or send rain_forecast at all.

Get a free API key at: https://openweathermap.org/api  (free tier: 60
calls/minute, 1,000,000 calls/month — far more than one device checking
every 10 minutes needs).
"""

import os
import requests

OWM_API_KEY = os.environ.get("OWM_API_KEY", "")
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Set these to your farm's actual coordinates (fixed per deployment).
# You can get lat/lon from Google Maps: right-click a location -> the
# numbers shown are (lat, lon).
DEVICE_LAT = float(os.environ.get("DEVICE_LAT", "18.5204"))  # placeholder: Pune, India
DEVICE_LON = float(os.environ.get("DEVICE_LON", "73.8567"))

# Simple in-memory cache so we don't hit the weather API every single
# 10-minute sensor cycle — weather forecasts don't change that fast, and
# this keeps you well inside the free tier even with multiple devices.
_cache = {"timestamp": 0, "rain_mm": None}
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes


def get_rain_forecast_mm(lat=None, lon=None, force_refresh=False):
    """
    Returns the total expected rainfall (mm) over the next 24 hours for
    the given coordinates (defaults to DEVICE_LAT/DEVICE_LON).

    Returns 0.0 (assume no rain) if the API key is missing or the request
    fails — irrigation decisions should never be blocked just because the
    weather API had a hiccup.
    """
    import time

    lat = lat if lat is not None else DEVICE_LAT
    lon = lon if lon is not None else DEVICE_LON

    now = time.time()
    if not force_refresh and _cache["rain_mm"] is not None:
        if now - _cache["timestamp"] < CACHE_TTL_SECONDS:
            return _cache["rain_mm"]

    if not OWM_API_KEY:
        print("WARNING: OWM_API_KEY not set — skipping weather fetch, assuming 0mm rain.")
        return 0.0

    try:
        response = requests.get(
            OWM_FORECAST_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": OWM_API_KEY,
                "units": "metric",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # The free /forecast endpoint returns 3-hour interval blocks.
        # Next 24h = next 8 blocks. Sum any "rain" volume in each block.
        total_rain_mm = 0.0
        for block in data.get("list", [])[:8]:
            rain_info = block.get("rain", {})
            # OpenWeatherMap reports this as "3h" (mm of rain in that 3h window)
            total_rain_mm += rain_info.get("3h", 0.0)

        _cache["timestamp"] = now
        _cache["rain_mm"] = round(total_rain_mm, 1)
        return _cache["rain_mm"]

    except requests.exceptions.RequestException as e:
        print(f"WARNING: Weather fetch failed ({e}) — assuming 0mm rain this cycle.")
        return 0.0
    except (KeyError, ValueError) as e:
        print(f"WARNING: Weather response parsing failed ({e}) — assuming 0mm rain.")
        return 0.0


if __name__ == "__main__":
    # Quick manual test: python3 weather.py
    print(f"Using coordinates: {DEVICE_LAT}, {DEVICE_LON}")
    rain = get_rain_forecast_mm(force_refresh=True)
    print(f"Forecast rain over next 24h: {rain} mm")
