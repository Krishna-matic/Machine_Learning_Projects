"""
app.py
------
Flask backend for the Rainfall Prediction web application.

This module loads the trained Random Forest model along with the two
artifacts produced during training (`feature_columns.pkl` and
`feature_defaults.pkl`) and exposes a small web app that:

  1. Serves a "Basic" prediction form (a handful of the most important
     weather fields) with an optional "Advanced" section for the rest.
  2. Reproduces -- feature for feature -- the same preprocessing pipeline
     used in the training notebook:
        - Missing/omitted value imputation (medians/modes learned from
          the training data, with a location+month aware lookup for
          `Evaporation` and `Sunshine`).
        - Date-based feature engineering (Year, Month, Day, DayOfWeek,
          Season).
        - Label encoding of `RainToday`.
        - One-hot encoding of the nominal categorical columns.
        - Reindexing to the exact 122-column layout the model was
          trained on, filling any column absent from a given request
          with 0.
  3. Runs the trained model on the resulting feature vector and renders
     a result page with the prediction, a confidence score, and simple
     actionable suggestions.

Nothing here is hardcoded to "122 features" -- the number of expected
features is always read from `feature_columns.pkl` and cross-checked
against `model.n_features_in_` at start-up, so the app stays correct
even if the model is retrained with a different feature set.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "rainfall_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
FEATURE_DEFAULTS_PATH = MODELS_DIR / "feature_defaults.pkl"

# The nominal categorical columns that were one-hot encoded during
# training (via `pd.get_dummies(..., drop_first=True)`). Any category
# that has no matching "<column>_<value>" entry in `feature_columns.pkl`
# is the dropped baseline category and is correctly represented by
# leaving every one-hot column for that field at 0.
ONE_HOT_BASE_COLUMNS = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "DayOfWeek",
    "Season",
]

# RainToday was label-encoded with scikit-learn's LabelEncoder, which
# sorts classes alphabetically. For {"No", "Yes"} that always yields
# No -> 0, Yes -> 1.
RAIN_TODAY_MAP = {"No": 0, "Yes": 1}

# Fields collected directly on the Basic form (HTML field name -> raw
# training-data column name).
BASIC_FIELD_MAP = {
    "location": "Location",
    "date": "Date",
    "min_temp": "MinTemp",
    "max_temp": "MaxTemp",
    "rainfall": "Rainfall",
    "humidity_9am": "Humidity9am",
    "humidity_3pm": "Humidity3pm",
    "pressure_9am": "Pressure9am",
    "pressure_3pm": "Pressure3pm",
    "wind_gust_speed": "WindGustSpeed",
    "wind_gust_dir": "WindGustDir",
    "cloud_3pm": "Cloud3pm",
    "rain_today": "RainToday",
}

# Fields collected only when the user expands "Advanced Inputs". Any of
# these left blank/omitted are filled in from feature_defaults.pkl.
ADVANCED_FIELD_MAP = {
    "evaporation": "Evaporation",
    "sunshine": "Sunshine",
    "wind_dir_9am": "WindDir9am",
    "wind_dir_3pm": "WindDir3pm",
    "wind_speed_9am": "WindSpeed9am",
    "wind_speed_3pm": "WindSpeed3pm",
    "cloud_9am": "Cloud9am",
    "temp_9am": "Temp9am",
    "temp_3pm": "Temp3pm",
}

# Raw columns that are numeric vs. categorical (needed to know whether a
# blank Advanced field should be filled from a median or a mode).
NUMERIC_RAW_COLUMNS = {
    "Evaporation",
    "Sunshine",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Cloud9am",
    "Temp9am",
    "Temp3pm",
}
CATEGORICAL_RAW_COLUMNS = {"WindDir9am", "WindDir3pm"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Model + artifact loading (done once, at start-up)
# ----------------------------------------------------------------------

def _load_artifacts():
    """Load the trained model and its companion preprocessing artifacts.

    Raises a clear, actionable error immediately at start-up if any file
    is missing or inconsistent, rather than failing confusingly on the
    first prediction request.
    """
    for path in (MODEL_PATH, FEATURE_COLUMNS_PATH, FEATURE_DEFAULTS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}. "
                "Make sure rainfall_model.pkl, feature_columns.pkl, and "
                "feature_defaults.pkl all live in the 'models/' directory."
            )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    feature_defaults = joblib.load(FEATURE_DEFAULTS_PATH)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and expected_features != len(feature_columns):
        raise ValueError(
            f"Mismatch between the trained model and feature_columns.pkl: "
            f"model.n_features_in_ = {expected_features}, but "
            f"feature_columns.pkl has {len(feature_columns)} entries. "
            "Regenerate feature_columns.pkl from the same training run "
            "that produced rainfall_model.pkl."
        )

    logger.info(
        "Loaded model expecting %d features (feature_columns.pkl has %d entries).",
        expected_features if expected_features is not None else len(feature_columns),
        len(feature_columns),
    )
    return model, feature_columns, feature_defaults


MODEL, FEATURE_COLUMNS, FEATURE_DEFAULTS = _load_artifacts()

# The number of features the model expects, read directly from the
# artifacts rather than hardcoded, per the project's consistency
# requirement.
N_EXPECTED_FEATURES = len(FEATURE_COLUMNS)

# Columns in feature_columns.pkl that are NOT one-hot encoded (i.e. they
# are used as-is: raw numeric measurements, the label-encoded RainToday,
# and the engineered Year/Month/Day integers). Derived dynamically so the
# app keeps working even if the training pipeline changes slightly.
DIRECT_COLUMNS = [
    col
    for col in FEATURE_COLUMNS
    if not any(col.startswith(f"{base}_") for base in ONE_HOT_BASE_COLUMNS)
]


# ----------------------------------------------------------------------
# Dropdown option lists for the templates (display only -- these do not
# affect preprocessing, they only populate <select> elements in the UI).
# ----------------------------------------------------------------------

# The 16 standard compass directions used throughout the dataset.
WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def _build_locations():
    """Derive the full list of selectable locations from feature_columns.pkl.

    Every location the model was trained on has a `Location_<name>`
    one-hot column EXCEPT the alphabetically-first one, which
    `pd.get_dummies(..., drop_first=True)` drops as the baseline
    category during training. That baseline ("Adelaide" in this
    dataset) is still a perfectly valid, selectable location -- picking
    it simply means every `Location_*` column stays at 0, which
    `build_model_input` already handles correctly.
    """
    onehot_locations = {
        col.split("Location_", 1)[1] for col in FEATURE_COLUMNS if col.startswith("Location_")
    }
    baseline_candidates = onehot_locations | {"Adelaide"}
    return sorted(baseline_candidates)


def _humanize_location(name: str) -> str:
    """Turn a raw location code like 'CoffsHarbour' into a readable
    label like 'Coffs Harbour', purely for display in the dropdown.
    """
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    label = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", label)
    return label


LOCATIONS = _build_locations()
LOCATION_OPTIONS = [{"value": loc, "label": _humanize_location(loc)} for loc in LOCATIONS]


# ----------------------------------------------------------------------
# Feature engineering helpers
# ----------------------------------------------------------------------

def get_season(month: int) -> str:
    """Map a calendar month to an Australian season, exactly as the
    training notebook does.
    """
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def _to_float(value, field_name: str) -> float:
    """Safely convert a form value to float, raising a friendly error."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a number, got: {value!r}") from exc


# ----------------------------------------------------------------------
# Preprocessing pipeline
# ----------------------------------------------------------------------

def extract_raw_features(form: dict) -> dict:
    """Read the incoming form data and return a dict of raw (pre-encoding)
    feature values, filling in any omitted Advanced field with the
    appropriate default computed from the training data.

    Basic fields are always required. Advanced fields fall back to:
      - the (Location, Month) median for Evaporation/Sunshine,
      - the global median for the other numeric Advanced fields,
      - the global mode for the categorical Advanced fields (wind
        directions at 9am/3pm).
    """
    raw: dict = {}

    # --- Basic (required) fields -----------------------------------
    location = form.get("location", "").strip()
    if not location:
        raise ValueError("'location' is required.")
    raw["Location"] = location

    date_str = form.get("date", "").strip()
    if not date_str:
        raise ValueError("'date' is required.")
    try:
        observation_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"'date' must be in YYYY-MM-DD format, got: {date_str!r}"
        ) from exc

    raw["Year"] = observation_date.year
    raw["Month"] = observation_date.month
    raw["Day"] = observation_date.day
    raw["DayOfWeek"] = observation_date.strftime("%A")  # e.g. "Monday"
    raw["Season"] = get_season(observation_date.month)

    numeric_basic_fields = {
        "min_temp": "MinTemp",
        "max_temp": "MaxTemp",
        "rainfall": "Rainfall",
        "humidity_9am": "Humidity9am",
        "humidity_3pm": "Humidity3pm",
        "pressure_9am": "Pressure9am",
        "pressure_3pm": "Pressure3pm",
        "wind_gust_speed": "WindGustSpeed",
        "cloud_3pm": "Cloud3pm",
    }
    for form_field, raw_col in numeric_basic_fields.items():
        raw[raw_col] = _to_float(form.get(form_field, ""), form_field)

    wind_gust_dir = form.get("wind_gust_dir", "").strip()
    if not wind_gust_dir:
        raise ValueError("'wind_gust_dir' is required.")
    raw["WindGustDir"] = wind_gust_dir

    rain_today = form.get("rain_today", "").strip()
    if rain_today not in RAIN_TODAY_MAP:
        raise ValueError("'rain_today' must be 'Yes' or 'No'.")
    raw["RainToday"] = RAIN_TODAY_MAP[rain_today]

    # --- Advanced (optional) fields ---------------------------------
    numerical_medians = FEATURE_DEFAULTS["numerical_medians"]
    categorical_modes = FEATURE_DEFAULTS["categorical_modes"]
    location_month_medians = FEATURE_DEFAULTS["location_month_medians"]

    for form_field, raw_col in ADVANCED_FIELD_MAP.items():
        value = form.get(form_field, "")
        value = value.strip() if isinstance(value, str) else value

        if raw_col in NUMERIC_RAW_COLUMNS:
            if value:
                raw[raw_col] = _to_float(value, form_field)
            elif raw_col in location_month_medians:
                # Evaporation / Sunshine: location+month aware default,
                # falling back to the global median if that specific
                # (location, month) pair was never observed in training.
                fallback = numerical_medians[raw_col]
                raw[raw_col] = location_month_medians[raw_col].get(
                    (location, observation_date.month), fallback
                )
            else:
                raw[raw_col] = numerical_medians[raw_col]
        elif raw_col in CATEGORICAL_RAW_COLUMNS:
            raw[raw_col] = value if value else categorical_modes[raw_col]

    return raw


def build_model_input(raw: dict) -> pd.DataFrame:
    """Turn a dict of raw (engineered but not yet encoded) feature values
    into a single-row DataFrame whose columns exactly match
    `feature_columns.pkl`, in the same order, with the same one-hot
    encoding scheme used during training.
    """
    # Start every column at 0; one-hot columns that stay 0 correctly
    # represent either the dropped baseline category or a category the
    # model never saw during training.
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Direct (already-numeric / label-encoded) columns.
    for col in DIRECT_COLUMNS:
        if col in raw:
            row[col] = raw[col]

    # One-hot encoded columns: set the single matching "<base>_<value>"
    # column to 1 if the model was trained on that category.
    for base in ONE_HOT_BASE_COLUMNS:
        value = raw.get(base)
        if value is None:
            continue
        dummy_col = f"{base}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1
        else:
            logger.warning(
                "Category '%s' for '%s' was not seen during training "
                "(or is the dropped baseline category) -- treating it "
                "as the reference category.",
                value,
                base,
            )

    # Build the final single-row DataFrame in the exact training column
    # order the model expects.
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    if input_df.shape[1] != N_EXPECTED_FEATURES:
        # Defensive check -- should be unreachable given the DataFrame
        # was constructed directly from FEATURE_COLUMNS, but this keeps
        # the failure mode explicit if that ever changes.
        raise ValueError(
            f"Built {input_df.shape[1]} features but the model expects "
            f"{N_EXPECTED_FEATURES}."
        )

    return input_df


def predict_rainfall(form: dict) -> dict:
    """Run the full pipeline end-to-end: parse form -> engineer features
    -> encode -> predict. Returns a small result dict ready for the
    template.
    """
    raw_features = extract_raw_features(form)
    model_input = build_model_input(raw_features)

    prediction = MODEL.predict(model_input)[0]
    probabilities = MODEL.predict_proba(model_input)[0]

    will_rain = bool(prediction == 1)
    confidence = probabilities[1] if will_rain else probabilities[0]

    return {
        "will_rain": will_rain,
        "confidence": round(float(confidence) * 100, 1),
        "location": raw_features["Location"],
        "date": form.get("date", ""),
    }


# ----------------------------------------------------------------------
# Flask application
# ----------------------------------------------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the prediction form (Basic + collapsible Advanced section)."""
    return render_template(
        "index.html", locations=LOCATION_OPTIONS, wind_directions=WIND_DIRECTIONS
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle a form submission: preprocess, run the model, show the
    result page. Falls back to a friendly error message on bad input
    rather than a raw stack trace.
    """
    try:
        result = predict_rainfall(request.form)
    except ValueError as exc:
        logger.warning("Invalid prediction request: %s", exc)
        return (
            render_template(
                "index.html",
                error=str(exc),
                locations=LOCATION_OPTIONS,
                wind_directions=WIND_DIRECTIONS,
            ),
            400,
        )
    except Exception:  # noqa: BLE001 - surface a safe message, log details
        logger.exception("Unexpected error while generating a prediction.")
        return (
            render_template(
                "index.html",
                error="Something went wrong while generating your prediction. Please try again.",
                locations=LOCATION_OPTIONS,
                wind_directions=WIND_DIRECTIONS,
            ),
            500,
        )

    return render_template("result.html", **result)


if __name__ == "__main__":
    # debug=True is convenient for local development; set to False (or
    # use a production WSGI server such as gunicorn) when deploying.
    app.run(debug=True)
