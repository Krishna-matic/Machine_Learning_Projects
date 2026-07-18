# Rainscope — Rainfall Prediction for Australian Weather Stations

A Flask web application that predicts whether it will rain **tomorrow** at any of
49 Australian weather stations, based on today's observed conditions. Predictions
are served by a Random Forest Classifier trained on ten years of historical
observations from the Australian Bureau of Meteorology.

---

## Overview

Rainscope wraps a scikit-learn Random Forest model in a clean, modern web
interface. Users can get a prediction from just the handful of weather
measurements most people actually have on hand (temperature, humidity,
pressure, wind, cloud cover), or expand an "Advanced" panel to supply the
full set of features the model was trained on for maximum precision.

Anything left blank in Advanced mode is filled in automatically using
statistics learned from the training data itself (median/mode values, with
a location- and month-aware lookup for evaporation and sunshine), so the
model always receives a complete, correctly-shaped feature vector.

---

## Features

- **Two prediction modes**
  - **Basic** — the 13 most important fields, shown by default.
  - **Advanced** — a smoothly collapsible panel exposing the remaining
    9 raw weather features, all optional.
- **Automatic feature engineering** — `Year`, `Month`, `Day`, `DayOfWeek`,
  and `Season` are derived from the date you pick, exactly as during
  training.
- **Automatic imputation** — any Advanced field left blank is filled in
  from `feature_defaults.pkl`, generated directly from the training data.
- **Consistent, model-driven feature count** — the app reads the expected
  number of features from `feature_columns.pkl` and cross-checks it
  against `model.n_features_in_` at start-up, rather than hardcoding it.
- **Premium glassmorphism UI** — animated gradient sky background, glass
  cards, a hand-built SVG icon set, and a barometer-style confidence
  gauge on the results page.
- **Fully responsive** — desktop, tablet, and mobile layouts, built with
  plain HTML5, CSS3, and vanilla JavaScript (no frameworks, no
  build step).
- **Friendly error handling** — missing/invalid input is caught before
  it ever reaches the model, with a clear message shown back on the form.

---

## Dataset

- **Source:** `weatherAUS.csv` — the Australian Weather dataset
  (Bureau of Meteorology daily observations, 2007–2017).
- **Size:** 145,460 rows × 23 columns.
- **Target:** `RainTomorrow` (`Yes` / `No`) — whether measurable rain fell
  on the following day.
- **Coverage:** 49 weather stations across Australia.
- **Class balance:** roughly 78% "No" / 22% "Yes" — a meaningfully
  imbalanced target, which is why accuracy alone was not used to pick the
  final model (see below).

---

## Machine Learning Workflow

The full workflow — EDA, imputation, feature engineering, encoding,
modeling, and evaluation — lives in `Classification_ML_Project.ipynb`.
Summary of the pipeline the Flask app reproduces at inference time:

1. **Missing value handling**
   - Rows with a missing target (`RainTomorrow`) are dropped.
   - All other numeric columns are imputed with their **median**.
   - All other categorical columns are imputed with their **mode**.
   - `Evaporation` and `Sunshine` receive a more precise treatment: median
     imputed **per (Location, Month)** group, falling back to the global
     median for any group with no observed values.
2. **Feature engineering** — `Date` is expanded into `Year`, `Month`,
   `Day`, `DayOfWeek`, and `Season` (Australian seasons: Dec–Feb Summer,
   Mar–May Autumn, Jun–Aug Winter, Sep–Nov Spring), then dropped.
3. **Encoding**
   - `RainToday` / `RainTomorrow` — label encoded (`No` → 0, `Yes` → 1).
   - `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `DayOfWeek`,
     `Season` — one-hot encoded with `drop_first=True`.
4. **Train/test split** — 80/20, stratified on the target.
5. **Model comparison** — Logistic Regression, KNN, Linear SVM, Decision
   Tree, Random Forest, and Gaussian Naive Bayes were all trained and
   tuned with `GridSearchCV` (5-fold cross-validation).
6. **Model selection** — Random Forest gave the best overall balance of
   accuracy, precision, recall, and ROC-AUC and was chosen as the final
   model.
7. **Serialization** — the trained model and the exact training column
   order were persisted with `joblib`:
   - `models/rainfall_model.pkl`
   - `models/feature_columns.pkl`

The Flask app adds one more artifact, generated the same way from the same
CSV, so Basic-mode predictions can fill in omitted fields consistently:
- `models/feature_defaults.pkl` — global medians/modes plus the
  (Location, Month) lookup for `Evaporation`/`Sunshine`.

---

## Model Used

**Random Forest Classifier** (scikit-learn), selected over Logistic
Regression, KNN, Linear SVM, Decision Tree, and Gaussian Naive Bayes.

| Metric | Test score |
|---|---|
| Accuracy | ~85.8% |
| Precision | ~78.9% |
| Recall | ~49.8% |
| F1-Score | ~61.1% |
| ROC-AUC | ~88.9% |

The model consumes **122 engineered features** per prediction — this
number is never hardcoded in the app; it's read from `feature_columns.pkl`
and verified against `model.n_features_in_` at start-up.

---

## Project Structure

```
Rainfall_Prediction/
│
├── app.py                      # Flask application (routes + full preprocessing pipeline)
├── requirements.txt
├── README.md
│
├── models/
│   ├── rainfall_model.pkl      # Trained Random Forest Classifier
│   ├── feature_columns.pkl     # Exact 122-column training feature order
│   └── feature_defaults.pkl    # Medians/modes for Basic-mode auto-fill
│
├── templates/
│   ├── index.html              # Basic + Advanced prediction form
│   └── result.html             # Prediction result page
│
└── static/
    ├── css/
    │   └── style.css           # Glassmorphism UI, gradients, animations
    ├── js/
    │   └── script.js           # Collapsible panel, validation, gauge animation
    └── images/                 # (reserved for future static assets)
```

> `feature_defaults.pkl` is generated automatically from `weatherAUS.csv`
> and is not meant to be edited by hand. If you retrain the model, see
> [Regenerating the model artifacts](#regenerating-the-model-artifacts).

---

## Installation

**Prerequisites:** Python 3.10+ and `pip`.

```bash
# 1. Clone or copy this project, then move into it
cd Rainfall_Prediction

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Flask Application

```bash
python app.py
```

By default this starts the Flask development server at
`http://127.0.0.1:5000/`. Open that address in a browser, fill in the
Basic prediction form (optionally expanding Advanced Inputs), and submit
to see tomorrow's forecast.

For a production deployment, run it behind a real WSGI server instead of
the built-in dev server:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

### Regenerating the model artifacts

If you retrain the model in `Classification_ML_Project.ipynb`, re-export
`rainfall_model.pkl` and `feature_columns.pkl` from the notebook, then
regenerate `feature_defaults.pkl` from the same training CSV so all three
artifacts stay in sync (a `compute_defaults.py` script mirroring the
notebook's exact imputation logic can be used for this — see the project
history / notebook for the reference implementation).

---

## Screenshots

> _Add screenshots of the running application here._

| Basic prediction form | Advanced inputs expanded | Result — rain expected | Result — no rain expected |
|---|---|---|---|
| _add image_ | _add image_ | _add image_ | _add image_ |

---

## Future Improvements

- Add a `/api/predict` JSON endpoint for programmatic access alongside
  the HTML form.
- Cache recent predictions per location to reduce repeated computation.
- Add a lightweight test suite (`pytest`) covering `extract_raw_features`
  and `build_model_input` with edge cases (unseen locations, boundary
  dates, malformed input).
- Explore recall-focused thresholding or class-weighting, since the
  training data is imbalanced (~78% No / ~22% Yes) and the current model
  favors precision over recall.
- Add a 5-day outlook by chaining predictions with forecast weather APIs.
- Containerize with Docker for simpler deployment.

---

## License

This project is provided for educational purposes as part of a machine
learning capstone project. Add your preferred license (e.g. MIT) here
before distributing.
