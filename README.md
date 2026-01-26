# StatsBomb xG Model Dashboard

An interactive Expected Goals (xG) model and dashboard built using StatsBomb's open data from La Liga 2015/16.

**[Live Demo](https://xg-model-statsbomb.onrender.com)** | **[Portfolio](https://donalhill.github.io)**

## Overview

This project demonstrates:
1. **xG Model Development** - XGBoost classifier with isotonic calibration
2. **Feature Engineering** - Distance, angle, GK positioning, defender proximity, goal visibility
3. **Temporal Validation** - Train on Aug-Dec 2015, test on Jan-May 2016 (no data leakage)
4. **Interactive Dashboard** - Dash app with model comparison, spatial analysis, and player breakdowns
5. **SHAP Explainability** - Feature impact visualisation

## Data

- **Source**: [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- **Competition**: La Liga 2015/16 (380 matches)
- **Shots**: ~9,000 shots with freeze-frame data

## Live Dashboard

The dashboard is hosted on Render: **https://xg-model-statsbomb.onrender.com**

Features:
- Model performance metrics (AUC, Brier score, calibration)
- SHAP feature importance analysis
- Spatial xG distribution heatmaps (Our Model vs StatsBomb)
- Team xG difference rankings
- Player analysis with shot maps and cumulative xG charts

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python -m statsbomb_xg.app
```

Open http://localhost:8050

## Project Structure

```
├── statsbomb_xg/
│   ├── app.py              # Dash dashboard
│   ├── config.py           # Feature columns, paths
│   ├── data.py             # StatsBomb data loading
│   ├── features.py         # Feature engineering
│   ├── model.py            # XGBoost training with calibration
│   ├── evaluate.py         # Metrics and evaluation
│   └── main.py             # Training pipeline
├── data/                   # Cached shot data
├── models/                 # Trained model
├── output/                 # Model metrics
└── render.yaml             # Render deployment config
```

## Model Performance (Test Set: Jan-May 2016)

| Metric | Our Model | StatsBomb |
|--------|-----------|-----------|
| AUC | 0.83 | 0.84 |
| Brier Score | 0.081 | 0.080 |

Near-parity with StatsBomb's xG. The small gap is expected given we train on only half a season (~4,500 shots) while StatsBomb uses years of data across multiple leagues. Shot height data is also not available in the open dataset.

## Features Used

- **Geometry**: Distance to goal, angle to goal
- **Body part**: Header vs foot
- **Context**: Counter attack, set piece, first time, under pressure
- **GK position**: Distance from goal line, distance from centre, positioning error
- **Defenders**: Nearest defender distance, nearest blocker distance, goal visible percentage

## Attribution

Data provided by [StatsBomb](https://statsbomb.com/) under their open data license.
