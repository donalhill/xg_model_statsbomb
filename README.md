# StatsBomb xG Analysis

Building an Expected Goals (xG) model using StatsBomb's open data from La Liga 2015/16.

## Overview

This project demonstrates:
1. **xG Model Development** - XGBoost classifier with isotonic calibration trained on ~9,000 shots
2. **Feature Engineering** - Continuous features following industry best practice: GK positioning, goal visibility, defender proximity
3. **Temporal Validation** - Train on Aug-Dec 2015, test on Jan-May 2016
4. **Player Analysis** - Focus on Antoine Griezmann's elite finishing

## Data

- **Source**: [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- **Competition**: La Liga 2015/16 (full season, 380 matches)
- **Shots**: ~9,000 shots with freeze-frame data

## Setup

```bash
# Install dependencies
pip install -r statsbomb_xg/requirements.txt
```

## Usage

### Run the full pipeline
```bash
python -m statsbomb_xg.main
```

This will:
1. Load and cache StatsBomb data
2. Engineer features (distance, angle, freeze-frame features)
3. Train XGBoost model with isotonic calibration
4. Evaluate with temporal validation
5. Generate analysis plots and metrics
6. Run Griezmann analysis

### Launch the dashboard
```bash
python -m statsbomb_xg.app
```

Open http://localhost:8050 to explore:
- Model metrics comparison
- Spatial xG distribution heatmap
- Player shot maps and cumulative xG charts

## Project Structure

```
├── statsbomb_xg/           # Main package
│   ├── config.py           # Constants, feature lists
│   ├── data.py             # StatsBomb data loading and caching
│   ├── features.py         # Feature engineering
│   ├── model.py            # XGBoost training with calibration
│   ├── evaluate.py         # Temporal validation, metrics, plots
│   ├── player_analysis.py  # Player-level analysis
│   ├── app.py              # Dash visualization app
│   ├── main.py             # Main pipeline script
│   └── README.md           # Detailed documentation
├── output/                 # Generated plots and metrics
├── models/                 # Trained models
└── data/                   # Cached data
```

## Key Findings

### Model Performance (Test Set)

| Model | ROC AUC | Brier Score | Gap |
|-------|---------|-------------|-----|
| Our XGBoost | 0.828 | 0.081 | - |
| StatsBomb | 0.835 | 0.080 | 0.007 |

Near-parity with StatsBomb (0.007 AUC gap). Remaining difference likely due to shot height data not in open dataset.

### Feature Importance

Top features for predicting goal probability:
1. GK distance to shot (13.1%)
2. Foot shot (11.1%)
3. Goal visible proportion (10.2%)
4. Angle to goal (8.8%)
5. GK distance from goal line (6.0%)

### Griezmann Analysis

Antoine Griezmann in La Liga 2015/16:
- **92 shots**, **22 goals** (23.9% conversion)
- **14.6 xG** → overperformed by **+7.4 goals**
- Elite finishing ability demonstrated

## Attribution

Data provided by [StatsBomb](https://statsbomb.com/). Used under their open data license for non-commercial research.
