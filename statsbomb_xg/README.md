# xG Model - StatsBomb Open Data

An Expected Goals (xG) model using StatsBomb open data from La Liga 2015/16.

## Overview

This project builds an xG model using XGBoost with isotonic calibration, trained on shots from the first half of La Liga 2015/16. The model predicts the probability that a shot will result in a goal based on shot location, body part, situation, goalkeeper positioning, and defensive context.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python -m statsbomb_xg.app
# Then open http://localhost:8050
```

## Project Structure

```
statsbomb_xg/
├── config.py              # Constants, feature lists
├── data.py                # Load and cache StatsBomb data
├── features.py            # Feature engineering
├── model.py               # XGBoost training with calibration
├── evaluate.py            # Temporal validation, metrics, plots
├── app.py                 # Dash visualisation dashboard
├── main.py                # Run full pipeline
└── requirements.txt
```

## Methodology

### Features

Following industry best practices (StatsBomb methodology), we use **continuous measures** rather than discrete counts:

**Shot Geometry**
- `distance_to_goal`: Euclidean distance from shot to goal centre
- `angle_to_goal`: Angle subtended by goal posts

**Goalkeeper Position**
- `gk_distance_from_goal_line`: How far GK is off their line
- `gk_distance_from_center`: Lateral displacement from goal centre
- `gk_positioning_error`: How far GK is from optimal position

**Defender Position**
- `goal_visible_pct`: Proportion of goal not blocked (0-1)
- `dist_nearest_defender`: Distance to closest outfield opponent
- `dist_nearest_blocker`: Distance to closest opponent in shooting cone

**Shot Context**
- `is_header`: Header vs foot
- `is_penalty`, `is_set_piece`, `is_counter`: Shot type
- `is_first_time`, `under_pressure`: Situational factors

### Why Continuous Features?

From [StatsBomb's research](https://statsbomb.com/articles/soccer/upgrading-expected-goals/):

> "There's a big discontinuity in xG when the goalkeeper is on the edge of the triangle... a tiny change in goalkeeper position doesn't result in such a dramatic change in real goalscoring likelihood."

Continuous measures like "proportion of goal visible" better reflect the smooth relationship between positioning and scoring probability.

### Model

- **Algorithm**: XGBoost (gradient boosted trees)
- **Calibration**: Isotonic regression via `CalibratedClassifierCV`
- **Hyperparameter tuning**: Optuna (100 trials)
- **No class weights**: Would distort probability calibration

### Validation

- **Temporal split**: Train on Aug-Dec 2015, test on Jan-May 2016
- **Metrics**: ROC AUC, Brier score, log loss

## Model Performance

| Metric | Our Model | StatsBomb |
|--------|-----------|-----------|
| AUC | 0.83 | 0.84 |
| Brier | 0.081 | 0.080 |

Near-parity with StatsBomb. The small gap is expected given we train on only half a season while StatsBomb uses years of multi-league data. Shot height is also not available in the open dataset.

## Dashboard Features

The Dash app provides:
- Model performance metrics (AUC, Brier, calibration curve)
- SHAP feature importance analysis
- Spatial xG distribution heatmaps (Our Model vs StatsBomb)
- Team xG difference rankings
- Player analysis with shot maps and cumulative xG charts

## Why No Class Weights?

Class weights artificially inflate predicted probability for the minority class (goals, ~10%). This improves recall but **destroys probability calibration**.

For an xG model, we care about calibrated probabilities, not classification. A 0.15 xG shot should score 15% of the time.

## What is Isotonic Calibration?

Isotonic regression learns a non-parametric, monotonic mapping from raw model probabilities to calibrated probabilities:

1. Sort predictions by probability
2. Apply Pool Adjacent Violators (PAV) algorithm
3. Ensure calibrated probs match observed frequencies

## Data Source

[StatsBomb Open Data](https://github.com/statsbomb/open-data) - Free football event data including freeze frames for player positions.

## License

This project uses StatsBomb open data under their non-commercial license.
