# xG Model - StatsBomb Open Data

A focused Expected Goals (xG) model using StatsBomb open data from La Liga 2015/16.

## Overview

This project builds an xG model using XGBoost with isotonic calibration, trained on ~9,000 shots from La Liga 2015/16. The model predicts the probability that a shot will result in a goal based on shot location, body part, situation, goalkeeper positioning, and defensive context.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m statsbomb_xg.main

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
├── player_analysis.py     # Player-level analysis
├── app.py                 # Dash visualization dashboard
├── main.py                # Run full pipeline
├── requirements.txt
└── README.md
```

## Methodology

### Features

Following industry best practices (StatsBomb methodology), we use **continuous measures** rather than discrete counts to avoid binary artifacts:

**Shot Geometry**
- `distance_to_goal`: Euclidean distance from shot to goal center
- `angle_to_goal`: Angle subtended by goal posts (larger = more goal visible)

**Goalkeeper Position (continuous)**
- `gk_distance_from_goal_line`: How far GK is off their line
- `gk_distance_from_center`: Lateral displacement from goal center
- `gk_distance_to_shot`: Distance from shooter to GK
- `gk_positioning_error`: How far GK is from optimal position on shot line

**Defender Position (continuous)**
- `goal_visible_pct`: Proportion of goal not blocked (ray-casting, 0-1)
- `dist_nearest_defender`: Distance to closest outfield opponent
- `dist_nearest_blocker`: Distance to closest opponent in shooting cone

**Shot Context**
- `is_header`, `is_foot`: Body part
- `is_open_play`, `is_penalty`, `is_set_piece`, `is_counter`: Shot type
- `is_first_time`, `under_pressure`: Situational factors

### Why Continuous Features?

From [StatsBomb's research](https://statsbomb.com/articles/soccer/upgrading-expected-goals/):

> "There's a big discontinuity in xG when the goalkeeper is on the edge of the triangle... a tiny change in goalkeeper position doesn't result in such a dramatic change in real goalscoring likelihood."

Discrete features (counts, binary flags) create artificial step-changes in xG. Continuous measures like "proportion of goal visible" and "distance to nearest blocker" better reflect the smooth relationship between positioning and scoring probability.

### Model

- **Algorithm**: XGBoost (gradient boosted trees)
- **Calibration**: Isotonic regression via `CalibratedClassifierCV`
- **No class weights**: Class weights would distort probability calibration

### Validation

- **Temporal split**: Train on Aug-Dec 2015, test on Jan-May 2016
- **Metrics**: ROC AUC, Brier score (proper scoring rule), log loss

## Key Findings

### Model Performance

| Metric    | Our Model | StatsBomb | Gap |
|-----------|-----------|-----------|-----|
| ROC AUC   | 0.828     | 0.835     | 0.007 |
| Brier     | 0.081     | 0.080     | 0.001 |

The remaining ~0.007 AUC gap is likely due to:
- Shot height (low, medium, high) - not in open data
- Proprietary model tuning

### Feature Importance

Top predictive features:
1. `gk_distance_to_shot` (13.1%) - GK proximity to shooter
2. `is_foot` (11.1%) - Foot vs other body part
3. `goal_visible_pct` (10.2%) - Proportion of goal unblocked
4. `angle_to_goal` (8.8%) - Shot angle geometry
5. `gk_distance_from_goal_line` (6.0%) - GK off their line

### Griezmann Analysis

Antoine Griezmann in La Liga 2015/16:
- **92 shots**, **22 goals** (23.9% conversion)
- **14.6 xG** → overperformed by **+7.4 goals**
- Elite finishing ability demonstrated

## Why No Class Weights?

Class weights artificially inflate predicted probability for the minority class (goals, ~10%). This improves recall but **destroys probability calibration**.

For an xG model, we care about calibrated probabilities, not classification. A 0.15 xG shot should score 15% of the time, not be classified as "goal" or "no goal".

## What is Isotonic Calibration?

Isotonic regression learns a non-parametric, monotonic mapping from raw model probabilities to calibrated probabilities. It fixes any monotonic miscalibration by:

1. Sorting predictions by probability
2. Applying Pool Adjacent Violators (PAV) algorithm
3. Ensuring calibrated probs match observed frequencies

## Dashboard

The Dash app provides:
- Model metrics comparison table
- Spatial xG heatmap
- Player selector dropdown
- Shot map for selected player
- Cumulative xG vs goals chart

## Data Source

[StatsBomb Open Data](https://github.com/statsbomb/open-data) - Free football event data including freeze frames for player positions.

## License

This project uses StatsBomb open data under their non-commercial license.
