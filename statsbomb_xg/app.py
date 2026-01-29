"""
Dash application for xG model visualisation.
"""
import json
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mplsoccer import VerticalPitch
from scipy.ndimage import gaussian_filter
from scipy.stats import theilslopes
from PIL import Image
import urllib.request

from sklearn.metrics import roc_curve, roc_auc_score
import shap

from .config import OUTPUT_DIR, MODEL_DIR, FEATURE_COLUMNS
from .data import load_shots
from .features import engineer_features
from .model import load_model, predict_xg


# =============================================================================
# Design System - Color Palette (Navy/Blue Theme)
# =============================================================================
COLORS = {
    'bg_primary': '#F8FAFC',
    'bg_card': '#FFFFFF',
    'border': '#E2E8F0',
    'shadow': 'rgba(0, 0, 0, 0.05)',
    'accent_primary': '#1E3A5F',      # Navy
    'accent_secondary': '#2563EB',    # Bright blue
    'accent_light': '#3B82F6',        # Light blue
    'text_primary': '#1E293B',
    'text_secondary': '#64748B',
    'success': '#10B981',
    'danger': '#EF4444',
    'chart_gray': '#94A3B8',
    'grid': '#E2E8F0',
}

# Clean feature labels for display
FEATURE_LABELS = {
    'distance_to_goal': 'Distance To Goal (m)',
    'angle_to_goal': 'Visible Goal Angle (°)',
    'is_header': 'Is Header',
    'gk_distance_from_goal_line': 'GK Distance From Goal (m)',
    'goal_visible_pct': 'Goal Visibility (0-1)',
}

# Custom CSS for the dashboard
CUSTOM_CSS = """
/* CSS Custom Properties */
:root {
    --bg-primary: #F8FAFC;
    --bg-card: #FFFFFF;
    --border-color: #E2E8F0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
    --shadow-hover: 0 10px 25px rgba(30,58,95,0.15);
    --accent-primary: #1E3A5F;
    --accent-secondary: #2563EB;
    --accent-light: #3B82F6;
    --text-primary: #1E293B;
    --text-secondary: #64748B;
    --radius-lg: 0.75rem;
}

/* Base styles */
body {
    background-color: var(--bg-primary) !important;
    font-family: Inter, system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hero section */
.hero-section {
    background: linear-gradient(135deg, rgba(30,58,95,0.08) 0%, rgba(37,99,235,0.08) 100%);
    border-bottom: 3px solid var(--accent-primary);
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.hero-badge {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
    margin: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* Card styling */
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    overflow: hidden;
}

.chart-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.chart-card .card-header {
    background: var(--bg-card);
    border-bottom: 2px solid var(--accent-primary);
    padding: 1rem 1.25rem;
}

.chart-card .card-header h4 {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 1.1rem;
    margin: 0;
}

.chart-card .card-body {
    padding: 1.25rem;
}

/* Stat cards */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.stat-card:hover {
    border-color: var(--accent-primary);
    box-shadow: 0 4px 12px rgba(249,115,22,0.1);
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    line-height: 1.2;
}

.stat-value.gold {
    color: var(--accent-secondary);
}

.stat-value.success {
    color: #10B981;
}

.stat-value.danger {
    color: #EF4444;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    letter-spacing: 0.02em;
    margin-top: 0.25rem;
}

/* Feature bars */
.feature-bar-container {
    margin-bottom: 0.5rem;
}

.feature-bar-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
}

.feature-bar {
    height: 8px;
    background: var(--border-color);
    border-radius: 4px;
    overflow: hidden;
}

.feature-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* Section headers */
.section-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent-primary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent-primary);
    display: inline-block;
}

/* Dropdown styling */
.Select-control {
    border-color: var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
}

.Select-control:hover {
    border-color: var(--accent-primary) !important;
}

.Select.is-focused > .Select-control {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.1) !important;
}

/* Footer */
.footer {
    border-top: 1px solid var(--border-color);
    padding: 1.5rem 0;
    margin-top: 2rem;
    text-align: center;
}

.footer a {
    color: var(--accent-primary);
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.chart-card {
    animation: fadeIn 0.3s ease-out;
}

/* Model summary list styling */
.model-summary-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.model-summary-list li {
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border-color);
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.model-summary-list li:last-child {
    border-bottom: none;
}

/* Player stats row */
.player-stats-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    justify-content: center;
}
"""

# Initialize app with minimal Bootstrap theme + custom CSS
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>La Liga 2015/16 - xG Model Dashboard</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>''' + CUSTOM_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load data and model
print("Loading data for Dash app...")
PENALTY_XG = 0.76  # Fixed xG for penalties based on historical conversion rate

try:
    df = load_shots()
    df = engineer_features(df)
    model = load_model()
    df['our_xg'] = predict_xg(model, df[FEATURE_COLUMNS])
    # Override penalty xG with fixed value
    df.loc[df['is_penalty'] == 1, 'our_xg'] = PENALTY_XG
    DATA_LOADED = True
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()
    model = None
    DATA_LOADED = False

# Load metrics if available
metrics_path = OUTPUT_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {}


# SHAP values computed lazily on first request
_shap_cache = {'values': None, 'data': None, 'computed': False}


def _compute_shap_values():
    """Load pre-computed SHAP values from disk, or compute if not available."""
    if _shap_cache['computed']:
        return _shap_cache['values'], _shap_cache['data']

    # Try to load pre-computed SHAP values
    shap_path = MODEL_DIR / 'shap_values.npz'
    if shap_path.exists():
        try:
            print("Loading pre-computed SHAP values...")
            data = np.load(shap_path, allow_pickle=True)
            _shap_cache['values'] = data['shap_values']
            _shap_cache['data'] = data['feature_data']
            _shap_cache['computed'] = True
            print("SHAP values loaded.")
            return _shap_cache['values'], _shap_cache['data']
        except Exception as e:
            print(f"Could not load SHAP values: {e}")

    # Fall back to computing if pre-computed not available
    if not DATA_LOADED or model is None:
        _shap_cache['computed'] = True
        return None, None

    try:
        print("Computing SHAP values...")
        test_df = df[df['match_date'] >= '2016-01-01']
        X_test = test_df[FEATURE_COLUMNS]

        base_model = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_model)
        _shap_cache['values'] = explainer.shap_values(np.asarray(X_test))
        _shap_cache['data'] = X_test
        print("SHAP values computed.")
    except Exception as e:
        print(f"Could not compute SHAP values: {e}")
        _shap_cache['values'] = None
        _shap_cache['data'] = None

    _shap_cache['computed'] = True
    return _shap_cache['values'], _shap_cache['data']


def get_chart_layout(title='', height=350, show_legend=True, legend_pos='top'):
    """Get consistent chart layout for all Plotly charts."""
    legend_config = dict(
        x=0.02 if legend_pos == 'top' else 0.5,
        y=0.98 if legend_pos == 'top' else 0.1,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=COLORS['border'],
        borderwidth=1,
        font=dict(size=11)
    )

    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=COLORS['text_primary'], family='Inter, system-ui'),
            x=0.5,
            xanchor='center'
        ),
        height=height,
        plot_bgcolor=COLORS['bg_card'],
        paper_bgcolor=COLORS['bg_card'],
        font=dict(
            family='Inter, system-ui, sans-serif',
            color=COLORS['text_primary'],
            size=12
        ),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            title_font=dict(color=COLORS['text_secondary'], size=11)
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            title_font=dict(color=COLORS['text_secondary'], size=11)
        ),
        legend=legend_config if show_legend else dict(visible=False),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(
            bgcolor=COLORS['bg_card'],
            bordercolor=COLORS['border'],
            font=dict(color=COLORS['text_primary'])
        )
    )


def create_shap_summary_image():
    """Create SHAP summary (beeswarm) plot showing feature effects on xG."""
    shap_values, shap_data = _compute_shap_values()
    if shap_values is None or shap_data is None:
        return None

    # Clear any existing figures to prevent contamination
    plt.close('all')

    # Create figure with updated styling
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use consistent feature labels
    feature_names = [FEATURE_LABELS.get(f, f.replace('_', ' ').title()) for f in FEATURE_COLUMNS]

    # Create SHAP summary plot with diverging colormap
    shap.summary_plot(
        shap_values,
        shap_data,
        feature_names=feature_names,
        show=False,
        plot_size=None,
        color_bar_label='Feature Value',
        cmap='RdBu_r'  # Diverging colormap (reversed)
    )

    plt.title('', fontsize=12)  # Remove title - card header will have it
    plt.xlabel('Impact on xG (SHAP value)', fontsize=10, color=COLORS['text_secondary'])
    ax.tick_params(colors=COLORS['text_secondary'])
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor=COLORS['bg_card'])
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close('all')

    return f'data:image/png;base64,{img_base64}'


def create_correlation_matrix_figure():
    """Create feature correlation matrix heatmap."""
    if not DATA_LOADED:
        return go.Figure()

    # Use training set
    train_df = df[df['match_date'] < '2016-01-01']

    # Get feature correlations
    corr = train_df[FEATURE_COLUMNS].corr()

    # Use nice labels
    labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Corr', side='right'),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        )
    ))

    layout = get_chart_layout('', height=350)
    layout['xaxis']['tickangle'] = 45
    layout['xaxis']['tickfont'] = dict(size=8)
    layout['yaxis']['tickfont'] = dict(size=8)
    layout['yaxis']['scaleanchor'] = 'x'  # Make it square
    layout['margin'] = dict(l=120, r=50, t=20, b=100)
    fig.update_layout(**layout)

    return fig


def create_feature_distribution_figure(feature='distance_to_goal'):
    """Create histogram showing feature distribution for goals vs non-goals (training set).

    Uses Plotly (not matplotlib) so won't interfere with SHAP/mplsoccer.
    """
    if not DATA_LOADED:
        return go.Figure()

    # Use training set, exclude penalties
    train_df = df[(df['match_date'] < '2016-01-01') & (df['is_penalty'] == 0)]

    if feature not in train_df.columns:
        return go.Figure()

    feature_label = FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())
    fig = go.Figure()

    # Handle binary features (like is_header) with grouped bar chart
    if feature == 'is_header':
        # Calculate percentages for each class
        goal_pct = train_df[train_df['is_goal'] == 1][feature].value_counts(normalize=True) * 100
        miss_pct = train_df[train_df['is_goal'] == 0][feature].value_counts(normalize=True) * 100

        categories = ['Foot/Body', 'Header']

        fig.add_trace(go.Bar(
            x=categories,
            y=[miss_pct.get(0, 0), miss_pct.get(1, 0)],
            name='Miss/Saved',
            marker_color='rgba(100, 116, 139, 0.7)',
            marker_line=dict(color='#475569', width=1),
        ))

        fig.add_trace(go.Bar(
            x=categories,
            y=[goal_pct.get(0, 0), goal_pct.get(1, 0)],
            name='Goal',
            marker_color=COLORS['accent_secondary'],
            marker_line=dict(color=COLORS['accent_primary'], width=1),
        ))

        layout = get_chart_layout(f'{feature_label} (Training Set)', height=300)
        layout['xaxis']['title'] = ''
        layout['yaxis']['title'] = 'Percentage (%)'
        layout['barmode'] = 'group'
        layout['margin'] = dict(l=50, r=30, t=40, b=70)
        layout['legend'] = dict(x=0.5, y=-0.25, xanchor='center', orientation='h',
                               bgcolor='rgba(255,255,255,0.9)',
                               bordercolor=COLORS['border'], borderwidth=1)
        fig.update_layout(**layout)
        return fig

    # For continuous features
    goals = train_df[train_df['is_goal'] == 1][feature]
    misses = train_df[train_df['is_goal'] == 0][feature]

    # Clip extreme values for GK distance (>15m is very rare)
    if feature == 'gk_distance_from_goal_line':
        goals = goals.clip(upper=15)
        misses = misses.clip(upper=15)
        all_vals = train_df[feature].clip(upper=15)
    else:
        all_vals = train_df[feature]

    # Calculate shared bin edges for both classes (ensures equal bins)
    min_val, max_val = all_vals.min(), all_vals.max()
    n_bins = 25
    bin_size = (max_val - min_val) / n_bins

    # Histogram for misses - use explicit xbins for consistent binning
    fig.add_trace(go.Histogram(
        x=misses,
        name='Miss/Saved',
        marker_color='rgba(100, 116, 139, 0.6)',
        marker_line=dict(color='#475569', width=1),
        xbins=dict(start=min_val, end=max_val, size=bin_size),
        histnorm='percent'
    ))

    # Histogram for goals - same xbins
    fig.add_trace(go.Histogram(
        x=goals,
        name='Goal',
        marker_color=COLORS['accent_secondary'],
        marker_line=dict(color=COLORS['accent_primary'], width=1),
        opacity=0.65,
        xbins=dict(start=min_val, end=max_val, size=bin_size),
        histnorm='percent'
    ))

    layout = get_chart_layout(f'{feature_label} (Training Set)', height=300)
    layout['xaxis']['title'] = feature_label
    layout['yaxis']['title'] = 'Percentage (%)'
    layout['barmode'] = 'overlay'
    layout['margin'] = dict(l=50, r=30, t=40, b=70)
    layout['legend'] = dict(x=0.5, y=-0.25, xanchor='center', orientation='h',
                           bgcolor='rgba(255,255,255,0.9)',
                           bordercolor=COLORS['border'], borderwidth=1)
    fig.update_layout(**layout)

    return fig


def create_calibration_figure():
    """Create calibration curve: predicted xG vs actual goal rate per bin with error bars."""
    if not DATA_LOADED:
        return go.Figure()

    # Use only test set, exclude penalties
    test_df = df[(df['match_date'] >= '2016-01-01') & (df['is_penalty'] == 0)]

    fig = go.Figure()

    # Diagonal line (perfect calibration)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color=COLORS['grid'], dash='dash', width=2),
        name='Perfect',
        showlegend=False
    ))

    # Uniform 0.1 bins
    bin_edges = np.arange(0, 0.8, 0.1)

    chi2_results = {}

    # Updated colors: blue for our model, gray for StatsBomb
    for xg_col, name, color in [
        ('our_xg', 'Our Model', COLORS['accent_secondary']),
        ('statsbomb_xg', 'StatsBomb', COLORS['chart_gray'])
    ]:
        bin_centers = []
        conversion_rates = []
        y_errors = []

        for i in range(len(bin_edges) - 1):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            mask = (test_df[xg_col] >= bin_edges[i]) & (test_df[xg_col] < bin_edges[i + 1])
            if i == len(bin_edges) - 2:
                mask = (test_df[xg_col] >= bin_edges[i]) & (test_df[xg_col] <= bin_edges[i + 1])
            bucket = test_df[mask]
            if len(bucket) >= 10:
                n = len(bucket)
                p = bucket['is_goal'].mean()
                se = np.sqrt(p * (1 - p) / n) if p > 0 and p < 1 else 0.01

                bin_centers.append(bin_center)
                conversion_rates.append(p)
                y_errors.append(se)

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=conversion_rates,
            mode='markers',
            marker=dict(size=12, color=color, line=dict(color='white', width=1)),
            error_y=dict(type='data', array=y_errors, color=color, thickness=2),
            name=name
        ))

    layout = get_chart_layout('Calibration (excl. penalties)', height=350)
    layout['xaxis']['title'] = 'Predicted xG (bin)'
    layout['yaxis']['title'] = 'Actual Conversion Rate'
    layout['xaxis']['range'] = [0, 1]
    layout['yaxis']['range'] = [0, 1]
    fig.update_layout(**layout)

    return fig


def create_roc_curve_figure():
    """Create ROC curve comparison: Our model vs StatsBomb (TEST SET ONLY)."""
    if not DATA_LOADED:
        return go.Figure()

    # Use only test set (matches from 2016 onwards)
    test_df = df[df['match_date'] >= '2016-01-01']
    y_true = test_df['is_goal']

    # Our model ROC
    fpr_ours, tpr_ours, _ = roc_curve(y_true, test_df['our_xg'])
    auc_ours = roc_auc_score(y_true, test_df['our_xg'])

    # StatsBomb ROC
    fpr_sb, tpr_sb, _ = roc_curve(y_true, test_df['statsbomb_xg'])
    auc_sb = roc_auc_score(y_true, test_df['statsbomb_xg'])

    fig = go.Figure()

    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color=COLORS['grid'], dash='dash', width=2),
        name='Random',
        showlegend=False
    ))

    # Our model - blue
    fig.add_trace(go.Scatter(
        x=fpr_ours, y=tpr_ours,
        mode='lines',
        line=dict(color=COLORS['accent_secondary'], width=3),
        name=f'Our Model (AUC={auc_ours:.3f})',
        fill='tozeroy',
        fillcolor='rgba(37,99,235,0.1)'
    ))

    # StatsBomb - gray
    fig.add_trace(go.Scatter(
        x=fpr_sb, y=tpr_sb,
        mode='lines',
        line=dict(color=COLORS['chart_gray'], width=2),
        name=f'StatsBomb (AUC={auc_sb:.3f})'
    ))

    layout = get_chart_layout('ROC Curve (Test Set)', height=350, legend_pos='bottom')
    layout['xaxis']['title'] = 'False Positive Rate'
    layout['yaxis']['title'] = 'True Positive Rate'
    layout['xaxis']['range'] = [0, 1]
    layout['yaxis']['range'] = [0, 1]
    fig.update_layout(**layout)

    return fig


def create_xg_histogram_figure(model='our_xg'):
    """Create histogram of xG values for goals vs non-goals (TEST SET ONLY, excludes penalties)."""
    if not DATA_LOADED:
        return go.Figure()

    # Use only test set, exclude penalties
    test_df = df[(df['match_date'] >= '2016-01-01') & (df['is_penalty'] == 0)]

    xg_col = model if model in ['our_xg', 'statsbomb_xg'] else 'our_xg'
    model_name = 'Our Model' if xg_col == 'our_xg' else 'StatsBomb'

    goals = test_df[test_df['is_goal'] == 1][xg_col]
    misses = test_df[test_df['is_goal'] == 0][xg_col]

    fig = go.Figure()

    # Histogram for misses - darker gray fill with outline
    fig.add_trace(go.Histogram(
        x=misses,
        name='Miss/Saved',
        marker_color='rgba(100, 116, 139, 0.6)',
        marker_line=dict(color='#475569', width=1.5),
        xbins=dict(start=0, end=1, size=0.1),
        histnorm='percent'
    ))

    # Histogram for goals
    fig.add_trace(go.Histogram(
        x=goals,
        name='Goal',
        marker_color=COLORS['accent_secondary'],
        marker_line=dict(color=COLORS['accent_primary'], width=1),
        opacity=0.65,
        xbins=dict(start=0, end=1, size=0.1),
        histnorm='percent'
    ))

    layout = get_chart_layout(f'xG Distribution: {model_name}', height=350)
    layout['xaxis']['title'] = 'xG'
    layout['yaxis']['title'] = 'Percentage (%)'
    layout['xaxis']['range'] = [0, 1]
    layout['barmode'] = 'overlay'
    layout['legend'] = dict(x=0.7, y=0.95, bgcolor='rgba(255,255,255,0.9)',
                           bordercolor=COLORS['border'], borderwidth=1)
    fig.update_layout(**layout)

    return fig


def _compute_smoothed_xg_grid(test_df, xg_col):
    """Compute smoothed xG grid - same logic as create_spatial_xg_image."""
    x_bins = np.linspace(60, 120, 21)
    y_bins = np.linspace(0, 80, 17)

    x_idx = np.digitize(test_df['x'], x_bins) - 1
    y_idx = np.digitize(test_df['y'], y_bins) - 1
    x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
    y_idx = np.clip(y_idx, 0, len(y_bins) - 2)

    xg_sum = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    xg_count = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

    for xi, yi, xg in zip(x_idx, y_idx, test_df[xg_col]):
        xg_sum[yi, xi] += xg
        xg_count[yi, xi] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        xg_mean = np.where(xg_count > 0, xg_sum / xg_count, np.nan)

    # Same smoothing as the heatmaps
    xg_filled = np.nan_to_num(xg_mean, nan=0.0)
    xg_smoothed = gaussian_filter(xg_filled, sigma=1.0)
    count_smoothed = gaussian_filter((xg_count > 0).astype(float), sigma=1.0)
    xg_final = np.where(count_smoothed > 0.1, xg_smoothed, np.nan)

    return xg_final, x_bins, y_bins


def create_spatial_diff_image():
    """Create spatial xG difference heatmap using same smoothed values as individual heatmaps."""
    if not DATA_LOADED:
        return None

    # Use only test set
    test_df = df[df['match_date'] >= '2016-01-01']

    # Get smoothed grids (same as what's displayed in the heatmaps)
    our_grid, x_bins, y_bins = _compute_smoothed_xg_grid(test_df, 'our_xg')
    sb_grid, _, _ = _compute_smoothed_xg_grid(test_df, 'statsbomb_xg')

    # Difference of smoothed values
    diff = our_grid - sb_grid

    # Create pitch with clean styling
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        half=True,
        pitch_color=COLORS['bg_card'],
        line_color=COLORS['text_secondary'],
        line_zorder=2,
        linewidth=1
    )

    fig, ax = pitch.draw(figsize=(5, 5))

    X, Y = np.meshgrid(x_bins, y_bins)

    # Fixed scale centered at 0, ±0.05 range
    vmax = 0.05

    pcm = ax.pcolormesh(
        Y, X,
        diff,
        cmap='RdBu_r',  # Red = we're higher, Blue = StatsBomb higher
        shading='flat',
        vmin=-vmax,
        vmax=vmax,
        zorder=1
    )

    # Colorbar with updated styling
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Our − SB (xG)', fontsize=10, color=COLORS['text_secondary'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['text_secondary'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text_secondary'])

    ax.set_title('Difference', fontsize=12, fontweight='600', color=COLORS['text_primary'], pad=10)

    fig.patch.set_facecolor(COLORS['bg_card'])

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor=COLORS['bg_card'])
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return f'data:image/png;base64,{img_base64}'


def create_spatial_xg_image(model='our_xg'):
    """
    Create spatial xG heatmap using mplsoccer.

    Manually bins shots into grid cells and calculates mean xG per cell.
    Returns base64 encoded PNG image for embedding in Dash.
    """
    if not DATA_LOADED:
        return None

    xg_col = model if model in ['our_xg', 'statsbomb_xg'] else 'our_xg'
    model_name = 'Our Model' if xg_col == 'our_xg' else 'StatsBomb'

    # Use only test set
    test_df = df[df['match_date'] >= '2016-01-01']

    # Use perceptually uniform colormap - set pitch to match viridis low end
    cmap_heat = 'viridis'
    pitch_color = '#440154'  # viridis lowest color (dark purple)

    pitch = VerticalPitch(
        pitch_type='statsbomb',
        half=True,
        pitch_color=pitch_color,
        line_color='#CCCCCC',  # Light gray lines visible on dark background
        line_zorder=2,
        linewidth=1
    )

    fig, ax = pitch.draw(figsize=(5, 5))

    # Define grid for binning
    x_bins = np.linspace(60, 120, 21)
    y_bins = np.linspace(0, 80, 17)

    # Bin the shots manually
    x_idx = np.digitize(test_df['x'], x_bins) - 1
    y_idx = np.digitize(test_df['y'], y_bins) - 1

    # Clip to valid range
    x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
    y_idx = np.clip(y_idx, 0, len(y_bins) - 2)

    # Create grid for mean xG
    xg_sum = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    xg_count = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

    for i, (xi, yi, xg) in enumerate(zip(x_idx, y_idx, test_df[xg_col])):
        xg_sum[yi, xi] += xg
        xg_count[yi, xi] += 1

    # Calculate mean, set empty cells to NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        xg_mean = np.where(xg_count > 0, xg_sum / xg_count, np.nan)

    # Apply Gaussian smoothing
    xg_filled = np.nan_to_num(xg_mean, nan=0.0)
    xg_smoothed = gaussian_filter(xg_filled, sigma=1.0)

    count_smoothed = gaussian_filter((xg_count > 0).astype(float), sigma=1.0)
    xg_final = np.where(count_smoothed > 0.1, xg_smoothed, np.nan)

    X, Y = np.meshgrid(x_bins, y_bins)

    pcm = ax.pcolormesh(
        Y, X,
        xg_final,
        cmap=cmap_heat,
        shading='flat',
        vmin=0.0,
        vmax=0.4,
        zorder=1
    )

    # Colorbar with updated styling
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Mean xG', fontsize=10, color=COLORS['text_primary'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['text_primary'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text_primary'])

    # Title
    n_shots = int(xg_count.sum())
    ax.set_title(
        f'{model_name}\n({n_shots:,} shots)',
        fontsize=12, fontweight='600', color=COLORS['text_primary'], pad=10
    )

    fig.patch.set_facecolor(COLORS['bg_card'])  # White background for figure

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor=COLORS['bg_card'])
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return f'data:image/png;base64,{img_base64}'


def get_player_stats(player_name):
    """Get stats for a player (TEST SET ONLY)."""
    if not DATA_LOADED or player_name is None:
        return None

    test_df = df[df['match_date'] >= '2016-01-01']
    player_df = test_df[test_df['player'].str.contains(player_name, case=False, na=False)]
    if len(player_df) == 0:
        return None

    n_games = player_df['match_id'].nunique()
    n_shots = len(player_df)
    n_goals = player_df['is_goal'].sum()
    our_xg = player_df['our_xg'].sum()
    sb_xg = player_df['statsbomb_xg'].sum()

    return {
        'games': n_games,
        'shots': n_shots,
        'goals': n_goals,
        'our_xg': our_xg,
        'sb_xg': sb_xg,
        'conversion': n_goals / n_shots if n_shots > 0 else 0,
        'goals_per_game': n_goals / n_games if n_games > 0 else 0,
        'xg_per_game': our_xg / n_games if n_games > 0 else 0,
        'shots_per_game': n_shots / n_games if n_games > 0 else 0,
        'xg_diff': n_goals - our_xg,
    }


def create_shot_map_image(player_name):
    """Create shot map for selected player using mplsoccer (TEST SET ONLY)."""
    if not DATA_LOADED or player_name is None:
        return None

    test_df = df[df['match_date'] >= '2016-01-01']
    player_df = test_df[test_df['player'].str.contains(player_name, case=False, na=False)]

    if len(player_df) == 0:
        return None

    # Use perceptually uniform colormap
    cmap_shot = 'viridis'
    pitch_color = COLORS['bg_card']

    pitch = VerticalPitch(
        pitch_type='statsbomb',
        half=True,
        pitch_color=pitch_color,
        line_color=COLORS['text_secondary'],
        line_zorder=2,
        linewidth=1.5
    )

    fig, ax = pitch.draw(figsize=(8, 8))

    # Separate goals and misses
    goals = player_df[player_df['is_goal'] == 1]
    misses = player_df[player_df['is_goal'] == 0]

    # Plot misses (circles) - colored by xG with subtle edge
    if len(misses) > 0:
        pitch.scatter(
            misses['x'], misses['y'],
            c=misses['our_xg'],
            cmap=cmap_shot,
            s=120,
            edgecolors=COLORS['text_secondary'],
            linewidth=0.5,
            ax=ax,
            zorder=3,
            vmin=0, vmax=0.5,
            alpha=0.85
        )

    # Plot goals (stars) - colored by xG with dark edge
    if len(goals) > 0:
        pitch.scatter(
            goals['x'], goals['y'],
            c=goals['our_xg'],
            cmap=cmap_shot,
            s=400,
            marker='*',
            edgecolors=COLORS['text_primary'],
            linewidth=1.5,
            ax=ax,
            zorder=4,
            vmin=0, vmax=0.5
        )

    ax.set_title(
        f"{player_name}",
        fontsize=14, fontweight='600', color=COLORS['text_primary'], pad=10
    )

    # Legend with updated styling (viridis colors)
    ax.scatter([], [], c='#31688e', s=80, marker='o',
               edgecolors=COLORS['text_secondary'], label='Miss/Saved')
    ax.scatter([], [], c='#fde725', s=150, marker='*',
               edgecolors=COLORS['text_primary'], linewidth=1.5, label='Goal')
    ax.legend(loc='center left', bbox_to_anchor=(0.08, 0.18), facecolor=pitch_color,
              edgecolor=COLORS['border'], labelcolor=COLORS['text_primary'], fontsize=10)

    fig.patch.set_facecolor(pitch_color)

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor=pitch_color)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return f'data:image/png;base64,{img_base64}'


def get_team_stats_df():
    """Calculate xG for and against per team."""
    if not DATA_LOADED:
        return None

    # Use test set
    test_df = df[df['match_date'] >= '2016-01-01']

    # Get unique matches and calculate xG for/against per team
    team_stats = {}

    for match_id in test_df['match_id'].unique():
        match_df = test_df[test_df['match_id'] == match_id]

        # Get teams in this match
        teams = match_df['team'].unique()
        if len(teams) != 2:
            continue

        for team in teams:
            if team not in team_stats:
                team_stats[team] = {'xg_for': 0, 'xg_against': 0, 'goals_for': 0, 'goals_against': 0}

            # xG for this team
            team_shots = match_df[match_df['team'] == team]
            opp_shots = match_df[match_df['team'] != team]

            team_stats[team]['xg_for'] += team_shots['our_xg'].sum()
            team_stats[team]['xg_against'] += opp_shots['our_xg'].sum()
            team_stats[team]['goals_for'] += team_shots['is_goal'].sum()
            team_stats[team]['goals_against'] += opp_shots['is_goal'].sum()

    return team_stats


def _get_team_stats(xg_column='our_xg'):
    """Calculate all team stats from test set (cached computation)."""
    if not DATA_LOADED:
        return {}

    test_df = df[df['match_date'] >= '2016-01-01']
    team_stats = {}

    for match_id in test_df['match_id'].unique():
        match_df = test_df[test_df['match_id'] == match_id]
        teams = match_df['team'].unique()
        if len(teams) != 2:
            continue

        for team in teams:
            if team not in team_stats:
                team_stats[team] = {
                    'goals_for': 0, 'goals_against': 0,
                    'xg_for': 0, 'xg_against': 0, 'n_games': 0
                }

            team_shots = match_df[match_df['team'] == team]
            opp_shots = match_df[match_df['team'] != team]

            team_stats[team]['goals_for'] += team_shots['is_goal'].sum()
            team_stats[team]['goals_against'] += opp_shots['is_goal'].sum()
            team_stats[team]['xg_for'] += team_shots[xg_column].sum()
            team_stats[team]['xg_against'] += opp_shots[xg_column].sum()
            team_stats[team]['n_games'] += 1

    return team_stats


def create_team_bar_chart(metric='xg_diff'):
    """Create ranked bar chart for selected team metric."""
    team_stats = _get_team_stats('our_xg')
    if not team_stats:
        return go.Figure()

    # Calculate values based on metric
    if metric == 'goal_diff':
        team_values = {
            team: (stats['goals_for'] - stats['goals_against']) / stats['n_games']
            for team, stats in team_stats.items()
        }
        xlabel = 'Goal Diff per 90'
    elif metric == 'g_minus_xg':
        team_values = {
            team: (stats['goals_for'] - stats['xg_for']) / stats['n_games']
            for team, stats in team_stats.items()
        }
        xlabel = 'Goals − xG per 90'
    elif metric == 'xga_minus_ga':
        team_values = {
            team: (stats['xg_against'] - stats['goals_against']) / stats['n_games']
            for team, stats in team_stats.items()
        }
        xlabel = 'xGA − Goals per 90'
    else:  # xg_diff
        team_values = {
            team: (stats['xg_for'] - stats['xg_against']) / stats['n_games']
            for team, stats in team_stats.items()
        }
        xlabel = 'xG Diff per 90'

    # Sort (best at top)
    sorted_teams = sorted(team_values.items(), key=lambda x: x[1], reverse=True)
    teams = [t[0] for t in sorted_teams]
    values = [t[1] for t in sorted_teams]

    # Create colors
    max_abs = max(abs(min(values)), abs(max(values))) if values else 1
    colors = []
    for val in values:
        norm = val / max_abs if max_abs > 0 else 0
        if norm >= 0:
            intensity = norm
            colors.append(f'rgba(37, 99, 235, {0.4 + 0.6 * intensity})')
        else:
            intensity = -norm
            colors.append(f'rgba(239, 68, 68, {0.4 + 0.6 * intensity})')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=teams,
        x=values,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f'{v:+.2f}' for v in values],
        textposition='outside',
        textfont=dict(size=9, color=COLORS['text_secondary']),
        hovertemplate='<b>%{y}</b><br>' + xlabel + ': %{x:+.2f}<extra></extra>'
    ))

    fig.add_vline(x=0, line=dict(color=COLORS['text_secondary'], width=1))

    min_val = min(values) if values else 0
    max_val = max(values) if values else 0
    x_padding = 0.25
    x_range = [min_val - x_padding, max_val + x_padding]

    layout = get_chart_layout('', height=550)
    layout['xaxis']['title'] = xlabel
    layout['xaxis']['range'] = x_range
    layout['yaxis']['title'] = ''
    layout['yaxis']['autorange'] = 'reversed'
    layout['margin'] = dict(l=120, r=50, t=20, b=40)

    fig.update_layout(**layout)

    return fig


def create_team_scatter_chart(metric='xg'):
    """Create scatter plot of xG or Goals per 90 vs conceded."""
    team_stats = _get_team_stats('our_xg')
    if not team_stats:
        return go.Figure()

    teams = list(team_stats.keys())

    # Calculate per 90 values based on metric
    if metric == 'goals':
        x_vals = np.array([team_stats[t]['goals_for'] / team_stats[t]['n_games'] for t in teams])
        y_vals = np.array([team_stats[t]['goals_against'] / team_stats[t]['n_games'] for t in teams])
        x_label = 'Goals per 90'
        y_label = 'Conceded per 90'
        title = 'Goals Scored vs Conceded (per 90)'
        hover_x = 'Goals/90'
        hover_y = 'Conceded/90'
    else:
        x_vals = np.array([team_stats[t]['xg_for'] / team_stats[t]['n_games'] for t in teams])
        y_vals = np.array([team_stats[t]['xg_against'] / team_stats[t]['n_games'] for t in teams])
        x_label = 'xG per 90'
        y_label = 'xGA per 90'
        title = 'xG Created vs Conceded (per 90)'
        hover_x = 'xG/90'
        hover_y = 'xGA/90'

    fig = go.Figure()

    # Calculate axis ranges first (needed for regression line)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    # Robust regression (Theil-Sen) - less sensitive to outliers like Barca
    slope, intercept, _, _ = theilslopes(y_vals, x_vals)
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 100)
    y_line = slope * x_line + intercept

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='rgba(37, 99, 235, 0.6)', width=2),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        text=teams,
        textposition='top center',
        textfont=dict(size=10, color=COLORS['text_secondary']),
        marker=dict(
            size=12,
            color=COLORS['accent_primary'],
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{text}</b><br>' + hover_x + ': %{x:.2f}<br>' + hover_y + ': %{y:.2f}<extra></extra>'
    ))

    layout = get_chart_layout(title, height=550)
    layout['xaxis']['title'] = x_label
    layout['yaxis']['title'] = y_label
    layout['xaxis']['range'] = [x_min - x_pad, x_max + x_pad]
    layout['yaxis']['range'] = [y_min - y_pad, y_max + y_pad]
    layout['margin'] = dict(l=60, r=30, t=50, b=50)
    layout['showlegend'] = False

    fig.update_layout(**layout)

    return fig


def create_cumulative_xg_figure(player_name):
    """Create cumulative xG vs goals chart for selected player (TEST SET ONLY)."""
    if not DATA_LOADED or player_name is None:
        return go.Figure()

    test_df = df[df['match_date'] >= '2016-01-01']
    player_df = test_df[test_df['player'].str.contains(player_name, case=False, na=False)]

    if len(player_df) == 0:
        return go.Figure()

    # Sort by date
    player_df = player_df.sort_values('match_date')

    cumulative_goals = player_df['is_goal'].cumsum().values
    cumulative_xg = player_df['our_xg'].cumsum().values
    cumulative_sb_xg = player_df['statsbomb_xg'].cumsum().values

    fig = go.Figure()

    shots = list(range(1, len(player_df) + 1))

    # Create shaded regions for over/under performance
    # Green fill where goals > xG (overperformance)
    over_y = np.maximum(cumulative_goals, cumulative_xg)
    fig.add_trace(go.Scatter(
        x=shots, y=over_y,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=shots, y=cumulative_xg,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(16, 185, 129, 0.25)',
        showlegend=False, hoverinfo='skip'
    ))

    # Red fill where goals < xG (underperformance)
    under_y = np.minimum(cumulative_goals, cumulative_xg)
    fig.add_trace(go.Scatter(
        x=shots, y=cumulative_xg,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=shots, y=under_y,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(239, 68, 68, 0.25)',
        showlegend=False, hoverinfo='skip'
    ))

    # Our xG - blue line
    fig.add_trace(go.Scatter(
        x=shots,
        y=cumulative_xg,
        mode='lines',
        name='Our xG',
        line=dict(color=COLORS['accent_secondary'], width=2)
    ))

    # Actual goals - dark slate, thicker line
    fig.add_trace(go.Scatter(
        x=shots,
        y=cumulative_goals,
        mode='lines',
        name='Actual Goals',
        line=dict(color=COLORS['text_primary'], width=3)
    ))

    # StatsBomb xG - gray dashed
    fig.add_trace(go.Scatter(
        x=shots,
        y=cumulative_sb_xg,
        mode='lines',
        name='StatsBomb xG',
        line=dict(color=COLORS['chart_gray'], width=2, dash='dash')
    ))

    layout = get_chart_layout(f'{player_name} - Cumulative Goals vs xG', height=400)
    layout['xaxis']['title'] = 'Shot Number'
    layout['yaxis']['title'] = 'Cumulative Goals / xG'
    fig.update_layout(**layout)

    return fig


# Calculate penalty stats for display
if DATA_LOADED:
    all_pens = df[df['is_penalty'] == 1]
    n_pens = len(all_pens)
    conv_rate = all_pens['is_goal'].mean()
    # Binomial standard error: sqrt(p * (1-p) / n)
    se = np.sqrt(conv_rate * (1 - conv_rate) / n_pens) if n_pens > 0 else 0
    PENALTY_STATS = {
        'n_penalties': n_pens,
        'n_goals': int(all_pens['is_goal'].sum()),
        'conversion': conv_rate,
        'se': se,
    }
else:
    PENALTY_STATS = {'n_penalties': 0, 'n_goals': 0, 'conversion': 0, 'se': 0}

# Calculate dataset stats for display
if DATA_LOADED:
    train_df = df[df['match_date'] < '2016-01-01']
    test_df_stats = df[df['match_date'] >= '2016-01-01']
    DATASET_STATS = {
        'total_shots': len(df),
        'total_goals': int(df['is_goal'].sum()),
        'total_matches': df['match_id'].nunique(),
        'train_shots': len(train_df),
        'train_goals': int(train_df['is_goal'].sum()),
        'test_shots': len(test_df_stats),
        'test_goals': int(test_df_stats['is_goal'].sum()),
    }
else:
    DATASET_STATS = {
        'total_shots': 0, 'total_goals': 0, 'total_matches': 0,
        'train_shots': 0, 'train_goals': 0, 'test_shots': 0, 'test_goals': 0
    }

# Generate SHAP image ONCE at startup (before any other matplotlib operations)
# This prevents matplotlib state contamination from callbacks
print("Generating SHAP image...")
SHAP_IMAGE = create_shap_summary_image() if DATA_LOADED else None
print("SHAP image generated.")

# Get unique player names for dropdown, ranked by xG overperformance (TEST SET ONLY)
if DATA_LOADED:
    test_df = df[df['match_date'] >= '2016-01-01']
    player_stats = test_df.groupby('player').agg({
        'is_goal': 'sum',
        'our_xg': 'sum',
        'player': 'count'
    }).rename(columns={'player': 'shots', 'is_goal': 'goals'})
    player_stats['xg_diff'] = player_stats['goals'] - player_stats['our_xg']

    # Filter to players with >= 10 shots in test period, sort by xG diff descending
    top_players_df = player_stats[player_stats['shots'] >= 10].sort_values('xg_diff', ascending=False)
    top_players = top_players_df.index.tolist()

    # Create labels showing the diff
    player_options = [
        {'label': f"{p} ({top_players_df.loc[p, 'xg_diff']:+.1f})", 'value': p}
        for p in top_players
    ]
else:
    top_players = []
    player_options = []

# Feature options for dropdown
feature_options = [
    {'label': FEATURE_LABELS.get(f, f.replace('_', ' ').title()), 'value': f}
    for f in FEATURE_COLUMNS
]


# App layout with modern design
app.layout = html.Div([
    # Hero Header Section
    html.Div([
        dbc.Container([
            html.H1("La Liga 2015/16 - xG Model Dashboard", className="hero-title text-center"),
            html.Div([
                html.Span("StatsBomb Open Data", className="hero-badge"),
                html.Span("Training Data: Aug–Dec 2015", className="hero-badge"),
                html.Span("Test Data: Jan–May 2016", className="hero-badge"),
            ], className="text-center hero-subtitle")
        ], fluid=True)
    ], className="hero-section"),

    # Main Content
    dbc.Container([
        # About the Data section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("About the Data", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Data Source", className="section-header"),
                                html.P([
                                    "This dashboard uses ",
                                    html.A("StatsBomb Open Data",
                                           href="https://github.com/statsbomb/open-data",
                                           target="_blank",
                                           style={'color': COLORS['accent_secondary']}),
                                    ", a free dataset of event-level football data released for research and education."
                                ], style={'fontSize': '0.9rem', 'color': COLORS['text_secondary']}),
                            ], md=4),
                            dbc.Col([
                                html.H6("Dataset", className="section-header"),
                                html.P([
                                    f"{DATASET_STATS['total_shots']:,} shots ({DATASET_STATS['total_goals']:,} goals) ",
                                    f"from all {DATASET_STATS['total_matches']} matches of ",
                                    html.Strong("La Liga 2015/16"),
                                    ". Each shot includes location, body part, "
                                    "technique, game state, and StatsBomb's own xG."
                                ], style={'fontSize': '0.9rem', 'color': COLORS['text_secondary']}),
                            ], md=4),
                            dbc.Col([
                                html.H6("Train/Test Split", className="section-header"),
                                html.P([
                                    html.Strong("Training: "), f"Aug–Dec 2015 ({DATASET_STATS['train_shots']:,} shots)", html.Br(),
                                    html.Strong("Test: "), f"Jan–May 2016 ({DATASET_STATS['test_shots']:,} shots)", html.Br(),
                                    html.Small("Temporal split prevents data leakage.",
                                              style={'fontStyle': 'italic'})
                                ], style={'fontSize': '0.9rem', 'color': COLORS['text_secondary']}),
                            ], md=4),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Model", className="section-header mt-2"),
                                html.P([
                                    "XGBoost classifier with isotonic calibration, tuned via Optuna (100 trials). "
                                    "Features include distance to goal, shot angle, body part, and game context."
                                ], style={'fontSize': '0.9rem', 'color': COLORS['text_secondary']}),
                            ], md=8),
                            dbc.Col([
                                html.H6("Penalties", className="section-header mt-2"),
                                html.P([
                                    f"Fixed xG of {PENALTY_XG} from historical rates. In 2015/16 La Liga, there were "
                                    f"{PENALTY_STATS['n_penalties']} penalties with ",
                                    html.Strong(f"{PENALTY_STATS['conversion']:.1%} \u00B1 {PENALTY_STATS['se']:.1%}"),
                                    " conversion."
                                ], style={'fontSize': '0.9rem', 'color': COLORS['text_secondary']}),
                            ], md=4),
                        ], className="mt-2"),
                    ])
                ], className="chart-card mb-4")
            ])
        ]),

        # Feature Analysis section - three equal columns
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Feature Analysis", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            # Feature Distribution
                            dbc.Col([
                                html.Div([
                                    html.Label("Distribution", style={
                                        'fontSize': '0.75rem',
                                        'color': COLORS['text_secondary'],
                                        'textTransform': 'uppercase',
                                        'letterSpacing': '0.05em',
                                        'marginBottom': '0.25rem',
                                        'display': 'block'
                                    }),
                                    dcc.Dropdown(
                                        id='feature-dist-dropdown',
                                        options=feature_options,
                                        value='distance_to_goal',
                                        clearable=False,
                                        style={'marginBottom': '0.5rem'}
                                    )
                                ]),
                                dcc.Graph(id='feature-dist-chart',
                                         figure=create_feature_distribution_figure('distance_to_goal'),
                                         config={'displayModeBar': False})
                            ], lg=4),
                            # SHAP
                            dbc.Col([
                                html.Label("SHAP Impact", style={
                                    'fontSize': '0.75rem',
                                    'color': COLORS['text_secondary'],
                                    'textTransform': 'uppercase',
                                    'letterSpacing': '0.05em',
                                    'marginBottom': '0.5rem',
                                    'display': 'block'
                                }),
                                html.Img(
                                    src=SHAP_IMAGE,
                                    style={'maxWidth': '100%', 'height': 'auto'},
                                    className="mx-auto d-block"
                                ) if SHAP_IMAGE else html.P("SHAP not available.", className="text-muted")
                            ], lg=4),
                            # Correlations
                            dbc.Col([
                                html.Label("Correlations", style={
                                    'fontSize': '0.75rem',
                                    'color': COLORS['text_secondary'],
                                    'textTransform': 'uppercase',
                                    'letterSpacing': '0.05em',
                                    'marginBottom': '0.5rem',
                                    'display': 'block'
                                }),
                                dcc.Graph(figure=create_correlation_matrix_figure(),
                                         config={'displayModeBar': False})
                            ], lg=4),
                        ])
                    ])
                ], className="chart-card mb-4")
            ])
        ]),

        # Model Performance section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H4("Model Performance", className="mb-0 d-inline"),
                            html.Div([
                                html.Span([
                                    html.Strong("AUC: "), f"{metrics.get('roc_auc', 0):.3f}"
                                ], className="hero-badge", style={'marginLeft': '1rem'}),
                                html.Span([
                                    html.Strong("Brier: "), f"{metrics.get('brier', 0):.3f}"
                                ], className="hero-badge"),
                                html.Span([
                                    html.Strong("LogLoss: "), f"{metrics.get('log_loss', 0):.3f}"
                                ], className="hero-badge"),
                            ], className="d-inline")
                        ], className="d-flex align-items-center flex-wrap")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=create_roc_curve_figure(),
                                         config={'displayModeBar': False})
                            ], lg=4),
                            dbc.Col([
                                dcc.Graph(figure=create_calibration_figure(),
                                         config={'displayModeBar': False})
                            ], lg=4),
                            dbc.Col([
                                html.Div([
                                    html.Label("Model", style={
                                        'fontSize': '0.75rem',
                                        'color': COLORS['text_secondary'],
                                        'textTransform': 'uppercase',
                                        'letterSpacing': '0.05em',
                                        'marginBottom': '0.25rem'
                                    }),
                                    dcc.Dropdown(
                                        id='model-dropdown',
                                        options=[
                                            {'label': 'Our Model', 'value': 'our_xg'},
                                            {'label': 'StatsBomb', 'value': 'statsbomb_xg'}
                                        ],
                                        value='our_xg',
                                        clearable=False,
                                        style={'marginBottom': '10px'}
                                    ),
                                ], style={'marginBottom': '0.5rem'}),
                                dcc.Graph(id='xg-histogram', figure=create_xg_histogram_figure('our_xg'), config={'displayModeBar': False})
                            ], lg=4),
                        ])
                    ])
                ], className="chart-card mb-4")
            ])
        ]),

        # Team Analysis section - bar chart with dropdown + scatter plot
        dbc.Row([
            dbc.Col([
                html.H4("Team Performance", className="mb-2"),
                html.Small("Test set: Jan–May 2016 | Using Our Model", className="text-muted d-block mb-3"),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dcc.Dropdown(
                            id='team-metric-dropdown',
                            options=[
                                {'label': 'Goal Difference', 'value': 'goal_diff'},
                                {'label': 'Goals − xG (Attack)', 'value': 'g_minus_xg'},
                                {'label': 'xGA − Goals (Defence)', 'value': 'xga_minus_ga'},
                                {'label': 'xG Difference', 'value': 'xg_diff'},
                            ],
                            value='xg_diff',
                            clearable=False,
                            style={'width': '220px'}
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='team-bar-chart', figure=create_team_bar_chart('xg_diff'), config={'displayModeBar': False})
                    ])
                ], className="chart-card mb-4")
            ], lg=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dcc.Dropdown(
                            id='scatter-metric-dropdown',
                            options=[
                                {'label': 'xG vs xGA', 'value': 'xg'},
                                {'label': 'Goals vs Conceded', 'value': 'goals'},
                            ],
                            value='xg',
                            clearable=False,
                            style={'width': '180px'}
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='team-scatter-chart', figure=create_team_scatter_chart('xg'), config={'displayModeBar': False})
                    ])
                ], className="chart-card mb-4")
            ], lg=6),
        ]),

        # Spatial xG section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Spatial xG Distribution", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Img(
                                    src=create_spatial_xg_image('our_xg'),
                                    style={'maxHeight': '350px', 'width': 'auto'},
                                    className="mx-auto d-block"
                                ) if DATA_LOADED else html.P("No data available.", className="text-muted")
                            ], lg=4, className="text-center"),
                            dbc.Col([
                                html.Img(
                                    src=create_spatial_xg_image('statsbomb_xg'),
                                    style={'maxHeight': '350px', 'width': 'auto'},
                                    className="mx-auto d-block"
                                ) if DATA_LOADED else html.P("No data available.", className="text-muted")
                            ], lg=4, className="text-center"),
                            dbc.Col([
                                html.Img(
                                    src=create_spatial_diff_image(),
                                    style={'maxHeight': '350px', 'width': 'auto'},
                                    className="mx-auto d-block"
                                ) if DATA_LOADED else html.P("No data available.", className="text-muted")
                            ], lg=4, className="text-center"),
                        ])
                    ], className="py-3")
                ], className="chart-card mb-4")
            ])
        ]),

        # Player analysis section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Player Analysis", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Player", style={
                                    'fontSize': '0.75rem',
                                    'color': COLORS['text_secondary'],
                                    'textTransform': 'uppercase',
                                    'letterSpacing': '0.05em',
                                    'marginBottom': '0.25rem',
                                    'display': 'block'
                                }),
                                html.Small("Ranked by Goals − xG", style={
                                    'color': COLORS['text_secondary'],
                                    'display': 'block',
                                    'marginBottom': '0.5rem'
                                }),
                                dcc.Dropdown(
                                    id='player-dropdown',
                                    options=player_options,
                                    value=top_players[0] if top_players else None,
                                    className="mb-3"
                                )
                            ], lg=3),
                            dbc.Col([
                                html.Div(id='player-stats-box')
                            ], lg=9, className="d-flex align-items-center")
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.Img(id='shot-map', src=create_shot_map_image(top_players[0] if top_players else None), style={'maxWidth': '100%', 'height': 'auto'})
                            ], lg=5, className="text-center"),
                            dbc.Col([
                                dcc.Graph(id='cumulative-xg', figure=create_cumulative_xg_figure(top_players[0] if top_players else None), config={'displayModeBar': False})
                            ], lg=7)
                        ])
                    ])
                ], className="chart-card mb-4")
            ])
        ]),

    ], fluid=True),

    # Footer
    html.Div([
        dbc.Container([
            html.P([
                "Built with ",
                html.A("StatsBomb Open Data", href="https://github.com/statsbomb/open-data",
                      target="_blank", style={'color': COLORS['accent_secondary']})
            ], className="text-center mb-0", style={'color': COLORS['text_secondary'], 'fontSize': '0.9rem'})
        ], fluid=True)
    ], className="footer")

], style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh'})


# Callbacks
@callback(
    Output('xg-histogram', 'figure'),
    Input('model-dropdown', 'value')
)
def update_xg_histogram(model):
    return create_xg_histogram_figure(model)


@callback(
    Output('feature-dist-chart', 'figure'),
    Input('feature-dist-dropdown', 'value')
)
def update_feature_distribution(feature):
    return create_feature_distribution_figure(feature)


@callback(
    Output('team-bar-chart', 'figure'),
    Input('team-metric-dropdown', 'value')
)
def update_team_bar_chart(metric):
    return create_team_bar_chart(metric)


@callback(
    Output('team-scatter-chart', 'figure'),
    Input('scatter-metric-dropdown', 'value')
)
def update_team_scatter_chart(metric):
    return create_team_scatter_chart(metric)


@callback(
    Output('shot-map', 'src'),
    Input('player-dropdown', 'value')
)
def update_shot_map(player_name):
    return create_shot_map_image(player_name)


@callback(
    Output('cumulative-xg', 'figure'),
    Input('player-dropdown', 'value')
)
def update_cumulative_xg(player_name):
    return create_cumulative_xg_figure(player_name)


@callback(
    Output('player-stats-box', 'children'),
    Input('player-dropdown', 'value')
)
def update_player_stats(player_name):
    stats = get_player_stats(player_name)
    if stats is None:
        return html.P("No data available", className="text-muted")

    # Determine diff class
    diff_class = 'success' if stats['xg_diff'] > 0 else 'danger'

    # Create modern stat cards
    stat_items = [
        ('Games', f"{stats['games']}", None),
        ('Shots', f"{stats['shots']}", None),
        ('Goals', f"{stats['goals']}", None),
        ('xG', f"{stats['our_xg']:.1f}", None),
        ('G-xG', f"{stats['xg_diff']:+.1f}", diff_class),
        ('Conv%', f"{stats['conversion']:.0%}", None),
        ('G/game', f"{stats['goals_per_game']:.2f}", None),
        ('xG/game', f"{stats['xg_per_game']:.2f}", None),
    ]

    return html.Div([
        html.Div([
            html.Div([
                html.Div(value, className=f"stat-value {color_class or ''}"),
                html.Div(label, className="stat-label")
            ], className="stat-card")
            for label, value, color_class in stat_items
        ], className="player-stats-row")
    ])


def run_app(debug=True, port=8050):
    """Run the Dash application."""
    import os
    # Use PORT env variable for Render/production deployment
    port = int(os.environ.get('PORT', port))
    host = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
    app.run(debug=debug, port=port, host=host)


# Expose server for gunicorn
server = app.server


if __name__ == '__main__':
    run_app()
