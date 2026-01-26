"""
Export xG dashboard as a static HTML report.

This script generates a standalone HTML file with:
- Interactive Plotly charts (zoom, pan, hover work client-side)
- Embedded matplotlib images as base64
- JavaScript-based dropdowns for chart variations
- Same visual styling as the Dash app
"""
import base64
from pathlib import Path
from datetime import datetime

import plotly.io as pio

# Import everything from the app module
from .app import (
    COLORS,
    CUSTOM_CSS,
    PENALTY_XG,
    PENALTY_STATS,
    DATASET_STATS,
    DATA_LOADED,
    df,
    model,
    metrics,
    SHAP_VALUES,
    player_options,
    create_shap_summary_image,
    create_calibration_figure,
    create_roc_curve_figure,
    create_xg_histogram_figure,
    create_spatial_xg_image,
    create_spatial_diff_image,
    create_team_xg_diff_chart,
    create_shot_map_image,
    create_cumulative_xg_figure,
    get_player_stats,
    create_model_summary,
)
from .config import OUTPUT_DIR


def fig_to_json(fig):
    """Convert Plotly figure to JSON for deferred rendering."""
    return fig.to_json()


def fig_to_div(div_id, height=350):
    """Create a placeholder div for a Plotly chart."""
    return f'<div id="{div_id}" style="height:{height}px; width:100%;"></div>'


def generate_html_report(output_path=None):
    """Generate static HTML report."""
    if not DATA_LOADED:
        print("Error: Data not loaded. Cannot generate report.")
        return None

    print("Generating HTML report...")

    # Generate all figures
    print("  Creating calibration chart...")
    calibration_fig = create_calibration_figure()

    print("  Creating ROC curve...")
    roc_fig = create_roc_curve_figure()

    print("  Creating xG histograms...")
    histogram_our = create_xg_histogram_figure('our_xg')
    histogram_sb = create_xg_histogram_figure('statsbomb_xg')

    print("  Creating team xG diff charts...")
    team_diff_our = create_team_xg_diff_chart('our_xg')
    team_diff_sb = create_team_xg_diff_chart('statsbomb_xg')

    print("  Creating SHAP image...")
    shap_img = create_shap_summary_image()

    print("  Creating spatial xG images...")
    spatial_our = create_spatial_xg_image('our_xg')
    spatial_sb = create_spatial_xg_image('statsbomb_xg')
    spatial_diff = create_spatial_diff_image()

    # Get top 5 players for the report
    print("  Creating player analyses...")
    top_players = player_options[:5] if len(player_options) >= 5 else player_options
    player_data = []
    for player in top_players:
        player_name = player['value']
        stats = get_player_stats(player_name)
        shot_map = create_shot_map_image(player_name)
        cumulative_fig = create_cumulative_xg_figure(player_name)
        player_data.append({
            'name': player_name,
            'label': player['label'],
            'stats': stats,
            'shot_map': shot_map,
            'cumulative_fig': cumulative_fig,
        })

    # Build HTML
    print("  Assembling HTML...")

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>La Liga 2015/16 - xG Model Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
{CUSTOM_CSS}

        /* Additional styles for static report */
        .chart-toggle {{
            display: inline-flex;
            background: {COLORS['bg_primary']};
            border-radius: 6px;
            padding: 3px;
            gap: 2px;
        }}
        .chart-toggle button {{
            border: none;
            background: transparent;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: {COLORS['text_secondary']};
            cursor: pointer;
            transition: all 0.2s;
        }}
        .chart-toggle button.active {{
            background: {COLORS['accent_primary']};
            color: white;
        }}
        .chart-toggle button:hover:not(.active) {{
            background: {COLORS['border']};
        }}
        .chart-container {{
            display: none;
        }}
        .chart-container.active {{
            display: block;
        }}
        .player-section {{
            display: none;
        }}
        .player-section.active {{
            display: block;
        }}
        .player-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 1rem;
        }}
        .player-tab {{
            padding: 8px 16px;
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}
        .player-tab:hover {{
            border-color: {COLORS['accent_secondary']};
        }}
        .player-tab.active {{
            background: {COLORS['accent_primary']};
            color: white;
            border-color: {COLORS['accent_primary']};
        }}
        .report-meta {{
            text-align: center;
            color: {COLORS['text_secondary']};
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS['border']};
        }}
    </style>
</head>
<body>
    <!-- Hero Section -->
    <div class="hero-section">
        <h1 class="hero-title text-center">La Liga 2015/16 - xG Model Dashboard</h1>
        <div class="text-center">
            <span class="hero-badge">StatsBomb Open Data</span>
            <span class="hero-badge">Training Data: Aug-Dec 2015</span>
            <span class="hero-badge">Test Data: Jan-May 2016</span>
        </div>
    </div>

    <div class="container">
        <!-- About the Data -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card chart-card">
                    <div class="card-header"><h4 class="mb-0">About the Data</h4></div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <h6 class="section-header">Data Source</h6>
                                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}">
                                    This dashboard uses <a href="https://github.com/statsbomb/open-data" target="_blank" style="color: {COLORS['accent_secondary']}">StatsBomb Open Data</a>,
                                    a free dataset of event-level football data released for research and education.
                                </p>
                            </div>
                            <div class="col-md-4">
                                <h6 class="section-header">Dataset</h6>
                                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}">
                                    {DATASET_STATS['total_shots']:,} shots ({DATASET_STATS['total_goals']:,} goals) from all {DATASET_STATS['total_matches']} matches of
                                    <strong>La Liga 2015/16</strong>. Each shot includes location, body part, technique, game state, and StatsBomb's own xG.
                                </p>
                            </div>
                            <div class="col-md-4">
                                <h6 class="section-header">Train/Test Split</h6>
                                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}">
                                    <strong>Training:</strong> Aug-Dec 2015 ({DATASET_STATS['train_shots']:,} shots)<br>
                                    <strong>Test:</strong> Jan-May 2016 ({DATASET_STATS['test_shots']:,} shots)<br>
                                    <em style="font-size: 0.85rem">Temporal split prevents data leakage.</em>
                                </p>
                            </div>
                        </div>
                        <div class="row mt-2">
                            <div class="col-md-8">
                                <h6 class="section-header">Model</h6>
                                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}">
                                    XGBoost classifier with isotonic calibration, tuned via Optuna (100 trials).
                                    Features include distance to goal, shot angle, body part, and game context.
                                </p>
                            </div>
                            <div class="col-md-4">
                                <h6 class="section-header">Penalties</h6>
                                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}">
                                    Fixed xG of {PENALTY_XG} from historical rates. In 2015/16 La Liga, there were
                                    {PENALTY_STATS['n_penalties']} penalties with {PENALTY_STATS['conversion']:.1%} (&plusmn;{PENALTY_STATS['se']:.1%}) conversion.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Summary and SHAP -->
        <div class="row mb-4">
            <div class="col-lg-7">
                <div class="card chart-card h-100">
                    <div class="card-header"><h4 class="mb-0">Model Summary</h4></div>
                    <div class="card-body">
                        {create_model_summary_html()}
                    </div>
                </div>
            </div>
            <div class="col-lg-5">
                <div class="card chart-card h-100">
                    <div class="card-header"><h4 class="mb-0">SHAP Feature Impact</h4></div>
                    <div class="card-body text-center py-2">
                        <img src="{shap_img}" style="max-width: 100%; height: auto;">
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Performance Charts -->
        <div class="row mb-4">
            <div class="col-lg-4">
                <div class="card chart-card">
                    <div class="card-header"><h4 class="mb-0">Calibration</h4></div>
                    <div class="card-body">
                        {fig_to_html(calibration_fig, 'calibration-chart', include_plotlyjs=False)}
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card chart-card">
                    <div class="card-header"><h4 class="mb-0">ROC Curve</h4></div>
                    <div class="card-body">
                        {fig_to_html(roc_fig, 'roc-chart', include_plotlyjs=False)}
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card chart-card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h4 class="mb-0">xG Distribution</h4>
                        <div class="chart-toggle" id="histogram-toggle">
                            <button class="active" onclick="toggleChart('histogram', 'our')">Our Model</button>
                            <button onclick="toggleChart('histogram', 'sb')">StatsBomb</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="histogram-our" class="chart-container active">
                            {fig_to_html(histogram_our, 'histogram-our-chart', include_plotlyjs=False)}
                        </div>
                        <div id="histogram-sb" class="chart-container">
                            {fig_to_html(histogram_sb, 'histogram-sb-chart', include_plotlyjs=False)}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Team xG Difference -->
        <div class="row mb-4">
            <div class="col-lg-6">
                <div class="card chart-card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <div>
                            <h4 class="mb-0">Team xG Difference per 90</h4>
                            <small class="text-muted">Test set: Jan-May 2016</small>
                        </div>
                        <div class="chart-toggle" id="team-toggle">
                            <button class="active" onclick="toggleChart('team', 'our')">Our Model</button>
                            <button onclick="toggleChart('team', 'sb')">StatsBomb</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="team-our" class="chart-container active">
                            {fig_to_html(team_diff_our, 'team-our-chart', include_plotlyjs=False)}
                        </div>
                        <div id="team-sb" class="chart-container">
                            {fig_to_html(team_diff_sb, 'team-sb-chart', include_plotlyjs=False)}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Spatial xG Distribution -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card chart-card">
                    <div class="card-header"><h4 class="mb-0">Spatial xG Distribution</h4></div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-lg-4 text-center">
                                <p class="text-muted mb-2" style="font-size: 0.9rem">Our Model</p>
                                <img src="{spatial_our}" style="max-height: 350px; width: auto;">
                            </div>
                            <div class="col-lg-4 text-center">
                                <p class="text-muted mb-2" style="font-size: 0.9rem">StatsBomb</p>
                                <img src="{spatial_sb}" style="max-height: 350px; width: auto;">
                            </div>
                            <div class="col-lg-4 text-center">
                                <p class="text-muted mb-2" style="font-size: 0.9rem">Difference (Ours - SB)</p>
                                <img src="{spatial_diff}" style="max-height: 350px; width: auto;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Player Analysis -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card chart-card">
                    <div class="card-header">
                        <h4 class="mb-0">Player Analysis</h4>
                        <small class="text-muted">Top 5 players by xG overperformance (test set)</small>
                    </div>
                    <div class="card-body">
                        <div class="player-tabs">
                            {generate_player_tabs(player_data)}
                        </div>
                        {generate_player_sections(player_data)}
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="report-meta">
            <p>
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} |
                Data: <a href="https://github.com/statsbomb/open-data" target="_blank">StatsBomb Open Data</a> |
                <a href="https://github.com/donalhill/statsbomb_free_data" target="_blank">Source Code</a>
            </p>
        </div>
    </div>

    <script>
        function toggleChart(chartType, version) {{
            // Hide all containers for this chart type
            document.querySelectorAll('#' + chartType + '-our, #' + chartType + '-sb').forEach(el => {{
                el.classList.remove('active');
            }});
            // Show selected
            document.getElementById(chartType + '-' + version).classList.add('active');

            // Update toggle buttons
            const toggle = document.getElementById(chartType + '-toggle');
            toggle.querySelectorAll('button').forEach((btn, idx) => {{
                btn.classList.remove('active');
                if ((version === 'our' && idx === 0) || (version === 'sb' && idx === 1)) {{
                    btn.classList.add('active');
                }}
            }});

            // Trigger Plotly resize for the newly visible chart
            window.dispatchEvent(new Event('resize'));
        }}

        function showPlayer(index) {{
            // Hide all player sections
            document.querySelectorAll('.player-section').forEach(el => {{
                el.classList.remove('active');
            }});
            // Show selected
            document.getElementById('player-' + index).classList.add('active');

            // Update tabs
            document.querySelectorAll('.player-tab').forEach((tab, idx) => {{
                tab.classList.toggle('active', idx === index);
            }});

            // Trigger Plotly resize
            window.dispatchEvent(new Event('resize'));
        }}

        // Initialize first player as active
        document.addEventListener('DOMContentLoaded', function() {{
            showPlayer(0);
        }});
    </script>
</body>
</html>'''

    # Write output
    if output_path is None:
        output_path = OUTPUT_DIR / 'xg_report.html'
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    print(f"Report saved to: {output_path}")
    return output_path


def create_model_summary_html():
    """Generate HTML for model summary section."""
    # Feature importance
    feature_items = ""
    if metrics and 'feature_importance' in metrics:
        sorted_features = sorted(
            metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:6]
        max_importance = max(v for _, v in sorted_features) if sorted_features else 1

        for feat, imp in sorted_features:
            pct = (imp / max_importance) * 100
            feature_items += f'''
                <div class="mb-2">
                    <div class="d-flex justify-content-between" style="font-size: 0.85rem;">
                        <span>{feat.replace('_', ' ').title()}</span>
                        <span class="text-muted">{imp:.3f}</span>
                    </div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar" style="width: {pct}%; background: {COLORS['accent_primary']};"></div>
                    </div>
                </div>'''

    # Test metrics
    metrics_html = ""
    if metrics:
        metric_items = [
            ('AUC', f"{metrics.get('roc_auc', 0):.2f}"),
            ('Brier', f"{metrics.get('brier', 0):.3f}"),
            ('LogLoss', f"{metrics.get('log_loss', 0):.2f}"),
        ]
        for label, value in metric_items:
            metrics_html += f'''
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>'''

    return f'''
        <div class="row">
            <div class="col-md-6">
                <h6 class="section-header">Feature Importance</h6>
                {feature_items}
            </div>
            <div class="col-md-6">
                <h6 class="section-header">Test Performance</h6>
                <div class="d-flex flex-wrap gap-3">
                    {metrics_html}
                </div>
            </div>
        </div>
    '''


def generate_player_tabs(player_data):
    """Generate player selection tabs."""
    tabs = ""
    for i, player in enumerate(player_data):
        active = "active" if i == 0 else ""
        tabs += f'<button class="player-tab {active}" onclick="showPlayer({i})">{player["name"]}</button>'
    return tabs


def generate_player_sections(player_data):
    """Generate player analysis sections."""
    sections = ""
    for i, player in enumerate(player_data):
        active = "active" if i == 0 else ""
        stats = player['stats']

        if stats:
            diff_class = 'success' if stats['xg_diff'] > 0 else 'danger'
            stats_html = f'''
                <div class="d-flex flex-wrap gap-3 mb-3">
                    <div class="stat-card"><div class="stat-value">{stats['games']}</div><div class="stat-label">Games</div></div>
                    <div class="stat-card"><div class="stat-value">{stats['shots']}</div><div class="stat-label">Shots</div></div>
                    <div class="stat-card"><div class="stat-value">{stats['goals']}</div><div class="stat-label">Goals</div></div>
                    <div class="stat-card"><div class="stat-value">{stats['our_xg']:.1f}</div><div class="stat-label">xG</div></div>
                    <div class="stat-card"><div class="stat-value text-{diff_class}">{stats['xg_diff']:+.1f}</div><div class="stat-label">G-xG</div></div>
                    <div class="stat-card"><div class="stat-value">{stats['conversion']:.0%}</div><div class="stat-label">Conv%</div></div>
                    <div class="stat-card"><div class="stat-value">{stats['goals_per_game']:.2f}</div><div class="stat-label">G/game</div></div>
                    <div class="stat-card"><div class="stat-value">{stats['xg_per_game']:.2f}</div><div class="stat-label">xG/game</div></div>
                </div>
            '''
        else:
            stats_html = '<p class="text-muted">No stats available</p>'

        cumulative_html = fig_to_html(player['cumulative_fig'], f'cumulative-{i}', include_plotlyjs=False)

        sections += f'''
            <div id="player-{i}" class="player-section {active}">
                {stats_html}
                <div class="row">
                    <div class="col-lg-5 text-center mb-3">
                        <img src="{player['shot_map']}" style="max-width: 100%; height: auto;">
                    </div>
                    <div class="col-lg-7">
                        {cumulative_html}
                    </div>
                </div>
            </div>
        '''
    return sections


if __name__ == '__main__':
    generate_html_report()
