"""
Interactive QC dashboard.

Renders the three datasets ProcessData produces (sourced from the same
revision-history walk `visualize_data/graph_errors.py` uses for the
static PNGs):

  - Stacked Area: outstanding queries per day, by Timepoint
  - Line Graph:   new / outstanding / resolved / priority over time
                  (with optional priority-share-over-time overlay)
  - Site Bar:     per-site over_thirty / under_thirty alongside
                  priority / non_priority

Color schemes match `graph_errors.py` so the interactive charts look
like the published PNGs. The stacked-area assigns colors by a stable
{timepoint: color} map so a given timepoint keeps the same hue across
filter selections and network switches.

Network selector includes a "Both" option that overlays PRONET and
PRESCIENT on shared axes (line graph + stacked area) or facets them
side-by-side (site bar).

Run:
  python create_dashboard.py                     # Dash server on :5000
  python create_dashboard.py --refresh           # rebuild cache first
  python create_dashboard.py --export-html PATH  # write standalone HTML
  python create_dashboard.py --export-html-only PATH  # export and exit
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from process_data import ProcessData  # noqa: E402


# Chart palettes — tailwind-derived, tuned for the dark surface and
# tested for >= 4.5:1 contrast. The line palette rotates cool / warm /
# cool / warm so all four series stay distinguishable; the site palette
# uses two semantic pairs (severity-ramp for detection age,
# attention-ramp for priority) on a unified hue wheel.
LINE_COLORS = {
    'new':         '#60a5fa',  # sky-400
    'outstanding': '#f59e0b',  # amber-500
    'resolved':    '#34d399',  # emerald-400
    'priority':    '#f472b6',  # pink-400
}
SITE_BAR_COLORS = {
    'over_thirty':  '#dc2626',  # red-600 (severe)
    'under_thirty': '#fbbf24',  # amber-400 (recent)
    'priority':     '#8b5cf6',  # violet-500 (flagged)
    'non_priority': '#475569',  # slate-600 (baseline)
}
NETWORK_DASH = {  # for the Both overlay
    'PRONET': 'solid',
    'PRESCIENT': 'dot',
}

# 3-tier surface palette: page → card → elevated (legend, hoverlabel,
# toggle pills, info chips). Spreading the tonal hierarchy keeps cards
# lifted off the page and elevated UI lifted off cards.
DARK_BG = '#0a0e13'      # page background (--surface-0)
CARD_BG = '#131a23'      # card surface     (--surface-1)
ELEVATED_BG = '#1c2531'  # legend / hover / chips (--surface-2)
ACCENT = '#60a5fa'       # focus / active-state outline
TEXT_PRIMARY = '#f1f5f9'
TEXT_SECONDARY = '#cbd5e1'
TEXT_MUTED = '#94a3b8'

GRID_RGBA = 'rgba(255,255,255,0.13)'
ZEROLINE_RGBA = 'rgba(255,255,255,0.28)'
AXIS_LINE_RGBA = 'rgba(255,255,255,0.25)'
STALE_THRESHOLD_HOURS = 36

# Canonical timepoint order used by `_timepoint_color_map` so the
# stacked area's color ramp follows time, not alphabetical sort.
TIMEPOINT_ORDER = [
    'screening', 'baseline',
    'month1', 'month2', 'month3', 'month4', 'month5', 'month6',
    'month7', 'month8', 'month9', 'month10', 'month11', 'month12',
    'month18', 'month24',
]

CHART_OPTIONS = [
    {'label': 'Outstanding by Timepoint (stacked area)', 'value': 'stacked'},
    {'label': 'Queries Over Time (new / outstanding / resolved / priority)',
     'value': 'line'},
    {'label': 'Queries Per Site (today)', 'value': 'site'},
]
SERIES_ORDER = ['new', 'outstanding', 'resolved', 'priority']

# Plotly modebar config shared by the static export so the toolbar
# only shows the buttons that make sense for these charts.
PLOTLY_CONFIG = {
    'displaylogo': False,
    'displayModeBar': 'hover',
    'modeBarButtonsToRemove': [
        'lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines',
    ],
    'toImageButtonOptions': {
        'format': 'png', 'filename': 'qc_dashboard', 'scale': 2,
    },
}


# -----------------------------
# Module-level DATA store
# -----------------------------
# Held in-process and mutated in place by `refresh()` so callbacks can
# reference it directly without a JSON round-trip through dcc.Store. A
# small `refresh-counter` Store fires the dependent callbacks. This
# assumes a single-process server — fine for the dev server and a
# single-worker gunicorn deployment, which is what this dashboard is
# sized for. If we ever scale out to multiple workers, swap to a shared
# cache (Redis, etc.) keyed on `build_info.built_at`.
DATA = {
    'stacked': pd.DataFrame(),
    'line': pd.DataFrame(),
    'site': pd.DataFrame(),
    'info': {'sources': {}, 'built_at': None},
}

# Module-level refresh state. `_refresh_lock` serializes concurrent
# force-refresh calls (so two users clicking Refresh don't both kick off
# a Dropbox walk). `_last_force_refresh_ts` rate-limits force refreshes
# to one per `REFRESH_MIN_INTERVAL_S` to prevent accidental DoS against
# Dropbox quota.
REFRESH_MIN_INTERVAL_S = 60
_refresh_lock = Lock()
_last_force_refresh_ts = 0.0


def refresh(force_refresh=False):
    """
    Repopulate the module-level DATA dict. With `force_refresh=True` a
    cache-bypassing rebuild kicks off — guarded by a per-process lock
    and rate limit so back-to-back clicks don't pile on Dropbox.
    """
    global _last_force_refresh_ts
    if force_refresh:
        now = time.monotonic()
        if now - _last_force_refresh_ts < REFRESH_MIN_INTERVAL_S:
            wait = REFRESH_MIN_INTERVAL_S - (now - _last_force_refresh_ts)
            print(f'[refresh] rate-limited; '
                  f'next force-refresh available in {wait:.0f}s')
            return DATA
        if not _refresh_lock.acquire(blocking=False):
            print('[refresh] another refresh is in progress; skipping.')
            return DATA
        try:
            _last_force_refresh_ts = now
            pd_inst = ProcessData(use_cache=False)
            stacked, line, site = pd_inst.build(force_refresh=True)
        finally:
            _refresh_lock.release()
    else:
        pd_inst = ProcessData(use_cache=True)
        stacked, line, site = pd_inst.build(force_refresh=False)

    # Normalize the `date` column once so callbacks don't pay
    # pd.to_datetime each interaction. Parquet cache already returns
    # datetime64; freshly built frames return strings.
    for df in (stacked, line):
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    DATA['stacked'] = stacked
    DATA['line'] = line
    DATA['site'] = site
    DATA['info'] = pd_inst.build_info
    return DATA


# Prime DATA at startup (cache hit on subsequent boots).
refresh(force_refresh=False)


# -----------------------------
# Figure helpers
# -----------------------------

def _empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(size=16, color='#cbd5e1'),
        xref='paper', yref='paper', x=0.5, y=0.5,
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        template='plotly_dark',
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _apply_layout(fig, title):
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=title, x=0.02, y=0.985, yanchor='top',
            font=dict(size=18, color=TEXT_PRIMARY,
                      family='Inter, system-ui, sans-serif'),
            pad=dict(t=4, b=4),
        ),
        # Generous top margin reserves space for the optional network
        # toggle band above the chart; right margin holds the vertical
        # legend; bottom keeps rotated tick labels uncropped.
        margin=dict(l=64, r=200, t=80, b=72),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family='Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial',
                  size=13, color=TEXT_SECONDARY),
        # Default chart height matches the static export's card; the
        # live Dash app overrides this implicitly via the dcc.Graph
        # container.
        height=560,
        # Vertical legend on the right — clears the rotated x-axis
        # ticks. Higher bg alpha + brighter border lift it off the
        # card surface; `itemsizing='constant'` keeps swatch sizes
        # stable when traces are toggled.
        legend=dict(
            orientation='v', yanchor='top', y=1.0,
            xanchor='left', x=1.02,
            bgcolor='rgba(28,37,49,0.92)',
            bordercolor='rgba(255,255,255,0.18)', borderwidth=1,
            font=dict(color=TEXT_SECONDARY, size=12,
                      family='Inter, system-ui, sans-serif'),
            itemsizing='constant',
            itemwidth=40,
            tracegroupgap=4,
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(28,37,49,0.97)',
            bordercolor='rgba(255,255,255,0.14)',
            font=dict(color=TEXT_PRIMARY, size=12,
                      family='Inter, system-ui, sans-serif'),
            align='left',
            namelength=-1,
        ),
        modebar=dict(
            bgcolor='rgba(0,0,0,0)',
            color='#64748b',
            activecolor=TEXT_PRIMARY,
            orientation='v',
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID_RGBA,
        tickangle=-30, automargin=True,
        tickfont=dict(color=TEXT_SECONDARY, size=11,
                      family='Inter, system-ui, sans-serif'),
        ticks='outside', ticklen=5, tickwidth=1,
        tickcolor=AXIS_LINE_RGBA,
        linecolor=AXIS_LINE_RGBA,
        title_font=dict(color=TEXT_PRIMARY, size=12),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID_RGBA, automargin=True,
        tickfont=dict(color=TEXT_SECONDARY, size=11,
                      family='Inter, system-ui, sans-serif'),
        ticks='outside', ticklen=4, tickwidth=1,
        tickcolor=AXIS_LINE_RGBA,
        linecolor=AXIS_LINE_RGBA,
        separatethousands=True,
        zerolinecolor=ZEROLINE_RGBA, zerolinewidth=1,
        title_font=dict(color=TEXT_PRIMARY, size=12),
    )
    return fig


def _add_network_toggle(fig):
    """
    Add a PRONET / PRESCIENT / Both button group above the chart that
    toggles trace visibility by network. Each trace must carry its
    network in `meta={'network': ...}` — the figure builders below
    set that for every trace they emit.

    Plotly hard-codes a white fill on the active updatemenu button,
    which clashes with the dark theme. `_static_html_css()` injects
    `!important` rules targeting `.updatemenu-item-rect` /
    `.updatemenu-active-button` to fix the active state — keep this
    method's color tokens in sync with that CSS.

    If the figure only has one network's worth of traces (or no meta),
    no buttons are added.
    """
    meta_networks = []
    for t in fig.data:
        m = getattr(t, 'meta', None)
        if isinstance(m, dict) and 'network' in m:
            meta_networks.append(m['network'])
        else:
            meta_networks.append(None)
    distinct = {n for n in meta_networks if n}
    if len(distinct) < 2:
        return fig

    visible_both = [True] * len(meta_networks)
    visible_pronet = [n == 'PRONET' for n in meta_networks]
    visible_prescient = [n == 'PRESCIENT' for n in meta_networks]

    # Each button bundles a trace-update AND a relayout that bumps
    # `updatemenus[0].active` to its own index. Without that second
    # arg, Plotly leaves `active=0` after every click so the
    # `.updatemenu-active-button` SVG class stays glued to the first
    # button and the CSS highlight never moves.
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            showactive=True,
            active=0,
            x=0.5, xanchor='center',
            y=1.10, yanchor='top',
            bgcolor='rgba(0,0,0,0)',  # let the CSS pill bg show through
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(color=TEXT_SECONDARY, size=12,
                      family='Inter, system-ui, sans-serif'),
            pad=dict(l=6, r=6, t=2, b=2),
            buttons=[
                dict(label='Both', method='update',
                     args=[{'visible': visible_both},
                           {'updatemenus[0].active': 0}]),
                dict(label='PRONET only', method='update',
                     args=[{'visible': visible_pronet},
                           {'updatemenus[0].active': 1}]),
                dict(label='PRESCIENT only', method='update',
                     args=[{'visible': visible_prescient},
                           {'updatemenus[0].active': 2}]),
            ],
        )],
        # Reserve room above the chart for the title + button band.
        margin=dict(t=130),
    )
    return fig


def _timepoint_color_map(all_timepoints):
    """
    Stable {timepoint: color} so the same timepoint keeps its color
    across filter changes and network switches. Sort by canonical
    timepoint order (screening → baseline → month1 → … → month24) and
    sample from Viridis so the color gradient reads as "earlier vs
    later" — alphabetical sort + qualitative palette would put month12
    and month6 in randomly unrelated hues.
    """
    def _order_key(tp):
        s = str(tp)
        if s in TIMEPOINT_ORDER:
            return (0, TIMEPOINT_ORDER.index(s))
        return (1, s)  # unknowns sorted after, in alpha order

    ordered = sorted(all_timepoints, key=_order_key)
    if not ordered:
        return {}
    samples = px.colors.sample_colorscale(
        'Viridis', [i / max(len(ordered) - 1, 1) for i in range(len(ordered))]
    )
    return {tp: samples[i] for i, tp in enumerate(ordered)}


def _resolve_networks(network_value):
    """Translate the network dropdown value into a list of networks."""
    if network_value == 'BOTH':
        return ['PRONET', 'PRESCIENT']
    return [network_value] if network_value else []


def _filter_date(df, start, end):
    if df.empty or 'date' not in df.columns:
        return df
    return df[(df['date'] >= start) & (df['date'] <= end)]


# -----------------------------
# Figure builders
# -----------------------------

def build_stacked_area(df, networks, start_date, end_date, timepoints,
                       tp_color_map):
    """
    Per-timepoint outstanding curves, stacked. Mirrors
    GraphErrors.create_stacked_line_graph but interactive. With
    `networks=['PRONET','PRESCIENT']`, each timepoint gets two stacks
    (solid PRONET + dotted PRESCIENT) on the same axes.
    """
    if df is None or df.empty:
        return _empty_figure('No stacked-area data available.')
    d = _filter_date(df[df['network'].isin(networks)].copy(), start_date, end_date)
    if timepoints:
        d = d[d['timepoint'].isin(timepoints)]
    if d.empty:
        return _empty_figure('No rows match the current filters.')

    fig = go.Figure()
    for net in networks:
        sub = d[d['network'] == net]
        if sub.empty:
            continue
        pivot = (
            sub.pivot_table(index='date', columns='timepoint',
                            values='outstanding', aggfunc='sum',
                            fill_value=0)
            .sort_index()
        )
        order = (
            pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
        )
        for tp in order:
            color = tp_color_map.get(str(tp), '#888')
            tp_label = str(tp) if str(tp) else '(blank)'
            y_vals = pivot[tp].values
            fig.add_trace(go.Scatter(
                x=pivot.index, y=y_vals,
                # `stackgroup` replaces the plotted y with the cumulative
                # stack height, and Plotly's `%{y}` in hover shows that
                # cumulative value — not the per-timepoint count the user
                # cares about. Pass the original per-component values as
                # `customdata` and reference `%{customdata}` so the
                # tooltip shows the actual outstanding count for THIS
                # timepoint at THIS date.
                customdata=y_vals,
                mode='lines',
                stackgroup=net,  # one stack per network
                name=(f'{tp_label} ({net})' if len(networks) > 1
                      else tp_label),
                legendgroup=str(tp),
                line=dict(width=0.5, color=color,
                          dash=NETWORK_DASH.get(net, 'solid')),
                # Self-contained tooltip — `hovermode='closest'` shows
                # one tooltip at a time, so we put the full context
                # (timepoint, network, date, value) inside this template
                # instead of relying on the unified-mode aggregation.
                # NB: Plotly's text parser only honors `style="..."` on
                # <span>, not <b> — so inline color via <b> is dropped.
                # The trace's color is already conveyed via the legend
                # swatch + hoverlabel border (set on the hoverlabel
                # config per-trace below).
                hovertemplate=(
                    f'<b>{tp_label}</b>  ({net})'
                    f'<br>%{{x|%b %d, %Y}}'
                    '<br>Outstanding: <b>%{customdata:,}</b>'
                    '<extra></extra>'
                ),
                # Per-trace bordercolor on the hoverlabel ties the
                # tooltip back to the timepoint visually.
                hoverlabel=dict(bordercolor=color),
                meta={'network': net},
            ))
    fig.update_yaxes(title='Outstanding Queries')
    fig.update_xaxes(title='Date', tickformat='%b %d, %Y')
    title_nets = ' + '.join(networks) if networks else 'No network'
    fig = _apply_layout(
        fig, f'{title_nets} Outstanding Queries Over Time by Timepoint'
    )
    # `x unified` produces a single tooltip listing ALL traces at the
    # cursor x — with 14 traces (BOTH × 7 timepoints) that tooltip is
    # tall enough to clip against the SVG boundary. `closest` shows
    # one tooltip for the trace under the cursor, which is exactly
    # what the user wants ("outstanding queries for that component").
    fig.update_layout(hovermode='closest')
    return fig


def build_line_graph(df, networks, start_date, end_date, series,
                     priority_share=False):
    """
    New / outstanding / resolved / priority over time. With multiple
    networks, each series is drawn once per network with distinct dash
    pattern. Optional priority-share overlay shows the
    priority/outstanding ratio on a secondary y-axis.
    """
    if df is None or df.empty:
        return _empty_figure('No line-graph data available.')
    d = _filter_date(df[df['network'].isin(networks)].copy(), start_date, end_date)
    if series:
        d = d[d['series'].isin(series)]
    if d.empty:
        return _empty_figure('No rows match the current filters.')

    fig = go.Figure()
    label_for = {
        'new': 'New Queries',
        'outstanding': 'Outstanding Queries',
        'resolved': 'Resolved Queries',
        'priority': 'Outstanding Priority Queries',
    }
    for net in networks:
        for s in ('outstanding', 'priority', 'new', 'resolved'):
            sub = d[(d['network'] == net) & (d['series'] == s)]
            if sub.empty:
                continue
            sub = sub.sort_values('date')
            label = label_for.get(s, s.title())
            name = f'{label} ({net})' if len(networks) > 1 else label
            fig.add_trace(go.Scatter(
                x=sub['date'], y=sub['count'],
                mode='lines+markers', name=name,
                legendgroup=s,
                line=dict(color=LINE_COLORS.get(s, '#aaa'), width=2,
                          dash=NETWORK_DASH.get(net, 'solid')),
                marker=dict(size=4),
                hovertemplate=(
                    f'%{{x|%Y-%m-%d}}<br>'
                    f'<b>{net}</b> — {label}: %{{y:,}}'
                    '<extra></extra>'
                ),
                meta={'network': net},
            ))

    if priority_share:
        for net in networks:
            net_df = df[df['network'] == net].copy()
            net_df = _filter_date(net_df, start_date, end_date)
            pivot = (
                net_df.pivot_table(index='date', columns='series',
                                   values='count', aggfunc='sum')
                .sort_index()
            )
            if 'priority' in pivot and 'outstanding' in pivot:
                ratio = (pivot['priority']
                         / pivot['outstanding'].replace(0, np.nan)) * 100
                fig.add_trace(go.Scatter(
                    x=ratio.index, y=ratio.values,
                    mode='lines',
                    name=(f'Priority share % ({net})'
                          if len(networks) > 1 else 'Priority share %'),
                    line=dict(color='#facc15', width=2,
                              dash=NETWORK_DASH.get(net, 'solid')),
                    yaxis='y2',
                    hovertemplate=(
                        f'%{{x|%Y-%m-%d}}<br>'
                        f'<b>{net}</b> — Priority share: '
                        '%{y:.1f}%<extra></extra>'
                    ),
                    meta={'network': net},
                ))
        fig.update_layout(
            yaxis2=dict(
                title='Priority share (%)', overlaying='y', side='right',
                rangemode='tozero', gridcolor='rgba(255,255,255,0.04)',
            )
        )

    fig.update_yaxes(title='Query Count')
    fig.update_xaxes(title='Date', tickformat='%b %d, %Y')
    title_nets = ' + '.join(networks) if networks else 'No network'
    return _apply_layout(fig, f'{title_nets} Queries Over Time')


def build_site_bar(df, networks):
    """
    Side-by-side stacked bars per site. With multiple networks, the
    chart emits 4 traces PER NETWORK (over_thirty, under_thirty,
    priority, non_priority) so the network-toggle button can hide /
    show per-network groups cleanly. Sites are prefixed by network
    only when both networks are visible.
    """
    if df is None or df.empty:
        return _empty_figure('No per-site data available.')
    d = df[df['network'].isin(networks)].copy()
    if d.empty:
        return _empty_figure(f'No per-site data for {networks}.')

    d['_total'] = d['over_thirty'] + d['under_thirty']
    d = d.sort_values(['network', '_total'],
                      ascending=[True, False]).drop(columns=['_total'])

    fig = go.Figure()
    bar_width = 0.4
    show_net_in_x = len(networks) > 1
    # Show network label in legend only when both are present so the
    # toggle and legend stay readable when only one is selected.
    show_net_in_legend = len(networks) > 1

    for net_idx, net in enumerate(networks):
        sub = d[d['network'] == net]
        if sub.empty:
            continue
        labels = ([f'{net} / {s}' for s in sub['site']]
                  if show_net_in_x else sub['site'].tolist())
        suffix = f' ({net})' if show_net_in_legend else ''

        fig.add_trace(go.Bar(
            x=labels, y=sub['over_thirty'],
            name=f'Detected Over 30 Days Ago{suffix}',
            marker_color=SITE_BAR_COLORS['over_thirty'],
            offset=-bar_width, width=bar_width,
            offsetgroup=f'age_{net}',
            legendgroup='over_thirty',
            showlegend=(net_idx == 0),
            hovertemplate=(
                f'%{{x}}<br><b>{net}</b> — >30d: %{{y:,}}'
                '<extra></extra>'
            ),
            meta={'network': net},
        ))
        fig.add_trace(go.Bar(
            x=labels, y=sub['under_thirty'],
            name=f'Detected Under 30 Days Ago{suffix}',
            marker_color=SITE_BAR_COLORS['under_thirty'],
            offset=-bar_width, width=bar_width,
            offsetgroup=f'age_{net}',
            base=sub['over_thirty'],
            legendgroup='under_thirty',
            showlegend=(net_idx == 0),
            hovertemplate=(
                f'%{{x}}<br><b>{net}</b> — ≤30d: %{{y:,}}'
                '<extra></extra>'
            ),
            meta={'network': net},
        ))
        fig.add_trace(go.Bar(
            x=labels, y=sub['priority'],
            name=f'High Priority{suffix}',
            marker_color=SITE_BAR_COLORS['priority'],
            offset=0, width=bar_width,
            offsetgroup=f'pri_{net}',
            legendgroup='priority',
            showlegend=(net_idx == 0),
            hovertemplate=(
                f'%{{x}}<br><b>{net}</b> — Priority: %{{y:,}}'
                '<extra></extra>'
            ),
            meta={'network': net},
        ))
        fig.add_trace(go.Bar(
            x=labels, y=sub['non_priority'],
            name=f'Lower Priority{suffix}',
            marker_color=SITE_BAR_COLORS['non_priority'],
            offset=0, width=bar_width,
            offsetgroup=f'pri_{net}',
            base=sub['priority'],
            legendgroup='non_priority',
            showlegend=(net_idx == 0),
            hovertemplate=(
                f'%{{x}}<br><b>{net}</b> — Non-priority: %{{y:,}}'
                '<extra></extra>'
            ),
            meta={'network': net},
        ))
    fig.update_layout(barmode='stack')
    fig.update_yaxes(title='Unresolved Queries')
    fig.update_xaxes(title='Site')

    title_nets = ' + '.join(networks) if networks else 'No network'
    fig = _apply_layout(
        fig,
        f'{title_nets} Queries Per Site '
        '(forms with unresolved queries are not uploaded to the NDA)'
    )
    # Site labels at -55deg + smaller font + tall bottom margin fit
    # ~30 BOTH-mode labels (`PRONET / BI`, …) without overlap.
    fig.update_xaxes(tickangle=-55, tickfont=dict(size=10))
    fig.update_layout(
        hovermode='closest',
        height=600,
        margin=dict(b=160),
    )
    return fig


# -----------------------------
# Dash app
# -----------------------------

pio.templates.default = 'plotly_dark'

external_stylesheets = [
    dbc.themes.CYBORG,
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap',
]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title='AMP SCZ QC Dashboard',
    suppress_callback_exceptions=True,
)
server = app.server

# Disable Dash's in-browser callback error overlay in production —
# users should see an empty figure with a generic message, not a
# Python traceback. Re-enable with `--debug` (which is already
# gated against 0.0.0.0 binds and shared hosts at __main__).
app.enable_dev_tools(
    debug=False, dev_tools_ui=False, dev_tools_props_check=False,
    dev_tools_serve_dev_bundles=False, dev_tools_hot_reload=False,
)


@server.route('/healthz')
def healthz():
    """Liveness probe: the process is up and serving."""
    return ('ok', 200, {'Content-Type': 'text/plain'})


@server.route('/readyz')
def readyz():
    """
    Readiness probe: returns 200 only if DATA has been built and the
    build is not older than `STALE_THRESHOLD_HOURS`. Returns 503
    otherwise so a load balancer / k8s won't route traffic until the
    cache is warm.
    """
    info = DATA.get('info', {})
    built_at = info.get('built_at')
    if not built_at:
        return ('not ready: no build', 503, {'Content-Type': 'text/plain'})
    try:
        built_dt = datetime.fromisoformat(built_at.replace('Z', ''))
        age_h = (datetime.now(timezone.utc).replace(tzinfo=None) - built_dt.replace(tzinfo=None)).total_seconds() / 3600
    except Exception:
        return ('not ready: bad build_at', 503,
                {'Content-Type': 'text/plain'})
    if age_h > STALE_THRESHOLD_HOURS:
        return (f'not ready: build is {age_h:.1f}h old (stale)',
                503, {'Content-Type': 'text/plain'})
    return (f'ok ({age_h:.1f}h old)', 200, {'Content-Type': 'text/plain'})

app.index_string = (
    """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      body{background-color:#0b0f14!important; font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
      .card{border:none; background-color:#0f1720}
      .Select-menu-outer{background-color:#0f1720 !important}
      .stale-fresh{color:#86efac}
      .stale-warn{color:#fca5a5}
    </style>
    </head><body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body></html>"""
)


def _kpi_card(label, value_id):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, id=f'{value_id}-label', className='text-muted small'),
        html.H4(id=value_id, className='fw-bold mb-0'),
    ]), className='shadow-sm'), md=3, sm=6, xs=12)


def build_sidebar():
    return dbc.Card([
        html.H2('AMP SCZ QC Dashboard', className='mb-1 fw-bold'),
        html.Hr(),
        html.H6('Chart', className='mt-2'),
        dcc.RadioItems(
            id='chart-menu',
            options=CHART_OPTIONS,
            value='stacked',
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block', 'marginBottom': '6px'},
            className='mb-3',
        ),
        html.H6('Network', className='mt-3'),
        dcc.Dropdown(
            id='network-select',
            options=[],   # populated by callback
            value='PRONET',
            clearable=False,
            className='mb-3',
        ),
        html.Div([
            html.H6('Date range', className='mt-3'),
            dcc.DatePickerRange(
                id='date-range',
                display_format='MMM D, YYYY',
                className='mb-3',
            ),
        ], id='date-range-wrap'),
        html.Div([
            dbc.Row([
                dbc.Col(html.H6('Timepoints', className='mt-3 mb-0'), width=6),
                dbc.Col([
                    dbc.Button('All', id='tp-all', size='sm',
                               color='secondary', outline=True,
                               className='me-1'),
                    dbc.Button('None', id='tp-none', size='sm',
                               color='secondary', outline=True),
                ], width=6, className='text-end pt-3'),
            ]),
            dcc.Dropdown(
                id='timepoint-select',
                options=[],
                value=[],
                multi=True,
                placeholder='All timepoints',
                className='mb-3',
            ),
        ], id='timepoint-wrap'),
        html.Div([
            dbc.Row([
                dbc.Col(html.H6('Series', className='mt-3 mb-0'), width=6),
                dbc.Col([
                    dbc.Button('All', id='ser-all', size='sm',
                               color='secondary', outline=True,
                               className='me-1'),
                    dbc.Button('None', id='ser-none', size='sm',
                               color='secondary', outline=True),
                ], width=6, className='text-end pt-3'),
            ]),
            dcc.Checklist(
                id='series-select',
                options=[{'label': ' ' + s.title(), 'value': s}
                         for s in SERIES_ORDER],
                value=SERIES_ORDER,
                labelStyle={'display': 'block'},
                inputStyle={'marginRight': '6px'},
                className='mb-2',
            ),
            dbc.Checklist(
                id='priority-share-toggle',
                options=[{'label': '  Overlay priority share %',
                          'value': 'on'}],
                value=[],
                inputStyle={'marginRight': '6px'},
                className='mb-3',
            ),
        ], id='series-wrap'),
        html.Hr(),
        dbc.Button('Refresh data', id='refresh-btn', color='secondary',
                   size='sm', className='w-100 mb-2'),
        dbc.Alert([
            html.Small(
                'Refresh walks Dropbox revisions newer than the cache '
                'and merges. After the first full walk, refreshes are '
                'fast (seconds). To force a full re-walk, delete '
                'cache/revisions/.'
            ),
        ], color='dark', className='mb-0 small'),
    ], body=True, className='shadow-sm')


def build_main():
    return dbc.Card([
        dbc.Row([
            dbc.Col([
                html.H4(id='chart-title', className='fw-semibold mb-0'),
                html.P(id='chart-subtitle', className='text-muted mb-0'),
            ], width=12),
        ], align='center', className='mb-3'),
        dbc.Row([
            _kpi_card('KPI 1', 'kpi-1'),
            _kpi_card('KPI 2', 'kpi-2'),
            _kpi_card('KPI 3', 'kpi-3'),
            _kpi_card('KPI 4', 'kpi-4'),
        ], className='g-2 mb-3'),
        dcc.Loading(
            id='loading-graph', type='circle',
            children=[dcc.Graph(id='main-graph',
                                config={'displaylogo': False,
                                        'toImageButtonOptions': {
                                            'format': 'png',
                                            'filename': 'qc_dashboard',
                                            'scale': 2}})],
        ),
        html.Div(id='data-source-line', className='text-muted small mt-2'),
        html.Hr(className='mt-3'),
        dbc.Row([
            dbc.Col(html.H6('Underlying data', className='mb-2'), width=8),
            dbc.Col(
                dbc.Button('Download CSV', id='download-btn',
                           color='secondary', size='sm', outline=True),
                width=4, className='text-end'),
        ]),
        dcc.Download(id='download-data'),
        html.Div(id='data-table-wrap'),
    ], body=True, className='shadow-sm')


app.layout = dbc.Container([
    dcc.Store(id='refresh-counter', data=0),
    dbc.Navbar(dbc.Container([
        dbc.NavbarBrand('QC Dashboard', className='fw-bold'),
        html.Div(id='freshness-banner', className='ms-3 small'),
        dbc.NavbarToggler(id='navbar-toggler'),
        dbc.Nav([
            dbc.NavLink('Plotly', href='https://plotly.com/python/',
                        target='_blank'),
        ], className='ms-auto'),
    ]), color='dark', dark=True, sticky='top', className='mb-4 shadow-sm'),
    dbc.Row([
        dbc.Col(build_sidebar(), md=3, lg=3, xl=3, className='mb-4'),
        dbc.Col(build_main(), md=9, lg=9, xl=9, className='mb-4'),
    ], className='g-3'),
], fluid=True, className='px-3')


# -----------------------------
# Callbacks
# -----------------------------

@app.callback(
    Output('refresh-counter', 'data'),
    Output('freshness-banner', 'children'),
    Input('refresh-btn', 'n_clicks'),
    State('refresh-counter', 'data'),
)
def on_refresh(n_clicks, counter):
    """
    Increment the refresh-counter Store so dependent callbacks fire.
    On startup `n_clicks` is None — emit the initial freshness banner
    without rebuilding the cache (DATA is already primed at import).
    """
    if n_clicks:
        refresh(force_refresh=True)
        counter = (counter or 0) + 1
    info = DATA.get('info', {})
    built_at = info.get('built_at')
    sources = info.get('sources', {})
    if built_at:
        try:
            built_dt = datetime.fromisoformat(built_at.replace('Z', ''))
            age_h = (datetime.now(timezone.utc).replace(tzinfo=None) - built_dt.replace(tzinfo=None)).total_seconds() / 3600
            stale = age_h > STALE_THRESHOLD_HOURS
            cls = 'stale-warn' if stale else 'stale-fresh'
            age_label = (f'{age_h:.1f}h old'
                         if age_h < 24 else f'{age_h/24:.1f}d old')
        except Exception:
            age_label, cls = '(age unknown)', ''
    else:
        age_label, cls = '(never built)', 'stale-warn'

    src_summary = ' / '.join(f'{n}: {s}' for n, s in sources.items()) \
        or 'no networks'
    banner = html.Span([
        html.Span('Data: ', className='text-muted'),
        html.Span(age_label, className=cls),
        html.Span(f' — {src_summary}', className='text-muted'),
    ])
    return counter, banner


@app.callback(
    Output('timepoint-select', 'options'),
    Output('timepoint-select', 'value'),
    Output('network-select', 'options'),
    Output('network-select', 'value'),
    Output('date-range', 'min_date_allowed'),
    Output('date-range', 'max_date_allowed'),
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date'),
    Input('refresh-counter', 'data'),
    State('network-select', 'value'),
    State('timepoint-select', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
)
def update_filter_options(_counter, current_net, current_tps,
                          current_start, current_end):
    """
    Rebuild sidebar dropdown options whenever DATA changes (initial
    load + every refresh). Preserves the user's current selection when
    possible — e.g. if the user picked 'baseline' and refresh adds a
    new timepoint, 'baseline' stays selected.
    """
    stacked = DATA['stacked']
    site = DATA['site']
    line = DATA['line']

    timepoints_all = sorted(stacked['timepoint'].dropna().unique().tolist()) \
        if not stacked.empty else []
    networks_all = sorted({
        *(stacked['network'].unique().tolist() if not stacked.empty else []),
        *(site['network'].unique().tolist() if not site.empty else []),
        *(line['network'].unique().tolist() if not line.empty else []),
    })

    if not stacked.empty and 'date' in stacked.columns:
        dates = pd.to_datetime(stacked['date'], errors='coerce').dropna()
        min_date = dates.min().date() if not dates.empty else None
        max_date = dates.max().date() if not dates.empty else None
    else:
        min_date, max_date = None, None

    tp_options = [{'label': tp if tp else '(blank)', 'value': tp}
                  for tp in timepoints_all]
    # Preserve selection across refresh; default to all on first load
    # when current_tps is None or [].
    if current_tps:
        kept = [tp for tp in current_tps if tp in timepoints_all]
        tp_value = kept if kept else timepoints_all
    else:
        tp_value = timepoints_all

    net_options = [{'label': n, 'value': n} for n in networks_all]
    if len(networks_all) > 1:
        net_options.append({'label': 'Both (overlay)', 'value': 'BOTH'})
    if current_net and (current_net in networks_all or current_net == 'BOTH'):
        net_value = current_net
    elif networks_all:
        net_value = networks_all[0]
    else:
        net_value = 'PRONET'

    # Date pickers — preserve user range when still within bounds.
    start_value = current_start or (min_date.isoformat() if min_date else None)
    end_value = current_end or (max_date.isoformat() if max_date else None)
    return (
        tp_options, tp_value,
        net_options, net_value,
        min_date, max_date, start_value, end_value,
    )


# Select-all / clear buttons for the timepoint multi-select.
@app.callback(
    Output('timepoint-select', 'value', allow_duplicate=True),
    Input('tp-all', 'n_clicks'),
    Input('tp-none', 'n_clicks'),
    State('timepoint-select', 'options'),
    prevent_initial_call=True,
)
def on_tp_buttons(all_clicks, none_clicks, options):
    from dash import ctx
    if not ctx.triggered_id:
        return no_update
    if ctx.triggered_id == 'tp-all':
        return [opt['value'] for opt in options]
    return []


# Select-all / clear for the series checklist.
@app.callback(
    Output('series-select', 'value', allow_duplicate=True),
    Input('ser-all', 'n_clicks'),
    Input('ser-none', 'n_clicks'),
    prevent_initial_call=True,
)
def on_series_buttons(all_clicks, none_clicks):
    from dash import ctx
    if ctx.triggered_id == 'ser-all':
        return SERIES_ORDER
    return []


# Filter visibility — hide controls that don't apply to the active chart.
@app.callback(
    Output('date-range-wrap', 'style'),
    Output('timepoint-wrap', 'style'),
    Output('series-wrap', 'style'),
    Input('chart-menu', 'value'),
)
def toggle_filter_visibility(chart_kind):
    shown, hidden = {}, {'display': 'none'}
    if chart_kind == 'stacked':
        return shown, shown, hidden
    if chart_kind == 'line':
        return shown, hidden, shown
    return hidden, hidden, hidden  # site bar


@app.callback(
    Output('main-graph', 'figure'),
    Output('chart-title', 'children'),
    Output('chart-subtitle', 'children'),
    Output('data-source-line', 'children'),
    Output('kpi-1', 'children'),
    Output('kpi-1-label', 'children'),
    Output('kpi-2', 'children'),
    Output('kpi-2-label', 'children'),
    Output('kpi-3', 'children'),
    Output('kpi-3-label', 'children'),
    Output('kpi-4', 'children'),
    Output('kpi-4-label', 'children'),
    Output('data-table-wrap', 'children'),
    Input('refresh-counter', 'data'),
    Input('chart-menu', 'value'),
    Input('network-select', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('timepoint-select', 'value'),
    Input('series-select', 'value'),
    Input('priority-share-toggle', 'value'),
)
def update_chart(_counter, chart_kind, network, start_date, end_date,
                 timepoints, series, priority_share_value):
    """
    Single render path for all three chart types. Wrapped in try/except
    so a builder fault returns an empty figure with a generic error
    message rather than letting Dash spill a traceback to the browser.
    """
    try:
        return _update_chart_inner(
            chart_kind, network, start_date, end_date,
            timepoints, series, priority_share_value,
        )
    except Exception as e:
        # Log the full traceback for the operator; surface a generic
        # message to the user.
        print(f'[update_chart] error: {type(e).__name__}: {e}')
        print(traceback.format_exc())
        return (
            _empty_figure('An error occurred rendering this chart. '
                          'Check the server log.'),
            'Error', '', f'Server error ({type(e).__name__}).',
            '—', 'KPI 1', '—', 'KPI 2', '—', 'KPI 3', '—', 'KPI 4',
            html.Div('—', className='text-muted small'),
        )


def _update_chart_inner(chart_kind, network, start_date, end_date,
                        timepoints, series, priority_share_value):
    stacked = DATA['stacked']
    line = DATA['line']
    site = DATA['site']
    info = DATA['info']

    networks = _resolve_networks(network)
    if not networks:
        return (
            _empty_figure('No network selected.'),
            'No data', '', '',
            '—', 'KPI 1', '—', 'KPI 2', '—', 'KPI 3', '—', 'KPI 4',
            html.Div(),
        )

    # Date range with safe defaults.
    if start_date:
        start_ts = pd.to_datetime(start_date)
    elif not stacked.empty:
        start_ts = stacked['date'].min()
    else:
        start_ts = pd.Timestamp('2026-04-15')
    if end_date:
        end_ts = pd.to_datetime(end_date)
    elif not stacked.empty:
        end_ts = stacked['date'].max()
    else:
        end_ts = pd.Timestamp(datetime.today().date())

    tp_color_map = _timepoint_color_map(
        stacked['timepoint'].dropna().unique().tolist()
        if not stacked.empty else []
    )

    if chart_kind == 'stacked':
        fig = build_stacked_area(
            stacked, networks, start_ts, end_ts, timepoints, tp_color_map,
        )
        title = f'{" + ".join(networks)} — Outstanding by Timepoint'
        filtered = stacked[
            stacked['network'].isin(networks)
            & (stacked['date'] >= start_ts)
            & (stacked['date'] <= end_ts)
        ]
        if timepoints:
            filtered = filtered[filtered['timepoint'].isin(timepoints)]
        n_rows = len(filtered)
    elif chart_kind == 'line':
        priority_share = 'on' in (priority_share_value or [])
        fig = build_line_graph(
            line, networks, start_ts, end_ts, series, priority_share,
        )
        title = f'{" + ".join(networks)} — Queries Over Time'
        filtered = line[
            line['network'].isin(networks)
            & (line['date'] >= start_ts)
            & (line['date'] <= end_ts)
        ]
        if series:
            filtered = filtered[filtered['series'].isin(series)]
        n_rows = len(filtered)
    else:
        fig = build_site_bar(site, networks)
        title = f'{" + ".join(networks)} — Queries Per Site'
        filtered = site[site['network'].isin(networks)]
        n_rows = len(filtered)

    if chart_kind in ('stacked', 'line'):
        subtitle = (
            f'{n_rows:,} rows | '
            f'{start_ts.date()} → {end_ts.date()}'
        )
    else:
        subtitle = f'{n_rows:,} sites'

    src_lookup = {
        'history': 'Dropbox revision history (full lifecycle)',
        'tracker': 'current tracker snapshot only (no revision walk)',
        'synthetic': 'SYNTHETIC DATA — Dropbox/trackers unreachable',
    }
    src_parts = []
    for net in networks:
        s = info.get('sources', {}).get(net, 'unknown')
        src_parts.append(f'{net}: {src_lookup.get(s, s)}')
    source_line = ' • '.join(src_parts)

    kpis = _compute_kpis(chart_kind, line, site, networks, end_ts)

    table = _build_data_table(filtered, chart_kind)
    return (
        fig, title, subtitle, source_line,
        kpis[0][0], kpis[0][1],
        kpis[1][0], kpis[1][1],
        kpis[2][0], kpis[2][1],
        kpis[3][0], kpis[3][1],
        table,
    )


def _compute_kpis(chart_kind, line, site, networks, as_of):
    """
    KPIs differ by chart context so the strip stays relevant:
      - stacked / line: Outstanding (today), Priority (today),
        New 7d avg, Resolved 7d avg
      - site: Total sites, Total outstanding, Top site, %Sites > 30 priority
    Returns 4-tuple of (value_str, label_str) tuples.
    """
    if chart_kind == 'site':
        if site.empty:
            return [('—', 'Total sites')] * 4
        d = site[site['network'].isin(networks)]
        if d.empty:
            return [('—', 'Total sites')] * 4
        total_sites = len(d)
        total_outstanding = int(
            (d['over_thirty'] + d['under_thirty']).sum()
        )
        d_total = d.assign(_t=d['over_thirty'] + d['under_thirty'])
        top_idx = d_total['_t'].idxmax()
        top_site = (f'{d_total.loc[top_idx, "site"]} '
                    f'({int(d_total.loc[top_idx, "_t"]):,})')
        high_priority_sites = (d['priority'] > 30).sum()
        share_pct = (high_priority_sites / total_sites * 100
                     if total_sites else 0)
        return [
            (f'{total_sites:,}', 'Total sites'),
            (f'{total_outstanding:,}', 'Total outstanding'),
            (top_site, 'Top site (by volume)'),
            (f'{share_pct:.0f}%', 'Sites with >30 priority'),
        ]

    if line.empty:
        return [
            ('—', 'Outstanding (today)'),
            ('—', 'Priority (today)'),
            ('—', 'New (7d avg)'),
            ('—', 'Resolved (7d avg)'),
        ]
    d = line[line['network'].isin(networks)]

    def _today(s):
        sub = d[d['series'] == s]
        if sub.empty:
            return '—'
        sub = sub.sort_values('date')
        # Sum across networks for "Both" mode.
        last_day = sub['date'].max()
        return f'{int(sub[sub["date"] == last_day]["count"].sum()):,}'

    def _seven_day_avg(s):
        sub = d[d['series'] == s]
        if sub.empty:
            return '—'
        sub = sub.sort_values('date')
        cutoff = sub['date'].max() - pd.Timedelta(days=6)
        recent = sub[sub['date'] >= cutoff]
        if recent.empty:
            return '—'
        # Per-day sum across networks first, then mean across days.
        per_day = recent.groupby('date')['count'].sum()
        return f'{per_day.mean():.1f}'

    return [
        (_today('outstanding'), 'Outstanding (today)'),
        (_today('priority'), 'Priority (today)'),
        (_seven_day_avg('new'), 'New (7d avg)'),
        (_seven_day_avg('resolved'), 'Resolved (7d avg)'),
    ]


TABLE_PAGE_SIZE = 15


def _build_data_table(df, chart_kind):
    """
    Sortable / filterable data table beneath the chart, paginated and
    filtered server-side. The previous client-side `head(500)` cap
    silently truncated filters at large dataset sizes — a user who
    filtered for "month12" while a 100k-row slice was loaded would
    only see rows present in the first 500. Server-side pagination
    fixes that: the table receives just the current page, and the
    full slice is the filter/sort target.

    A dcc.Store (`table-source-store`) holds the JSON of the active
    slice along with the chart_kind, so the paging callback knows
    which columns to surface. Capped at 50k rows in the store to
    bound the per-client payload; the Download CSV button bypasses
    the cap.
    """
    if df is None or df.empty:
        return html.Div('No rows in current selection.',
                        className='text-muted small')
    display_df = df.copy()
    if 'date' in display_df.columns:
        display_df['date'] = display_df['date'].dt.date.astype(str)

    # Cap stored payload — 50k rows × ~6 columns at ~10 bytes each is
    # ~3 MB of JSON, an acceptable per-client transfer.
    max_store_rows = 50_000
    truncated = len(display_df) > max_store_rows
    stored_df = display_df.head(max_store_rows) if truncated else display_df

    note_text = (
        f'Filter / sort apply to the full {len(display_df):,}-row slice. '
        f'Browser cache limited to {max_store_rows:,} rows.'
        if truncated else
        f'{len(display_df):,} rows. Filter / sort apply to the full slice.'
    )

    columns = [{'name': c, 'id': c} for c in stored_df.columns]
    return html.Div([
        html.Div(note_text, className='text-muted small mb-2'),
        dcc.Store(id='table-source-store',
                  data=stored_df.to_dict('records')),
        dash_table.DataTable(
            id='data-table',
            columns=columns,
            data=[],
            sort_action='custom',
            sort_mode='single',
            sort_by=[],
            filter_action='custom',
            filter_query='',
            page_action='custom',
            page_current=0,
            page_size=TABLE_PAGE_SIZE,
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': CARD_BG,
                'color': '#e5e7eb',
                'border': '1px solid rgba(255,255,255,0.05)',
                'fontSize': '12px',
                'padding': '6px',
            },
            style_header={
                'backgroundColor': '#1a2330',
                'fontWeight': 'bold',
                'border': '1px solid rgba(255,255,255,0.1)',
            },
            style_filter={'backgroundColor': '#15202b'},
        ),
    ])


@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'page_count'),
    Input('data-table', 'page_current'),
    Input('data-table', 'page_size'),
    Input('data-table', 'sort_by'),
    Input('data-table', 'filter_query'),
    State('table-source-store', 'data'),
    prevent_initial_call=False,
)
def page_data_table(page_current, page_size, sort_by, filter_query,
                    source_data):
    """
    Apply filter -> sort -> page to the full stored slice. Filter
    syntax is Dash's stock `{column} contains "x"` form; we translate
    it to a pandas boolean mask. Handles missing source data
    gracefully (empty page).
    """
    if not source_data:
        return [], 1
    df = pd.DataFrame(source_data)
    if df.empty:
        return [], 1

    if filter_query:
        for token in filter_query.split(' && '):
            token = token.strip()
            if not token:
                continue
            try:
                col, op_value = _parse_filter_token(token)
                if col not in df.columns:
                    continue
                df = _apply_filter(df, col, op_value)
            except Exception:
                continue

    if sort_by:
        df = df.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            na_position='last',
        )

    page_size = page_size or TABLE_PAGE_SIZE
    total = max(1, (len(df) + page_size - 1) // page_size)
    page_current = min(max(0, page_current or 0), total - 1)
    start = page_current * page_size
    page = df.iloc[start:start + page_size]
    return page.to_dict('records'), total


def _parse_filter_token(token):
    """
    Parse a Dash filter token into (column, (op, value)). Recognizes
    `contains`, `=`, `>`, `>=`, `<`, `<=`. Falls back to substring
    match. Returns None for malformed tokens.
    """
    import re
    m = re.match(r'\{([^}]+)\}\s*(contains|>=|<=|=|>|<)\s*"?([^"]*)"?',
                 token)
    if not m:
        m2 = re.match(r'\{([^}]+)\}\s*(contains|>=|<=|=|>|<)\s*(\S+)',
                      token)
        if not m2:
            raise ValueError(f'cannot parse filter token: {token}')
        col, op, val = m2.group(1), m2.group(2), m2.group(3)
    else:
        col, op, val = m.group(1), m.group(2), m.group(3)
    return col, (op, val)


def _apply_filter(df, col, op_value):
    op, val = op_value
    if op == 'contains':
        return df[df[col].astype(str).str.contains(val, case=False, na=False)]
    try:
        val_num = float(val)
        col_num = pd.to_numeric(df[col], errors='coerce')
        if op == '=':
            return df[col_num == val_num]
        if op == '>':
            return df[col_num > val_num]
        if op == '>=':
            return df[col_num >= val_num]
        if op == '<':
            return df[col_num < val_num]
        if op == '<=':
            return df[col_num <= val_num]
    except ValueError:
        # Non-numeric comparison: string equality only.
        if op == '=':
            return df[df[col].astype(str) == val]
    return df


@app.callback(
    Output('download-data', 'data'),
    Input('download-btn', 'n_clicks'),
    State('chart-menu', 'value'),
    State('network-select', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('timepoint-select', 'value'),
    State('series-select', 'value'),
    prevent_initial_call=True,
)
def on_download(n_clicks, chart_kind, network, start_date, end_date,
                timepoints, series):
    if not n_clicks:
        return no_update
    networks = _resolve_networks(network)
    if chart_kind == 'stacked':
        df = DATA['stacked']
        d = df[df['network'].isin(networks)]
        if start_date:
            d = d[d['date'] >= pd.to_datetime(start_date)]
        if end_date:
            d = d[d['date'] <= pd.to_datetime(end_date)]
        if timepoints:
            d = d[d['timepoint'].isin(timepoints)]
        name = f'stacked_{"_".join(networks)}.csv'
    elif chart_kind == 'line':
        df = DATA['line']
        d = df[df['network'].isin(networks)]
        if start_date:
            d = d[d['date'] >= pd.to_datetime(start_date)]
        if end_date:
            d = d[d['date'] <= pd.to_datetime(end_date)]
        if series:
            d = d[d['series'].isin(series)]
        name = f'line_{"_".join(networks)}.csv'
    else:
        df = DATA['site']
        d = df[df['network'].isin(networks)]
        name = f'site_{"_".join(networks)}.csv'
    return dcc.send_data_frame(d.to_csv, name, index=False)


# -----------------------------
# Static HTML export
# -----------------------------

def export_static_html(out_path='qc_dashboard.html', network='BOTH'):
    """
    Bundle the three datasets + Plotly JS into a self-contained HTML
    file. Default is `BOTH` so each chart includes both networks plus
    a "PRONET / PRESCIENT / Both" toggle button group at the top.

    `network='PRONET'` or `'PRESCIENT'` bakes in a single network with
    no toggle (smaller file).
    """
    out_path = Path(out_path)
    stacked = DATA['stacked']
    line = DATA['line']
    site = DATA['site']

    networks = _resolve_networks(network)
    if stacked.empty:
        start_ts = pd.Timestamp('2026-04-15')
        end_ts = pd.Timestamp(datetime.today().date())
    else:
        start_ts = stacked['date'].min()
        end_ts = stacked['date'].max()

    tp_color_map = _timepoint_color_map(
        stacked['timepoint'].dropna().unique().tolist()
        if not stacked.empty else []
    )

    stacked_fig = build_stacked_area(
        stacked, networks, start_ts, end_ts, None, tp_color_map,
    )
    line_fig = build_line_graph(
        line, networks, start_ts, end_ts, None, priority_share=False,
    )
    site_fig = build_site_bar(site, networks)

    # Attach per-chart PRONET / PRESCIENT / Both toggle buttons. The
    # `_add_network_toggle` helper is a no-op when only one network is
    # present, so single-network exports skip it gracefully.
    for fig in (stacked_fig, line_fig, site_fig):
        _add_network_toggle(fig)

    # Per-chart heights tuned for the static export: time-series want
    # vertical room for ~14 traces of legend; the site bar wants room
    # for ~30 rotated labels at the bottom.
    stacked_fig.update_layout(height=620)
    line_fig.update_layout(height=620)
    # site_fig height already set inside build_site_bar.

    pretty_now = datetime.now().strftime('%Y-%m-%d %H:%M')
    date_range_label = (
        f'{start_ts.strftime("%b %d")} → '
        f'{end_ts.strftime("%b %d, %Y")}'
    )

    def _subtitle(text):
        return f'<p class="card-subtitle">{text}</p>'

    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>AMP SCZ QC Dashboard</title>',
        # Inline SVG favicon — three accent bars matching the page
        # palette. Self-contained, no extra request.
        '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;utf8,'
        '<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22>'
        '<rect width=%2232%22 height=%2232%22 rx=%226%22 fill=%22%230a0e13%22/>'
        '<rect x=%227%22 y=%2218%22 width=%224%22 height=%228%22 fill=%22%2360a5fa%22/>'
        '<rect x=%2214%22 y=%2212%22 width=%224%22 height=%2214%22 fill=%22%23a78bfa%22/>'
        '<rect x=%2221%22 y=%226%22 width=%224%22 height=%2220%22 fill=%22%23f472b6%22/>'
        '</svg>">',
        # Inter font + Bootswatch (the previous version named Inter in
        # CSS but never `<link>`d the font — silent fallback to system).
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">',
        '<link href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/cyborg/bootstrap.min.css" rel="stylesheet">',
        _static_html_css(),
        '</head><body>',
        # Sticky header: title bar + metadata chips ride the top of the
        # viewport while the user scrolls through the charts. The
        # network names render as a small uppercase pill instead of an
        # inline span so they read as metadata, not a weak title echo.
        '<header class="page-header">',
        '<h1>AMP SCZ QC Dashboard '
        f'<span class="net-tag">{" / ".join(networks)}</span>'
        '</h1>',
        _meta_chips_html(networks),
        '</header>',
        '<main class="container">',
        '<section class="card">',
        _subtitle(f'Outstanding queries per timepoint &middot; {date_range_label}'),
        stacked_fig.to_html(
            full_html=False, include_plotlyjs='cdn',
            config=PLOTLY_CONFIG,
        ),
        '</section><section class="card">',
        _subtitle(f'Daily new, outstanding, resolved &amp; priority &middot; {date_range_label}'),
        line_fig.to_html(
            full_html=False, include_plotlyjs=False,
            config=PLOTLY_CONFIG,
        ),
        '</section><section class="card">',
        _subtitle('Per-site totals as of today &middot; '
                  'sites with unresolved queries are held back from the NDA'),
        site_fig.to_html(
            full_html=False, include_plotlyjs=False,
            config=PLOTLY_CONFIG,
        ),
        '</section>',
        '</main>',
        '<footer class="page-footer">'
        f'<span>AMP SCZ QC Dashboard &middot; rendered {pretty_now}</span>'
        f'<span>Source: <code>{"/".join(networks)}_Output_V2.xlsx</code> '
        'via Dropbox revision history</span>'
        '</footer>',
        '</body></html>',
    ]
    out_path.write_text(''.join(parts), encoding='utf-8')
    return out_path


def _static_html_css():
    """
    Tokens + component styles for the static export. Plotly hard-codes
    a white fill on the active updatemenu button; the `.updatemenu-*`
    rules are `!important` overrides — keep colors in sync with the
    `ELEVATED_BG` / `ACCENT` constants above.
    """
    return f"""<style>
:root {{
  --surface-0: {DARK_BG};
  --surface-1: {CARD_BG};
  --surface-2: {ELEVATED_BG};
  --surface-3: #232f3f;
  --text-primary: {TEXT_PRIMARY};
  --text-secondary: {TEXT_SECONDARY};
  --text-muted: {TEXT_MUTED};
  --accent: {ACCENT};
  --accent-soft: rgba(96, 165, 250, 0.18);
  --accent-glow: rgba(96, 165, 250, 0.35);
  --border-soft: rgba(255, 255, 255, 0.06);
  --border-strong: rgba(255, 255, 255, 0.14);
  --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.45),
                 0 8px 22px rgba(0, 0, 0, 0.28);
  --shadow-card-hover: 0 2px 4px rgba(0, 0, 0, 0.5),
                       0 14px 32px rgba(0, 0, 0, 0.35);
  --transition-fast: 120ms cubic-bezier(0.2, 0.7, 0.2, 1);
  --transition-base: 200ms cubic-bezier(0.2, 0.7, 0.2, 1);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  padding: 0;
  /* Layered background: radial accent + faint dot-grid + base surface.
     The 1.8%-opacity dots give the dark surface a real-product texture
     instead of a flat void (Linear / Vercel / Stripe all do something
     similar at ~1-2% opacity). */
  background:
    radial-gradient(1200px 600px at 50% -200px,
                    rgba(96, 165, 250, 0.06),
                    transparent 60%),
    radial-gradient(circle at center,
                    rgba(255, 255, 255, 0.018) 1px,
                    transparent 1.5px) 0 0 / 22px 22px,
    var(--surface-0);
  color: var(--text-secondary);
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto,
               Helvetica, Arial, sans-serif;
  font-size: 0.875rem;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-feature-settings: "ss01", "cv01", "cv11", "tnum";
  /* Tabular nums everywhere a number sits — axes ticks, legend
     entries, hovertemplate counts. Single biggest "looks like a
     real product" lever on a numeric dashboard. */
  font-variant-numeric: tabular-nums;
}}
.js-plotly-plot text {{
  font-variant-numeric: tabular-nums;
}}
main.container {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 1rem 2rem;
}}
header.page-header {{
  position: sticky;
  top: 0;
  z-index: 20;
  margin: 0 auto 1.5rem;
  padding: 1.25rem 1rem 1rem;
  max-width: 1280px;
  background: linear-gradient(180deg,
                              rgba(10, 14, 19, 0.95) 0%,
                              rgba(10, 14, 19, 0.85) 70%,
                              rgba(10, 14, 19, 0) 100%);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}}
header.page-header h1 {{
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.15;
  color: var(--text-primary);
  margin: 0 0 0.6rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}}
header.page-header h1::before {{
  content: '';
  display: inline-block;
  width: 3px;
  height: 1.6rem;
  background: var(--accent);
  border-radius: 2px;
  box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.18);
}}
.net-tag {{
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 0.18rem 0.55rem;
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  vertical-align: 0.25em;
  background: rgba(28, 37, 49, 0.6);
}}
.meta-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}}
.meta-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.3rem 0.75rem;
  background: var(--surface-2);
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  font-size: 0.75rem;
  color: var(--text-secondary);
  transition: background var(--transition-fast),
              border-color var(--transition-fast),
              transform var(--transition-fast);
}}
.meta-chip:hover {{
  background: var(--surface-3);
  border-color: var(--border-strong);
  transform: translateY(-1px);
}}
.meta-chip strong {{
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.625rem;
}}
/* 1px vertical rule separator between the strong-label and value.
   Middle-dot looks fussy, en-dash reads like a hyphen at this size. */
.meta-chip strong::after {{
  content: '';
  display: inline-block;
  width: 1px;
  height: 0.8em;
  background: var(--border-strong);
  margin: 0 0.55rem -0.1em;
  opacity: 0.7;
}}
.meta-chip .dot {{
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent-glow);
  flex-shrink: 0;
}}
.meta-chip[data-source="synthetic"] .dot {{ background: #f59e0b; }}
.meta-chip[data-source="tracker"]   .dot {{ background: #a78bfa; }}
.meta-chip[data-source="history"]   .dot {{ background: #34d399; }}
/* Synthetic data is a "fix me" state — a soft pulse on the amber dot
   so it's impossible to mistake fixture data for real history. The
   pulse self-extinguishes the moment the source flips to `history`
   because the rule is data-attribute-scoped. */
@keyframes synthetic-pulse {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.55); }}
  50%      {{ box-shadow: 0 0 0 5px rgba(245, 158, 11, 0); }}
}}
.meta-chip[data-source="synthetic"] .dot {{
  animation: synthetic-pulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}}
/* Network line-style legend chip — shows the dash convention (solid
   PRONET vs dotted PRESCIENT) right in the header so users don't
   need to discover it from the right-side legend. */
.meta-chip svg.netline {{
  width: 22px; height: 6px; flex-shrink: 0;
}}

.card-subtitle {{
  margin: -0.25rem 0 0.75rem;
  font-size: 0.8125rem;
  color: var(--text-muted);
  letter-spacing: 0;
}}

section.card {{
  position: relative;
  background: var(--surface-1);
  border: 1px solid var(--border-soft);
  border-radius: 14px;
  margin-bottom: 1.5rem;
  padding: 1.25rem 1.5rem 1rem;
  /* `visible` so Plotly's hover tooltip layer isn't clipped by the
     card boundary when the cursor lands near an edge. */
  overflow: visible;
  box-shadow: var(--shadow-card);
  transition: transform var(--transition-base),
              box-shadow var(--transition-base),
              border-color var(--transition-base);
  animation: card-rise 420ms cubic-bezier(0.2, 0.7, 0.2, 1) both;
}}
/* Staggered fade-up on first paint — the signature "settle into
   place" feel. `both` fill-mode means cards stay in their final
   state under prefers-reduced-motion (the universal override at the
   bottom of this stylesheet zeroes the animation duration). */
@keyframes card-rise {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
section.card:nth-of-type(1) {{ animation-delay:  40ms; }}
section.card:nth-of-type(2) {{ animation-delay: 120ms; }}
section.card:nth-of-type(3) {{ animation-delay: 200ms; }}
section.card:hover {{
  transform: translateY(-1px);
  border-color: rgba(255, 255, 255, 0.10);
  box-shadow: var(--shadow-card-hover);
}}
/* Top accent line — subtle gradient bar that gives each card a touch
   of identity without competing with the chart palette. */
section.card::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 1.25rem;
  right: 1.25rem;
  height: 2px;
  background: linear-gradient(90deg,
                              transparent 0%,
                              rgba(96, 165, 250, 0.4) 18%,
                              rgba(167, 139, 250, 0.4) 50%,
                              rgba(244, 114, 182, 0.4) 82%,
                              transparent 100%);
  border-radius: 0 0 2px 2px;
  opacity: 0.7;
}}
/* Plotly's inner divs sometimes clip; make sure tooltips can escape.
   SVG defaults to overflow:hidden per CSS spec — letting the hover
   layer extend beyond the SVG bounds is the only reliable way to
   stop tooltips clipping when the cursor is near an edge. */
.js-plotly-plot,
.plot-container,
.plot-container .svg-container,
.js-plotly-plot .main-svg,
.js-plotly-plot .hoverlayer {{
  overflow: visible !important;
}}
.js-plotly-plot .hoverlabel {{
  font-family: 'Inter', system-ui, sans-serif !important;
}}
.js-plotly-plot .hoverlabel text {{
  font-variant-numeric: tabular-nums;
}}

footer.page-footer {{
  max-width: 1280px;
  margin: 2rem auto 0;
  padding: 1rem 1rem 0.75rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  /* Gradient bookend mirroring the card top accent — the page opens
     with a blue accent bar on the H1 and closes with the same
     blue→violet→pink ramp above the footer. */
  border-top: none;
  background-image: linear-gradient(90deg,
                                    transparent 0%,
                                    rgba(96, 165, 250, 0.4) 18%,
                                    rgba(167, 139, 250, 0.4) 50%,
                                    rgba(244, 114, 182, 0.4) 82%,
                                    transparent 100%);
  background-size: 100% 1px;
  background-repeat: no-repeat;
  background-position: top;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}}
footer.page-footer code {{
  font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.72rem;
  padding: 0.1rem 0.4rem;
  background: var(--surface-2);
  border: 1px solid var(--border-soft);
  border-radius: 4px;
  color: var(--text-secondary);
}}
footer.page-footer a {{
  color: var(--text-secondary);
  text-decoration: none;
  border-bottom: 1px dashed var(--border-strong);
  transition: color var(--transition-fast),
              border-color var(--transition-fast);
}}
footer.page-footer a:hover {{
  color: var(--text-primary);
  border-color: var(--accent);
}}

/* ------ Plotly updatemenu pill overrides ------ */
.updatemenu-item-rect {{
  fill: var(--surface-2) !important;
  stroke: var(--border-strong) !important;
  rx: 8 !important;
  ry: 8 !important;
}}
.updatemenu-item-text {{
  fill: var(--text-secondary) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-weight: 500 !important;
}}
/* Subtle hover feedback even on the inactive pills. */
.updatemenu-button:hover .updatemenu-item-rect {{
  fill: var(--surface-3) !important;
  stroke: rgba(255, 255, 255, 0.22) !important;
}}
/* Faint accent highlight on the actually-selected button. The button
   tracks user clicks because each button bundles
   `{{updatemenus[0].active: idx}}` in its args. Opacity bumped from
   0.18 to 0.28 + stroke 2px so the selected pill reads as definitely
   selected on the dark surface without being aggressive. */
.updatemenu-button.updatemenu-active-button .updatemenu-item-rect {{
  fill: rgba(96, 165, 250, 0.28) !important;
  stroke: var(--accent) !important;
  stroke-width: 2 !important;
}}
.updatemenu-button.updatemenu-active-button .updatemenu-item-text {{
  fill: #cfe1ff !important;
  font-weight: 600 !important;
}}

/* Keyboard focus rings — required for WCAG 2.4.7. The dashboard's
   meta-chips are spans (non-tabbable unless a future change adds
   `tabindex`), but the rule is forward-compatible. */
.meta-chip:focus-visible,
.updatemenu-button:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 999px;
}}
footer.page-footer a:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 3px;
  border-bottom-color: transparent;
}}

/* Hide bootstrap's default heading reset that the cyborg theme drags
   in — keeps our tokens authoritative. */
h1, h2, h3, h4, h5, h6 {{ font-family: inherit; }}

/* ------ Print stylesheet ------ */
@media print {{
  body {{
    background: white;
    color: #111;
  }}
  header.page-header {{
    position: static;
    background: white;
    border-bottom-color: #ddd;
    backdrop-filter: none;
  }}
  header.page-header h1 {{ color: #111; }}
  header.page-header h1::before {{
    background: #111;
    box-shadow: none;
  }}
  section.card {{
    background: white;
    border-color: #ddd;
    box-shadow: none;
    page-break-inside: avoid;
  }}
  section.card:hover {{ transform: none; box-shadow: none; }}
  section.card::before {{ display: none; }}
  .meta-chip {{
    background: white;
    border-color: #ddd;
    color: #333;
  }}
  footer.page-footer {{ border-color: #ddd; color: #555; }}
}}

/* ------ Reduced motion ------ */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    transition-duration: 0ms !important;
    animation-duration: 0ms !important;
  }}
  html {{ scroll-behavior: auto; }}
}}
</style>"""


def _meta_chips_html(networks):
    """
    Build metadata as pill chips with a colored status dot per network
    (green=history, purple=tracker, amber=synthetic). The dot color is
    driven by the chip's `data-source` attribute via CSS so the styles
    live in `_static_html_css()`. A trailing chip per network legends
    the line-dash convention (solid PRONET vs dotted PRESCIENT) so
    users learn it from the header instead of the right-side legend.
    """
    info = DATA.get('info', {})
    built_at = info.get('built_at') or 'unknown'
    # Trim ISO timestamp to "2026-05-13 17:42 UTC" for readability.
    pretty_built = built_at
    try:
        if built_at and built_at != 'unknown':
            t = built_at.replace('Z', '').split('.')[0]
            t = datetime.fromisoformat(t)
            pretty_built = t.strftime('%Y-%m-%d %H:%M UTC')
    except Exception:
        pass
    sources = info.get('sources') or {}
    chips = [
        f'<span class="meta-chip"><span class="dot"></span>'
        f'<strong>Built</strong>{pretty_built}</span>'
    ]
    for net in networks:
        s = sources.get(net, 'unknown')
        chips.append(
            f'<span class="meta-chip" data-source="{s}">'
            f'<span class="dot"></span>'
            f'<strong>{net}</strong>{s}</span>'
        )
    # Line-style legend chips. Only emitted in BOTH mode where the
    # solid/dotted convention actually distinguishes networks.
    if len(networks) > 1:
        for net in networks:
            dash = NETWORK_DASH.get(net, 'solid')
            dash_attr = 'stroke-dasharray="3,3"' if dash != 'solid' else ''
            chips.append(
                f'<span class="meta-chip">'
                f'<svg class="netline" viewBox="0 0 22 6">'
                f'<line x1="0" y1="3" x2="22" y2="3" '
                f'stroke="#cbd5e1" stroke-width="1.5" {dash_attr}/>'
                f'</svg><strong>{net}</strong>line style</span>'
            )
    return f'<div class="meta-row">{"".join(chips)}</div>'


# -----------------------------
# Entry point
# -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true',
                        help='Rebuild data cache before starting (slow).')
    parser.add_argument('--export-html', nargs='?', const='qc_dashboard.html',
                        help='Also write a standalone HTML.')
    parser.add_argument('--export-html-only', nargs='?',
                        const='qc_dashboard.html',
                        help='Write standalone HTML and exit.')
    parser.add_argument('--export-network', default='BOTH',
                        help='Network for static HTML export (PRONET / '
                             'PRESCIENT / BOTH). Default is BOTH so the '
                             'HTML carries both networks with a per-chart '
                             'toggle button group.')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--debug', action='store_true',
                        help='Enable Dash debug UI. Requires '
                             '--allow-unsafe-debug to acknowledge that '
                             'the Werkzeug debugger is RCE-via-PIN on '
                             'any local user, even on 127.0.0.1.')
    parser.add_argument('--allow-unsafe-debug', action='store_true',
                        help='Acknowledge the Werkzeug debugger risks.')
    args = parser.parse_args()

    # Refuse to serve the Werkzeug debugger without explicit
    # acknowledgement and a loopback bind. Combined with the dev-tools
    # disable at import time, this means production deploys cannot
    # accidentally expose the interactive debugger.
    if args.debug:
        if args.host == '0.0.0.0':
            parser.error('refuse to run --debug on 0.0.0.0; bind to '
                         '127.0.0.1 for debug mode.')
        if not args.allow_unsafe_debug:
            parser.error('--debug requires --allow-unsafe-debug; the '
                         'Werkzeug debugger is RCE-via-PIN for any local '
                         'user on this host.')

    if args.refresh:
        refresh(force_refresh=True)

    if args.export_html_only:
        out = export_static_html(args.export_html_only, args.export_network)
        print(f'[export] wrote {out.resolve()}')
        sys.exit(0)

    if args.export_html:
        out = export_static_html(args.export_html, args.export_network)
        print(f'[export] wrote {out.resolve()}')

    app.run(host=args.host, port=args.port, debug=args.debug)
