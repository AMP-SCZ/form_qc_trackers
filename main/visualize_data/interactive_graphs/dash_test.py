from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import argparse
from process_data import ProcessData


# Load data (from ProcessData), with safe fallbacks
def _coerce_df(obj) -> pd.DataFrame :
    """Try common patterns to get a DataFrame out of an object."""
    if isinstance(obj, pd.DataFrame):
        return obj
    # attributes commonly used
    for attr in ("df", "dataframe", "data"):
        if hasattr(obj, attr) and isinstance(getattr(obj, attr), pd.DataFrame):
            return getattr(obj, attr)
    # methods commonly used
    for name in ("get_df", "get_dataframe", "to_dataframe", "to_df", "get_data", "load"):
        if hasattr(obj, name):
            out = getattr(obj, name)()
            if isinstance(out, pd.DataFrame):
                return out
    return None

def load_base_df() -> pd.DataFrame:
    # Try to instantiate and extract a DataFrame from ProcessData()
    try:
        pd_inst = ProcessData()  # if your class needs args, add them here
        df = _coerce_df(pd_inst())
        if df is None:
            # Some classes return DF on call, try that:
            if callable(pd_inst):
                maybe = pd_inst()
                df = _coerce_df(maybe) or maybe
        if isinstance(df, pd.DataFrame):
            print("[data] Loaded DataFrame from ProcessData")
            return df.copy()
        print("[data] ProcessData did not yield a DataFrame; falling back to synthetic sample.")
    except Exception as e:
        print(f"[data] Failed to load from ProcessData: {e}. Falling back to synthetic sample.")

    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    groups = ["Alpha", "Beta", "Gamma"]
    base = pd.DataFrame({
        "date": np.tile(dates, len(groups)),
        "group": np.repeat(groups, len(dates)),
    })
    base["measure1"] = (
        np.sin(np.linspace(0, 8 * np.pi, len(base))) * 10
        + base["group"].map({"Alpha": 0, "Beta": 5, "Gamma": -4}).to_numpy()
        + np.random.normal(scale=2.0, size=len(base))
    )
    base["measure2"] = base["measure1"] ** 2 / 10 + np.random.normal(scale=3.0, size=len(base))
    base["category"] = np.random.choice(["A", "B", "C", "D"], size=len(base))
    base.to_csv('base.csv',index = False)
    return base

base = load_base_df()

# Ensure required columns; add safe defaults if missing
if "date" not in base.columns:
    raise ValueError("Your DataFrame must include a 'date' column.")
if "group" not in base.columns:
    base["group"] = "All"
if "measure1" not in base.columns:
    raise ValueError("Your DataFrame must include a numeric 'measure1' column.")

# Convert types / add optional columns
base["date"] = pd.to_datetime(base["date"], errors="coerce")
base = base.dropna(subset=["date"]).copy()
if "measure2" not in base.columns:
    base["measure2"] = 0.0
if "category" not in base.columns:
    base["category"] = "Uncategorized"

# These drive controls; derive from *your* data
groups_all = sorted(base["group"].dropna().unique().tolist())
dates_all = base["date"].sort_values()
min_date = dates_all.min().date()
max_date = dates_all.max().date()

external_stylesheets = [
    dbc.themes.CYBORG,  # Dark Bootswatch theme
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap",
]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="AMP SCZ QC Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server

# Use Plotly dark template globally
pio.templates.default = "plotly_dark"

# Minimal dark page chrome (keeps dbc theme in charge)
app.index_string = (
    """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      body{background-color:#0b0f14!important; font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
      .card{border:none}
    </style>
    </head><body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body></html>"""
)


def apply_figure_style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig

def build_stacked_area(df: pd.DataFrame, color_by: str) -> go.Figure:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    # Sum per day per series
    agg = d.groupby(["date", color_by], as_index=False)["measure1"].sum()

    # Fill missing days with 0 per series (avoids weird stacking)
    full_idx = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    filled = []
    for key, sub in agg.groupby(color_by):
        sub = sub.set_index("date").reindex(full_idx)
        sub[color_by] = key
        sub["measure1"] = sub["measure1"].fillna(0)
        sub = sub.reset_index().rename(columns={"index": "date"})
        filled.append(sub)
    long_df = pd.concat(filled, ignore_index=True)

    fig = px.area(
        long_df, x="date", y="measure1", color=color_by,
        labels={"measure1": "QC Queries", "date": "Date"},
        # groupnorm="fraction",  # <- uncomment for 100% stacked
    )
    return apply_figure_style(fig, f"Stacked Area: QC Queries by {color_by}")

CHARTS = {
    "stacked": {"label": "Stacked Area (m1 by color)", "fn": lambda df, **kw: build_stacked_area(df, kw.get("color_by"))},
}

def build_sidebar():
    return dbc.Card(
        [
            html.Div([
                html.H2("AMP SCZ QC Dashboard", className="mb-1 fw-bold"),
            ]),
            html.Hr(),
            html.H6("Chart selector", className="mt-2"),
            dcc.RadioItems(
                id="chart-menu",
                options=[{"label": v["label"], "value": k} for k, v in CHARTS.items()],
                value="line",
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "6px"},
                className="mb-3",
            ),
            html.H6("Filters & options", className="mt-3"),
            dcc.Dropdown(
                id="group-select",
                options=[{"label": g, "value": g} for g in groups_all],
                value=groups_all,
                multi=True,
                placeholder="Select groups",
                className="mb-2",
            ),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                display_format="MMM D, YYYY",
                className="mb-2",
            ),
            html.Hr(),
            dbc.Alert([
                html.Div("Visualization of form QC flags over time for each network and timepoint", className="mb-1"),
            ], color="secondary", className="mb-0"),
        ],
        body=True,
        className="shadow-sm border-0",
    )


def build_main():
    return dbc.Card(
        [
            dbc.Row([
                dbc.Col([
                    html.H4("Overview", className="fw-semibold mb-0"),
                    html.P(id="chart-subtitle", className="text-muted mb-0"),
                ], width=8),
                dbc.Col([
                    dcc.Dropdown(
                        id="color-by",
                        options=[
                            {"label": "Color by group", "value": "group"},
                            {"label": "Color by category", "value": "category"},
                        ],
                        value="group",
                        clearable=False,
                    )
                ], width=4),
            ], align="center", className="mb-2"),
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=[dcc.Graph(id="main-graph", config={"displaylogo": False})],
            ),
        ],
        body=True,
        className="shadow-sm border-0",
    )


app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("QC Dashboard", className="fw-bold"),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Nav([
                dbc.NavLink("Docs", href="https://dash.plotly.com/", target="_blank", className="me-3"),
                dbc.NavLink("Plotly", href="https://plotly.com/python/", target="_blank"),
            ], className="ms-auto"),
        ]),
        color="dark", dark=True, sticky="top", className="mb-4 shadow-sm",
    ),
    dbc.Row([
        dbc.Col(build_sidebar(), md=3, lg=3, xl=3, className="mb-4"),
        dbc.Col(build_main(),    md=9, lg=9, xl=9, className="mb-4"),
    ], className="g-3"),
], fluid=True, className="px-3")


@app.callback(
    Output("main-graph", "figure"),
    Output("chart-subtitle", "children"),
    Input("chart-menu", "value"),
    Input("group-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("color-by", "value"),
)
def update_chart(chart_kind, selected_groups, start_date, end_date, color_by):
    if not selected_groups:
        selected_groups = groups_all

    mask = (
        base["group"].isin(selected_groups)
        & (base["date"] >= pd.to_datetime(start_date))
        & (base["date"] <= pd.to_datetime(end_date))
    )
    d = base.loc[mask].copy()

    subtitle = (
        f"Showing {len(d):,} rows | Groups: {', '.join(sorted(set(d['group'])))} | "
        f"{pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}"
    )

    entry = CHARTS.get(chart_kind, CHARTS["stacked"])  # fallback safe
    fig = entry["fn"](d, color_by=color_by)
    return fig, subtitle


def export_static_html(df: pd.DataFrame, out_path="qc_dashboard.html") -> Path:
    out_path = Path(out_path)

    d = df.copy()
    if "group" not in d:     d["group"] = "All"
    if "category" not in d:  d["category"] = "Uncategorized"
    if "measure2" not in d:  d["measure2"] = 0.0

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).copy()

    # strings for JS
    d["date_str"] = d["date"].dt.date.astype(str)

    # Build safe JSON payload (avoid numpy types/NaN by using to_json → loads)
    records_json = d[["date_str","group","category","measure1","measure2"]].rename(
        columns={"date_str":"date"}
    ).to_json(orient="records")
    records = json.loads(records_json)  # Python list/dicts, NaN→null handled

    groups_vals = sorted(d["group"].dropna().astype(str).unique().tolist())
    cats_vals   = sorted(d["category"].dropna().astype(str).unique().tolist())

    # Safe min/max date strings
    min_dt = d["date"].min()
    max_dt = d["date"].max()
    min_s  = min_dt.date().isoformat() if pd.notna(min_dt) else "2000-01-01"
    max_s  = max_dt.date().isoformat() if pd.notna(max_dt) else "2100-12-31"

    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AMP SCZ QC Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/cyborg/bootstrap.min.css" rel="stylesheet" />
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root { --muted: #94a3b8; }
    body { background:#0b0f14; color:#e5e7eb; font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    .navbar { border-bottom: 1px solid rgba(255,255,255,0.08); }
    .sidebar { position: sticky; top: 1rem; width; overflow:auto }
    .card { border: none; background-color:#0f1720; }
    .form-label { font-weight: 600; color: var(--muted); }
    footer { color: #9ca3af; }
    .chart-radio label { display:block; margin: 4px 0; }
    .select-multi { height: 120px; }
  </style>
</head>
<body>
  <nav class="navbar navbar-dark bg-dark mb-4">
    <div class="container-fluid">
      <span class="navbar-brand fw-bold">QC Dashboard</span>
      <div class="d-flex">
        <a class="nav-link" href="https://dash.plotly.com/" target="_blank">Docs</a>
        <a class="nav-link" href="https://plotly.com/javascript/" target="_blank">Plotly.js</a>
      </div>
    </div>
  </nav>

  <div class="container-xxl">
    <div class="row g-3">
      <div class="col-12 col-lg-3">
        <div class="card sidebar">
          <div class="card-body">
            <h2 class="h4 fw-bold mb-1">AMP SCZ QC Dashboard</h2>
            <hr/>
            <label class="form-label">Chart selector</label>
            <div class="chart-radio mb-3" id="chartMenu">
              <div class="form-check">
                <input class="form-check-input" type="radio" name="chartKind" id="chartStacked" value="stacked">
              <label class="form-check-label" for="chartStacked">QC Queries Over Time</label>
              </div>
            </div>

            <label class="form-label">Filters & options</label>
            <div class="mb-2">
              <label for="groupSelect" class="form-label">Groups</label>
              <select multiple class="form-select select-multi" id="groupSelect"></select>
            </div>

            <div class="mb-2">
              <label class="form-label">Date range</label>
              <div class="d-flex gap-2">
                <input type="date" class="form-control" id="startDate">
                <input type="date" class="form-control" id="endDate">
              </div>
            </div>

            <div class="mb-3" id="smoothContainer">
              <label for="smoothWindow" class="form-label">Smoothing (days)</label>
              <input type="range" class="form-range" id="smoothWindow" min="1" max="21" step="1" value="7">
              <div class="text-muted small"><span id="smoothValue">7</span> day rolling mean overlay</div>
            </div>

            <div class="mb-2">
              <label class="form-label" for="colorBy">Color by</label>
              <select id="colorBy" class="form-select">
                <option value="group" selected>Color by group</option>
                <option value="category">Color by category</option>
              </select>
            </div>

            <hr/>
            <div class="alert alert-secondary mb-0">
              <div class="mb-1">Visualization of form QC flags over time for each network and timepoint.</div>
            </div>
          </div>
        </div>
      </div>

      <div class="col-12 col-lg-9">
        <div class="card">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <div>
                <h4 class="fw-semibold mb-0">Overview</h4>
                <p id="chartSubtitle" class="text-muted mb-0"></p>
              </div>
              <div class="text-muted small" id="statusLine"></div>
            </div>
            <div id="mainGraph" style="height: 520px;"></div>
          </div>
        </div>
      </div>
    </div>

    <footer class="my-4">
      <small> <span id="yearNow"></span></small>
    </footer>
  </div>

<script>
// -----------------------------
// Utilities
// -----------------------------
function parseDate(s) { return new Date(s + "T00:00:00"); }
function formatDateISO(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}
function monthStart(date) { return new Date(date.getFullYear(), date.getMonth(), 1); }
function rollingMean(values, window) {
  const out = []; let sum = 0; const buf = [];
  for (let i=0;i<values.length;i++) { buf.push(values[i]); sum += values[i]; if (buf.length>window) sum -= buf.shift(); out.push(sum/buf.length); }
  return out;
}
function groupBy(arr, keyFn) { const m = new Map(); for (const row of arr) { const k = keyFn(row); if (!m.has(k)) m.set(k, []); m.get(k).push(row);} return m; }

// -----------------------------
// Embedded data (filled by Python below)
// -----------------------------
const groupsAll = __GROUPS__;
const categoriesAll = __CATS__;
const base = __RECORDS__;
base.forEach(r => r.date = parseDate(r.date));

document.getElementById("yearNow").textContent = new Date().getFullYear();

// Populate group multiselect
const groupSelect = document.getElementById("groupSelect");
for (const g of groupsAll) { const opt = document.createElement("option"); opt.value = g; opt.textContent = g; opt.selected = true; groupSelect.appendChild(opt); }

// Date range defaults
const startDateEl = document.getElementById("startDate");
const endDateEl = document.getElementById("endDate");
startDateEl.value = "__MIN_DATE__";
endDateEl.value = "__MAX_DATE__";

// Smoothing slider
const smoothSlider = document.getElementById("smoothWindow");
const smoothValue = document.getElementById("smoothValue");
smoothSlider.addEventListener("input", () => { smoothValue.textContent = smoothSlider.value; });

function getSelectedChartKind() { const radios = document.querySelectorAll('input[name="chartKind"]'); for (const r of radios) if (r.checked) return r.value; return "stacked"; }
function getSelectedGroups() { return Array.from(groupSelect.selectedOptions).map(o => o.value); }
function getColorBy() { return document.getElementById("colorBy").value; }

function filterData() {
  const gs = getSelectedGroups(); const start = parseDate(startDateEl.value); const end = parseDate(endDateEl.value);
  return base.filter(r => gs.includes(r.group) && r.date >= start && r.date <= end);
}
function setSubtitle(d) {
  const gs = Array.from(new Set(d.map(x => x.group))).sort().join(", ");
  const s = `${d.length.toLocaleString()} rows | Groups: ${gs || "—"} | ${formatDateISO(parseDate(startDateEl.value))} → ${formatDateISO(parseDate(endDateEl.value))}`;
  document.getElementById("chartSubtitle").textContent = s;
}

function buildTimeSeries(d, colorBy, smoothWindow) {
  d.sort((a,b)=>a.date-b.date);
  const keyFn = colorBy === "group" ? (r)=>r.group : (r)=>r.category;
  const grouped = groupBy(d, keyFn);
  const traces = [];
  for (const [key, rows] of grouped.entries()) {
    const x = rows.map(r=>r.date); const y = rows.map(r=>r.measure1);
    traces.push({ type: "scatter", mode: "lines", name: key, x, y, hovertemplate: "%{y:.2f}<extra>"+key+"</extra>" });
  }
  if (smoothWindow && smoothWindow > 1) {
    for (const [key, rows] of grouped.entries()) {
      const x = rows.map(r=>r.date); const y = rows.map(r=>r.measure1); const ySmooth = rollingMean(y, smoothWindow);
      traces.push({ type: "scatter", mode: "lines", name: key+" (smoothed)", x, y: ySmooth, line: { dash: "dash", width: 2 }, hovertemplate: "%{y:.2f}<extra>Smoothed</extra>" });
    }
  }
  const layout = { template: "plotly_dark", title: { text: "Time Series of Measure 1", x: 0.02 }, margin: { l:40,r:20,t:60,b:40 }, legend: { orientation:"h", y:-0.2, x:0 }, xaxis:{showgrid:true, gridcolor:"rgba(255,255,255,0.08)"}, yaxis:{showgrid:true, gridcolor:"rgba(255,255,255,0.08)", title:"Measure 1"}, paper_bgcolor:'#0b0f14', plot_bgcolor:'#0b0f14' };
  Plotly.react("mainGraph", traces, layout, {displaylogo:false, responsive:true});
}

function buildStackedArea(d, colorBy) {
  const keyFn = colorBy === "group" ? (r)=>r.group : (r)=>r.category;
  const dayKey = (dt)=> {
    const d0 = new Date(dt.getFullYear(), dt.getMonth(), dt.getDate());
    return d0.toISOString().slice(0,10);
  };
  const acc = new Map(); // "YYYY-MM-DD|series" -> sum
  const seriesSet = new Set(); const daySet = new Set();
  for (const r of d) {
    const day = dayKey(r.date); const key = keyFn(r);
    const k = `${day}|${key}`;
    seriesSet.add(key); daySet.add(day);
    acc.set(k, (acc.get(k) || 0) + (Number(r.measure1) || 0));
  }
  const days = Array.from(daySet).sort();
  const series = Array.from(seriesSet).sort();
  const traces = series.map(s => {
    const y = days.map(day => acc.get(`${day}|${s}`) || 0);
    return {
      type: "scatter",
      mode: "lines",
      stackgroup: "one",
      // groupnorm: "fraction", // <- 100% stacked
      name: s,
      x: days.map(dv => new Date(dv+"T00:00:00")),
      y,
      hovertemplate: "%{y:.2f}<extra>" + s + "</extra>"
    };
  });
  const layout = {
    template: "plotly_dark",
    title: { text: "Queries Over Time", x: 0.02 },
    margin: { l:40, r:20, t:60, b:40 },
    legend: { orientation: "h", y: -0.2, x: 0 },
    xaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.08)" },
    yaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.08)", title: "Queries" },
    paper_bgcolor: "#0b0f14",
    plot_bgcolor: "#0b0f14"
  };
  Plotly.react("mainGraph", traces, layout, { displaylogo: false, responsive: true });
}


const statusLine = document.getElementById("statusLine");
function render() {
  const d = filterData(); setSubtitle(d); statusLine.textContent = "";
  const chart = getSelectedChartKind(); const colorBy = getColorBy(); const smooth = parseInt(document.getElementById("smoothWindow").value, 10);
 if (chart === "stacked") { document.getElementById("smoothContainer").style.display = "none"; buildStackedArea(d, colorBy); }
}

document.getElementById("chartMenu").addEventListener("change", render);
groupSelect.addEventListener("change", render);
startDateEl.addEventListener("change", render);
endDateEl.addEventListener("change", render);
document.getElementById("colorBy").addEventListener("change", render);
document.getElementById("smoothWindow").addEventListener("change", render);
render();
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    html = (
        template
        .replace("__GROUPS__", json.dumps(groups_vals))
        .replace("__CATS__", json.dumps(cats_vals))
        .replace("__RECORDS__", json.dumps(records))
        .replace("__MIN_DATE__", min_s)
        .replace("__MAX_DATE__", max_s)
    )

    out_path.write_text(html, encoding="utf-8")
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dash app and/or export static HTML dashboard.")
    parser.add_argument(
        "--export-html",
        nargs="?",
        const="research_dashboard.html",
        help=("Write a standalone HTML dashboard to this path (default: research_dashboard.html) "
              "and continue running the server."),
    )
    parser.add_argument(
        "--export-html-only",
        nargs="?",
        const="research_dashboard.html",
        help="Write the standalone HTML and exit without starting the server.",
    )
    args = parser.parse_args()

    """if args.export_html_only:
        out = export_static_html(base, args.export_html_only)
        print(f"[export] Saved static dashboard to: {out.resolve()}")
    else:
        out_path = args.export_html or Path("research_dashboard.html")
        out = export_static_html(base, out_path)
        print(f"[export] Saved static dashboard to: {out.resolve()}")
        app.run(debug=True)"""

    out = export_static_html(base)
    print(f"[export] Saved static dashboard to: {out.resolve()}")
