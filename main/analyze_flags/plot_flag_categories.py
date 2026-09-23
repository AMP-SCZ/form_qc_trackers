"""Render a bar chart of flag counts per proposed category.

Reads the `proposed_summary` sheet of template_mapping_recategorized.xlsx (the
output of build_proposed_categories.py). Writes a self-contained interactive
HTML chart by default, or a static PNG/PDF for pasting into a report.
Re-run after changing the merge map to refresh the chart.

    python plot_flag_categories.py --input proposed.xlsx                 # HTML
    python plot_flag_categories.py --input proposed.xlsx --png           # PNG
    python plot_flag_categories.py --input proposed.xlsx --png --pdf out.png

Both renderers read the same sheet and use the same validated palette, so the
static export and the HTML always agree. PNG/PDF need matplotlib (already a
dependency of site_network_radar.py); the HTML path needs nothing extra.
"""

import argparse
from pathlib import Path

import pandas as pd

SRC = Path(__file__).parent / "template_mapping_recategorized.xlsx"

# Single-series categorical slot 1. Validated with the dataviz palette
# validator against both surfaces: lightness band, chroma floor and >=3:1
# contrast all PASS in light (#2a78d6 on #fcfcfb) and dark (#3987e5 on #1a1a19).
SERIES_LIGHT, SERIES_DARK = "#2a78d6", "#3987e5"

# Bars are scaled so the longest reaches this share of its column, leaving the
# remainder for the value label at the tip. Every bar uses the same factor, so
# relative length still reads truthfully.
BAR_MAX_PCT = 82.0

# Called out in the row label; everything else kept its name.
MERGED = {"Medication Course Record Error"}


def rows_html(rows, total):
    longest = max(r[1] for r in rows)
    out = []
    for name, entries, templates in rows:
        pct_total = 100 * entries / total
        width = BAR_MAX_PCT * entries / longest
        tag = ' <span class="tag">merged</span>' if name in MERGED else ""
        out.append(
            f'<div class="row" tabindex="0" role="listitem" '
            f'data-name="{name}" data-entries="{entries:,}" '
            f'data-templates="{templates:,}" data-pct="{pct_total:.2f}%">'
            f'<div class="lbl">{name}{tag}</div>'
            f'<div class="track"><div class="bar" style="width:{width:.4f}%"></div>'
            f'<span class="val">{entries:,}</span></div>'
            f"</div>"
        )
    return "\n".join(out)


def table_html(rows, total):
    body = "\n".join(
        f"<tr><td>{n}</td><td class=num>{e:,}</td><td class=num>{t:,}</td>"
        f"<td class=num>{100*e/total:.2f}%</td></tr>"
        for n, e, t in rows
    )
    return (
        "<table><thead><tr><th>Category</th><th class=num>Flags</th>"
        "<th class=num>Templates</th><th class=num>Share</th></tr></thead>"
        f"<tbody>{body}</tbody><tfoot><tr><th>Total</th>"
        f"<th class=num>{total:,}</th><th class=num>"
        f"{sum(r[2] for r in rows):,}</th><th class=num>100%</th></tr></tfoot></table>"
    )


CSS = """
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:28px 20px}
.wrap{max-width:940px;margin:0 auto}
h1{font-size:20px;font-weight:600;margin:0 0 6px}
.sub{color:var(--text-secondary);margin:0 0 26px;max-width:70ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:20px 22px 22px;margin-bottom:20px}
h2{font-size:14px;font-weight:600;margin:0 0 3px}
.note{color:var(--text-secondary);font-size:12.5px;margin:0 0 18px}
.row{display:grid;grid-template-columns:minmax(120px,236px) 1fr;gap:14px;
  align-items:center;padding:4px 0;border-radius:5px;outline:none}
.row:hover,.row:focus-visible{background:var(--hover)}
.lbl{color:var(--text-secondary);font-size:12.5px;text-align:right;
  overflow-wrap:anywhere}
.tag{color:var(--muted);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.04em;white-space:nowrap}
.track{display:flex;align-items:center;min-width:0}
.bar{height:16px;background:var(--series);border-radius:0 4px 4px 0;min-width:2px}
.val{margin-left:9px;color:var(--text-secondary);font-size:12px;
  font-variant-numeric:tabular-nums;white-space:nowrap}
.axis{grid-column:2;border-top:1px solid var(--baseline);margin-top:9px;
  padding-top:5px;color:var(--muted);font-size:11px}
details{margin-top:4px}
summary{cursor:pointer;color:var(--text-secondary);font-size:12.5px;padding:6px 0}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:12.5px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--gridline)}
thead th,tfoot th{color:var(--text-secondary);font-weight:600}
tfoot th{border-top:1px solid var(--baseline);border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--surface);border:1px solid var(--border);border-radius:7px;
  padding:8px 11px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.13);z-index:9}
#tip b{display:block;margin-bottom:3px;color:var(--text-primary);font-weight:600}
#tip span{color:var(--text-secondary);font-variant-numeric:tabular-nums}
"""

TOKENS_LIGHT = f"""--page:#f9f9f7;--surface:#fcfcfb;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--gridline:#e1e0d9;--baseline:#c3c2b7;
--series:{SERIES_LIGHT};--border:rgba(11,11,11,.10);--hover:rgba(11,11,11,.035);"""

TOKENS_DARK = f"""--page:#0d0d0d;--surface:#1a1a19;--text-primary:#fff;
--text-secondary:#c3c2b7;--muted:#898781;--gridline:#2c2c2a;--baseline:#383835;
--series:{SERIES_DARK};--border:rgba(255,255,255,.10);--hover:rgba(255,255,255,.05);"""

JS = """
const tip=document.getElementById('tip');
const show=el=>{const d=el.dataset;
  tip.innerHTML=`<b>${d.name}</b><span>${d.entries} flags &middot; ${d.pct} of all flags<br>${d.templates} templates</span>`;
  tip.style.opacity=1;const r=el.getBoundingClientRect();
  tip.style.left=Math.min(r.left+14,innerWidth-tip.offsetWidth-12)+'px';
  tip.style.top=Math.max(8,r.top-tip.offsetHeight-6)+'px';};
const hide=()=>tip.style.opacity=0;
document.querySelectorAll('.row').forEach(el=>{
  el.addEventListener('mouseenter',()=>show(el));
  el.addEventListener('mouseleave',hide);
  el.addEventListener('focus',()=>show(el));
  el.addEventListener('blur',hide);});
"""


def load_rows(input_path=SRC):
    s = pd.read_excel(input_path, sheet_name="proposed_summary").sort_values(
        "n_entries", ascending=False)
    rows = [(r.proposed_category, int(r.n_entries), int(r.n_templates))
            for r in s.itertuples()]
    return rows, sum(r[1] for r in rows)


# ------------------------------------------------------------------ static
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"


def _panel(fig, ax, rows, heading, note):
    """Draw one panel. Headings are figure-level text flush with the main
    title — an axes title would start after the (wide) category-label gutter
    and read as floating mid-chart."""
    longest = rows[0][1]
    labels = [f"{n}  (merged)" if n in MERGED else n for n, _, _ in rows]
    values = [e for _, e, _ in rows]
    y = range(len(rows))

    ax.barh(y, values, height=0.52, color=SERIES_LIGHT, zorder=3)
    for i, v in enumerate(values):
        ax.text(v + longest * 0.012, i, f"{v:,}", va="center", ha="left",
                fontsize=8.5, color=INK2, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK2)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, longest * 1.16)
    ax.set_xticks([])
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="y", length=0, pad=6)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["left"].set_linewidth(0.8)

    pos = ax.get_position()
    fig.text(0.022, pos.y1 + 0.030, heading, ha="left", va="baseline",
             fontsize=10.5, fontweight="600", color=INK)
    fig.text(0.022, pos.y1 + 0.012, note, ha="left", va="baseline",
             fontsize=8.5, color=INK2)


def render_static(out_path, rows, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    top, tail = rows[0], rows[1:]

    fig, axes = plt.subplots(
        2, 1, figsize=(9.4, 8.0), dpi=200, facecolor=PAGE,
        gridspec_kw={"height_ratios": [len(rows), len(tail)],
                     "hspace": 0.20, "left": 0.285, "right": 0.985,
                     "top": 0.855, "bottom": 0.035})

    _panel(fig, axes[0], rows, f"All {len(rows)} categories",
           f"{top[0]} alone is {100*top[1]/total:.1f}% of every flag ever raised.")
    _panel(fig, axes[1], tail, f"Excluding {top[0]}",
           "The other categories on their own scale, so the ones the first "
           "panel flattens are readable.")

    fig.text(0.022, 0.955, "Flags per proposed QC category", ha="left",
             va="baseline", fontsize=15, fontweight="600", color=INK)
    fig.text(0.022, 0.928,
             f"{total:,} flag entries across {len(rows)} categories, after the "
             "reassignments and the one merge in FLAG_CATEGORY_MERGE_REVIEW.md. "
             "Bars within a panel share one linear scale.",
             ha="left", va="baseline", fontsize=8.5, color=INK2)

    for path in out_path:
        fig.savefig(path, facecolor=PAGE)
        print(f"wrote {path}")
    plt.close(fig)


# -------------------------------------------------------------------- html
def main(out_path, input_path=SRC):
    out_path = Path(out_path).expanduser().resolve()
    rows, total = load_rows(input_path)
    top = rows[0]
    tail = rows[1:]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flags per proposed QC category</title>
<style>
:root{{color-scheme:light;{TOKENS_LIGHT}}}
@media (prefers-color-scheme:dark){{
  :root:where(:not([data-theme="light"])){{color-scheme:dark;{TOKENS_DARK}}}}}
:root[data-theme="dark"]{{color-scheme:dark;{TOKENS_DARK}}}
{CSS}</style></head><body>
<div class="wrap">
<h1>Flags per proposed QC category</h1>
<p class="sub">{total:,} flag entries across {len(rows)} categories, after the
reassignments and the one merge in <code>FLAG_CATEGORY_MERGE_REVIEW.md</code>.
Two panels because the range spans four orders of magnitude &mdash; the first is
the honest picture, the second makes the tail readable.</p>

<div class="card">
<h2>All {len(rows)} categories</h2>
<p class="note">{top[0]} alone is {100*top[1]/total:.1f}% of every flag ever raised.</p>
<div role="list">
{rows_html(rows, total)}
<div class="axis">flag entries &mdash; bars share one linear scale</div>
</div></div>

<div class="card">
<h2>Excluding {top[0]}</h2>
<p class="note">The other {len(tail)} categories on their own scale, so the ones
the first panel flattens are readable.</p>
<div role="list">
{rows_html(tail, total)}
<div class="axis">flag entries &mdash; bars share one linear scale</div>
</div></div>

<div class="card"><details open><summary>Table view &mdash; all values</summary>
{table_html(rows, total)}</details></div>
</div>
<div id="tip" role="status"></div>
<script>{JS}</script></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({len(rows)} categories, {total:,} entries)")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="recategorized workbook containing proposed_summary")
    parser.add_argument("output", nargs="?", type=Path,
                        help="HTML output or basename for static outputs")
    parser.add_argument("--png", action="store_true", help="write a PNG figure")
    parser.add_argument("--pdf", action="store_true", help="write a PDF figure")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if args.png or args.pdf:
        stem = (args.output or Path("flag_categories_chart")).expanduser().resolve().with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        targets = ([str(stem) + ".png"] if args.png else []) + \
                  ([str(stem) + ".pdf"] if args.pdf else [])
        rows, total = load_rows(input_path)
        render_static(targets, rows, total)
    else:
        main(args.output or Path("flag_categories_chart.html"), input_path)


if __name__ == "__main__":
    cli()
