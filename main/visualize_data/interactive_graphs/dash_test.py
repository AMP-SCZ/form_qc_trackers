"""
Retired — this file used to host an early prototype of the QC
dashboard and a separate static-HTML export. Both are now in
`create_dashboard.py`, which:

  - Reads V2 trackers and walks Dropbox revisions through the same
    helpers `graph_errors.py` uses (so the interactive charts match
    the published PNGs).
  - Renders three chart types (stacked area, line graph, site bar)
    with network compare ("Both") mode.
  - Exposes the same static-HTML export via `--export-html PATH` /
    `--export-html-only PATH` flags.

Running this file just forwards to `create_dashboard.py` so existing
shortcuts / scripts don't break. Delete this file freely.
"""
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    target = here / 'create_dashboard.py'
    print(f'[dash_test.py] retired — forwarding to {target}')
    # Pass through any CLI args the user passed.
    os.execvp(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
