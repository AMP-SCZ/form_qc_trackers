# AMPSCZ form QC pipeline


## Quick start

1. Configure `config.json` at the project root (`paths.*`, `testing_enabled`,
   `pipeline_networks`). Set `pipeline_networks` to `["PRONET", "PRESCIENT"]`
   for both networks, or a one-item list for a single network. This setting is
   required unless the `QC_NETWORKS` environment variable supplies a one-off
   override (for example, `QC_NETWORKS=PRESCIENT`).
2. Ensure combined day1 CSVs exist under `paths.combined_csv_path` and
   dependencies under `paths.dependencies_path`.
3. Run the pipeline:

```bash
python run_qc.py
```
