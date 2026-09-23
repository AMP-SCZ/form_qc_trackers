"""Run the main QC pipeline under instrumentation and report its compute cost.

The production entry point (``run_prescient_qc.py``) hard-requires the live
Dropbox configuration, and its third stage rotates
``combined_outputs/current_output`` into ``old_output``.  Neither is acceptable
for a benchmark, so this harness calls the same stage callables directly,
defaults to the two compute-bound stages, and snapshots/restores every artifact
the run would overwrite.

Usage (PowerShell)::

    $env:QC_NETWORKS = "PRESCIENT"
    py -3.9 measure_qc_run.py --label prescient_baseline

Metrics per stage: wall clock, CPU seconds (user/kernel split), mean core
occupancy, and peak working set sampled on a background thread.  Results are
printed and written to ``compute_profiles/<label>.json``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import threading
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IS_WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------
# Process metrics.  psutil is not installed in this workspace, so read the
# same counters straight from the Win32 API.  Falls back to resource(2) so the
# harness still works if it is ever run on the Linux production host.
# --------------------------------------------------------------------------

if IS_WINDOWS:

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _CURRENT_PROCESS = ctypes.c_void_p(_kernel32.GetCurrentProcess())

    def memory_counters():
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not _psapi.GetProcessMemoryInfo(
            _CURRENT_PROCESS, ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize, counters.PeakWorkingSetSize

    def cpu_times():
        """(user_seconds, kernel_seconds) for this process, all threads."""
        creation = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not _kernel32.GetProcessTimes(
            _CURRENT_PROCESS,
            ctypes.byref(creation),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        def _to_seconds(filetime):
            # FILETIME counts 100-nanosecond intervals.
            return (
                (filetime.dwHighDateTime << 32) | filetime.dwLowDateTime
            ) / 1e7

        return _to_seconds(user), _to_seconds(kernel)

    def total_physical_memory():
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        _kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys

else:  # pragma: no cover - production host path
    import resource

    def memory_counters():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is kilobytes on Linux; there is no live working-set
        # counter, so the peak doubles as the instantaneous sample.
        return usage.ru_maxrss * 1024, usage.ru_maxrss * 1024

    def cpu_times():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime, usage.ru_stime

    def total_physical_memory():
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


class MemorySampler(threading.Thread):
    """Poll the working set so each stage gets its own peak, not a running max."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stage_peak = 0
        self.samples = 0

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                current, _ = memory_counters()
            except OSError:
                continue
            with self._lock:
                if current > self._stage_peak:
                    self._stage_peak = current
                self.samples += 1

    def reset_stage_peak(self):
        with self._lock:
            current, _ = memory_counters()
            self._stage_peak = current

    def stage_peak(self):
        with self._lock:
            return self._stage_peak

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Artifact protection.
# --------------------------------------------------------------------------

def artifact_paths(config_info, testing):
    """Shared state a prep+qc run overwrites, so it can be put back afterwards.

    Under ``testing_enabled = "True"`` every stage suffixes output_path with
    ``testing/`` (qc_forms_main, calculate_resolved_errors, create_trackers all
    do this independently), so the real combined_outputs tree is never touched
    and only the dependency directory is shared.  ``dependencies_path`` gets no
    such suffix in either mode, so it always needs protecting.
    """
    paths = [Path(config_info["paths"]["dependencies_path"])]
    if testing:
        return paths
    combined = Path(config_info["paths"]["output_path"]) / "combined_outputs"
    paths.extend(
        [
            combined / "new_output",
            combined / "old_output",
            combined / "current_output",
        ]
    )
    return paths


def append_snapshot(manifest, source, snapshot_root, index):
    """Copy one file or directory aside, recording whether it existed."""
    destination = snapshot_root / "{:02d}_{}".format(index, source.name)
    if source.is_dir():
        kind, existed = "dir", True
        shutil.copytree(source, destination)
    elif source.is_file():
        kind, existed = "file", True
        shutil.copyfile(source, destination)
    else:
        kind, existed = ("dir" if not source.suffix else "file"), False
    manifest.append(
        {
            "source": str(source),
            "snapshot": str(destination),
            "existed": existed,
            "kind": kind,
        }
    )
    print("[measure] snapshot {}: {}".format(
        "saved" if existed else "absent", source))
    return manifest


def write_manifest(manifest, snapshot_root):
    (snapshot_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def snapshot(paths, snapshot_root, manifest=None):
    """Copy each path aside, recording which ones did not exist."""
    snapshot_root.mkdir(parents=True, exist_ok=True)
    manifest = list(manifest or [])
    for path in paths:
        append_snapshot(manifest, path, snapshot_root, len(manifest))
    write_manifest(manifest, snapshot_root)
    return manifest


def restore(manifest):
    """Put the workspace back exactly as the snapshot found it."""
    for entry in manifest:
        source = Path(entry["source"])
        kind = entry.get("kind", "dir")
        if kind == "dir":
            if source.is_dir():
                shutil.rmtree(source)
            if entry["existed"]:
                shutil.copytree(entry["snapshot"], source)
        else:
            if source.is_file():
                source.unlink()
            if entry["existed"]:
                shutil.copyfile(entry["snapshot"], source)
        print("[measure] restored: {}".format(source))


# --------------------------------------------------------------------------
# Stages.
# --------------------------------------------------------------------------

def stage_prep():
    from run_prescient_qc import _run_combined_data_stage

    _run_combined_data_stage()


def stage_qc():
    from main.qc_forms.qc_forms_main import QCFormsMain

    QCFormsMain().run_script()


def stage_reports():
    from main.generate_reports.generate_reports_main import GenerateReports

    GenerateReports().run_script()


STAGES = {
    "prep": ("process_variables (dependency build)", stage_prep),
    "qc": ("qc_forms_main (per-row checks)", stage_qc),
    "reports": ("generate_reports (Excel + Dropbox upload)", stage_reports),
}


def input_bytes(combined_csv_path, networks):
    """Size of the combined CSVs this run actually reads."""
    root = Path(combined_csv_path)
    if not root.is_dir():
        return 0
    tokens = {"PRONET": "pronet", "PRESCIENT": "prescient"}
    wanted = {tokens[network] for network in networks}
    total = 0
    for csv_file in root.glob("*.csv"):
        name = csv_file.name.casefold()
        if any(token in name for token in wanted):
            total += csv_file.stat().st_size
    return total


def output_rows(config_info):
    path = (
        Path(config_info["paths"]["output_path"])
        / "combined_outputs"
        / "new_output"
        / "combined_qc_flags.parquet"
    )
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception:
        return None


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def write_report(run, report_path):
    """Flush the report after every stage.

    A Windows TerminateProcess cannot be caught, so a kill mid-run never
    reaches the rollback/report code in ``finally``. Writing incrementally
    means an interrupted run still leaves the stages that did finish on disk.
    """
    report_path.write_text(json.dumps(run, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="prep,qc",
        help=(
            "Comma-separated subset of prep,qc,reports. 'reports' uploads to "
            "live Dropbox and rotates current_output -> old_output; it is "
            "excluded by default and needs --allow-dropbox-upload."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Name for the report file (default: network scope + timestamp).",
    )
    parser.add_argument(
        "--allow-dropbox-upload",
        action="store_true",
        help="Required to include the 'reports' stage.",
    )
    parser.add_argument(
        "--testing-config",
        action="store_true",
        help=(
            "Set testing_enabled='True' in config.json for the duration of "
            "the run, so outputs land in <output_path>/testing/ and uploads "
            "go to the Dropbox refactoring_tests folder instead of the live "
            "tracker folder. config.json is snapshotted and restored."
        ),
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Keep the artifacts this run produced instead of rolling back.",
    )
    parser.add_argument(
        "--restore-only",
        metavar="LABEL",
        default=None,
        help=(
            "Run no stages; just roll a leftover snapshot back. Needed when a "
            "run is killed, since the rollback in the finally block never gets "
            "to execute."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Also run the qc stage under cProfile and dump a .prof file. "
            "Profiled wall time is inflated; the unprofiled numbers stand."
        ),
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="Seconds between working-set samples (default 0.25).",
    )
    parser.add_argument(
        "--hourly-rate",
        type=float,
        default=0.0,
        help=(
            "Optional $/hour for the machine class you would rent, used only "
            "to translate wall time into a cost line."
        ),
    )
    args = parser.parse_args()

    if args.restore_only:
        snapshot_root = (
            PROJECT_ROOT / "compute_profiles" / ".snapshot_{}".format(
                args.restore_only)
        )
        manifest_path = snapshot_root / "manifest.json"
        if not manifest_path.is_file():
            parser.error("no snapshot manifest at {}".format(manifest_path))
        restore(json.loads(manifest_path.read_text(encoding="utf-8")))
        shutil.rmtree(snapshot_root, ignore_errors=True)
        print("[measure] snapshot {} rolled back and removed".format(
            args.restore_only))
        return 0

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [stage for stage in requested if stage not in STAGES]
    if unknown:
        parser.error(
            "unknown stage(s): {}; choose from {}".format(unknown, list(STAGES)))
    if "reports" in requested and not args.allow_dropbox_upload:
        parser.error(
            "the 'reports' stage uploads trackers to live Dropbox and rotates "
            "current_output into old_output; pass --allow-dropbox-upload to "
            "confirm that is intended."
        )

    # Resolve the network scope the way Utils does, but without constructing
    # it: Utils snapshots config.json at construction time, and --testing-config
    # has to edit that file first.
    config_path = PROJECT_ROOT / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = os.environ.get("QC_NETWORKS")
    if configured:
        scope = [n.strip() for n in configured.split(",") if n.strip()]
    else:
        scope = raw_config.get("pipeline_networks")
    if not isinstance(scope, (list, tuple)) or not scope:
        parser.error(
            "set config.json pipeline_networks to a non-empty list, or "
            "provide a comma-separated QC_NETWORKS override")
    networks = [str(network).strip().upper() for network in scope]

    label = args.label or "{}_{}".format(
        "-".join(networks).casefold(), datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    report_dir = PROJECT_ROOT / "compute_profiles"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "{}.json".format(label)
    snapshot_root = report_dir / ".snapshot_{}".format(label)

    manifest = []
    if args.testing_config:
        snapshot_root.mkdir(parents=True, exist_ok=True)
        append_snapshot(manifest, config_path, snapshot_root, 0)
        write_manifest(manifest, snapshot_root)
        raw_config["testing_enabled"] = "True"
        config_path.write_text(
            json.dumps(raw_config, indent=2) + os.linesep, encoding="utf-8"
        )
        print("[measure] config.json: testing_enabled -> True "
              "(outputs under <output_path>/testing/, Dropbox uploads to "
              "/Apps/Automated QC Trackers/refactoring_tests/)")

    from main.utils.utils import Utils

    utils = Utils()
    config_info = utils.config_info
    networks = list(utils.pipeline_networks)
    testing = config_info.get("testing_enabled") == "True"

    print("[measure] networks: {}".format(", ".join(networks)))
    print("[measure] stages:   {}".format(", ".join(requested)))
    print("[measure] testing:  {}".format(testing))
    print("[measure] report:   {}".format(report_path))

    protected = artifact_paths(config_info, testing)
    manifest = snapshot(protected, snapshot_root, manifest)

    machine = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "total_ram_gib": round(gib(total_physical_memory()), 2),
    }

    sampler = MemorySampler(args.sample_interval)
    sampler.start()

    run = {
        "label": label,
        "started": datetime.now().astimezone().isoformat(),
        "networks": networks,
        "stages_requested": requested,
        "machine": machine,
        "input_bytes": input_bytes(
            config_info["paths"]["combined_csv_path"], networks
        ),
        "stages": [],
    }

    run_wall_start = time.perf_counter()
    run_user_start, run_kernel_start = cpu_times()
    failure = None

    try:
        for stage_name in requested:
            description, stage_callable = STAGES[stage_name]
            print("\n[measure] ===== {}: {} =====".format(stage_name, description),
                  flush=True)
            sampler.reset_stage_peak()
            user_start, kernel_start = cpu_times()
            wall_start = time.perf_counter()
            try:
                stage_callable()
                status = "ok"
            except SystemExit as exc:
                # The pipeline's own fatal guards call sys.exit(1). Record the
                # stage rather than letting it kill the harness mid-measurement.
                status = "sys.exit({})".format(exc.code)
                failure = "{} exited with code {}".format(stage_name, exc.code)
            wall = time.perf_counter() - wall_start
            user_end, kernel_end = cpu_times()
            user = user_end - user_start
            kernel = kernel_end - kernel_start
            peak = sampler.stage_peak()
            entry = {
                "stage": stage_name,
                "description": description,
                "status": status,
                "wall_seconds": round(wall, 2),
                "cpu_user_seconds": round(user, 2),
                "cpu_kernel_seconds": round(kernel, 2),
                "cpu_total_seconds": round(user + kernel, 2),
                "mean_cores_used": round((user + kernel) / wall, 2) if wall else None,
                "peak_working_set_gib": round(gib(peak), 3),
            }
            run["stages"].append(entry)
            write_report(run, report_path)
            print(
                "[measure] {stage}: wall {wall_seconds}s  cpu {cpu_total_seconds}s"
                "  cores {mean_cores_used}  peak RSS {peak_working_set_gib} GiB"
                "  [{status}]".format(**entry),
                flush=True,
            )
            if failure:
                break

        if args.profile and "qc" in requested and not failure:
            import cProfile
            import pstats

            profile_path = report_dir / "{}_qc.prof".format(label)
            print("\n[measure] ===== profiling qc stage -> {} =====".format(
                profile_path))
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                stage_qc()
            except SystemExit as exc:
                print("[measure] profiled qc stage exited: {}".format(exc.code))
            finally:
                profiler.disable()
                profiler.dump_stats(str(profile_path))
            run["profile_path"] = str(profile_path)

            stats = pstats.Stats(str(profile_path))
            print("\n[measure] top 30 by cumulative time (profiled run only):")
            stats.sort_stats("cumulative").print_stats(30)
    finally:
        sampler.stop()
        total_wall = time.perf_counter() - run_wall_start
        run_user_end, run_kernel_end = cpu_times()
        total_cpu = (run_user_end - run_user_start) + (
            run_kernel_end - run_kernel_start
        )
        _, process_peak = memory_counters()

        run["output_flag_rows"] = output_rows(config_info)
        run["totals"] = {
            "wall_seconds": round(total_wall, 2),
            "wall_minutes": round(total_wall / 60, 2),
            "cpu_seconds": round(total_cpu, 2),
            "cpu_core_hours": round(total_cpu / 3600, 4),
            "mean_cores_used": round(total_cpu / total_wall, 2) if total_wall else None,
            "peak_working_set_gib": round(gib(process_peak), 3),
            "memory_samples": sampler.samples,
        }
        if run["input_bytes"] and total_cpu:
            run["totals"]["input_mib_per_cpu_second"] = round(
                (run["input_bytes"] / (1024 ** 2)) / total_cpu, 2
            )
        if args.hourly_rate:
            run["totals"]["assumed_hourly_rate_usd"] = args.hourly_rate
            run["totals"]["estimated_cost_usd"] = round(
                args.hourly_rate * total_wall / 3600, 4
            )
        if failure:
            run["failure"] = failure

        write_report(run, report_path)
        print("\n[measure] wrote {}".format(report_path))

        if args.no_restore:
            print(
                "[measure] --no-restore: workspace left as the run produced it; "
                "snapshot kept at {}".format(snapshot_root)
            )
        else:
            restore(manifest)
            shutil.rmtree(snapshot_root, ignore_errors=True)

    totals = run["totals"]
    print("\n" + "=" * 72)
    print("COMPUTE PROFILE  {}".format(label))
    print("=" * 72)
    print("  machine            {}".format(machine["cpu_model"]))
    print("                     {} logical CPUs, {} GiB RAM".format(
        machine["logical_cpus"], machine["total_ram_gib"]))
    print("  networks           {}".format(", ".join(networks)))
    print("  input read         {:.2f} GiB of combined CSV".format(
        gib(run["input_bytes"])))
    print("  flag rows written  {}".format(run["output_flag_rows"]))
    print("  wall time          {} min".format(totals["wall_minutes"]))
    print("  CPU time           {} s ({} core-hours)".format(
        totals["cpu_seconds"], totals["cpu_core_hours"]))
    print("  mean cores busy    {} of {}".format(
        totals["mean_cores_used"], machine["logical_cpus"]))
    print("  peak memory        {} GiB".format(totals["peak_working_set_gib"]))
    if "estimated_cost_usd" in totals:
        print("  est. cost          ${} at ${}/hr".format(
            totals["estimated_cost_usd"], totals["assumed_hourly_rate_usd"]))
    print("=" * 72)

    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
