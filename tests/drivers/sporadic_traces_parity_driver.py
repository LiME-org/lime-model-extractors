#!/usr/bin/env python3

import concurrent.futures
import importlib.util
import json
import os
import sys
from pathlib import Path

from rt_model_inference import (
    infer_delta_max,
    infer_delta_max_hi,
    infer_delta_max_lo,
    infer_delta_min,
    infer_delta_min_hi,
    infer_delta_min_lo,
)
from rt_model_inference.extractors import (
    DeltaMaxExtractor,
    DeltaMaxHiExtractor,
    DeltaMaxLoExtractor,
    DeltaMinExtractor,
    DeltaMinHiExtractor,
    DeltaMinLoExtractor,
)


def load_python_test_module(reference_dir: Path):
    test_path = reference_dir / "tests" / "test_sporadic_traces.py"
    spec = importlib.util.spec_from_file_location("test_sporadic_traces", test_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_trace_overrides(module, test_all_traces: bool) -> None:
    if test_all_traces:
        module.TRACE_FILE_NAMES = module.ALL_TRACE_FILE_NAMES
        module.SPORADIC_CUTOFF = None


NMAX = 5
_WORKER_MODULE = None


def model_for_trace(module, trace_file_name: str, nmax: int):
    releases_and_windows = module.load_release_trace(module.TRACE_DIR / trace_file_name)
    if module.SPORADIC_CUTOFF is not None:
        releases_and_windows = releases_and_windows[: module.SPORADIC_CUTOFF]

    releases = [r for r, _ in releases_and_windows]
    windows = [w for _, w in releases_and_windows]

    dmin = infer_delta_min(releases, nmax=nmax)
    dmax = infer_delta_max(releases, nmax=nmax)
    dmin_hi = infer_delta_min_hi(windows, nmax=nmax)
    dmin_lo = infer_delta_min_lo(windows, nmax=nmax)
    dmax_hi = infer_delta_max_hi(windows, nmax=nmax)
    dmax_lo = infer_delta_max_lo(windows, nmax=nmax)

    ex_dmin = DeltaMinExtractor(nmax=nmax)
    ex_dmin.feed(releases)

    ex_dmax = DeltaMaxExtractor(nmax=nmax)
    ex_dmax.feed(releases)

    ex_dmin_hi = DeltaMinHiExtractor(nmax=nmax)
    ex_dmin_hi.feed(windows)

    ex_dmin_lo = DeltaMinLoExtractor(nmax=nmax)
    ex_dmin_lo.feed(windows)

    ex_dmax_hi = DeltaMaxHiExtractor(nmax=nmax)
    ex_dmax_hi.feed(windows)

    ex_dmax_lo = DeltaMaxLoExtractor(nmax=nmax)
    ex_dmax_lo.feed(windows)

    return {
        "trace_file_name": trace_file_name,
        "dmin": dmin,
        "dmax": dmax,
        "dmin_hi": dmin_hi,
        "dmin_lo": dmin_lo,
        "dmax_hi": dmax_hi,
        "dmax_lo": dmax_lo,
        "ex_dmin": ex_dmin.current_model,
        "ex_dmax": ex_dmax.current_model,
        "ex_dmin_hi": ex_dmin_hi.current_model,
        "ex_dmin_lo": ex_dmin_lo.current_model,
        "ex_dmax_hi": ex_dmax_hi.current_model,
        "ex_dmax_lo": ex_dmax_lo.current_model,
    }


def resolve_worker_count(num_cases: int) -> int:
    requested = os.getenv("PARITY_DRIVER_WORKERS", "auto").strip().lower()
    if requested == "auto":
        workers = os.cpu_count() or 1
    else:
        workers = int(requested)

    return max(1, min(workers, num_cases))


def init_worker(reference_dir: str, test_all_traces: bool) -> None:
    global _WORKER_MODULE
    _WORKER_MODULE = load_python_test_module(Path(reference_dir))
    apply_trace_overrides(_WORKER_MODULE, test_all_traces)


def model_for_trace_worker(trace_file_name: str):
    assert _WORKER_MODULE is not None
    return model_for_trace(_WORKER_MODULE, trace_file_name, NMAX)


def collect_trace_cases(reference_dir: Path, trace_file_names: list[str], test_all_traces: bool):
    workers = resolve_worker_count(len(trace_file_names))
    if workers == 1:
        return [
            model_for_trace(_WORKER_MODULE, trace_file_name, NMAX)
            for trace_file_name in trace_file_names
        ]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(reference_dir), test_all_traces),
    ) as executor:
        return list(executor.map(model_for_trace_worker, trace_file_names))


def main() -> None:
    reference_dir = Path(sys.argv[1])
    test_all_traces = bool(os.getenv("TEST_ALL_TRACES"))

    global _WORKER_MODULE
    _WORKER_MODULE = load_python_test_module(reference_dir)
    apply_trace_overrides(_WORKER_MODULE, test_all_traces)
    trace_file_names = list(_WORKER_MODULE.TRACE_FILE_NAMES)

    payload = {
        "test_all_traces": test_all_traces,
        "trace_dir": str(_WORKER_MODULE.TRACE_DIR),
        "trace_file_names": trace_file_names,
        "sporadic_cutoff": _WORKER_MODULE.SPORADIC_CUTOFF,
        "nmax": NMAX,
        "cases": collect_trace_cases(reference_dir, trace_file_names, test_all_traces),
    }

    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
