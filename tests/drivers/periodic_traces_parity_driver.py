#!/usr/bin/env python3

import concurrent.futures
import importlib.util
import json
import os
import sys
from pathlib import Path

from rt_model_inference import (
    infer_certain_fit_periodic_model,
    infer_delta_max,
    infer_delta_max_lo,
    infer_delta_min,
    infer_delta_min_hi,
    infer_possible_fit_periodic_model,
)
from rt_model_inference.certain_periodic import infer_periodic_model
from rt_model_inference.extractors import (
    CertainFitPeriodicExtractor,
    DeltaMaxExtractor,
    DeltaMaxLoExtractor,
    DeltaMinExtractor,
    DeltaMinHiExtractor,
    PeriodicExtractor,
    PossibleFitPeriodicExtractor,
)


def load_python_test_module(reference_dir: Path):
    test_path = reference_dir / "tests" / "test_periodic_traces.py"
    spec = importlib.util.spec_from_file_location("test_periodic_traces", test_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_trace_overrides(module, test_all_traces: bool) -> None:
    if test_all_traces:
        module.TRACE_FILE_NAMES = module.ALL_TRACE_FILE_NAMES
        module.SPORADIC_CUTOFF = None


ONE_MILLISECOND = 1_000_000
EXACT_CONFIG = {
    "batch_size": 128,
    "negligible_jitter_threshold": ONE_MILLISECOND,
}
CERTAIN_FIT_CONFIG = {
    "batch_size": 512,
    "jitter_selection_threshold": 1.65,
}
POSSIBLE_FIT_CONFIG = {
    "batch_size": 512,
    "negligible_jitter_threshold": ONE_MILLISECOND,
}
NMAX = 5
_WORKER_MODULE = None


def model_dict(model):
    return {
        "period": model.period,
        "offset": model.offset,
        "jitter": model.jitter,
    }


def trace_case(
    module,
    trace_file_name: str,
    nmax: int,
    exact_config: dict,
    certain_fit_config: dict,
    possible_fit_config: dict,
):
    releases_and_windows = module.load_release_trace(module.TRACE_DIR / trace_file_name)
    releases = [r for r, _ in releases_and_windows]
    windows = [w for _, w in releases_and_windows]
    expected_period = module.expected_period(trace_file_name)

    exact = infer_periodic_model(releases, **exact_config)
    certain_fit = infer_certain_fit_periodic_model(windows, **certain_fit_config)
    possible_fit = infer_possible_fit_periodic_model(windows, **possible_fit_config)

    ex_exact = PeriodicExtractor(**exact_config)
    ex_exact.feed(releases)

    ex_cf = CertainFitPeriodicExtractor(**certain_fit_config)
    ex_cf.feed(windows)

    ex_pf = PossibleFitPeriodicExtractor(**possible_fit_config)
    ex_pf.feed(windows)

    sporadic_trace = releases_and_windows
    if module.SPORADIC_CUTOFF is not None:
        sporadic_trace = sporadic_trace[: module.SPORADIC_CUTOFF]
    sporadic_releases = [r for r, _ in sporadic_trace]
    sporadic_windows = [w for _, w in sporadic_trace]

    dmin = infer_delta_min(sporadic_releases, nmax=nmax)
    dmax = infer_delta_max(sporadic_releases, nmax=nmax)
    dmin_hi = infer_delta_min_hi(sporadic_windows, nmax=nmax)
    dmax_lo = infer_delta_max_lo(sporadic_windows, nmax=nmax)

    ex_dmin = DeltaMinExtractor(nmax=nmax)
    ex_dmin.feed(sporadic_releases)

    ex_dmax = DeltaMaxExtractor(nmax=nmax)
    ex_dmax.feed(sporadic_releases)

    ex_dmin_hi = DeltaMinHiExtractor(nmax=nmax)
    ex_dmin_hi.feed(sporadic_windows)

    ex_dmax_lo = DeltaMaxLoExtractor(nmax=nmax)
    ex_dmax_lo.feed(sporadic_windows)

    return {
        "trace_file_name": trace_file_name,
        "expected_period": expected_period,
        "exact": model_dict(exact),
        "certain_fit": model_dict(certain_fit),
        "possible_fit": model_dict(possible_fit),
        "extractor_exact": model_dict(ex_exact.current_model),
        "extractor_certain_fit": model_dict(ex_cf.current_model),
        "extractor_possible_fit": model_dict(ex_pf.current_model),
        "dmin": dmin,
        "dmax": dmax,
        "dmin_hi": dmin_hi,
        "dmax_lo": dmax_lo,
        "ex_dmin": ex_dmin.current_model,
        "ex_dmax": ex_dmax.current_model,
        "ex_dmin_hi": ex_dmin_hi.current_model,
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


def trace_case_worker(trace_file_name: str):
    assert _WORKER_MODULE is not None
    return trace_case(
        _WORKER_MODULE,
        trace_file_name,
        NMAX,
        EXACT_CONFIG,
        CERTAIN_FIT_CONFIG,
        POSSIBLE_FIT_CONFIG,
    )


def collect_trace_cases(reference_dir: Path, trace_file_names: list[str], test_all_traces: bool):
    workers = resolve_worker_count(len(trace_file_names))
    if workers == 1:
        return [
            trace_case(
                _WORKER_MODULE,
                trace_file_name,
                NMAX,
                EXACT_CONFIG,
                CERTAIN_FIT_CONFIG,
                POSSIBLE_FIT_CONFIG,
            )
            for trace_file_name in trace_file_names
        ]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(reference_dir), test_all_traces),
    ) as executor:
        return list(executor.map(trace_case_worker, trace_file_names))


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
