mod common;

use std::path::{Path, PathBuf};

use common::{
    load_release_trace, progress_bar, python_reference_dir, python_reference_payload,
    python_reference_python,
};
use lime_model_extractors::{
    extractors::{
        DeltaMaxExtractor, DeltaMaxHiExtractor, DeltaMaxLoExtractor, DeltaMinExtractor,
        DeltaMinHiExtractor, DeltaMinLoExtractor,
    },
    infer_delta_max, infer_delta_max_hi, infer_delta_max_lo, infer_delta_min, infer_delta_min_hi,
    infer_delta_min_lo,
    time::Duration,
};
use rayon::prelude::*;
use serde::Deserialize;
use similar_asserts::assert_eq;

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct TraceCase {
    trace_file_name: String,
    dmin: Vec<Duration>,
    dmax: Vec<Duration>,
    dmin_hi: Vec<Duration>,
    dmin_lo: Vec<Duration>,
    dmax_hi: Vec<Duration>,
    dmax_lo: Vec<Duration>,
    ex_dmin: Vec<Duration>,
    ex_dmax: Vec<Duration>,
    ex_dmin_hi: Vec<Duration>,
    ex_dmin_lo: Vec<Duration>,
    ex_dmax_hi: Vec<Duration>,
    ex_dmax_lo: Vec<Duration>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct TracePayload {
    test_all_traces: bool,
    trace_dir: PathBuf,
    trace_file_names: Vec<String>,
    sporadic_cutoff: Option<usize>,
    nmax: usize,
    cases: Vec<TraceCase>,
}

#[test]
fn sporadic_traces_match_python_reference() {
    let reference_dir = python_reference_dir();
    let python_path = match python_reference_python(&reference_dir) {
        Some(path) => path,
        None => {
            eprintln!(
                "skipping sporadic traces parity test: reference virtualenv is not initialized"
            );
            return;
        }
    };
    let driver_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("drivers")
        .join("sporadic_traces_parity_driver.py");

    if !reference_dir.join("pyproject.toml").exists() {
        eprintln!(
            "skipping sporadic traces parity test: Python reference checkout is not available"
        );
        return;
    }

    assert!(
        driver_path.exists(),
        "missing sporadic traces parity driver"
    );

    let payload: TracePayload = python_reference_payload(
        &python_path,
        &reference_dir,
        &driver_path,
        "sporadic trace parity",
    );
    if std::env::var_os("TEST_ALL_TRACES").is_some() {
        assert!(
            payload.test_all_traces,
            "driver did not observe TEST_ALL_TRACES"
        );
        assert!(
            payload.sporadic_cutoff.is_none(),
            "driver still reported a sporadic cutoff with TEST_ALL_TRACES enabled"
        );
        assert!(
            payload.trace_file_names.len() > 50,
            "driver still reported only {} traces with TEST_ALL_TRACES enabled",
            payload.trace_file_names.len()
        );
    }
    eprintln!(
        "[sporadic_traces] loaded {} traces from {} with cutoff {:?} and nmax {}",
        payload.cases.len(),
        payload.trace_dir.display(),
        payload.sporadic_cutoff,
        payload.nmax
    );

    let progress = progress_bar("sporadic_traces", payload.cases.len());
    payload.cases.par_iter().for_each(|case| {
        assert_sporadic_trace_case(&payload, case);
        progress.inc(1);
    });
    progress.finish_with_message("done");
}

fn assert_sporadic_trace_case(payload: &TracePayload, case: &TraceCase) {
    let trace_path = payload.trace_dir.join(&case.trace_file_name);
    let (releases, windows) = load_release_trace(&trace_path, payload.sporadic_cutoff);

    assert_eq!(
        infer_delta_min(releases.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmin,
        "dmin trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_max(releases.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmax,
        "dmax trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_min_hi(windows.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmin_hi,
        "dmin_hi trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_min_lo(windows.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmin_lo,
        "dmin_lo trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_max_hi(windows.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmax_hi,
        "dmax_hi trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_max_lo(windows.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmax_lo,
        "dmax_lo trace={}",
        case.trace_file_name
    );

    let mut ex_dmin = DeltaMinExtractor::new(Some(payload.nmax)).unwrap();
    ex_dmin.feed(releases.iter().copied()).unwrap();
    assert_eq!(
        ex_dmin.current_model(),
        case.ex_dmin,
        "ex_dmin trace={}",
        case.trace_file_name
    );

    let mut ex_dmax = DeltaMaxExtractor::new(Some(payload.nmax)).unwrap();
    ex_dmax.feed(releases.iter().copied()).unwrap();
    assert_eq!(
        ex_dmax.current_model(),
        case.ex_dmax,
        "ex_dmax trace={}",
        case.trace_file_name
    );

    let mut ex_dmin_hi = DeltaMinHiExtractor::new(Some(payload.nmax)).unwrap();
    ex_dmin_hi.feed(windows.iter().copied()).unwrap();
    assert_eq!(
        ex_dmin_hi.current_model(),
        case.ex_dmin_hi,
        "ex_dmin_hi trace={}",
        case.trace_file_name
    );

    let mut ex_dmin_lo = DeltaMinLoExtractor::new(Some(payload.nmax)).unwrap();
    ex_dmin_lo.feed(windows.iter().copied()).unwrap();
    assert_eq!(
        ex_dmin_lo.current_model(),
        case.ex_dmin_lo,
        "ex_dmin_lo trace={}",
        case.trace_file_name
    );

    let mut ex_dmax_hi = DeltaMaxHiExtractor::new(Some(payload.nmax)).unwrap();
    ex_dmax_hi.feed(windows.iter().copied()).unwrap();
    assert_eq!(
        ex_dmax_hi.current_model(),
        case.ex_dmax_hi,
        "ex_dmax_hi trace={}",
        case.trace_file_name
    );

    let mut ex_dmax_lo = DeltaMaxLoExtractor::new(Some(payload.nmax)).unwrap();
    ex_dmax_lo.feed(windows.iter().copied()).unwrap();
    assert_eq!(
        ex_dmax_lo.current_model(),
        case.ex_dmax_lo,
        "ex_dmax_lo trace={}",
        case.trace_file_name
    );
}
