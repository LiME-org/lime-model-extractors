mod common;

use std::path::{Path, PathBuf};

use common::{
    load_release_trace, progress_bar, python_reference_dir, python_reference_payload,
    python_reference_python,
};
use lime_model_extractors::{
    PeriodicConfig, PeriodicModel,
    extractors::{
        CertainFitPeriodicExtractor, DeltaMaxExtractor, DeltaMaxLoExtractor, DeltaMinExtractor,
        DeltaMinHiExtractor, PeriodicExtractor, PossibleFitPeriodicExtractor,
    },
    infer_certain_fit_periodic_model, infer_delta_max, infer_delta_max_lo, infer_delta_min,
    infer_delta_min_hi, infer_periodic_model, infer_possible_fit_periodic_model,
    time::Duration,
};
use rayon::prelude::*;
use serde::Deserialize;
use similar_asserts::assert_eq;

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct RawPeriodicModel {
    period: Duration,
    offset: i64,
    jitter: i64,
}

#[derive(Debug, Deserialize, PartialEq)]
struct TraceCase {
    trace_file_name: String,
    expected_period: Option<Duration>,
    exact: RawPeriodicModel,
    certain_fit: RawPeriodicModel,
    possible_fit: RawPeriodicModel,
    extractor_exact: RawPeriodicModel,
    extractor_certain_fit: RawPeriodicModel,
    extractor_possible_fit: RawPeriodicModel,
    dmin: Vec<Duration>,
    dmax: Vec<Duration>,
    dmin_hi: Vec<Duration>,
    dmax_lo: Vec<Duration>,
    ex_dmin: Vec<Duration>,
    ex_dmax: Vec<Duration>,
    ex_dmin_hi: Vec<Duration>,
    ex_dmax_lo: Vec<Duration>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Payload {
    test_all_traces: bool,
    trace_dir: PathBuf,
    trace_file_names: Vec<String>,
    sporadic_cutoff: Option<usize>,
    nmax: usize,
    cases: Vec<TraceCase>,
}

#[test]
fn periodic_traces_match_python_reference() {
    let reference_dir = python_reference_dir();
    let python_path = match python_reference_python(&reference_dir) {
        Some(path) => path,
        None => {
            eprintln!(
                "skipping periodic traces parity test: reference virtualenv is not initialized"
            );
            return;
        }
    };
    let driver_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("drivers")
        .join("periodic_traces_parity_driver.py");

    if !reference_dir.join("pyproject.toml").exists() {
        eprintln!(
            "skipping periodic traces parity test: Python reference checkout is not available"
        );
        return;
    }

    assert!(
        driver_path.exists(),
        "missing periodic traces parity driver"
    );

    let payload: Payload = python_reference_payload(
        &python_path,
        &reference_dir,
        &driver_path,
        "periodic trace parity",
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
        "[periodic_traces] loaded {} traces from {} with sporadic cutoff {:?} and nmax {}",
        payload.cases.len(),
        payload.trace_dir.display(),
        payload.sporadic_cutoff,
        payload.nmax
    );

    let progress = progress_bar("periodic_traces", payload.cases.len());
    payload.cases.par_iter().for_each(|case| {
        assert_periodic_trace_case(&payload, case);
        progress.inc(1);
    });
    progress.finish_with_message("done");
}

fn to_model(raw: &RawPeriodicModel) -> PeriodicModel {
    PeriodicModel {
        period: raw.period,
        offset: raw.offset,
        jitter: raw.jitter,
    }
}

fn assert_periodic_trace_case(payload: &Payload, case: &TraceCase) {
    let trace_path = payload.trace_dir.join(&case.trace_file_name);
    let (releases, windows) = load_release_trace(&trace_path, None);

    let exact_config = PeriodicConfig {
        batch_size: 128,
        negligible_jitter_threshold: 1_000_000,
        ..PeriodicConfig::default()
    };
    let exact = infer_periodic_model(releases.iter().copied(), &exact_config).unwrap();
    assert_eq!(
        exact,
        to_model(&case.exact),
        "exact trace={}",
        case.trace_file_name
    );

    let certain_fit_config = PeriodicConfig {
        batch_size: 512,
        jitter_selection_threshold: 1.65,
        ..PeriodicConfig::default()
    };
    let certain_fit =
        infer_certain_fit_periodic_model(windows.iter().copied(), &certain_fit_config).unwrap();
    assert_eq!(
        certain_fit,
        to_model(&case.certain_fit),
        "certain-fit trace={}",
        case.trace_file_name
    );

    let possible_fit_config = PeriodicConfig {
        batch_size: 512,
        negligible_jitter_threshold: 1_000_000,
        ..PeriodicConfig::default()
    };
    let possible_fit =
        infer_possible_fit_periodic_model(windows.iter().copied(), &possible_fit_config).unwrap();
    assert_eq!(
        possible_fit,
        to_model(&case.possible_fit),
        "possible-fit trace={}",
        case.trace_file_name
    );

    if let Some(expected_period) = case.expected_period {
        assert_eq!(
            exact.period, expected_period,
            "expected period trace={}",
            case.trace_file_name
        );
    }

    assert_feed_modes(
        &releases,
        Some(to_model(&case.extractor_exact)),
        || PeriodicExtractor::with_config(exact_config.clone()).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()),
        |extractor| extractor.current_model(),
        "extractor exact",
        &case.trace_file_name,
    );

    assert_feed_modes(
        &windows,
        Some(to_model(&case.extractor_certain_fit)),
        || CertainFitPeriodicExtractor::with_config(certain_fit_config.clone()).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()),
        |extractor| extractor.current_model(),
        "extractor certain-fit",
        &case.trace_file_name,
    );

    assert_feed_modes(
        &windows,
        Some(to_model(&case.extractor_possible_fit)),
        || PossibleFitPeriodicExtractor::with_config(possible_fit_config.clone()).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()),
        |extractor| extractor.current_model(),
        "extractor possible-fit",
        &case.trace_file_name,
    );

    let (sporadic_releases, sporadic_windows) =
        load_release_trace(&trace_path, payload.sporadic_cutoff);

    assert_eq!(
        infer_delta_min(sporadic_releases.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmin,
        "dmin trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_max(sporadic_releases.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmax,
        "dmax trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_min_hi(sporadic_windows.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmin_hi,
        "dmin_hi trace={}",
        case.trace_file_name
    );
    assert_eq!(
        infer_delta_max_lo(sporadic_windows.iter().copied(), Some(payload.nmax)).unwrap(),
        case.dmax_lo,
        "dmax_lo trace={}",
        case.trace_file_name
    );

    assert_feed_modes(
        &sporadic_releases,
        case.ex_dmin.clone(),
        || DeltaMinExtractor::new(Some(payload.nmax)).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()).unwrap(),
        |extractor| extractor.current_model(),
        "ex_dmin",
        &case.trace_file_name,
    );

    assert_feed_modes(
        &sporadic_releases,
        case.ex_dmax.clone(),
        || DeltaMaxExtractor::new(Some(payload.nmax)).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()).unwrap(),
        |extractor| extractor.current_model(),
        "ex_dmax",
        &case.trace_file_name,
    );

    assert_feed_modes(
        &sporadic_windows,
        case.ex_dmin_hi.clone(),
        || DeltaMinHiExtractor::new(Some(payload.nmax)).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()).unwrap(),
        |extractor| extractor.current_model(),
        "ex_dmin_hi",
        &case.trace_file_name,
    );

    assert_feed_modes(
        &sporadic_windows,
        case.ex_dmax_lo.clone(),
        || DeltaMaxLoExtractor::new(Some(payload.nmax)).unwrap(),
        |extractor, batch| extractor.feed(batch.iter().copied()).unwrap(),
        |extractor| extractor.current_model(),
        "ex_dmax_lo",
        &case.trace_file_name,
    );
}

const FEED_BATCH_SIZES: [usize; 4] = [16, 128, 512, 2048];

fn assert_feed_modes<E, T, O, New, Feed, Current>(
    inputs: &[T],
    expected: O,
    mut new_extractor: New,
    mut feed: Feed,
    mut current_model: Current,
    label: &str,
    trace_file_name: &str,
) where
    T: Copy,
    O: Clone + std::fmt::Debug + PartialEq,
    New: FnMut() -> E,
    Feed: FnMut(&mut E, &[T]),
    Current: FnMut(&E) -> O,
{
    let mut extractor = new_extractor();
    feed(&mut extractor, inputs);
    assert_eq!(
        current_model(&extractor),
        expected.clone(),
        "{label} feed_all trace={trace_file_name}"
    );

    for batch_size in FEED_BATCH_SIZES {
        let mut extractor = new_extractor();
        for chunk in inputs.chunks(batch_size) {
            feed(&mut extractor, chunk);
        }
        assert_eq!(
            current_model(&extractor),
            expected.clone(),
            "{label} feed_batches={batch_size} trace={trace_file_name}"
        );
    }
}
