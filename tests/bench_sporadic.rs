mod common;

use std::hint::black_box;

use common::{DEFAULT_NMAX, LoadedTrace, load_sporadic_traces};
use gungraun::{
    Callgrind, EventKind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use lime_model_extractors::{
    extractors::{
        DeltaMaxExtractor, DeltaMaxHiExtractor, DeltaMaxLoExtractor, DeltaMinExtractor,
        DeltaMinHiExtractor, DeltaMinLoExtractor,
    },
    infer_delta_max, infer_delta_max_hi, infer_delta_max_lo, infer_delta_min, infer_delta_min_hi,
    infer_delta_min_lo,
};

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_infer_delta_min(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (releases, _) in &traces {
        work += infer_delta_min(releases.iter().copied(), Some(DEFAULT_NMAX))
            .unwrap()
            .len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_infer_delta_max(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (releases, _) in &traces {
        work += infer_delta_max(releases.iter().copied(), Some(DEFAULT_NMAX))
            .unwrap()
            .len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_infer_delta_min_hi(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        work += infer_delta_min_hi(windows.iter().copied(), Some(DEFAULT_NMAX))
            .unwrap()
            .len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_infer_delta_min_lo(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        work += infer_delta_min_lo(windows.iter().copied(), Some(DEFAULT_NMAX))
            .unwrap()
            .len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_infer_delta_max_hi(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        work += infer_delta_max_hi(windows.iter().copied(), Some(DEFAULT_NMAX))
            .unwrap()
            .len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_infer_delta_max_lo(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        work += infer_delta_max_lo(windows.iter().copied(), Some(DEFAULT_NMAX))
            .unwrap()
            .len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_extract_delta_min(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (releases, _) in &traces {
        let mut extractor = DeltaMinExtractor::new(Some(DEFAULT_NMAX)).unwrap();
        extractor.feed(releases.iter().copied()).unwrap();
        work += extractor.current_model().len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_extract_delta_max(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (releases, _) in &traces {
        let mut extractor = DeltaMaxExtractor::new(Some(DEFAULT_NMAX)).unwrap();
        extractor.feed(releases.iter().copied()).unwrap();
        work += extractor.current_model().len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_extract_delta_min_hi(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        let mut extractor = DeltaMinHiExtractor::new(Some(DEFAULT_NMAX)).unwrap();
        extractor.feed(windows.iter().copied()).unwrap();
        work += extractor.current_model().len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_extract_delta_min_lo(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        let mut extractor = DeltaMinLoExtractor::new(Some(DEFAULT_NMAX)).unwrap();
        extractor.feed(windows.iter().copied()).unwrap();
        work += extractor.current_model().len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_extract_delta_max_hi(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        let mut extractor = DeltaMaxHiExtractor::new(Some(DEFAULT_NMAX)).unwrap();
        extractor.feed(windows.iter().copied()).unwrap();
        work += extractor.current_model().len();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_sporadic_traces)]
fn sporadic_extract_delta_max_lo(traces: Vec<LoadedTrace>) -> usize {
    let mut work = 0usize;
    for (_, windows) in &traces {
        let mut extractor = DeltaMaxLoExtractor::new(Some(DEFAULT_NMAX)).unwrap();
        extractor.feed(windows.iter().copied()).unwrap();
        work += extractor.current_model().len();
    }
    black_box(work)
}

library_benchmark_group!(
    name = sporadic_profiles;
    benchmarks =
        sporadic_infer_delta_min,
        sporadic_infer_delta_max,
        sporadic_infer_delta_min_hi,
        sporadic_infer_delta_min_lo,
        sporadic_infer_delta_max_hi,
        sporadic_infer_delta_max_lo,
        sporadic_extract_delta_min,
        sporadic_extract_delta_max,
        sporadic_extract_delta_min_hi,
        sporadic_extract_delta_min_lo,
        sporadic_extract_delta_max_hi,
        sporadic_extract_delta_max_lo
);

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::EstimatedCycles, 50f64)])),
    library_benchmark_groups = sporadic_profiles
);
