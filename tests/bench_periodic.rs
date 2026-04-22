mod common;

use std::hint::black_box;

use common::{LoadedTrace, load_periodic_traces};
use gungraun::{
    Callgrind, EventKind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use lime_model_extractors::{
    PeriodicConfig,
    extractors::{CertainFitPeriodicExtractor, PeriodicExtractor, PossibleFitPeriodicExtractor},
    infer_certain_fit_periodic_model, infer_periodic_model, infer_possible_fit_periodic_model,
};

fn exact_config() -> PeriodicConfig {
    PeriodicConfig {
        batch_size: 128,
        negligible_jitter_threshold: 1_000_000,
        ..PeriodicConfig::default()
    }
}

fn certain_fit_config() -> PeriodicConfig {
    PeriodicConfig {
        batch_size: 512,
        jitter_selection_threshold: 1.65,
        ..PeriodicConfig::default()
    }
}

fn possible_fit_config() -> PeriodicConfig {
    PeriodicConfig {
        batch_size: 512,
        negligible_jitter_threshold: 1_000_000,
        ..PeriodicConfig::default()
    }
}

#[library_benchmark]
#[bench::with_setup(setup = load_periodic_traces)]
fn periodic_infer_exact(traces: Vec<LoadedTrace>) -> usize {
    let config = exact_config();
    let mut work = 0usize;
    for (releases, _) in &traces {
        work += infer_periodic_model(releases.iter().copied(), &config)
            .unwrap()
            .period as usize;
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_periodic_traces)]
fn periodic_infer_certain_fit(traces: Vec<LoadedTrace>) -> usize {
    let config = certain_fit_config();
    let mut work = 0usize;
    for (_, windows) in &traces {
        work += infer_certain_fit_periodic_model(windows.iter().copied(), &config)
            .unwrap()
            .period as usize;
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_periodic_traces)]
fn periodic_infer_possible_fit(traces: Vec<LoadedTrace>) -> usize {
    let config = possible_fit_config();
    let mut work = 0usize;
    for (_, windows) in &traces {
        work += infer_possible_fit_periodic_model(windows.iter().copied(), &config)
            .unwrap()
            .period as usize;
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_periodic_traces)]
fn periodic_extract_exact(traces: Vec<LoadedTrace>) -> usize {
    let config = exact_config();
    let mut work = 0usize;
    for (releases, _) in &traces {
        let mut extractor = PeriodicExtractor::with_config(config.clone()).unwrap();
        extractor.feed(releases.iter().copied());
        work += extractor
            .current_model()
            .map(|model| model.period as usize)
            .unwrap_or_default();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_periodic_traces)]
fn periodic_extract_certain_fit(traces: Vec<LoadedTrace>) -> usize {
    let config = certain_fit_config();
    let mut work = 0usize;
    for (_, windows) in &traces {
        let mut extractor = CertainFitPeriodicExtractor::with_config(config.clone()).unwrap();
        extractor.feed(windows.iter().copied());
        work += extractor
            .current_model()
            .map(|model| model.period as usize)
            .unwrap_or_default();
    }
    black_box(work)
}

#[library_benchmark]
#[bench::with_setup(setup = load_periodic_traces)]
fn periodic_extract_possible_fit(traces: Vec<LoadedTrace>) -> usize {
    let config = possible_fit_config();
    let mut work = 0usize;
    for (_, windows) in &traces {
        let mut extractor = PossibleFitPeriodicExtractor::with_config(config.clone()).unwrap();
        extractor.feed(windows.iter().copied());
        work += extractor
            .current_model()
            .map(|model| model.period as usize)
            .unwrap_or_default();
    }
    black_box(work)
}

library_benchmark_group!(
    name = periodic_profiles;
    benchmarks =
        periodic_infer_exact,
        periodic_infer_certain_fit,
        periodic_infer_possible_fit,
        periodic_extract_exact,
        periodic_extract_certain_fit,
        periodic_extract_possible_fit
);

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::EstimatedCycles, 50f64)])),
    library_benchmark_groups = periodic_profiles
);
