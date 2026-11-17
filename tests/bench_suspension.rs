//! Benchmark suspension bounds, BOS coverage checks, and least-cost merges.

use std::hint::black_box;

use gungraun::{
    Callgrind, EventKind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use lime_model_extractors::{
    BagOfSegmentsModel, BasicJobSuspensionBehavior, BasicSuspensionModel,
    SegmentedJobSuspensionBehavior, SegmentedSuspensionModel,
    extractors::{
        BosSuspensionModelExtractor, MaxSuspensionModelExtractor, MinSuspensionModelExtractor,
        SegmentedSuspensionModelExtractor,
    },
};

fn basic_observations(nmax: Option<usize>) -> (Vec<BasicJobSuspensionBehavior>, Option<usize>) {
    let observations = (0..4096).map(|i| (i % 11, (i * 71 + 31) % 997)).collect();
    (observations, nmax)
}

fn segmented_observations() -> Vec<SegmentedJobSuspensionBehavior> {
    // Increasing the bounds keeps jobs uncovered so the full bag still has to select merges.
    (0..2048)
        .map(|job| {
            (0..8)
                .map(|segment| {
                    (
                        job * 3 + (job * 71 + segment * 31) % 101,
                        job + 1 + (job * 37 + segment * 53) % 101,
                    )
                })
                .collect()
        })
        .collect()
}

fn bos_observations(max_models: usize) -> (Vec<SegmentedJobSuspensionBehavior>, usize) {
    (segmented_observations(), max_models)
}

fn covered_observations() -> (Vec<SegmentedJobSuspensionBehavior>, usize) {
    (vec![vec![(5, 10); 8]; 16384], 8)
}

#[library_benchmark]
#[benches::limits(args = [Some(1), Some(32), Some(256), None], setup = basic_observations)]
fn maximum_suspension(
    (observations, nmax): (Vec<BasicJobSuspensionBehavior>, Option<usize>),
) -> Vec<BasicSuspensionModel> {
    let mut extractor = MaxSuspensionModelExtractor::new(nmax).unwrap();
    for batch in observations.chunks(32) {
        extractor.feed(batch.iter().copied());
    }
    black_box(extractor.into_model())
}

#[library_benchmark]
#[benches::limits(args = [Some(1), Some(32), Some(256), None], setup = basic_observations)]
fn minimum_suspension(
    (observations, nmax): (Vec<BasicJobSuspensionBehavior>, Option<usize>),
) -> Vec<BasicSuspensionModel> {
    let mut extractor = MinSuspensionModelExtractor::new(nmax).unwrap();
    for batch in observations.chunks(32) {
        extractor.feed(batch.iter().copied());
    }
    black_box(extractor.into_model())
}

#[library_benchmark]
#[bench::with_setup(setup = segmented_observations)]
fn segmented_suspension(
    observations: Vec<SegmentedJobSuspensionBehavior>,
) -> Option<SegmentedSuspensionModel> {
    let mut extractor = SegmentedSuspensionModelExtractor::new(Some(8)).unwrap();
    for batch in observations.chunks(32) {
        extractor.feed(batch).unwrap();
    }
    black_box(extractor.into_model())
}

#[library_benchmark]
#[benches::limits(args = [1, 8, 32, 64], setup = bos_observations)]
#[bench::already_covered(setup = covered_observations)]
fn bos_suspension(
    (observations, max_models): (Vec<SegmentedJobSuspensionBehavior>, usize),
) -> Option<BagOfSegmentsModel> {
    let mut extractor = BosSuspensionModelExtractor::new(Some(8), Some(max_models)).unwrap();
    for batch in observations.chunks(32) {
        extractor.feed(batch).unwrap();
    }
    black_box(extractor.into_model())
}

library_benchmark_group!(
    name = suspension_profiles;
    benchmarks = maximum_suspension, minimum_suspension, segmented_suspension, bos_suspension
);

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::EstimatedCycles, 50f64)])),
    library_benchmark_groups = suspension_profiles
);
