//! Check one-shot and streaming suspension models against Python at successive job prefixes.

mod common;

use std::path::Path;

use common::{python_reference_dir, python_reference_payload, python_reference_python};
use lime_model_extractors::{
    BasicJobSuspensionBehavior, BasicSuspensionModel, ObservedSegment, Segment,
    SegmentedJobSuspensionBehavior, SegmentedSuspensionModel, SuspensionError,
    basic_from_segmented_model, basic_from_segmented_observation,
    extractors::{
        BosSuspensionModelExtractor, MaxSuspensionModelExtractor, MinSuspensionModelExtractor,
        SegmentedSuspensionModelExtractor,
    },
    infer_bos_suspension_model, infer_max_suspension_model, infer_min_suspension_model,
    infer_segmented_suspension_model,
};
use serde::Deserialize;
use similar_asserts::assert_eq;

#[derive(Debug, Deserialize)]
struct BasicSnapshot {
    end: usize,
    maximum: Vec<BasicJobSuspensionBehavior>,
    minimum: Vec<BasicJobSuspensionBehavior>,
}

#[derive(Debug, Deserialize)]
struct BasicCase {
    seed: String,
    observations: Vec<BasicJobSuspensionBehavior>,
    nmax: Option<usize>,
    snapshots: Vec<BasicSnapshot>,
}

#[derive(Debug, Deserialize)]
struct Snapshot<M> {
    end: usize,
    model: Option<M>,
}

#[derive(Debug, Deserialize)]
struct SegmentedCase {
    seed: String,
    observations: Vec<SegmentedJobSuspensionBehavior>,
    max_segments: Option<usize>,
    basic: Vec<BasicJobSuspensionBehavior>,
    snapshots: Vec<Snapshot<Vec<ObservedSegment>>>,
}

#[derive(Debug, Deserialize)]
struct BosCase {
    seed: String,
    observations: Vec<SegmentedJobSuspensionBehavior>,
    max_segments: Option<usize>,
    max_models: Option<usize>,
    snapshots: Vec<Snapshot<Vec<Vec<ObservedSegment>>>>,
}

#[derive(Debug, Deserialize)]
struct Payload {
    revision: String,
    basic: Vec<BasicCase>,
    segmented: Vec<SegmentedCase>,
    bos: Vec<BosCase>,
}

fn basic_model(observations: &[BasicJobSuspensionBehavior]) -> Vec<BasicSuspensionModel> {
    observations
        .iter()
        .map(
            |&(number_of_suspensions, cumulative_duration)| BasicSuspensionModel {
                number_of_suspensions,
                cumulative_duration,
            },
        )
        .collect()
}

fn segmented_model(observations: &[ObservedSegment]) -> SegmentedSuspensionModel {
    observations.iter().copied().map(Segment::from).collect()
}

fn expected_feed_result(
    jobs: &[SegmentedJobSuspensionBehavior],
    max_segments: Option<usize>,
) -> Result<(), SuspensionError> {
    if let Some(limit) = max_segments
        && let Some(job) = jobs.iter().find(|job| job.len() > limit)
    {
        return Err(SuspensionError::SegmentLimitExceeded {
            observed: job.len(),
            limit,
        });
    }
    Ok(())
}

#[test]
fn suspension_models_match_python_reference() {
    let reference_dir = python_reference_dir();
    let python_path = python_reference_python(&reference_dir);
    let driver_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/drivers/suspension_parity_driver.py");
    let payload: Payload = python_reference_payload(
        &python_path,
        &reference_dir,
        &driver_path,
        "suspension parity",
    );
    eprintln!(
        "[suspension] Python revision {}; {} basic, {} segmented, {} BOS cases",
        payload.revision,
        payload.basic.len(),
        payload.segmented.len(),
        payload.bos.len()
    );
    assert!(!payload.basic.is_empty() && !payload.segmented.is_empty() && !payload.bos.is_empty());

    for case in payload.basic {
        let mut maximum = MaxSuspensionModelExtractor::new(case.nmax).unwrap();
        let mut minimum = MinSuspensionModelExtractor::new(case.nmax).unwrap();
        let mut start = 0;
        for snapshot in case.snapshots {
            let batch = &case.observations[start..snapshot.end];
            maximum.feed(batch.iter().copied());
            minimum.feed(batch.iter().copied());
            let expected_max = basic_model(&snapshot.maximum);
            let expected_min = basic_model(&snapshot.minimum);
            let observed = &case.observations[..snapshot.end];
            assert_eq!(
                maximum.current_model(),
                expected_max,
                "max seed={} end={}",
                case.seed,
                snapshot.end
            );
            assert_eq!(
                minimum.current_model(),
                expected_min,
                "min seed={} end={}",
                case.seed,
                snapshot.end
            );
            assert_eq!(
                infer_max_suspension_model(observed.iter().copied(), case.nmax).unwrap(),
                expected_max
            );
            assert_eq!(
                infer_min_suspension_model(observed.iter().copied(), case.nmax).unwrap(),
                expected_min
            );
            start = snapshot.end;
        }
    }
    for case in payload.segmented {
        for (job, &expected) in case.observations.iter().zip(&case.basic) {
            assert_eq!(basic_from_segmented_observation(job), expected);
            assert_eq!(
                basic_from_segmented_model(&segmented_model(job)),
                basic_model(&[expected])[0]
            );
        }
        let mut extractor = SegmentedSuspensionModelExtractor::new(case.max_segments).unwrap();
        let mut start = 0;
        for snapshot in case.snapshots {
            let observed = &case.observations[..snapshot.end];
            assert_eq!(
                extractor.feed(&observed[start..]),
                expected_feed_result(observed, case.max_segments)
            );
            let expected = snapshot.model.as_deref().map(segmented_model);
            assert_eq!(
                extractor.current_model(),
                expected,
                "segmented seed={} end={}",
                case.seed,
                snapshot.end
            );
            assert_eq!(
                infer_segmented_suspension_model(observed, case.max_segments).unwrap(),
                expected
            );
            start = snapshot.end;
        }
    }
    for case in payload.bos {
        let mut extractor =
            BosSuspensionModelExtractor::new(case.max_segments, case.max_models).unwrap();
        let mut start = 0;
        for snapshot in case.snapshots {
            let observed = &case.observations[..snapshot.end];
            assert_eq!(
                extractor.feed(&observed[start..]),
                expected_feed_result(observed, case.max_segments)
            );
            let expected = snapshot.model.map(|bag| {
                bag.iter()
                    .map(|model| segmented_model(model))
                    .collect::<Vec<_>>()
            });
            assert_eq!(
                extractor.current_model(),
                expected,
                "BOS seed={} end={} max_models={:?}",
                case.seed,
                snapshot.end,
                case.max_models
            );
            assert_eq!(
                infer_bos_suspension_model(observed, case.max_segments, case.max_models).unwrap(),
                expected
            );
            start = snapshot.end;
        }
    }
}
