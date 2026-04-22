//! Certain sporadic-task model inference and streaming extraction.
//!
//! This module mirrors the discrete-time logic from the Python
//! `rt_model_inference.certain_sporadic` implementation. Time is modeled as
//! discrete unsigned integers with a smallest unit of one tick.

use thiserror::Error;

use crate::{
    sporadic_core::{self, ApproximationMode, DeltaMaxExtractorCore, DeltaMinExtractorCore},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, SporadicError>;

/// Errors returned by certain-sporadic inference and extraction.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SporadicError {
    /// Releases were not provided in nondecreasing order.
    #[error("releases must be monotonic")]
    NonMonotonicReleases,
    /// Release-window lower bounds were not provided in nondecreasing order.
    #[error("release-window lower bounds must be monotonic")]
    NonMonotonicLowerBounds,
    /// Release-window upper bounds were not provided in nondecreasing order.
    #[error("release-window upper bounds must be monotonic")]
    NonMonotonicUpperBounds,
    /// At least two releases are needed to infer a classic sporadic model.
    #[error("need at least two releases to infer a sporadic task model")]
    InsufficientReleases,
    /// An invalid bound was requested.
    #[error("nmax must be at least {minimum}, got {actual}")]
    InvalidNmax { minimum: usize, actual: usize },
}

impl SporadicError {
    pub(crate) fn validate(nmax: Option<usize>, minimum: usize) -> std::result::Result<(), Self> {
        if let Some(actual) = nmax
            && actual < minimum
        {
            return Err(Self::InvalidNmax { minimum, actual });
        }

        Ok(())
    }
}

/// Infer a delta-min vector from a release sequence.
///
/// The returned vector `dmin` satisfies `dmin[n] = shortest interval containing
/// at least n releases`. For non-empty input, `dmin[0] == 0` and
/// `dmin[1] == 1`.
pub fn infer_delta_min<I>(releases: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    I: IntoIterator<Item = Instant>,
{
    sporadic_core::infer_delta_min::<PointRelease, _>(releases, nmax)
}

/// Infer the minimum observed separation between consecutive releases.
pub fn infer_sporadic_model<I>(releases: I) -> Result<Duration>
where
    I: IntoIterator<Item = Instant>,
{
    let mut extractor = SporadicExtractor::new();
    extractor.feed(releases)?;
    extractor
        .current_model()
        .ok_or(SporadicError::InsufficientReleases)
}

/// Given a delta-min vector and an interval length, determine the maximum number
/// of releases possible in any interval of that length.
///
/// Returns `None` when the vector does not cover the given interval.
pub fn max_releases(delta_min: &[Duration], delta: Duration) -> Option<usize> {
    if delta_min.is_empty() || delta >= *delta_min.last().unwrap() {
        return None;
    }

    Some(
        delta_min
            .partition_point(|&dmin| dmin <= delta)
            .saturating_sub(1),
    )
}

/// Infer a delta-max vector from a release sequence.
///
/// The returned vector `dmax` satisfies `dmax[n] = longest interval containing
/// at most n releases`.
pub fn infer_delta_max<I>(releases: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    I: IntoIterator<Item = Instant>,
{
    sporadic_core::infer_delta_max::<PointRelease, _>(releases, nmax)
}

/// Given a delta-max vector and an interval length, determine the minimum number
/// of releases present in any interval of that length.
///
/// Returns `None` when the vector does not cover the given interval.
pub fn min_releases(delta_max: &[Duration], delta: Duration) -> Option<usize> {
    if delta_max.is_empty() || delta > *delta_max.last().unwrap() {
        return None;
    }

    Some(delta_max.partition_point(|&dmax| dmax < delta))
}

/// Streaming extractor for the classic sporadic task model.
#[derive(Debug, Default, Clone)]
pub struct SporadicExtractor {
    last_release: Option<Instant>,
    min_separation: Option<Duration>,
}

impl SporadicExtractor {
    /// Construct an empty extractor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the extractor with additional releases.
    pub fn feed<I>(&mut self, releases: I) -> Result<()>
    where
        I: IntoIterator<Item = Instant>,
    {
        for r in releases {
            if let Some(last_release) = self.last_release {
                if r < last_release {
                    return Err(SporadicError::NonMonotonicReleases);
                }
                let separation = r - last_release;
                self.min_separation = if let Some(min_sep) = self.min_separation {
                    Some(min_sep.min(separation))
                } else {
                    Some(separation)
                };
            }
            self.last_release = Some(r);
        }

        Ok(())
    }

    /// Query the current model estimate.
    pub fn current_model(&self) -> Option<Duration> {
        self.min_separation
    }
}

/// Streaming extractor for delta-min vectors.
#[derive(Debug, Clone, Copy)]
pub struct PointRelease;

impl ApproximationMode for PointRelease {
    type Observation = Instant;

    fn ensure_monotonicity(observation: Instant, last: Option<&Instant>) -> Result<()> {
        if last.is_some_and(|last| *last > observation) {
            return Err(SporadicError::NonMonotonicReleases);
        }

        Ok(())
    }

    fn delta_min_interval_end(observation: Instant) -> Instant {
        observation
    }

    fn delta_min_interval_start(observation: Instant) -> Instant {
        observation
    }

    fn delta_max_interval_end(observation: Instant) -> Instant {
        observation
    }

    fn delta_max_interval_start(observation: Instant) -> Instant {
        observation
    }

    fn delta_max_initial_observation(observation: Instant) -> Instant {
        observation.saturating_sub(1)
    }

    fn delta_max_closing_observation(observation: Instant) -> Instant {
        observation.saturating_add(1)
    }
}

/// Streaming extractor for delta-min vectors.
pub type DeltaMinExtractor = DeltaMinExtractorCore<PointRelease>;

/// Streaming extractor for delta-max vectors.
pub type DeltaMaxExtractor = DeltaMaxExtractorCore<PointRelease>;

#[cfg(test)]
mod tests {
    use super::{
        DeltaMaxExtractor, DeltaMinExtractor, SporadicError, SporadicExtractor, infer_delta_max,
        infer_delta_min, infer_sporadic_model, max_releases, min_releases,
    };

    #[test]
    fn infer_delta_min_for_periodic_trace() {
        let releases: Vec<_> = (123..1000).step_by(50).collect();
        let dmin = infer_delta_min(releases, Some(5)).unwrap();
        assert_eq!(dmin, vec![0, 1, 51, 101, 151, 201]);
    }

    #[test]
    fn infer_delta_min_handles_irregular_bursts() {
        let dmin = infer_delta_min([0, 2, 4, 100, 101, 200], Some(4)).unwrap();
        assert_eq!(dmin, vec![0, 1, 2, 5, 100]);
    }

    #[test]
    fn infer_delta_min_handles_duplicates() {
        let dmin = infer_delta_min([5, 5, 6, 10, 10, 10], Some(3)).unwrap();
        assert_eq!(dmin, vec![0, 1, 1, 1]);
    }

    #[test]
    fn infer_delta_min_empty_sequence_is_empty() {
        let dmin = infer_delta_min([], None).unwrap();
        assert!(dmin.is_empty());
    }

    #[test]
    fn infer_delta_min_rejects_non_monotonic_input() {
        let err = infer_delta_min([0, 5, 3, 6], None).unwrap_err();
        assert_eq!(err, SporadicError::NonMonotonicReleases);
    }

    #[test]
    fn infer_sporadic_model_extracts_minimum_separation() {
        let min_sep = infer_sporadic_model([5, 8, 8, 14, 25]).unwrap();
        assert_eq!(min_sep, 0);
    }

    #[test]
    fn infer_sporadic_model_matches_delta_min_for_two_releases() {
        let releases = [0, 2, 4, 100, 101, 200];
        let min_sep = infer_sporadic_model(releases).unwrap();
        let dmin = infer_delta_min(releases, Some(2)).unwrap();
        assert_eq!(min_sep, dmin[2] - 1);
    }

    #[test]
    fn infer_sporadic_model_needs_two_releases() {
        let err = infer_sporadic_model([7]).unwrap_err();
        assert_eq!(err, SporadicError::InsufficientReleases);
    }

    #[test]
    fn infer_delta_max_for_periodic_trace() {
        let releases: Vec<_> = (123..1000).step_by(50).collect();
        let dmax = infer_delta_max(releases, Some(5)).unwrap();
        assert_eq!(dmax, vec![49, 99, 149, 199, 249, 299]);
    }

    #[test]
    fn infer_delta_max_handles_irregular_bursts() {
        let dmax = infer_delta_max([0, 2, 4, 100, 101, 200], Some(4)).unwrap();
        assert_eq!(dmax, vec![98, 99, 195, 197, 199]);
    }

    #[test]
    fn infer_delta_max_handles_duplicates() {
        let dmax = infer_delta_max([5, 5, 6, 10, 10, 10], Some(3)).unwrap();
        assert_eq!(dmax, vec![3, 4, 4, 5]);
    }

    #[test]
    fn infer_delta_max_empty_sequence_is_empty() {
        let dmax = infer_delta_max([], None).unwrap();
        assert!(dmax.is_empty());
    }

    #[test]
    fn release_count_queries_match_python_behavior() {
        let dmax = vec![9, 19, 29, 39, 49, 59];
        assert_eq!(min_releases(&dmax, 5), Some(0));
        assert_eq!(min_releases(&dmax, 20), Some(2));
        assert_eq!(min_releases(&dmax, 30), Some(3));

        let dmin = vec![0, 1, 11, 21, 31, 41, 51, 52];
        assert_eq!(max_releases(&dmin, 0), Some(0));
        assert_eq!(max_releases(&dmin, 11), Some(2));
        assert_eq!(max_releases(&dmin, 31), Some(4));
    }

    #[test]
    fn sporadic_extractor_matches_one_shot_inference() {
        let releases = [0, 2, 4, 100, 101, 200];
        let mut extractor = SporadicExtractor::new();

        extractor.feed([0, 2, 4]).unwrap();
        assert_eq!(extractor.current_model(), Some(2));

        extractor.feed([100, 101, 200]).unwrap();
        assert_eq!(
            extractor.current_model(),
            Some(infer_sporadic_model(releases).unwrap())
        );
    }

    #[test]
    fn delta_min_extractor_matches_one_shot_inference() {
        let releases = [0, 2, 4, 100, 101, 200];
        let mut extractor = DeltaMinExtractor::new(Some(4)).unwrap();

        extractor.feed([0, 2, 4]).unwrap();
        extractor.feed([100, 101, 200]).unwrap();

        assert_eq!(
            extractor.current_model(),
            infer_delta_min(releases, Some(4)).unwrap()
        );
    }

    #[test]
    fn delta_max_extractor_matches_one_shot_inference() {
        let releases = [0, 2, 4, 100, 101, 200];
        let mut extractor = DeltaMaxExtractor::new(Some(4)).unwrap();

        extractor.feed([0, 2, 4]).unwrap();
        extractor.feed([100, 101, 200]).unwrap();

        assert_eq!(
            extractor.current_model(),
            infer_delta_max(releases, Some(4)).unwrap()
        );
    }
}
