//! Uncertain sporadic-task inference and streaming extraction.
//!
//! This module mirrors `rt_model_inference.uncertain_sporadic` from the Python
//! reference implementation for release windows `(lo, hi)`.

use crate::time::{Duration, Instant};
use crate::{
    certain_sporadic::SporadicError,
    release_window::ReleaseWindow,
    sporadic_core::{self, ApproximationMode, DeltaMaxExtractorCore, DeltaMinExtractorCore},
};

type Result<T> = std::result::Result<T, UncertainSporadicError>;
pub type UncertainSporadicError = SporadicError;

/// Infer a delta-min-hi vector from uncertain release windows.
pub fn infer_delta_min_hi<I>(release_windows: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    I: IntoIterator<Item = ReleaseWindow>,
{
    sporadic_core::infer_delta_min::<DeltaHi, _>(release_windows, nmax)
}

/// Infer a delta-min-lo vector from uncertain release windows.
pub fn infer_delta_min_lo<I>(release_windows: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    I: IntoIterator<Item = ReleaseWindow>,
{
    sporadic_core::infer_delta_min::<DeltaLo, _>(release_windows, nmax)
}

/// Infer a delta-max-hi vector from uncertain release windows.
pub fn infer_delta_max_hi<I>(release_windows: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    I: IntoIterator<Item = ReleaseWindow>,
{
    sporadic_core::infer_delta_max::<DeltaHi, _>(release_windows, nmax)
}

/// Infer a delta-max-lo vector from uncertain release windows.
pub fn infer_delta_max_lo<I>(release_windows: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    I: IntoIterator<Item = ReleaseWindow>,
{
    sporadic_core::infer_delta_max::<DeltaLo, _>(release_windows, nmax)
}

#[derive(Debug, Clone, Copy)]
pub struct DeltaHi;

impl ApproximationMode for DeltaHi {
    type Observation = ReleaseWindow;

    fn ensure_monotonicity(observation: ReleaseWindow, last: Option<&ReleaseWindow>) -> Result<()> {
        observation.ensure_monotonicity(last)
    }

    fn delta_min_interval_end(observation: ReleaseWindow) -> Instant {
        observation.lo
    }

    fn delta_min_interval_start(observation: ReleaseWindow) -> Instant {
        observation.hi
    }

    fn delta_max_interval_end(observation: ReleaseWindow) -> Instant {
        observation.lo
    }

    fn delta_max_interval_start(observation: ReleaseWindow) -> Instant {
        observation.hi
    }

    fn delta_max_initial_observation(observation: ReleaseWindow) -> ReleaseWindow {
        ReleaseWindow::new(observation.lo, observation.hi.saturating_sub(1))
    }

    fn delta_max_closing_observation(observation: ReleaseWindow) -> ReleaseWindow {
        ReleaseWindow::new(observation.lo.saturating_add(1), observation.hi)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeltaLo;

impl ApproximationMode for DeltaLo {
    type Observation = ReleaseWindow;

    fn ensure_monotonicity(observation: ReleaseWindow, last: Option<&ReleaseWindow>) -> Result<()> {
        observation.ensure_monotonicity(last)
    }

    fn delta_min_interval_end(observation: ReleaseWindow) -> Instant {
        observation.hi
    }

    fn delta_min_interval_start(observation: ReleaseWindow) -> Instant {
        observation.lo
    }

    fn delta_max_interval_end(observation: ReleaseWindow) -> Instant {
        observation.hi
    }

    fn delta_max_interval_start(observation: ReleaseWindow) -> Instant {
        observation.lo
    }

    fn delta_max_initial_observation(observation: ReleaseWindow) -> ReleaseWindow {
        ReleaseWindow::new(observation.lo.saturating_sub(1), observation.hi)
    }

    fn delta_max_closing_observation(observation: ReleaseWindow) -> ReleaseWindow {
        ReleaseWindow::new(observation.lo, observation.hi.saturating_add(1))
    }
}

pub type DeltaMinHiExtractor = DeltaMinExtractorCore<DeltaHi>;

pub type DeltaMinLoExtractor = DeltaMinExtractorCore<DeltaLo>;

pub type DeltaMaxHiExtractor = DeltaMaxExtractorCore<DeltaHi>;

pub type DeltaMaxLoExtractor = DeltaMaxExtractorCore<DeltaLo>;

#[cfg(test)]
mod tests {
    use super::{
        DeltaMaxHiExtractor, DeltaMaxLoExtractor, DeltaMinHiExtractor, DeltaMinLoExtractor,
        ReleaseWindow, UncertainSporadicError, infer_delta_max_hi, infer_delta_max_lo,
        infer_delta_min_hi, infer_delta_min_lo,
    };
    use crate::{infer_delta_max, infer_delta_min};

    fn windows(raw: &[(u64, u64)]) -> Vec<ReleaseWindow> {
        raw.iter().copied().map(ReleaseWindow::from).collect()
    }

    #[test]
    fn exact_windows_match_certain_dmin() {
        let releases = [0, 2, 4, 100, 101, 200];
        let exact_windows = windows(&releases.map(|r| (r, r)));

        assert_eq!(
            infer_delta_min_hi(exact_windows.clone(), Some(4)).unwrap(),
            infer_delta_min(releases, Some(4)).unwrap()
        );
        assert_eq!(
            infer_delta_min_lo(exact_windows, Some(4)).unwrap(),
            infer_delta_min(releases, Some(4)).unwrap()
        );
    }

    #[test]
    fn exact_windows_match_certain_dmax() {
        let releases = [0, 2, 4, 100, 101, 200];
        let exact_windows = windows(&releases.map(|r| (r, r)));

        assert_eq!(
            infer_delta_max_hi(exact_windows.clone(), Some(4)).unwrap(),
            infer_delta_max(releases, Some(4)).unwrap()
        );
        assert_eq!(
            infer_delta_max_lo(exact_windows, Some(4)).unwrap(),
            infer_delta_max(releases, Some(4)).unwrap()
        );
    }

    #[test]
    fn uncertain_windows_affect_dmin_hi_and_lo() {
        let input = windows(&[(0, 3), (5, 7), (9, 10), (20, 21)]);

        assert_eq!(
            infer_delta_min_hi(input.clone(), None).unwrap(),
            vec![0, 1, 3, 7, 18]
        );
        assert_eq!(
            infer_delta_min_lo(input, None).unwrap(),
            vec![0, 1, 6, 11, 22]
        );
    }

    #[test]
    fn overlapping_windows_keep_positive_dmin_hi() {
        let input = windows(&[(0, 10), (1, 11), (2, 12)]);
        assert_eq!(infer_delta_min_hi(input, None).unwrap(), vec![0, 1, 1, 1]);
    }

    #[test]
    fn dmin_and_dmax_empty_inputs_are_empty() {
        assert!(
            infer_delta_min_hi(Vec::<ReleaseWindow>::new(), None)
                .unwrap()
                .is_empty()
        );
        assert!(
            infer_delta_min_lo(Vec::<ReleaseWindow>::new(), None)
                .unwrap()
                .is_empty()
        );
        assert!(
            infer_delta_max_hi(Vec::<ReleaseWindow>::new(), None)
                .unwrap()
                .is_empty()
        );
        assert!(
            infer_delta_max_lo(Vec::<ReleaseWindow>::new(), None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn monotonicity_errors_match_python_behavior() {
        let bad = windows(&[(0, 1), (5, 6), (4, 7)]);
        assert_eq!(
            infer_delta_min_hi(bad.clone(), None).unwrap_err(),
            UncertainSporadicError::NonMonotonicLowerBounds
        );
        assert_eq!(
            infer_delta_min_lo(bad.clone(), None).unwrap_err(),
            UncertainSporadicError::NonMonotonicLowerBounds
        );
        assert_eq!(
            infer_delta_max_hi(bad.clone(), None).unwrap_err(),
            UncertainSporadicError::NonMonotonicLowerBounds
        );
        assert_eq!(
            infer_delta_max_lo(bad, None).unwrap_err(),
            UncertainSporadicError::NonMonotonicLowerBounds
        );
    }

    #[test]
    fn upper_bound_monotonicity_errors_are_detected() {
        let bad = windows(&[(0, 3), (1, 4), (2, 3)]);
        assert_eq!(
            infer_delta_min_hi(bad.clone(), None).unwrap_err(),
            UncertainSporadicError::NonMonotonicUpperBounds
        );
        assert_eq!(
            infer_delta_max_lo(bad, None).unwrap_err(),
            UncertainSporadicError::NonMonotonicUpperBounds
        );
    }

    #[test]
    fn nmax_validation_matches_python_constraints() {
        assert_eq!(
            DeltaMinHiExtractor::new(Some(0)).unwrap_err(),
            UncertainSporadicError::InvalidNmax {
                minimum: 2,
                actual: 0
            }
        );
        assert_eq!(
            DeltaMinLoExtractor::new(Some(1)).unwrap_err(),
            UncertainSporadicError::InvalidNmax {
                minimum: 2,
                actual: 1
            }
        );
        assert_eq!(
            DeltaMaxHiExtractor::new(Some(0)).unwrap_err(),
            UncertainSporadicError::InvalidNmax {
                minimum: 1,
                actual: 0
            }
        );
        assert_eq!(
            DeltaMaxLoExtractor::new(Some(0)).unwrap_err(),
            UncertainSporadicError::InvalidNmax {
                minimum: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn hi_and_lo_extractors_match_one_shot_inference() {
        let input = windows(&[(0, 3), (5, 7), (9, 10), (20, 21), (30, 31), (32, 40)]);
        let chunk_size = 2;

        let mut dmin_hi = DeltaMinHiExtractor::new(Some(4)).unwrap();
        let mut dmin_lo = DeltaMinLoExtractor::new(Some(4)).unwrap();
        let mut dmax_hi = DeltaMaxHiExtractor::new(Some(4)).unwrap();
        let mut dmax_lo = DeltaMaxLoExtractor::new(Some(4)).unwrap();
        let mut observed = Vec::new();

        for batch in input.chunks(chunk_size) {
            dmin_hi.feed(batch.iter().copied()).unwrap();
            dmin_lo.feed(batch.iter().copied()).unwrap();
            dmax_hi.feed(batch.iter().copied()).unwrap();
            dmax_lo.feed(batch.iter().copied()).unwrap();
            observed.extend_from_slice(batch);

            assert_eq!(
                dmin_hi.current_model(),
                infer_delta_min_hi(observed.iter().copied(), Some(4)).unwrap()
            );
            assert_eq!(
                dmin_lo.current_model(),
                infer_delta_min_lo(observed.iter().copied(), Some(4)).unwrap()
            );
            assert_eq!(
                dmax_hi.current_model(),
                infer_delta_max_hi(observed.iter().copied(), Some(4)).unwrap()
            );
            assert_eq!(
                dmax_lo.current_model(),
                infer_delta_max_lo(observed.iter().copied(), Some(4)).unwrap()
            );
        }
    }
}
