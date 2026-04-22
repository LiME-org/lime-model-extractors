//! Uncertain periodic-task model inference and streaming extraction.
//!
//! This module mirrors LiME's Python `uncertain_periodic` logic for release
//! windows. It supports two variants:
//! - certain-fit: the model must fully cover every release window
//! - possible-fit: the model must intersect every release window

use crate::{
    certain_periodic::{PeriodicConfig, PeriodicError, PeriodicModel},
    periodic_core::{self, CertainFit, PeriodicExtractorCore, PossibleFit},
    release_window::ReleaseWindow,
};

type Result<T> = std::result::Result<T, PeriodicError>;

/// Alias for the shared periodic-model inference error type.
pub type UncertainPeriodicError = PeriodicError;

/// Infer a certain-fit periodic model from release windows.
pub fn infer_certain_fit_periodic_model<I>(
    release_windows: I,
    config: &PeriodicConfig,
) -> Result<PeriodicModel>
where
    I: IntoIterator<Item = ReleaseWindow>,
{
    periodic_core::infer_model::<CertainFit, _>(release_windows, config)
}

/// Infer a possible-fit periodic model from release windows.
pub fn infer_possible_fit_periodic_model<I>(
    release_windows: I,
    config: &PeriodicConfig,
) -> Result<PeriodicModel>
where
    I: IntoIterator<Item = ReleaseWindow>,
{
    periodic_core::infer_model::<PossibleFit, _>(release_windows, config)
}

/// Streaming extractor for certain-fit periodic models.
pub type CertainFitPeriodicExtractor = PeriodicExtractorCore<CertainFit>;

/// Streaming extractor for possible-fit periodic models.
pub type PossibleFitPeriodicExtractor = PeriodicExtractorCore<PossibleFit>;

#[cfg(test)]
mod tests {
    use crate::certain_periodic::{PeriodicExtractor, infer_periodic_model};

    use super::{
        CertainFitPeriodicExtractor, PeriodicConfig, PeriodicModel, PossibleFitPeriodicExtractor,
        ReleaseWindow, UncertainPeriodicError, infer_certain_fit_periodic_model,
        infer_possible_fit_periodic_model,
    };

    fn exact_windows(releases: &[u64]) -> Vec<ReleaseWindow> {
        releases
            .iter()
            .copied()
            .map(|release| ReleaseWindow::new(release, release))
            .collect()
    }

    fn model_covers_all(model: PeriodicModel, windows: &[ReleaseWindow]) -> bool {
        windows.iter().enumerate().all(|(index, window)| {
            let arrival = index as i64 * model.period as i64 + model.offset;
            arrival <= window.lo as i64 && window.hi as i64 <= arrival + model.jitter
        })
    }

    fn model_intersects_all(model: PeriodicModel, windows: &[ReleaseWindow]) -> bool {
        windows.iter().enumerate().all(|(index, window)| {
            let arrival = index as i64 * model.period as i64 + model.offset;
            let upper = arrival + model.jitter;
            !(arrival > window.hi as i64 || upper < window.lo as i64)
        })
    }

    #[test]
    fn exact_windows_match_certain_periodic_inference() {
        let releases: Vec<u64> = (0..100).map(|index| 7 + 13 * index).collect();
        let windows = exact_windows(&releases);
        let config = PeriodicConfig {
            batch_size: 11,
            overlap: 3,
            n_candidates: 20,
            candidate_dispersion: 2.0,
            rounding_adjustments: vec![-1, 0, 1],
            ..PeriodicConfig::default()
        };

        let exact = infer_periodic_model(releases.iter().copied(), &config).unwrap();
        assert_eq!(
            infer_certain_fit_periodic_model(windows.iter().copied(), &config).unwrap(),
            exact
        );
        assert_eq!(
            infer_possible_fit_periodic_model(windows.iter().copied(), &config).unwrap(),
            exact
        );
    }

    #[test]
    fn uncertain_models_reject_invalid_overlap() {
        let config = PeriodicConfig {
            overlap: 0,
            ..PeriodicConfig::default()
        };
        assert_eq!(
            infer_certain_fit_periodic_model(exact_windows(&[0, 10, 20]), &config).unwrap_err(),
            UncertainPeriodicError::InvalidOverlap
        );
        assert_eq!(
            infer_possible_fit_periodic_model(exact_windows(&[0, 10, 20]), &config).unwrap_err(),
            UncertainPeriodicError::InvalidOverlap
        );
    }

    #[test]
    fn certain_fit_covers_and_possible_fit_intersects_uncertain_windows() {
        let windows = vec![
            ReleaseWindow::new(2, 4),
            ReleaseWindow::new(12, 17),
            ReleaseWindow::new(23, 29),
            ReleaseWindow::new(34, 39),
            ReleaseWindow::new(42, 50),
        ];
        let config = PeriodicConfig {
            batch_size: 5,
            ..PeriodicConfig::default()
        };

        let certain = infer_certain_fit_periodic_model(windows.iter().copied(), &config).unwrap();
        let possible = infer_possible_fit_periodic_model(windows.iter().copied(), &config).unwrap();

        assert!(model_covers_all(certain, &windows));
        assert!(model_intersects_all(possible, &windows));
        assert!(certain.offset <= possible.offset || certain.jitter >= possible.jitter);
    }

    #[test]
    fn extractors_match_one_shot_inference() {
        let releases: Vec<u64> = (0..120).map(|index| 3 + 10 * index + index % 9).collect();
        let windows: Vec<_> = releases
            .iter()
            .enumerate()
            .map(|(index, &release)| {
                let spread = (index as u64 % 5) + 2;
                ReleaseWindow::new(release - spread, release + spread)
            })
            .collect();

        let certain_config = PeriodicConfig {
            batch_size: 16,
            ..PeriodicConfig::default()
        };
        let possible_config = PeriodicConfig {
            batch_size: 24,
            ..PeriodicConfig::default()
        };

        let expected_certain =
            infer_certain_fit_periodic_model(windows.iter().copied(), &certain_config).unwrap();
        let expected_possible =
            infer_possible_fit_periodic_model(windows.iter().copied(), &possible_config).unwrap();

        let mut certain = CertainFitPeriodicExtractor::with_config(certain_config).unwrap();
        let mut possible = PossibleFitPeriodicExtractor::with_config(possible_config).unwrap();

        certain.feed(windows[..40].iter().copied());
        certain.feed(windows[40..80].iter().copied());
        certain.feed(windows[80..].iter().copied());

        possible.feed(windows[..40].iter().copied());
        possible.feed(windows[40..80].iter().copied());
        possible.feed(windows[80..].iter().copied());

        assert_eq!(certain.current_model(), Some(expected_certain));
        assert_eq!(possible.current_model(), Some(expected_possible));

        let mut exact = PeriodicExtractor::with_config(PeriodicConfig {
            batch_size: 16,
            ..PeriodicConfig::default()
        })
        .unwrap();
        exact.feed(releases);
        assert!(exact.current_model().is_some());
    }
}
