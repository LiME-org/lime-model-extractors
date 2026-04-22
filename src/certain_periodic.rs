//! Certain periodic-task model inference and streaming extraction.
//!
//! This module mirrors LiME's Python `certain_periodic` heuristic while using
//! a Rust-native configuration API.

use thiserror::Error;

use crate::{
    periodic_core::{self, PeriodicExtractorCore, PointRelease},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, PeriodicError>;

/// A periodic model with offset and release jitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PeriodicModel {
    pub period: Duration,
    pub offset: i64,
    pub jitter: i64,
}

/// Configuration for periodic-model inference.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicConfig {
    pub batch_size: usize,
    pub overlap: usize,
    pub n_candidates: usize,
    pub candidate_dispersion: f64,
    pub rounding_adjustments: Vec<i64>,
    pub jitter_pruning_threshold: f64,
    pub jitter_selection_threshold: f64,
    pub negligible_jitter_threshold: i64,
}

impl Default for PeriodicConfig {
    fn default() -> Self {
        Self {
            batch_size: 4096,
            overlap: 1,
            n_candidates: 50,
            candidate_dispersion: 3.0,
            rounding_adjustments: vec![-2, -1, 0, 1, 2],
            jitter_pruning_threshold: 5.0,
            jitter_selection_threshold: 1.25,
            negligible_jitter_threshold: 0,
        }
    }
}

/// Errors returned by periodic-model inference.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PeriodicError {
    #[error("overlap must be at least 1")]
    InvalidOverlap,
    #[error("need at least 3 candidates")]
    InvalidCandidateCount,
    #[error("infeasible batch size {batch_size} with overlap {overlap}")]
    InfeasibleBatching { batch_size: usize, overlap: usize },
    #[error("need at least two releases to infer a periodic task model")]
    InsufficientReleases,
    #[error("unable to derive any periodic model candidates")]
    NoCandidateModels,
}

impl PeriodicConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.overlap == 0 {
            return Err(PeriodicError::InvalidOverlap);
        }
        if self.n_candidates < 3 {
            return Err(PeriodicError::InvalidCandidateCount);
        }
        if self.batch_size <= self.overlap {
            return Err(PeriodicError::InfeasibleBatching {
                batch_size: self.batch_size,
                overlap: self.overlap,
            });
        }
        Ok(())
    }
}

/// Infer a periodic model from a release sequence.
pub fn infer_periodic_model<I>(releases: I, config: &PeriodicConfig) -> Result<PeriodicModel>
where
    I: IntoIterator<Item = Instant>,
{
    periodic_core::infer_model::<PointRelease, _>(releases, config)
}

/// Streaming extractor for periodic models.
pub type PeriodicExtractor = PeriodicExtractorCore<PointRelease>;

#[cfg(test)]
mod tests {
    use super::{
        PeriodicConfig, PeriodicError, PeriodicExtractor, PeriodicModel, infer_periodic_model,
    };

    #[test]
    fn inference_rejects_invalid_config() {
        let err = infer_periodic_model(
            [0_u64, 10, 20],
            &PeriodicConfig {
                overlap: 0,
                ..PeriodicConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(err, PeriodicError::InvalidOverlap);
    }

    #[test]
    fn inference_matches_exact_periodic_trace() {
        let releases: Vec<u64> = (0..100).map(|index| 7 + 13 * index).collect();
        let model = infer_periodic_model(
            releases,
            &PeriodicConfig {
                batch_size: 11,
                overlap: 3,
                n_candidates: 20,
                candidate_dispersion: 2.0,
                rounding_adjustments: vec![-1, 0, 1],
                ..PeriodicConfig::default()
            },
        )
        .unwrap();

        assert_eq!(
            model,
            PeriodicModel {
                period: 13,
                offset: 7,
                jitter: 0,
            }
        );
    }

    #[test]
    fn inference_matches_jittered_trace() {
        let releases: Vec<u64> = (0..90).map(|index| 3 + 10 * index + index % 3).collect();
        let model = infer_periodic_model(
            releases,
            &PeriodicConfig {
                batch_size: 8,
                overlap: 1,
                n_candidates: 30,
                candidate_dispersion: 3.0,
                rounding_adjustments: vec![-2, -1, 0, 1, 2],
                ..PeriodicConfig::default()
            },
        )
        .unwrap();

        assert_eq!(
            model,
            PeriodicModel {
                period: 10,
                offset: 3,
                jitter: 2,
            }
        );
    }

    #[test]
    fn extractor_matches_one_shot_inference() {
        let releases: Vec<u64> = (0..90).map(|index| 3 + 10 * index + index % 27).collect();
        let config = PeriodicConfig {
            batch_size: 8,
            overlap: 1,
            n_candidates: 30,
            candidate_dispersion: 3.0,
            rounding_adjustments: vec![-2, -1, 0, 1, 2],
            ..PeriodicConfig::default()
        };

        let expected = infer_periodic_model(releases.iter().copied(), &config).unwrap();
        let mut extractor = PeriodicExtractor::with_config(config).unwrap();

        extractor.feed(releases[..30].iter().copied());
        extractor.feed(releases[30..60].iter().copied());
        extractor.feed(releases[60..].iter().copied());

        assert_eq!(extractor.current_model(), Some(expected));
    }
}
