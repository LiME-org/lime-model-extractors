//! Bounds on self-suspension counts and durations.

use thiserror::Error;

use crate::time::Duration;

type Result<T> = std::result::Result<T, SuspensionError>;

/// `(suspension count, total suspension time)` for one job.
/// Totals over consecutive jobs must fit in `u64`.
pub type BasicJobSuspensionBehavior = (u64, Duration);
/// Bounds on suspension count and time for a fixed number of consecutive jobs (a “window”).
/// The count and time bounds can come from different groups of that size.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BasicSuspensionModel {
    /// Bound on the total suspension count over consecutive jobs.
    pub number_of_suspensions: u64,
    /// Bound on the total suspension time over consecutive jobs.
    pub cumulative_duration: Duration,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SuspensionError {
    #[error("nmax must be positive")]
    InvalidNmax,
}

impl From<BasicJobSuspensionBehavior> for BasicSuspensionModel {
    fn from((number_of_suspensions, cumulative_duration): BasicJobSuspensionBehavior) -> Self {
        Self {
            number_of_suspensions,
            cumulative_duration,
        }
    }
}

impl BasicSuspensionModel {
    fn add(self, other: Self) -> Self {
        Self {
            number_of_suspensions: self.number_of_suspensions + other.number_of_suspensions,
            cumulative_duration: self.cumulative_duration + other.cumulative_duration,
        }
    }

    fn component_max(self, other: Self) -> Self {
        Self {
            number_of_suspensions: self.number_of_suspensions.max(other.number_of_suspensions),
            cumulative_duration: self.cumulative_duration.max(other.cumulative_duration),
        }
    }

    fn component_min(self, other: Self) -> Self {
        Self {
            number_of_suspensions: self.number_of_suspensions.min(other.number_of_suspensions),
            cumulative_duration: self.cumulative_duration.min(other.cumulative_duration),
        }
    }
}

/// Entry `n` gives the most suspensions and the longest total suspension time
/// observed over any `n` consecutive jobs.
#[derive(Debug, Clone, Default)]
pub struct MaxSuspensionModelExtractor {
    nmax: Option<usize>,
    suffix_totals: Vec<BasicSuspensionModel>,
    bounds: Vec<BasicSuspensionModel>,
}

impl MaxSuspensionModelExtractor {
    /// Track windows of at most `nmax` jobs. Use `None` for all lengths or a positive limit.
    pub fn new(nmax: Option<usize>) -> Result<Self> {
        if nmax == Some(0) {
            return Err(SuspensionError::InvalidNmax);
        }
        Ok(Self {
            nmax,
            ..Self::default()
        })
    }

    /// Update the extractor with additional job observations.
    pub fn feed<I>(&mut self, observations: I)
    where
        I: IntoIterator<Item = BasicJobSuspensionBehavior>,
    {
        for job in observations {
            let job = BasicSuspensionModel::from(job);
            if self.bounds.is_empty() {
                self.bounds.push(BasicSuspensionModel::default());
            }

            let mut shorter_total = BasicSuspensionModel::default();
            for (suffix, bound) in self.suffix_totals.iter_mut().zip(&mut self.bounds[1..]) {
                let total = shorter_total.add(job);
                shorter_total = *suffix;
                *suffix = total;
                *bound = bound.component_max(total);
            }
            if self.nmax != Some(self.suffix_totals.len()) {
                let total = shorter_total.add(job);
                self.suffix_totals.push(total);
                self.bounds.push(total);
            }
        }
    }

    /// Return the current upper bounds by window length.
    pub fn current_model(&self) -> Vec<BasicSuspensionModel> {
        self.bounds.clone()
    }

    /// Take the bounds by window length without copying them.
    pub fn into_model(self) -> Vec<BasicSuspensionModel> {
        self.bounds
    }
}

/// Tracks separate lower bounds on suspension count and time for each window length.
#[derive(Debug, Clone, Default)]
pub struct MinSuspensionModelExtractor {
    nmax: Option<usize>,
    suffix_totals: Vec<BasicSuspensionModel>,
    bounds: Vec<BasicSuspensionModel>,
}

impl MinSuspensionModelExtractor {
    /// Track windows of at most `nmax` jobs. Use `None` for all lengths or a positive limit.
    pub fn new(nmax: Option<usize>) -> Result<Self> {
        if nmax == Some(0) {
            return Err(SuspensionError::InvalidNmax);
        }
        Ok(Self {
            nmax,
            ..Self::default()
        })
    }

    /// Update the extractor with additional job observations.
    pub fn feed<I>(&mut self, observations: I)
    where
        I: IntoIterator<Item = BasicJobSuspensionBehavior>,
    {
        for job in observations {
            let job = BasicSuspensionModel::from(job);
            if self.bounds.is_empty() {
                self.bounds.push(BasicSuspensionModel::default());
            }

            let mut shorter_total = BasicSuspensionModel::default();
            for (suffix, bound) in self.suffix_totals.iter_mut().zip(&mut self.bounds[1..]) {
                let total = shorter_total.add(job);
                shorter_total = *suffix;
                *suffix = total;
                *bound = bound.component_min(total);
            }
            if self.nmax != Some(self.suffix_totals.len()) {
                let total = shorter_total.add(job);
                self.suffix_totals.push(total);
                self.bounds.push(total);
            }
        }
    }

    /// Return the current lower bounds by window length.
    pub fn current_model(&self) -> Vec<BasicSuspensionModel> {
        self.bounds.clone()
    }

    /// Take the bounds by window length without copying them.
    pub fn into_model(self) -> Vec<BasicSuspensionModel> {
        self.bounds
    }
}

/// Return the upper bounds on suspension count and time over `n` consecutive jobs at index `n`.
pub fn infer_max_suspension_model<I>(
    observations: I,
    nmax: Option<usize>,
) -> Result<Vec<BasicSuspensionModel>>
where
    I: IntoIterator<Item = BasicJobSuspensionBehavior>,
{
    let mut extractor = MaxSuspensionModelExtractor::new(nmax)?;
    extractor.feed(observations);
    Ok(extractor.into_model())
}

/// Return the lower bounds on suspension count and time over `n` consecutive jobs at index `n`.
pub fn infer_min_suspension_model<I>(
    observations: I,
    nmax: Option<usize>,
) -> Result<Vec<BasicSuspensionModel>>
where
    I: IntoIterator<Item = BasicJobSuspensionBehavior>,
{
    let mut extractor = MinSuspensionModelExtractor::new(nmax)?;
    extractor.feed(observations);
    Ok(extractor.into_model())
}

#[cfg(test)]
mod tests {
    use super::{
        BasicSuspensionModel, MaxSuspensionModelExtractor, MinSuspensionModelExtractor,
        SuspensionError, infer_max_suspension_model, infer_min_suspension_model,
    };

    fn basic_models(values: &[(u64, u64)]) -> Vec<BasicSuspensionModel> {
        values
            .iter()
            .map(
                |&(number_of_suspensions, cumulative_duration)| BasicSuspensionModel {
                    number_of_suspensions,
                    cumulative_duration,
                },
            )
            .collect()
    }

    #[test]
    fn basic_models_bound_consecutive_jobs_component_wise() {
        let observations = [(3, 10), (1, 7), (4, 1), (1, 5), (5, 2)];
        assert_eq!(
            infer_max_suspension_model(observations, None).unwrap(),
            basic_models(&[(0, 0), (5, 10), (6, 17), (10, 18), (11, 23), (14, 25)])
        );
        assert_eq!(
            infer_min_suspension_model(observations, None).unwrap(),
            basic_models(&[(0, 0), (1, 1), (4, 6), (6, 8), (9, 15), (14, 25)])
        );
    }

    #[test]
    fn basic_extractors_match_prefix_inference_with_bounded_history() {
        let observations: Vec<_> = (0..100).map(|i| (i % 7, i * 13 % 31)).collect();
        for nmax in [None, Some(1), Some(7), Some(150)] {
            let mut maximum = MaxSuspensionModelExtractor::new(nmax).unwrap();
            let mut minimum = MinSuspensionModelExtractor::new(nmax).unwrap();
            let mut observed = Vec::new();
            for batch in observations.chunks(3) {
                maximum.feed(batch.iter().copied());
                minimum.feed(batch.iter().copied());
                observed.extend_from_slice(batch);
                assert_eq!(
                    maximum.current_model(),
                    infer_max_suspension_model(observed.iter().copied(), nmax).unwrap()
                );
                assert_eq!(
                    minimum.current_model(),
                    infer_min_suspension_model(observed.iter().copied(), nmax).unwrap()
                );
                for totals in [&maximum.suffix_totals, &minimum.suffix_totals] {
                    assert!(totals.len() <= nmax.unwrap_or(observed.len()));
                }
            }
        }
    }

    #[test]
    fn empty_input_and_zero_observations_are_distinct() {
        assert!(infer_max_suspension_model([], None).unwrap().is_empty());
        assert!(infer_min_suspension_model([], None).unwrap().is_empty());
        assert!(
            MaxSuspensionModelExtractor::default()
                .current_model()
                .is_empty()
        );
        assert!(
            MinSuspensionModelExtractor::default()
                .current_model()
                .is_empty()
        );
        assert_eq!(
            infer_max_suspension_model([(0, 0); 2], None).unwrap(),
            basic_models(&[(0, 0); 3])
        );
        assert_eq!(
            infer_min_suspension_model([(0, 0); 2], None).unwrap(),
            basic_models(&[(0, 0); 3])
        );
    }

    #[test]
    fn suspension_limits_must_be_positive() {
        assert_eq!(
            MaxSuspensionModelExtractor::new(Some(0)).unwrap_err(),
            SuspensionError::InvalidNmax
        );
        assert_eq!(
            MinSuspensionModelExtractor::new(Some(0)).unwrap_err(),
            SuspensionError::InvalidNmax
        );
    }
}
