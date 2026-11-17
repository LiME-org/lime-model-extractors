//! Bounds on self-suspension counts, durations, and individual job segments.

use thiserror::Error;

use crate::time::Duration;

type Result<T> = std::result::Result<T, SuspensionError>;

/// `(suspension count, total suspension time)` for one job.
/// Totals over consecutive jobs must fit in `u64`.
pub type BasicJobSuspensionBehavior = (u64, Duration);
/// `(suspension time, execution time)`. The first segment's suspension includes release jitter.
pub type ObservedSegment = (Duration, Duration);
/// The segments of one observed job in execution order.
pub type SegmentedJobSuspensionBehavior = Vec<ObservedSegment>;

/// Bounds on suspension count and time for a fixed number of consecutive jobs (a “window”).
/// The count and time bounds can come from different groups of that size.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BasicSuspensionModel {
    /// Bound on the total suspension count over consecutive jobs.
    pub number_of_suspensions: u64,
    /// Bound on the total suspension time over consecutive jobs.
    pub cumulative_duration: Duration,
}

/// The first entry in a segmented model records the longest delay and the longest
/// execution time among the jobs' first segments. The second entry uses their second segments, and so on.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Segment {
    /// Maximum delay before this segment starts, including release jitter for the first segment.
    pub max_suspension_time: Duration,
    /// Maximum execution time observed for this segment.
    pub max_execution_time: Duration,
}

impl From<ObservedSegment> for Segment {
    fn from((max_suspension_time, max_execution_time): ObservedSegment) -> Self {
        Self {
            max_suspension_time,
            max_execution_time,
        }
    }
}

/// Upper bounds indexed by segment position within a job.
pub type SegmentedSuspensionModel = Vec<Segment>;
/// Alternative segmented models that together cover the observed jobs.
pub type BagOfSegmentsModel = Vec<SegmentedSuspensionModel>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SuspensionError {
    #[error("nmax must be positive")]
    InvalidNmax,
    #[error("max_segments must be positive")]
    InvalidMaxSegments,
    #[error("job has {observed} segments, exceeding the limit of {limit}")]
    SegmentLimitExceeded { observed: usize, limit: usize },
    #[error("max_models must be positive")]
    InvalidMaxModels,
    #[error("each observation must contain at least one segment")]
    EmptyObservation,
    #[error("segment must have positive suspension or execution time")]
    ZeroSegment,
}

/// Total suspension time is the sum of all delays, including initial release jitter.
/// Each boundary between segments counts as one suspension.
pub fn basic_from_segmented_observation(
    segments: &[ObservedSegment],
) -> BasicJobSuspensionBehavior {
    (
        segments.len().saturating_sub(1) as u64,
        segments.iter().map(|s| s.0).sum(),
    )
}

/// Sum the delay bounds and count the boundaries between segments.
pub fn basic_from_segmented_model(model: &[Segment]) -> BasicSuspensionModel {
    BasicSuspensionModel {
        number_of_suspensions: model.len().saturating_sub(1) as u64,
        cumulative_duration: model.iter().map(|s| s.max_suspension_time).sum(),
    }
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

pub(crate) fn merge_segments<I>(target: &mut SegmentedSuspensionModel, source: I)
where
    I: IntoIterator<Item = Segment>,
{
    let mut source = source.into_iter();
    for (bound, segment) in target.iter_mut().zip(source.by_ref()) {
        bound.max_suspension_time = bound.max_suspension_time.max(segment.max_suspension_time);
        bound.max_execution_time = bound.max_execution_time.max(segment.max_execution_time);
    }
    target.extend(source);
}

/// If a job exceeds `max_segments`, truncating it would discard observed work.
/// The extractor discards the model in that case.
#[derive(Debug, Clone)]
pub struct SegmentedSuspensionModelExtractor {
    max_segments: Option<usize>,
    model: Result<SegmentedSuspensionModel>,
}

impl Default for SegmentedSuspensionModelExtractor {
    fn default() -> Self {
        Self {
            max_segments: None,
            model: Ok(Vec::new()),
        }
    }
}

impl SegmentedSuspensionModelExtractor {
    /// Limit the number of segments per job. Use `None` for any length or a positive limit.
    pub fn new(max_segments: Option<usize>) -> Result<Self> {
        if max_segments == Some(0) {
            return Err(SuspensionError::InvalidMaxSegments);
        }
        Ok(Self {
            max_segments,
            ..Self::default()
        })
    }

    /// Update the extractor with additional job observations.
    /// If a job exceeds `max_segments`, the model is discarded. This and all later calls
    /// return [`SuspensionError::SegmentLimitExceeded`] without reading further jobs.
    /// Other errors leave earlier jobs from the same call in the model.
    pub fn feed<I, O>(&mut self, observations: I) -> Result<()>
    where
        I: IntoIterator<Item = O>,
        O: AsRef<[ObservedSegment]>,
    {
        let model = self.model.as_mut().map_err(|error| error.clone())?;
        for job in observations {
            let segments = job.as_ref();
            if let Some(limit) = self.max_segments
                && segments.len() > limit
            {
                let error = SuspensionError::SegmentLimitExceeded {
                    observed: segments.len(),
                    limit,
                };
                self.model = Err(error.clone());
                return Err(error);
            }
            if segments.is_empty() {
                return Err(SuspensionError::EmptyObservation);
            }
            if segments.contains(&(0, 0)) {
                return Err(SuspensionError::ZeroSegment);
            }
            merge_segments(model, segments.iter().copied().map(Segment::from));
        }
        Ok(())
    }

    /// Copy the bounds by segment position, or return `None` after an oversized job.
    pub fn current_model(&self) -> Option<SegmentedSuspensionModel> {
        self.model.as_ref().ok().cloned()
    }

    /// Take the bounds by segment position, or return `None` after an oversized job.
    pub fn into_model(self) -> Option<SegmentedSuspensionModel> {
        self.model.ok()
    }
}

/// Bound each segment position across jobs, returning `None` if any job exceeds `max_segments`.
pub fn infer_segmented_suspension_model<I, O>(
    observations: I,
    max_segments: Option<usize>,
) -> Result<Option<SegmentedSuspensionModel>>
where
    I: IntoIterator<Item = O>,
    O: AsRef<[ObservedSegment]>,
{
    let mut extractor = SegmentedSuspensionModelExtractor::new(max_segments)?;
    match extractor.feed(observations) {
        Ok(()) => Ok(extractor.into_model()),
        Err(SuspensionError::SegmentLimitExceeded { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BasicSuspensionModel, MaxSuspensionModelExtractor, MinSuspensionModelExtractor, Segment,
        SegmentedSuspensionModelExtractor, SuspensionError, basic_from_segmented_model,
        basic_from_segmented_observation, infer_max_suspension_model, infer_min_suspension_model,
        infer_segmented_suspension_model,
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
        assert_eq!(
            infer_segmented_suspension_model(Vec::<Vec<(u64, u64)>>::new(), None).unwrap(),
            Some(Vec::new())
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
        assert_eq!(
            SegmentedSuspensionModelExtractor::new(Some(0)).unwrap_err(),
            SuspensionError::InvalidMaxSegments
        );
    }

    #[test]
    fn conversions_count_initial_jitter_only_in_duration() {
        for observations in [vec![], vec![(3, 5)], vec![(3, 5), (7, 11), (13, 17)]] {
            let model: Vec<_> = observations.iter().copied().map(Segment::from).collect();
            let (count, duration) = basic_from_segmented_observation(&observations);
            assert_eq!(
                basic_from_segmented_model(&model),
                basic_models(&[(count, duration)])[0]
            );
        }
        assert_eq!(basic_from_segmented_observation(&[]), (0, 0));
        assert_eq!(basic_from_segmented_observation(&[(3, 5)]), (0, 3));
        assert_eq!(
            basic_from_segmented_observation(&[(3, 5), (7, 11), (13, 17)]),
            (2, 23)
        );
    }

    #[test]
    fn segmented_model_preserves_tail_and_merges_component_wise() {
        let observations = [
            vec![(1, 9), (8, 2)],
            vec![(3, 4)],
            vec![(2, 7), (5, 6), (11, 13)],
        ];
        let expected = vec![
            Segment::from((3, 9)),
            Segment::from((8, 6)),
            Segment::from((11, 13)),
        ];
        let mut extractor = SegmentedSuspensionModelExtractor::default();
        for job in &observations {
            extractor.feed([job]).unwrap();
        }
        assert_eq!(extractor.current_model(), Some(expected.clone()));
        assert_eq!(
            infer_segmented_suspension_model(&observations, Some(3)).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn segment_limit_invalidation_is_permanent_and_stops_consumption() {
        let mut extractor = SegmentedSuspensionModelExtractor::new(Some(1)).unwrap();
        let mut jobs = [vec![(1, 2)], vec![(3, 4), (5, 6)], vec![]].into_iter();
        let error = SuspensionError::SegmentLimitExceeded {
            observed: 2,
            limit: 1,
        };
        assert_eq!(extractor.feed(&mut jobs), Err(error.clone()));
        assert_eq!(extractor.current_model(), None);
        assert_eq!(extractor.feed(&mut jobs), Err(error));
        assert_eq!(jobs.next(), Some(vec![]));
        assert_eq!(extractor.into_model(), None);
    }

    #[test]
    fn empty_job_is_rejected_without_losing_previous_observations() {
        let mut extractor = SegmentedSuspensionModelExtractor::default();
        assert_eq!(
            extractor.feed([vec![(1, 2)], vec![]]),
            Err(SuspensionError::EmptyObservation)
        );
        assert_eq!(extractor.current_model(), Some(vec![Segment::from((1, 2))]));
        extractor.feed([[(3, 4)]]).unwrap();
        assert_eq!(extractor.current_model(), Some(vec![Segment::from((3, 4))]));
    }
}
