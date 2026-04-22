//! Shared periodic-task inference core for exact releases and release windows.

use std::{cmp::Reverse, marker::PhantomData};

use crate::{
    certain_periodic::{PeriodicConfig, PeriodicError, PeriodicModel},
    release_window::ReleaseWindow,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, PeriodicError>;

pub trait FitMode {
    /// The type representing individual release observations.
    type Observation: Copy + std::fmt::Debug + PartialEq + Eq;

    /// The point in time used to estimate inter-release gaps.
    fn gap_anchor(observation: Self::Observation) -> Duration;

    /// The estimate of the release time used when bounding the offset.
    fn estimate_release_for_offset_inference(observation: Self::Observation) -> i64;

    /// The estimate of the release time used when bounding the maximum jitter.
    fn estimate_release_for_jitter_inference(observation: Self::Observation) -> i64;

    /// The "minimum jitter" used to prune diverging model candidates.
    fn jitter_pruning_reference(candidates: &[PeriodicModel]) -> Option<i64> {
        candidates.iter().map(|candidate| candidate.jitter).min()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PointRelease;

impl FitMode for PointRelease {
    type Observation = Instant;

    fn gap_anchor(observation: Instant) -> Duration {
        observation
    }

    fn estimate_release_for_offset_inference(observation: Instant) -> i64 {
        observation as i64
    }

    fn estimate_release_for_jitter_inference(observation: Instant) -> i64 {
        observation as i64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CertainFit;

impl FitMode for CertainFit {
    type Observation = ReleaseWindow;

    fn gap_anchor(observation: ReleaseWindow) -> Duration {
        observation.hi
    }

    fn estimate_release_for_offset_inference(observation: ReleaseWindow) -> i64 {
        observation.lo as i64
    }

    fn estimate_release_for_jitter_inference(observation: ReleaseWindow) -> i64 {
        observation.hi as i64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PossibleFit;

impl FitMode for PossibleFit {
    type Observation = ReleaseWindow;

    fn gap_anchor(observation: ReleaseWindow) -> Duration {
        observation.hi
    }

    fn estimate_release_for_offset_inference(observation: ReleaseWindow) -> i64 {
        observation.hi as i64
    }

    fn estimate_release_for_jitter_inference(observation: ReleaseWindow) -> i64 {
        observation.lo as i64
    }

    fn jitter_pruning_reference(candidates: &[PeriodicModel]) -> Option<i64> {
        candidates
            .iter()
            .filter(|candidate| candidate.jitter > 0)
            .map(|candidate| candidate.jitter)
            .min()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Batch<F: FitMode>(Vec<(usize, F::Observation)>);

impl<F: FitMode> Batch<F> {
    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn push(&mut self, observation: (usize, F::Observation)) {
        self.0.push(observation);
    }

    fn retain_overlap(&mut self, config: &PeriodicConfig) {
        let keep_from = self.len() - config.overlap;
        self.0.copy_within(keep_from.., 0);
        self.0.truncate(config.overlap);
    }

    fn clean_slice_and_mean_gap(&self) -> (&[(usize, F::Observation)], f64) {
        let gaps: Vec<_> = self
            .0
            .windows(2)
            .map(|window| F::gap_anchor(window[1].1).saturating_sub(F::gap_anchor(window[0].1)))
            .collect();

        let (start, end) = first_and_last_nonoutlier(&gaps);
        let start = start.unwrap_or(0);
        let end = end.unwrap_or(self.len() - 1);

        let mean_gap = if start < end {
            let total: Duration = gaps[start..=end].iter().copied().sum();
            total as f64 / (end - start + 1) as f64
        } else {
            0.0
        };

        (&self.0[start..=(end + 1)], mean_gap)
    }

    /// Search for the period that minimizes the batch jitter.
    fn min_jitter_model(&self, search_range: (f64, f64)) -> PeriodicModel {
        let (cleaned, mean_gap) = self.clean_slice_and_mean_gap();
        let mut low = (mean_gap * search_range.0).floor().max(0.0) as Duration;
        let mut high = (mean_gap * search_range.1).ceil().max(0.0) as Duration;
        let mut optimum = None;

        while high - low > 1 {
            let mid = (high + low) / 2;
            let lower = PeriodicModel::from_observations::<F>(cleaned, mid);
            let upper = PeriodicModel::from_observations::<F>(cleaned, mid + 1);

            if lower.jitter < upper.jitter {
                high = mid;
                optimum = Some(lower);
            } else {
                low = mid;
                optimum = Some(upper);
            }
        }

        optimum.unwrap_or_else(|| {
            PeriodicModel::from_observations::<F>(cleaned, mean_gap.round() as Duration)
        })
    }

    /// Return the batch index of the last already-processed release.
    #[inline]
    fn last_processed_index(&self, config: &PeriodicConfig) -> usize {
        self.0[config.overlap - 1].0
    }
}

#[inline]
fn offset<F: FitMode>(batch: &[(usize, F::Observation)], period: Duration) -> i64 {
    batch
        .iter()
        .map(|(index, observation)| {
            let total_periods = *index as i64 * period as i64;
            F::estimate_release_for_offset_inference(*observation) - total_periods
        })
        .min()
        .unwrap_or(0)
}

#[inline]
fn jitter<F: FitMode>(batch: &[(usize, F::Observation)], offset: i64, period: Duration) -> i64 {
    batch
        .iter()
        .map(|(index, observation)| {
            let arrival = *index as i64 * period as i64 + offset;
            F::estimate_release_for_jitter_inference(*observation) - arrival
        })
        .max()
        .unwrap_or(0)
}

impl PeriodicModel {
    #[inline]
    fn from_observations<F: FitMode>(batch: &[(usize, F::Observation)], period: Duration) -> Self {
        let offset = offset::<F>(batch, period);
        let jitter = jitter::<F>(batch, offset, period);
        Self {
            period,
            offset,
            jitter,
        }
    }

    #[inline]
    fn update_observations<F: FitMode>(&mut self, batch: &Batch<F>) {
        let period = self.period as i64;
        let mut offset = self.offset;
        let mut upper_bound = self.offset + self.jitter;
        for &(index, observation) in &batch.0 {
            let expected = (index as i64).wrapping_mul(period);
            offset = offset.min(F::estimate_release_for_offset_inference(observation) - expected);
            upper_bound =
                upper_bound.max(F::estimate_release_for_jitter_inference(observation) - expected);
        }

        self.offset = offset;
        self.jitter = upper_bound - offset;
    }

    /// Generate the initial candidate periods used by the periodic heuristic.
    fn candidate_periods(&self, config: &PeriodicConfig) -> Vec<Duration> {
        let extension = config.candidate_dispersion * self.jitter.unsigned_abs() as f64;
        let mut periods: Vec<_> =
            evenly_spaced_around(self.period as f64, extension, config.n_candidates)
                .map(|value| value.round() as Duration)
                .filter(|&period| period > 0)
                .chain(rounded_at_each_granularity(
                    self.period,
                    &config.rounding_adjustments,
                ))
                .collect();
        periods.sort_unstable();
        periods.dedup();
        periods
    }
}

fn add_derived_model_candidate(
    model_candidates: &mut Vec<PeriodicModel>,
    candidate_count: usize,
    reference_period: Duration,
    last_processed_index: usize,
) {
    let idx = last_processed_index as Duration;
    let candidate = model_candidates
        .iter()
        .take(candidate_count)
        .min_by_key(|model| model.period.abs_diff(reference_period))
        .copied()
        .expect("derived_model_candidates requires non-empty model_candidates");

    if candidate.period == reference_period {
        return;
    }

    let (offset, jitter) = if candidate.period < reference_period {
        let shift = (idx * (reference_period - candidate.period)) as i64;
        let offset = candidate.offset - shift;
        let jitter = candidate.offset + candidate.jitter - offset;
        (offset, jitter)
    } else {
        let offset = candidate.offset;
        let jitter = candidate.jitter + (idx * (candidate.period - reference_period)) as i64;
        (offset, jitter)
    };

    model_candidates.push(PeriodicModel {
        period: reference_period,
        offset,
        jitter,
    });
}

fn approximate_median(values: &mut [Duration]) -> Option<Duration> {
    if values.is_empty() {
        return None;
    }

    let len = values.len();
    let mid = len / 2;
    let (_, median, _) = values.select_nth_unstable(mid);

    // Since this is just an approximation, we don't bother with the would
    // have to case of len being a multiple of 2, in which case we technically
    // average the two middle elememts. For our purposes, something close to
    // the median is good enough.
    Some(*median)
}

fn hampel_nonwindowed_stats(data: &[Duration]) -> Option<(Duration, Duration)> {
    let mut values = data.to_vec();
    let median_value = approximate_median(&mut values)?;

    values
        .iter_mut()
        .for_each(|value| *value = value.abs_diff(median_value));
    let mad = approximate_median(&mut values).unwrap_or(0);

    Some((median_value, mad))
}

fn first_and_last_nonoutlier(data: &[Duration]) -> (Option<usize>, Option<usize>) {
    let Some((median_value, mad)) = hampel_nonwindowed_stats(data) else {
        return (None, None);
    };
    if mad == 0 {
        return if data.is_empty() {
            (None, None)
        } else {
            (Some(0), Some(data.len() - 1))
        };
    }
    let threshold = mad.saturating_mul(3);

    data.iter()
        .enumerate()
        .filter_map(|(index, &value)| (value.abs_diff(median_value) <= threshold).then_some(index))
        .fold((None, None), |(first, _), index| {
            (Some(first.unwrap_or(index)), Some(index))
        })
}

/// Return the number of trailing zeroes in the decimal representation.
fn trailing_zeroes(mut n: Duration) -> usize {
    let mut count = 0;
    while n != 0 && n.is_multiple_of(10) {
        count += 1;
        n /= 10;
    }
    count
}

fn truncate_to_max_len(candidates: &mut Vec<PeriodicModel>, max_len: usize) {
    if candidates.len() > max_len {
        candidates.select_nth_unstable_by_key(max_len, |candidate| candidate.jitter);
        candidates.truncate(max_len);
    }
}

/// Yield candidate periods obtained by rounding at increasing decimal granularities.
fn rounded_at_each_granularity(
    period: Duration,
    adjustments: &[i64],
) -> impl Iterator<Item = Duration> + '_ {
    // look at granularities 10, 100, 1000, 10000, ... while not exceeding period
    std::iter::successors(
        (10 < period).then_some(10 as Duration),
        move |&granularity| granularity.checked_mul(10).filter(|&next| next < period),
    )
    .flat_map(move |granularity| {
        // try each adjustment with the given granularity
        rounded_candidates_for_granularity(period, adjustments, granularity, false).chain(
            (granularity.saturating_mul(10) >= period)
                .then(|| rounded_candidates_for_granularity(period, adjustments, granularity, true))
                .into_iter()
                .flatten(),
        )
    })
}

fn rounded_candidates_for_granularity(
    period: Duration,
    adjustments: &[i64],
    granularity: Duration,
    include_trailing_zeroes: bool,
) -> impl Iterator<Item = Duration> + '_ {
    let rounded = (period / granularity) as i64;
    adjustments.iter().copied().filter_map(move |delta| {
        let candidate = rounded + delta;
        if candidate <= 0 || (candidate % 10 == 0) != include_trailing_zeroes {
            None
        } else {
            (candidate as Duration).checked_mul(granularity)
        }
    })
}

/// Generate evenly spaced points around a center while always including the center.
fn evenly_spaced_around(center: f64, extension: f64, n: usize) -> impl Iterator<Item = f64> {
    debug_assert!(n >= 3);

    let low = center - extension;
    let spread = 2.0 * extension;
    let tail = (1..n).map(move |step| low + spread * (step - 1) as f64 / (n - 2) as f64);

    std::iter::once(center).chain(tail)
}

#[derive(Debug, Clone)]
pub struct PeriodicExtractorCore<F: FitMode> {
    config: PeriodicConfig,
    next_index: usize,
    batch_count: usize,
    current_batch: Batch<F>,
    candidates: Vec<PeriodicModel>,
    running_mean: f64,
    fit: PhantomData<F>,
}

impl<F: FitMode> PeriodicExtractorCore<F> {
    pub fn new() -> Result<Self> {
        Self::with_config(PeriodicConfig::default())
    }

    pub fn with_config(config: PeriodicConfig) -> Result<Self> {
        config.validate()?;
        let batch_size = config.batch_size;

        Ok(Self {
            config,
            next_index: 0,
            batch_count: 1,
            current_batch: Batch::with_capacity(batch_size),
            candidates: Vec::new(),
            running_mean: 0.0,
            fit: PhantomData,
        })
    }

    fn is_first_batch(&self) -> bool {
        self.candidates.is_empty()
    }

    fn batch_contains_only_overlap(&self) -> bool {
        !self.is_first_batch() && self.current_batch.len() == self.config.overlap
    }

    fn process_first_batch(&self) -> (f64, Vec<PeriodicModel>) {
        let min_jitter_model = self.current_batch.min_jitter_model((0.5, 2.0));
        let candidates = min_jitter_model
            .candidate_periods(&self.config)
            .into_iter()
            .map(|period| PeriodicModel::from_observations::<F>(&self.current_batch.0, period))
            .collect();

        (min_jitter_model.period as f64, candidates)
    }

    fn process_subsequent_batch(&self) -> (f64, Vec<PeriodicModel>) {
        let previous_len = self.candidates.len();

        let min_jitter_model = self.current_batch.min_jitter_model((0.5, 2.0));
        let running_mean = self.running_mean
            + (min_jitter_model.period as f64 - self.running_mean) / self.batch_count as f64;

        // make a copy of the candidates
        let mut updated_candidates = Vec::with_capacity(previous_len + 2);
        updated_candidates.extend(self.candidates.iter().copied());

        // add derived model candidates
        let last_processed_index = self.current_batch.last_processed_index(&self.config);
        let running_mean_period = running_mean.round() as Duration;
        add_derived_model_candidate(
            &mut updated_candidates,
            previous_len,
            running_mean_period,
            last_processed_index,
        );
        if min_jitter_model.period != running_mean_period {
            add_derived_model_candidate(
                &mut updated_candidates,
                previous_len,
                min_jitter_model.period,
                last_processed_index,
            );
        }

        // update all candidates
        for candidate in &mut updated_candidates {
            candidate.update_observations(&self.current_batch);
        }

        // prune diverging models
        let mut candidates: Vec<_> =
            if let Some(best_jitter) = F::jitter_pruning_reference(&updated_candidates) {
                updated_candidates
                    .into_iter()
                    .filter(|candidate| {
                        candidate.jitter <= self.config.negligible_jitter_threshold
                            || candidate.jitter as f64
                                <= best_jitter as f64 * self.config.jitter_pruning_threshold
                    })
                    .collect()
            } else {
                updated_candidates
            };

        // prevent unbounded growth
        truncate_to_max_len(&mut candidates, previous_len);

        (running_mean, candidates)
    }

    fn current_candidates(&self) -> Option<Vec<PeriodicModel>> {
        if self.is_first_batch() {
            if self.current_batch.len() < 2 {
                return None;
            }
            let (_, candidates) = self.process_first_batch();
            return Some(candidates);
        }

        if self.batch_contains_only_overlap() {
            return Some(self.candidates.clone());
        }

        let (_, candidates) = self.process_subsequent_batch();
        Some(candidates)
    }

    fn select_model(&self, candidates: &[PeriodicModel]) -> Option<PeriodicModel> {
        let best = candidates
            .iter()
            .min_by_key(|candidate| candidate.jitter)
            .copied()?;

        candidates
            .iter()
            .copied()
            .filter(|candidate| {
                candidate.jitter <= self.config.negligible_jitter_threshold
                    || candidate.jitter as f64
                        <= best.jitter as f64 * self.config.jitter_selection_threshold
            })
            .max_by_key(|candidate| (trailing_zeroes(candidate.period), Reverse(candidate.jitter)))
            .map(|model| PeriodicModel {
                jitter: model.jitter.max(0),
                ..model
            })
    }

    fn consume_batch(&mut self) {
        (self.running_mean, self.candidates) = if self.is_first_batch() {
            self.process_first_batch()
        } else {
            self.process_subsequent_batch()
        };

        self.current_batch.retain_overlap(&self.config);
        self.batch_count += 1;
    }

    pub fn feed<I>(&mut self, observations: I)
    where
        I: IntoIterator<Item = F::Observation>,
    {
        for observation in observations {
            self.current_batch.push((self.next_index, observation));
            self.next_index += 1;
            if self.current_batch.len() == self.config.batch_size {
                self.consume_batch();
            }
        }
    }

    pub fn observed_count(&self) -> usize {
        self.next_index
    }

    pub fn current_model(&self) -> Option<PeriodicModel> {
        let candidates = self.current_candidates()?;
        self.select_model(&candidates)
    }

    #[cfg(test)]
    fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}

pub(crate) fn infer_model<F, I>(observations: I, config: &PeriodicConfig) -> Result<PeriodicModel>
where
    F: FitMode,
    I: IntoIterator<Item = F::Observation>,
{
    let mut extractor = PeriodicExtractorCore::<F>::with_config(config.clone())?;
    extractor.feed(observations);

    extractor.current_model().ok_or_else(|| {
        if extractor.observed_count() < 2 {
            PeriodicError::InsufficientReleases
        } else {
            PeriodicError::NoCandidateModels
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        Batch, CertainFit, PeriodicExtractorCore, PointRelease, PossibleFit,
        add_derived_model_candidate, evenly_spaced_around, first_and_last_nonoutlier, infer_model,
        jitter, offset, rounded_at_each_granularity, trailing_zeroes, truncate_to_max_len,
    };
    use crate::{
        certain_periodic::{PeriodicConfig, PeriodicModel},
        release_window::ReleaseWindow,
        time::Instant,
    };

    type ObservedRelease = (usize, Instant);

    const SAMPLE_BATCH: [ObservedRelease; 4] = [(0, 3), (1, 14), (2, 24), (3, 35)];

    fn point_batch(entries: &[(usize, Instant)]) -> Batch<PointRelease> {
        Batch(entries.to_vec())
    }

    fn window_batch(entries: &[(usize, ReleaseWindow)]) -> Batch<CertainFit> {
        Batch(entries.to_vec())
    }

    #[test]
    fn batch_mean_gap_computes_average_gap() {
        assert!(
            (point_batch(&SAMPLE_BATCH).clean_slice_and_mean_gap().1 - 32.0 / 3.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn batch_offset_jitter_and_model_match_expected_values() {
        let batch = point_batch(&SAMPLE_BATCH);
        let offset = offset::<PointRelease>(&batch.0, 10);
        let jitter = jitter::<PointRelease>(&batch.0, offset, 10);

        assert_eq!(offset, 3);
        assert_eq!(jitter, 2);
        assert_eq!(
            PeriodicModel::from_observations::<PointRelease>(&batch.0, 10),
            PeriodicModel {
                period: 10,
                offset: 3,
                jitter: 2,
            }
        );
    }

    #[test]
    fn select_model_clamps_negative_jitter() {
        let extractor = PeriodicExtractorCore::<PointRelease>::new().unwrap();
        let selected = extractor.select_model(&[
            PeriodicModel {
                period: 1_000_000,
                offset: 12_592_534_687_571,
                jitter: -12_597,
            },
            PeriodicModel {
                period: 999_999,
                offset: 12_592_534_687_571,
                jitter: 0,
            },
        ]);

        assert_eq!(
            selected,
            Some(PeriodicModel {
                period: 1_000_000,
                offset: 12_592_534_687_571,
                jitter: 0,
            })
        );
    }

    #[test]
    fn possible_fit_selection_clamps_negative_jitter() {
        let extractor = PeriodicExtractorCore::<PossibleFit>::new().unwrap();
        let selected = extractor.select_model(&[
            PeriodicModel {
                period: 1_000_000,
                offset: 12_592_534_687_571,
                jitter: -12_597,
            },
            PeriodicModel {
                period: 999_999,
                offset: 12_592_534_687_571,
                jitter: 0,
            },
        ]);

        assert_eq!(
            selected,
            Some(PeriodicModel {
                period: 1_000_000,
                offset: 12_592_534_687_571,
                jitter: 0,
            })
        );
    }

    #[test]
    fn min_jitter_model_matches_exact_periodic_batch() {
        let batch = point_batch(&[(0, 7), (1, 20), (2, 33), (3, 46), (4, 59)]);
        assert_eq!(
            batch.min_jitter_model((0.5, 2.0)),
            PeriodicModel {
                period: 13,
                offset: 7,
                jitter: 0,
            }
        );
    }

    #[test]
    fn spaced_candidates_use_unique_rounded_periods() {
        let candidates = PeriodicModel {
            period: 10,
            offset: 3,
            jitter: 2,
        }
        .candidate_periods(&PeriodicConfig {
            n_candidates: 5,
            candidate_dispersion: 2.0,
            ..PeriodicConfig::default()
        });
        assert_eq!(
            BTreeSet::from_iter(candidates),
            BTreeSet::from([6, 9, 10, 11, 14])
        );
    }

    #[test]
    fn rounded_at_each_granularity_returns_expected_roundings() {
        assert_eq!(
            rounded_at_each_granularity(123, &[-1, 0, 1]).collect::<Vec<_>>(),
            vec![110, 120, 130, 100, 200]
        );
        assert_eq!(
            rounded_at_each_granularity(10, &[-1, 0, 1]).collect::<Vec<_>>(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn first_and_last_nonoutlier_end() {
        assert_eq!(
            first_and_last_nonoutlier(&[10, 11, 11, 12, 50]),
            (Some(0), Some(3))
        );
        assert_eq!(
            first_and_last_nonoutlier(&[10, 11, 12, 13]),
            (Some(0), Some(3))
        );
    }

    #[test]
    fn derived_candidates_include_expected_models() {
        let batch = point_batch(&[(5, 50), (6, 62), (7, 74)]);
        let config = PeriodicConfig {
            overlap: 2,
            ..PeriodicConfig::default()
        };
        let mut derived = vec![
            PeriodicModel {
                period: 10,
                offset: 40,
                jitter: 3,
            },
            PeriodicModel {
                period: 14,
                offset: 30,
                jitter: 5,
            },
            PeriodicModel {
                period: 20,
                offset: 5,
                jitter: 8,
            },
        ];
        let candidate_count = derived.len();
        add_derived_model_candidate(
            &mut derived,
            candidate_count,
            13,
            batch.last_processed_index(&config),
        );
        add_derived_model_candidate(
            &mut derived,
            candidate_count,
            22,
            batch.last_processed_index(&config),
        );

        assert_eq!(
            &derived[3..],
            vec![
                PeriodicModel {
                    period: 13,
                    offset: 30,
                    jitter: 11,
                },
                PeriodicModel {
                    period: 22,
                    offset: -7,
                    jitter: 20,
                },
            ]
        );
    }

    #[test]
    fn batch_update_expands_model_bounds_for_batch() {
        let batch = point_batch(&SAMPLE_BATCH);
        let mut model = PeriodicModel {
            period: 10,
            offset: 5,
            jitter: 1,
        };
        model.update_observations::<PointRelease>(&batch);
        assert_eq!(
            model,
            PeriodicModel {
                period: 10,
                offset: 3,
                jitter: 3,
            }
        );
    }

    #[test]
    fn helper_examples_return_expected_values() {
        assert_eq!(
            evenly_spaced_around(10.0, 2.0, 5).collect::<Vec<_>>(),
            vec![10.0, 8.0, 9.333333333333334, 10.666666666666666, 12.0]
        );

        let candidates = PeriodicModel {
            period: 10,
            offset: 3,
            jitter: 2,
        }
        .candidate_periods(&PeriodicConfig {
            n_candidates: 5,
            candidate_dispersion: 2.0,
            rounding_adjustments: vec![-1, 0, 1],
            ..PeriodicConfig::default()
        });
        assert_eq!(candidates, vec![6, 9, 10, 11, 14]);

        let rounded = PeriodicModel {
            period: 123,
            offset: 0,
            jitter: 0,
        }
        .candidate_periods(&PeriodicConfig {
            n_candidates: 5,
            candidate_dispersion: 2.0,
            rounding_adjustments: vec![-1, 0, 1],
            ..PeriodicConfig::default()
        });
        assert_eq!(rounded, vec![100, 110, 120, 123, 130, 200]);
        assert_eq!(trailing_zeroes(100000), 5);
    }

    #[test]
    fn truncate_to_max_len_removes_worst_jitter_bounds() {
        let mut candidates = vec![
            PeriodicModel {
                period: 10,
                offset: 3,
                jitter: 4,
            },
            PeriodicModel {
                period: 11,
                offset: 2,
                jitter: 1,
            },
            PeriodicModel {
                period: 12,
                offset: 1,
                jitter: 7,
            },
            PeriodicModel {
                period: 13,
                offset: 0,
                jitter: 2,
            },
        ];

        truncate_to_max_len(&mut candidates, 2);

        assert_eq!(candidates.len(), 2);
        assert_eq!(
            candidates.iter().map(|candidate| candidate.jitter).max(),
            Some(2)
        );
    }

    #[test]
    fn uncertain_helper_functions_return_expected_values() {
        let sample_batch = window_batch(&[
            (0, ReleaseWindow::new(3, 3)),
            (1, ReleaseWindow::new(14, 14)),
            (2, ReleaseWindow::new(24, 24)),
            (3, ReleaseWindow::new(35, 35)),
        ]);
        let config = PeriodicConfig {
            overlap: 2,
            ..PeriodicConfig::default()
        };

        assert!((sample_batch.clean_slice_and_mean_gap().1 - 32.0 / 3.0).abs() < f64::EPSILON);
        assert_eq!(sample_batch.last_processed_index(&config), 1);
        assert_eq!(
            {
                let mut derived_candidates = vec![
                    PeriodicModel {
                        period: 10,
                        offset: 40,
                        jitter: 3,
                    },
                    PeriodicModel {
                        period: 14,
                        offset: 30,
                        jitter: 5,
                    },
                    PeriodicModel {
                        period: 20,
                        offset: 5,
                        jitter: 8,
                    },
                ];
                let last_processed_index = window_batch(&[
                    (5, ReleaseWindow::new(50, 50)),
                    (6, ReleaseWindow::new(62, 62)),
                    (7, ReleaseWindow::new(74, 74)),
                ])
                .last_processed_index(&config);
                let candidate_count = derived_candidates.len();
                add_derived_model_candidate(
                    &mut derived_candidates,
                    candidate_count,
                    13,
                    last_processed_index,
                );
                add_derived_model_candidate(
                    &mut derived_candidates,
                    candidate_count,
                    22,
                    last_processed_index,
                );
                derived_candidates[3..].to_vec()
            },
            vec![
                PeriodicModel {
                    period: 13,
                    offset: 30,
                    jitter: 11
                },
                PeriodicModel {
                    period: 22,
                    offset: -7,
                    jitter: 20
                },
            ]
        );
    }

    #[test]
    fn extractor_candidate_count_does_not_grow_on_irregular_point_trace() {
        let gaps: Vec<u64> = (0..128)
            .map(|index| 3 + ((index * index) % 17) as u64)
            .collect();
        let mut release = 0u64;
        let releases: Vec<u64> = gaps
            .into_iter()
            .map(|gap| {
                release += gap;
                release
            })
            .collect();
        let config = PeriodicConfig {
            batch_size: 16,
            overlap: 2,
            n_candidates: 30,
            ..PeriodicConfig::default()
        };
        let mut extractor = PeriodicExtractorCore::<PointRelease>::with_config(config).unwrap();
        let mut previous_count = None;

        for chunk in releases.chunks(8) {
            extractor.feed(chunk.iter().copied());
            if extractor.candidate_count() != 0 {
                if let Some(previous_count) = previous_count {
                    assert!(extractor.candidate_count() <= previous_count);
                }
                previous_count = Some(extractor.candidate_count());
            }
        }
    }

    #[test]
    fn extractor_candidate_count_does_not_grow_on_irregular_window_trace() {
        let gaps: Vec<u64> = (0..128)
            .map(|index| 4 + ((index * 7) % 19) as u64)
            .collect();
        let mut release = 0u64;
        let windows: Vec<_> = gaps
            .into_iter()
            .enumerate()
            .map(|(index, gap)| {
                release += gap;
                let spread = (index as u64 % 5) + 1;
                ReleaseWindow::new(release.saturating_sub(spread), release + spread)
            })
            .collect();
        let config = PeriodicConfig {
            batch_size: 16,
            overlap: 2,
            n_candidates: 30,
            ..PeriodicConfig::default()
        };
        let mut certain = PeriodicExtractorCore::<CertainFit>::with_config(config.clone()).unwrap();
        let mut possible = PeriodicExtractorCore::<PossibleFit>::with_config(config).unwrap();
        let mut previous_certain = None;
        let mut previous_possible = None;

        for chunk in windows.chunks(8) {
            certain.feed(chunk.iter().copied());
            if certain.candidate_count() != 0 {
                if let Some(previous_certain) = previous_certain {
                    assert!(certain.candidate_count() <= previous_certain);
                }
                previous_certain = Some(certain.candidate_count());
            }

            possible.feed(chunk.iter().copied());
            if possible.candidate_count() != 0 {
                if let Some(previous_possible) = previous_possible {
                    assert!(possible.candidate_count() <= previous_possible);
                }
                previous_possible = Some(possible.candidate_count());
            }
        }
    }

    #[test]
    fn infer_model_matches_point_release_trace() {
        let releases: Vec<u64> = (0..100).map(|index| 7 + 13 * index).collect();
        let config = PeriodicConfig {
            batch_size: 11,
            overlap: 3,
            n_candidates: 20,
            candidate_dispersion: 2.0,
            rounding_adjustments: vec![-1, 0, 1],
            ..PeriodicConfig::default()
        };

        let model = infer_model::<PointRelease, _>(releases, &config).unwrap();
        assert_eq!(
            model,
            PeriodicModel {
                period: 13,
                offset: 7,
                jitter: 0,
            }
        );
    }
}
