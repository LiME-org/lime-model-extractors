//! Bag-of-segments (BOS) inference for jobs with different self-suspension patterns.

use std::cmp::Reverse;

use crate::suspension::{
    BagOfSegmentsModel, ObservedSegment, Segment, SegmentedSuspensionModel, SuspensionError,
    merge_segments,
};

type Result<T> = std::result::Result<T, SuspensionError>;

// Prefer increases in suspension-time bounds over increases in execution-time bounds,
// since execution-time bounds directly affect utilization.
fn merge_cost(target: &[Segment], source: &[Segment], cutoff: u64) -> u64 {
    let (shared, tail) = source.split_at(source.len().min(target.len()));
    let increases = shared
        .iter()
        .zip(target)
        .map(|(segment, bound)| {
            (
                segment
                    .max_suspension_time
                    .saturating_sub(bound.max_suspension_time),
                segment
                    .max_execution_time
                    .saturating_sub(bound.max_execution_time),
            )
        })
        .chain(
            tail.iter()
                .map(|s| (s.max_suspension_time, s.max_execution_time)),
        );

    // A squared u64 increase can overflow u64 before we compare it with the cutoff.
    let mut cost = 0_u128;
    for (suspension, execution) in increases {
        cost += u128::from(suspension) + u128::from(execution).pow(2);
        // The remaining terms are nonnegative, so they cannot make this pair cheaper.
        if cost > u128::from(cutoff) {
            return u64::MAX;
        }
    }
    cost as u64
}

#[inline]
fn covers(model: &[Segment], required: &[ObservedSegment]) -> bool {
    model.len() >= required.len()
        && model
            .iter()
            .zip(required)
            .all(|(bound, &(suspension, execution))| {
                suspension <= bound.max_suspension_time && execution <= bound.max_execution_time
            })
}

fn absorbed_by(source: &[Segment], incoming: &[ObservedSegment]) -> bool {
    source.len() <= incoming.len()
        && source
            .iter()
            .zip(incoming)
            .all(|(segment, &(suspension, execution))| {
                segment.max_suspension_time <= suspension && segment.max_execution_time <= execution
            })
}

#[derive(Debug, Clone, Default)]
struct ModelBag {
    models: BagOfSegmentsModel,
    // Cost and target index of the cheapest merge for each source model.
    best_merges: Vec<Option<(u64, Reverse<usize>)>>,
}

/// Builds a [`BagOfSegmentsModel`] whose models “together cover the observed jobs”.
#[derive(Debug, Clone)]
pub struct BosSuspensionModelExtractor {
    max_segments: Option<usize>,
    max_models: Option<usize>,
    bag: Result<ModelBag>,
}

impl Default for BosSuspensionModelExtractor {
    fn default() -> Self {
        Self {
            max_segments: None,
            max_models: None,
            bag: Ok(ModelBag::default()),
        }
    }
}

impl BosSuspensionModelExtractor {
    /// Limit segments per job and models per bag independently. Each limit is positive or `None`.
    pub fn new(max_segments: Option<usize>, max_models: Option<usize>) -> Result<Self> {
        if max_segments == Some(0) {
            return Err(SuspensionError::InvalidMaxSegments);
        }
        if max_models == Some(0) {
            return Err(SuspensionError::InvalidMaxModels);
        }
        Ok(Self {
            max_segments,
            max_models,
            ..Self::default()
        })
    }

    /// Update the extractor with additional job observations.
    /// If a job exceeds `max_segments`, the bag is discarded. This and all later calls
    /// return [`SuspensionError::SegmentLimitExceeded`] without reading further jobs.
    /// Other errors leave earlier jobs from the same call in the bag.
    pub fn feed<I, O>(&mut self, observations: I) -> Result<()>
    where
        I: IntoIterator<Item = O>,
        O: AsRef<[ObservedSegment]>,
    {
        let bag = self.bag.as_mut().map_err(|error| error.clone())?;
        for job in observations {
            let segments = job.as_ref();
            if let Some(limit) = self.max_segments
                && segments.len() > limit
            {
                let error = SuspensionError::SegmentLimitExceeded {
                    observed: segments.len(),
                    limit,
                };
                self.bag = Err(error.clone());
                return Err(error);
            }
            if segments.is_empty() {
                return Err(SuspensionError::EmptyObservation);
            }
            if segments.contains(&(0, 0)) {
                return Err(SuspensionError::ZeroSegment);
            }
            if !bag.models.iter().any(|model| covers(model, segments)) {
                bag.insert(segments, self.max_models);
            }
        }
        Ok(())
    }

    /// Return a copy of the bag, shortest models first, or `None` if a job exceeded `max_segments`.
    pub fn current_model(&self) -> Option<BagOfSegmentsModel> {
        // Keep insertion order in the bag to break ties between equal-cost merges.
        let mut models = self.bag.as_ref().ok()?.models.clone();
        sort_models(&mut models);
        Some(models)
    }

    /// Return the bag without copying it, in the same order as [`Self::current_model`].
    pub fn into_model(self) -> Option<BagOfSegmentsModel> {
        let mut models = self.bag.ok()?.models;
        sort_models(&mut models);
        Some(models)
    }
}

fn sort_models(models: &mut [SegmentedSuspensionModel]) {
    models.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
}

impl ModelBag {
    #[inline]
    fn insert(&mut self, observed: &[ObservedSegment], max_models: Option<usize>) {
        let segments = observed.iter().copied().map(Segment::from);
        match max_models {
            None => self.models.push(segments.collect()),
            Some(limit) => {
                if limit == 1
                    && let Some(model) = self.models.first_mut()
                {
                    merge_segments(model, segments);
                } else if self.models.len() == limit
                    && let Some(source) = self
                        .models
                        .iter()
                        .rposition(|model| absorbed_by(model, observed))
                {
                    // The new model covers this source, so merging them costs zero.
                    let mut model = self.remove_model(source);
                    merge_segments(&mut model, segments);
                    self.add_model(model);
                } else {
                    self.add_model(segments.collect());
                    if self.models.len() > limit {
                        self.consolidate();
                    }
                }
            }
        }
    }

    fn add_model(&mut self, model: SegmentedSuspensionModel) {
        let models = &mut self.models;
        let index = models.len();
        // Inserting a model leaves all old pair costs unchanged.
        // Merge costs are directional, so score the new model as both a target and a source.
        for (cached, source_model) in self.best_merges.iter_mut().zip(models.iter()) {
            let cutoff = cached.map_or(u64::MAX, |(cost, _)| cost);
            let cost = merge_cost(&model, source_model, cutoff);
            if cost <= cutoff {
                *cached = Some((cost, Reverse(index)));
            }
        }
        let mut best = None;
        // Visit newer targets first so a tie keeps the newer target.
        for (target, target_model) in models.iter().enumerate().rev() {
            let cutoff = best.map_or(u64::MAX, |(cost, _)| cost);
            let cost = merge_cost(target_model, &model, cutoff);
            if best.is_none() || cost < cutoff {
                best = Some((cost, Reverse(target)));
                if cost == 0 {
                    break;
                }
            }
        }
        self.best_merges.push(best);
        models.push(model);
    }

    fn cheapest_merge(&self) -> (usize, usize) {
        // Different pairs can have the same merge cost but produce different segment bounds.
        // To make the result deterministic, break ties by newest target, then newest source.
        let (_, Reverse(target), Reverse(source)) = self
            .best_merges
            .iter()
            .enumerate()
            .filter_map(|(source, best)| best.map(|(cost, target)| (cost, target, Reverse(source))))
            .min()
            .expect("consolidation requires at least two models");
        (target, source)
    }

    fn remove_model(&mut self, index: usize) -> SegmentedSuspensionModel {
        self.best_merges.remove(index);
        for best in &mut self.best_merges {
            if let Some((_, Reverse(target))) = best {
                if *target == index {
                    // The merged target covers the old winner, so it will match or beat
                    // every surviving target. Its insertion fills this slot.
                    *best = None;
                } else if *target > index {
                    *target -= 1;
                }
            }
        }
        self.models.remove(index)
    }

    fn consolidate(&mut self) {
        let (target, source) = self.cheapest_merge();
        let mut merged = self.remove_model(target.max(source));
        let mut other = self.remove_model(target.min(source));
        // Merging in either direction gives the same bounds, so reuse a buffer that fits the longer model.
        if merged.capacity() < other.len() {
            std::mem::swap(&mut merged, &mut other);
        }
        merge_segments(&mut merged, other);
        self.add_model(merged);
    }
}

/// Infer a bag of segmented models, merging the least-cost pair when the bag exceeds `max_models`.
/// Returns `None` if a job exceeds `max_segments`.
pub fn infer_bos_suspension_model<I, O>(
    observations: I,
    max_segments: Option<usize>,
    max_models: Option<usize>,
) -> Result<Option<BagOfSegmentsModel>>
where
    I: IntoIterator<Item = O>,
    O: AsRef<[ObservedSegment]>,
{
    let mut extractor = BosSuspensionModelExtractor::new(max_segments, max_models)?;
    match extractor.feed(observations) {
        Ok(()) => Ok(extractor.into_model()),
        Err(SuspensionError::SegmentLimitExceeded { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::{BosSuspensionModelExtractor, ModelBag, infer_bos_suspension_model, merge_cost};
    use crate::{Segment, SuspensionError, infer_segmented_suspension_model};

    #[test]
    fn merge_ties_favor_newer_models() {
        let jobs = [[(15, 7)], [(24, 3)], [(19, 5)], [(12, 9)], [(1, 11)]];
        let expected = vec![
            vec![Segment::from((15, 11))],
            vec![Segment::from((19, 5))],
            vec![Segment::from((24, 3))],
        ];
        let mut extractor = BosSuspensionModelExtractor::new(None, Some(3)).unwrap();
        for job in jobs {
            extractor.feed([job]).unwrap();
        }
        assert_eq!(extractor.current_model(), Some(expected.clone()));
        assert_eq!(
            infer_bos_suspension_model(jobs, None, Some(3)).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn one_model_matches_segmented_bounds() {
        let jobs = [
            vec![(1, 9), (8, 2)],
            vec![(3, 4)],
            vec![(2, 7), (5, 6), (11, 13)],
        ];
        let segmented = infer_segmented_suspension_model(&jobs, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            infer_bos_suspension_model(&jobs, None, Some(1)).unwrap(),
            Some(vec![segmented])
        );
    }

    #[test]
    fn coverage_and_sorting_match_python_one_shot() {
        let jobs = [[(2, 9)], [(1, 10)], [(0, 8)]];
        assert_eq!(
            infer_bos_suspension_model(jobs, None, None).unwrap(),
            Some(vec![
                vec![Segment::from((1, 10))],
                vec![Segment::from((2, 9))]
            ])
        );
    }

    #[test]
    fn bos_limits_and_empty_jobs_are_validated() {
        assert_eq!(
            BosSuspensionModelExtractor::new(Some(0), None).unwrap_err(),
            SuspensionError::InvalidMaxSegments
        );
        assert_eq!(
            BosSuspensionModelExtractor::new(None, Some(0)).unwrap_err(),
            SuspensionError::InvalidMaxModels
        );
        let mut extractor = BosSuspensionModelExtractor::default();
        assert_eq!(extractor.current_model(), Some(Vec::new()));
        assert_eq!(
            extractor.feed([vec![(1, 2)], vec![]]),
            Err(SuspensionError::EmptyObservation)
        );
        assert_eq!(
            extractor.current_model(),
            Some(vec![vec![Segment::from((1, 2))]])
        );
    }

    #[test]
    fn segment_limit_invalidation_is_permanent() {
        let mut extractor = BosSuspensionModelExtractor::new(Some(1), Some(2)).unwrap();
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
    fn model_snapshots_do_not_mutate_extractor_state() {
        let mut extractor = BosSuspensionModelExtractor::default();
        extractor.feed([[(1, 2)]]).unwrap();
        let mut snapshot = extractor.current_model().unwrap();
        snapshot[0].push(Segment::from((3, 4)));
        snapshot.clear();
        assert_eq!(
            extractor.current_model(),
            Some(vec![vec![Segment::from((1, 2))]])
        );
    }

    #[test]
    fn merge_cost_is_directional_and_saturates() {
        let a = [Segment::from((3, 5))];
        let b = [Segment::from((7, 2))];
        assert_eq!(merge_cost(&a, &b, u64::MAX), 4);
        assert_eq!(merge_cost(&b, &a, u64::MAX), 9);
        let wide = [Segment::from((0, 1_u64 << 40))];
        assert_eq!(merge_cost(&[], &wide, u64::MAX), u64::MAX);
    }

    #[test]
    fn cached_merge_search_matches_exhaustive_search() {
        for seed in 0..128_u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let models: Vec<Vec<_>> = (0..2 + seed % 9)
                .map(|_| {
                    (0..rng.random_range(1..=6))
                        .map(|_| Segment::from((rng.random_range(0..13), rng.random_range(0..13))))
                        .collect()
                })
                .collect();
            let expected = models
                .iter()
                .enumerate()
                .flat_map(|(target, target_model)| {
                    models
                        .iter()
                        .enumerate()
                        .filter(move |(source, _)| *source != target)
                        .map(move |(source, source_model)| {
                            let cost: u128 = source_model
                                .iter()
                                .enumerate()
                                .map(|(index, segment)| {
                                    let bound =
                                        target_model.get(index).copied().unwrap_or_default();
                                    let execution = u128::from(
                                        segment
                                            .max_execution_time
                                            .saturating_sub(bound.max_execution_time),
                                    );
                                    execution * execution
                                        + u128::from(
                                            segment
                                                .max_suspension_time
                                                .saturating_sub(bound.max_suspension_time),
                                        )
                                })
                                .sum();
                            (cost, Reverse(target), Reverse(source))
                        })
                })
                .min()
                .unwrap();
            let mut bag = ModelBag::default();
            for model in models {
                bag.add_model(model);
            }
            assert_eq!(
                bag.cheapest_merge(),
                (expected.1.0, expected.2.0),
                "seed={seed}"
            );
        }
    }
}
