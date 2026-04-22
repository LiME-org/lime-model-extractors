//! Shared sporadic-task inference core for exact releases and release windows.

use std::{collections::VecDeque, marker::PhantomData};

use crate::{
    certain_sporadic::SporadicError,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, SporadicError>;

/// Policy hook for delta-min extraction over different observation types.
///
/// Implementors define how the shared extractor:
/// - validates monotonicity between successive observations
/// - estimates the interval end indicated by the current observation
/// - estimates the interval start indicated by each prior observation
/// - derives the synthetic observations needed to initialize and close
///   delta-max extraction
pub trait ApproximationMode {
    /// The observation type consumed by the shared extractor core.
    type Observation: Copy;

    /// Reject a new observation if it violates the required monotonicity
    /// constraints with respect to the previously buffered observation.
    fn ensure_monotonicity(
        observation: Self::Observation,
        last: Option<&Self::Observation>,
    ) -> Result<()>;

    /// Estimate the interval end marked by the current observation when
    /// computing a delta-min candidate.
    fn delta_min_interval_end(observation: Self::Observation) -> Instant;

    /// Estimate the interval start marked by a prior observation when
    /// computing a delta-min candidate.
    fn delta_min_interval_start(observation: Self::Observation) -> Instant;

    /// Estimate the interval end marked by the current observation when
    /// computing a delta-max candidate.
    fn delta_max_interval_end(observation: Self::Observation) -> Instant;

    /// Estimate the interval start marked by a prior observation when
    /// computing a delta-max candidate.
    fn delta_max_interval_start(observation: Self::Observation) -> Instant;

    /// Derive the synthetic placeholder observation used before the first
    /// real observation for delta-max extraction.
    fn delta_max_initial_observation(observation: Self::Observation) -> Self::Observation;

    /// Derive the synthetic observation used to close the final interval when
    /// querying or consuming a delta-max extractor.
    fn delta_max_closing_observation(observation: Self::Observation) -> Self::Observation;
}

/// Infer a delta-min vector from a sequence of observations.
pub fn infer_delta_min<M, I>(observations: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    M: ApproximationMode,
    I: IntoIterator<Item = M::Observation>,
{
    let mut extractor = DeltaMinExtractorCore::<M>::new(nmax)?;
    extractor.feed(observations)?;
    Ok(extractor.into_model())
}

/// Infer a delta-max vector from a sequence of observations.
pub fn infer_delta_max<M, I>(observations: I, nmax: Option<usize>) -> Result<Vec<Duration>>
where
    M: ApproximationMode,
    I: IntoIterator<Item = M::Observation>,
{
    let mut extractor = DeltaMaxExtractorCore::<M>::new(nmax)?;
    extractor.feed(observations)?;
    Ok(extractor.into_model())
}

#[derive(Debug, Clone)]
pub struct DeltaMinExtractorCore<M: ApproximationMode> {
    max_history: Option<usize>,
    dmin: Vec<Duration>,
    buffer: VecDeque<M::Observation>,
    mode: PhantomData<M>,
}

impl<M: ApproximationMode> DeltaMinExtractorCore<M> {
    /// Construct an extractor with an optional bound on the inferred prefix size.
    pub fn new(nmax: Option<usize>) -> Result<Self> {
        SporadicError::validate(nmax, 2)?;

        let mut dmin = Vec::with_capacity(nmax.unwrap_or(2).max(2) + 1);
        dmin.extend([0, 1]);

        Ok(Self {
            max_history: nmax.map(|limit| limit - 1),
            dmin,
            buffer: nmax.map_or_else(VecDeque::new, |limit| VecDeque::with_capacity(limit - 1)),
            mode: PhantomData,
        })
    }

    #[inline]
    // Apply the delta-min update to one contiguous history slice.
    //
    // `VecDeque::as_slices()` can split the ring buffer into two slices, so the
    // outer update logic calls this helper once for each slice instead of
    // duplicating the same loop twice. Using `as_slices()` also avoids copying
    // the deque into a temporary buffer just to get slice-based iteration.
    fn update_dmin_chunk(
        history: &[M::Observation],
        dmin: &mut [Duration],
        interval_end: Instant,
        mut count: usize,
    ) {
        for &observation in history {
            let delta = interval_end
                .saturating_sub(M::delta_min_interval_start(observation))
                .saturating_add(1);
            dmin[count] = dmin[count].min(delta);
            count -= 1;
        }
    }

    #[inline]
    fn update_dmin(&mut self, observation: M::Observation) {
        let history_len = self.buffer.len();
        if history_len == 0 {
            return;
        }

        let max_count = history_len + 1;
        if self.dmin.len() <= max_count {
            self.dmin.resize(max_count + 1, u64::MAX);
        }

        // The history buffer is a `VecDeque`, so its logical contents may wrap
        // around the ring-buffer boundary. `as_slices()` exposes that as up to
        // two contiguous slices, which are processed in order with the same
        // helper. This keeps the update slice-based without calling
        // `make_contiguous()` or copying into a temporary `Vec`.
        let (head, tail) = self.buffer.as_slices();
        let dmin = &mut self.dmin;
        let interval_end = M::delta_min_interval_end(observation);

        Self::update_dmin_chunk(head, dmin, interval_end, max_count);
        Self::update_dmin_chunk(tail, dmin, interval_end, tail.len() + 1);
    }

    #[inline]
    fn limit_buffer_size(&mut self) {
        if let Some(limit) = self.max_history
            && self.buffer.len() == limit
        {
            self.buffer.pop_front();
        }
    }

    /// Update the extractor with additional releases or release windows.
    pub fn feed<I>(&mut self, observations: I) -> Result<()>
    where
        I: IntoIterator<Item = M::Observation>,
    {
        for observation in observations {
            M::ensure_monotonicity(observation, self.buffer.back())?;

            self.update_dmin(observation);
            self.limit_buffer_size();
            self.buffer.push_back(observation);
        }

        Ok(())
    }

    /// Query the current model estimate.
    pub fn current_model(&self) -> Vec<Duration> {
        if !self.buffer.is_empty() {
            self.dmin.clone()
        } else {
            Vec::new()
        }
    }

    /// Consume the extractor and return the inferred model without cloning.
    pub fn into_model(self) -> Vec<Duration> {
        if !self.buffer.is_empty() {
            self.dmin
        } else {
            Vec::new()
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeltaMaxExtractorCore<M: ApproximationMode> {
    nmax: Option<usize>,
    dmax: Vec<Duration>,
    buffer: VecDeque<M::Observation>,
    mode: PhantomData<M>,
}

impl<M: ApproximationMode> DeltaMaxExtractorCore<M> {
    /// Construct an extractor with an optional bound on the inferred prefix size.
    pub fn new(nmax: Option<usize>) -> Result<Self> {
        SporadicError::validate(nmax, 1)?;

        let dmax = nmax.map_or_else(Vec::new, |limit| Vec::with_capacity(limit + 1));

        Ok(Self {
            nmax,
            dmax,
            buffer: nmax.map_or_else(VecDeque::new, |limit| VecDeque::with_capacity(limit + 1)),
            mode: PhantomData,
        })
    }

    #[inline]
    fn update_dmax_chunk(
        history: &[M::Observation],
        dmax: &mut [Duration],
        interval_end: Instant,
        mut count: usize,
    ) {
        for &observation in history.iter().rev() {
            let delta = interval_end
                .saturating_sub(M::delta_max_interval_start(observation).saturating_add(1));
            dmax[count] = dmax[count].max(delta);
            count += 1;
        }
    }

    #[inline]
    fn update_dmax(
        nmax: Option<usize>,
        buffer: &VecDeque<M::Observation>,
        dmax: &mut Vec<Duration>,
        observation: M::Observation,
    ) {
        if buffer.is_empty() {
            return;
        }

        let history_len = buffer.len();
        let required = if let Some(limit) = nmax {
            history_len.min(limit + 1)
        } else {
            history_len
        };
        if dmax.len() < required {
            dmax.resize(required, 0);
        }

        let (head, tail) = buffer.as_slices();
        let interval_end = M::delta_max_interval_end(observation);

        Self::update_dmax_chunk(tail, dmax, interval_end, 0);
        Self::update_dmax_chunk(head, dmax, interval_end, tail.len());
    }

    #[inline]
    fn prefill_buffer(&mut self, observation: M::Observation) {
        if self.buffer.is_empty() {
            self.buffer
                .push_back(M::delta_max_initial_observation(observation));
        }
    }

    #[inline]
    fn limit_buffer_size(&mut self) {
        if let Some(limit) = self.nmax
            && self.buffer.len() == limit + 1
        {
            self.buffer.pop_front();
        }
    }

    /// Update the extractor with additional releases or release windows.
    pub fn feed<I>(&mut self, observations: I) -> Result<()>
    where
        I: IntoIterator<Item = M::Observation>,
    {
        for observation in observations {
            M::ensure_monotonicity(observation, self.buffer.back())?;

            self.prefill_buffer(observation);
            Self::update_dmax(self.nmax, &self.buffer, &mut self.dmax, observation);
            self.limit_buffer_size();
            self.buffer.push_back(observation);
        }

        Ok(())
    }

    pub fn current_model(&self) -> Vec<Duration> {
        let mut dmax = self.dmax.clone();

        if let Some(&last) = self.buffer.back() {
            Self::update_dmax(
                self.nmax,
                &self.buffer,
                &mut dmax,
                M::delta_max_closing_observation(last),
            );
        }

        dmax
    }

    /// Consume the extractor and return the inferred model without cloning.
    pub fn into_model(mut self) -> Vec<Duration> {
        if let Some(&last) = self.buffer.back() {
            Self::update_dmax(
                self.nmax,
                &self.buffer,
                &mut self.dmax,
                M::delta_max_closing_observation(last),
            );
        }

        self.dmax
    }
}
