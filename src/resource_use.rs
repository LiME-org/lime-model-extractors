//! Minimum and maximum cumulative resource use over consecutive jobs.
//! Entry `n` bounds the total for `n` jobs. Each window total must fit in `u64`.
//! The model stays empty until the first job, then entry zero represents the empty window.

use thiserror::Error;

pub type ResourceAmount = u64;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResourceUseError {
    #[error("nmax must be positive")]
    InvalidNmax,
}

/// Tracks the maximum total resource use for each window length.
#[derive(Debug, Clone, Default)]
pub struct MaxResourceUseExtractor {
    nmax: Option<usize>,
    suffix_totals: Vec<ResourceAmount>,
    bounds: Vec<ResourceAmount>,
}

impl MaxResourceUseExtractor {
    /// Track windows of at most `nmax` jobs. Use `None` for all lengths or a positive limit.
    pub fn new(nmax: Option<usize>) -> Result<Self, ResourceUseError> {
        if nmax == Some(0) {
            return Err(ResourceUseError::InvalidNmax);
        }
        Ok(Self {
            nmax,
            ..Self::default()
        })
    }

    /// Update the extractor with additional job observations.
    pub fn feed<I>(&mut self, observations: I)
    where
        I: IntoIterator<Item = ResourceAmount>,
    {
        for job in observations {
            if self.bounds.is_empty() {
                self.bounds.push(0);
            }

            let mut shorter_total = 0;
            for (suffix, bound) in self.suffix_totals.iter_mut().zip(&mut self.bounds[1..]) {
                let total = shorter_total + job;
                shorter_total = *suffix;
                *suffix = total;
                *bound = (*bound).max(total);
            }
            if self.nmax != Some(self.suffix_totals.len()) {
                let total = shorter_total + job;
                self.suffix_totals.push(total);
                self.bounds.push(total);
            }
        }
    }

    /// Return the current upper bounds by window length.
    pub fn current_model(&self) -> Vec<ResourceAmount> {
        self.bounds.clone()
    }

    /// Take the bounds by window length without copying them.
    pub fn into_model(self) -> Vec<ResourceAmount> {
        self.bounds
    }
}

/// Tracks the minimum total resource use for each window length.
#[derive(Debug, Clone, Default)]
pub struct MinResourceUseExtractor {
    nmax: Option<usize>,
    suffix_totals: Vec<ResourceAmount>,
    bounds: Vec<ResourceAmount>,
}

impl MinResourceUseExtractor {
    /// Track windows of at most `nmax` jobs. Use `None` for all lengths or a positive limit.
    pub fn new(nmax: Option<usize>) -> Result<Self, ResourceUseError> {
        if nmax == Some(0) {
            return Err(ResourceUseError::InvalidNmax);
        }
        Ok(Self {
            nmax,
            ..Self::default()
        })
    }

    /// Update the extractor with additional job observations.
    pub fn feed<I>(&mut self, observations: I)
    where
        I: IntoIterator<Item = ResourceAmount>,
    {
        for job in observations {
            if self.bounds.is_empty() {
                self.bounds.push(0);
            }

            let mut shorter_total = 0;
            for (suffix, bound) in self.suffix_totals.iter_mut().zip(&mut self.bounds[1..]) {
                let total = shorter_total + job;
                shorter_total = *suffix;
                *suffix = total;
                *bound = (*bound).min(total);
            }
            if self.nmax != Some(self.suffix_totals.len()) {
                let total = shorter_total + job;
                self.suffix_totals.push(total);
                self.bounds.push(total);
            }
        }
    }

    /// Return the current lower bounds by window length.
    pub fn current_model(&self) -> Vec<ResourceAmount> {
        self.bounds.clone()
    }

    /// Take the bounds by window length without copying them.
    pub fn into_model(self) -> Vec<ResourceAmount> {
        self.bounds
    }
}

/// Return the maximum total resource use over `n` consecutive jobs at index `n`.
pub fn infer_max_resource_use<I>(
    observations: I,
    nmax: Option<usize>,
) -> Result<Vec<ResourceAmount>, ResourceUseError>
where
    I: IntoIterator<Item = ResourceAmount>,
{
    let mut extractor = MaxResourceUseExtractor::new(nmax)?;
    extractor.feed(observations);
    Ok(extractor.into_model())
}

/// Return the minimum total resource use over `n` consecutive jobs at index `n`.
pub fn infer_min_resource_use<I>(
    observations: I,
    nmax: Option<usize>,
) -> Result<Vec<ResourceAmount>, ResourceUseError>
where
    I: IntoIterator<Item = ResourceAmount>,
{
    let mut extractor = MinResourceUseExtractor::new(nmax)?;
    extractor.feed(observations);
    Ok(extractor.into_model())
}
