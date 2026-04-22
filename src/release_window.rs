use crate::{SporadicError, time::Instant};

type Result<T> = std::result::Result<T, SporadicError>;

/// Inclusive release-time window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseWindow {
    pub lo: Instant,
    pub hi: Instant,
}

impl ReleaseWindow {
    pub const fn new(lo: Instant, hi: Instant) -> Self {
        Self { lo, hi }
    }

    pub(crate) fn ensure_monotonicity(self, last: Option<&Self>) -> Result<()> {
        if let Some(last) = last {
            if last.lo > self.lo {
                return Err(SporadicError::NonMonotonicLowerBounds);
            }
            if last.hi > self.hi {
                return Err(SporadicError::NonMonotonicUpperBounds);
            }
        }

        Ok(())
    }
}

impl From<(Instant, Instant)> for ReleaseWindow {
    fn from((lo, hi): (Instant, Instant)) -> Self {
        Self { lo, hi }
    }
}
