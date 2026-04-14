//! Computation resource management.
//!
//! [`ComputeContext`] controls timeout, memory limits, recursion depth,
//! and feature flags for CAS computations. Pass through computation
//! functions to enforce resource boundaries.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Error returned when a computation exceeds resource limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeError {
    /// Computation exceeded the time limit.
    Timeout {
        /// The configured timeout duration.
        limit: Duration,
    },
    /// Recursion depth exceeded.
    RecursionLimit {
        /// The configured maximum depth.
        limit: usize,
    },
    /// Computation was cancelled externally.
    Cancelled,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::Timeout { limit } => {
                write!(f, "computation timed out after {:?}", limit)
            }
            ComputeError::RecursionLimit { limit } => {
                write!(f, "recursion depth exceeded limit of {}", limit)
            }
            ComputeError::Cancelled => write!(f, "computation cancelled"),
        }
    }
}

impl std::error::Error for ComputeError {}

/// Result type for computations that can exceed resource limits.
pub type ComputeResult<T> = Result<T, ComputeError>;

/// Configuration and resource limits for CAS computations.
///
/// Create via [`ComputeContext::default()`] for reasonable defaults,
/// or use the builder methods to customize.
///
/// # Thread safety
///
/// The cancel flag is shared via `Arc<AtomicBool>`, so a context can
/// be cancelled from another thread.
///
/// # Usage
///
/// ```ignore
/// let ctx = ComputeContext::default().with_timeout(Duration::from_secs(5));
/// // Pass &ctx to computation functions
/// // Periodically call ctx.check()? inside loops
/// ```
#[derive(Clone, Debug)]
pub struct ComputeContext {
    /// When the computation started (set on creation).
    start: Instant,
    /// Maximum wall-clock time allowed.
    timeout: Option<Duration>,
    /// Maximum recursion depth.
    recursion_limit: usize,
    /// Current recursion depth.
    current_depth: usize,
    /// External cancellation flag.
    cancelled: Arc<AtomicBool>,
    /// Feature flags.
    features: FeatureFlags,
}

/// Feature flags controlling which algorithms are enabled.
#[derive(Clone, Debug)]
pub struct FeatureFlags {
    /// Allow use of floating-point approximations.
    pub allow_float: bool,
    /// Enable automatic simplification during computation.
    pub auto_simplify: bool,
    /// Enable Groebner basis computations (can be expensive).
    pub groebner_enabled: bool,
    /// Enable series expansion for limit computation.
    pub series_expansion: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        FeatureFlags {
            allow_float: false,
            auto_simplify: true,
            groebner_enabled: true,
            series_expansion: true,
        }
    }
}

impl Default for ComputeContext {
    fn default() -> Self {
        ComputeContext {
            start: Instant::now(),
            timeout: None,
            recursion_limit: 256,
            current_depth: 0,
            cancelled: Arc::new(AtomicBool::new(false)),
            features: FeatureFlags::default(),
        }
    }
}

impl ComputeContext {
    /// Create a context with a specific timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Create a context with a specific recursion limit.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Create a context with custom feature flags.
    pub fn with_features(mut self, features: FeatureFlags) -> Self {
        self.features = features;
        self
    }

    /// Get a handle that can cancel this computation from another thread.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            flag: self.cancelled.clone(),
        }
    }

    /// Check resource limits. Call periodically in computation loops.
    ///
    /// Returns `Ok(())` if within limits, or the appropriate error.
    pub fn check(&self) -> ComputeResult<()> {
        // Check cancellation first (cheapest)
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(ComputeError::Cancelled);
        }

        // Check timeout
        if let Some(limit) = self.timeout {
            if self.start.elapsed() >= limit {
                return Err(ComputeError::Timeout { limit });
            }
        }

        Ok(())
    }

    /// Enter a new recursion level. Returns a child context with
    /// incremented depth, or an error if the limit is exceeded.
    pub fn recurse(&self) -> ComputeResult<Self> {
        let new_depth = self.current_depth + 1;
        if new_depth > self.recursion_limit {
            return Err(ComputeError::RecursionLimit {
                limit: self.recursion_limit,
            });
        }
        Ok(ComputeContext {
            start: self.start,
            timeout: self.timeout,
            recursion_limit: self.recursion_limit,
            current_depth: new_depth,
            cancelled: self.cancelled.clone(),
            features: self.features.clone(),
        })
    }

    /// Current recursion depth.
    pub fn depth(&self) -> usize {
        self.current_depth
    }

    /// Remaining time before timeout, if a timeout is set.
    pub fn remaining(&self) -> Option<Duration> {
        self.timeout
            .map(|limit| limit.saturating_sub(self.start.elapsed()))
    }

    /// Reference to feature flags.
    pub fn features(&self) -> &FeatureFlags {
        &self.features
    }

    /// Elapsed time since context creation.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Handle to cancel a computation from another thread.
#[derive(Clone, Debug)]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
}

impl CancelHandle {
    /// Signal cancellation.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_default_context() {
        let ctx = ComputeContext::default();
        assert!(ctx.check().is_ok());
        assert_eq!(ctx.depth(), 0);
        assert!(ctx.remaining().is_none());
    }

    #[test]
    fn test_timeout() {
        let ctx = ComputeContext::default().with_timeout(Duration::from_millis(10));
        assert!(ctx.check().is_ok());
        thread::sleep(Duration::from_millis(20));
        match ctx.check() {
            Err(ComputeError::Timeout { limit }) => {
                assert_eq!(limit, Duration::from_millis(10));
            }
            other => panic!("expected Timeout, got {:?}", other),
        }
    }

    #[test]
    fn test_recursion_limit() {
        let ctx = ComputeContext::default().with_recursion_limit(3);

        let r1 = ctx.recurse().unwrap();
        assert_eq!(r1.depth(), 1);

        let r2 = r1.recurse().unwrap();
        assert_eq!(r2.depth(), 2);

        let r3 = r2.recurse().unwrap();
        assert_eq!(r3.depth(), 3);

        // Exceeds limit
        match r3.recurse() {
            Err(ComputeError::RecursionLimit { limit }) => {
                assert_eq!(limit, 3);
            }
            other => panic!("expected RecursionLimit, got {:?}", other),
        }
    }

    #[test]
    fn test_cancel_from_another_thread() {
        let ctx = ComputeContext::default();
        let handle = ctx.cancel_handle();

        assert!(ctx.check().is_ok());

        // Cancel from another thread
        let t = thread::spawn(move || {
            handle.cancel();
        });
        t.join().unwrap();

        match ctx.check() {
            Err(ComputeError::Cancelled) => {}
            other => panic!("expected Cancelled, got {:?}", other),
        }
    }

    #[test]
    fn test_cancel_handle_clone() {
        let ctx = ComputeContext::default();
        let h1 = ctx.cancel_handle();
        let h2 = h1.clone();

        h2.cancel();
        assert!(ctx.check().is_err());
    }

    #[test]
    fn test_remaining_time() {
        let ctx = ComputeContext::default().with_timeout(Duration::from_secs(10));
        let remaining = ctx.remaining().unwrap();
        assert!(remaining <= Duration::from_secs(10));
        assert!(remaining > Duration::from_secs(9));
    }

    #[test]
    fn test_feature_flags_default() {
        let f = FeatureFlags::default();
        assert!(!f.allow_float);
        assert!(f.auto_simplify);
        assert!(f.groebner_enabled);
        assert!(f.series_expansion);
    }

    #[test]
    fn test_custom_features() {
        let features = FeatureFlags {
            allow_float: true,
            auto_simplify: false,
            ..FeatureFlags::default()
        };
        let ctx = ComputeContext::default().with_features(features);
        assert!(ctx.features().allow_float);
        assert!(!ctx.features().auto_simplify);
    }

    #[test]
    fn test_error_display() {
        let e = ComputeError::Timeout {
            limit: Duration::from_secs(5),
        };
        assert!(e.to_string().contains("5s"));

        let e = ComputeError::RecursionLimit { limit: 100 };
        assert!(e.to_string().contains("100"));

        let e = ComputeError::Cancelled;
        assert_eq!(e.to_string(), "computation cancelled");
    }

    #[test]
    fn test_recurse_preserves_timeout() {
        let ctx = ComputeContext::default()
            .with_timeout(Duration::from_secs(5))
            .with_recursion_limit(10);

        let child = ctx.recurse().unwrap();
        // Child shares same start time and timeout
        assert!(child.remaining().is_some());
        assert_eq!(child.depth(), 1);
    }

    #[test]
    fn test_recurse_shares_cancel_flag() {
        let ctx = ComputeContext::default().with_recursion_limit(10);
        let child = ctx.recurse().unwrap();
        let handle = ctx.cancel_handle();

        handle.cancel();
        assert!(child.check().is_err());
    }
}
