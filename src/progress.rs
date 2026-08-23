// src/progress.rs

/// How often the pipeline emits `ReadsProcessed`, in reads.
pub const PROGRESS_INTERVAL: u64 = 100_000;

/// A progress event emitted by the core.
///
/// The core never renders progress. It hands these to a caller-supplied
/// callback, and each client decides what to do: the CLI drives `indicatif`,
/// Python forwards to `tqdm` or discards, WASM ignores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// Cumulative reads consumed so far.
    ReadsProcessed(u64),
    /// A cohort sample has started processing (0-based index).
    SampleStarted { index: usize, total: usize },
    /// A cohort sample has finished processing (0-based index).
    SampleFinished { index: usize, total: usize },
    /// Work is complete; carries the final read count.
    Finished { reads: u64 },
}

/// An optional progress callback.
///
/// `None` means silence, which is the default for library use. The callback is
/// invoked from worker threads, hence `Send + Sync`.
pub type ProgressFn<'a> = Option<&'a (dyn Fn(Progress) + Send + Sync)>;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_closure_can_be_used_as_a_progress_callback() {
        let seen: Arc<Mutex<Vec<Progress>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();

        let callback = move |event: Progress| {
            sink.lock().unwrap().push(event);
        };
        let progress: ProgressFn = Some(&callback);

        if let Some(f) = progress {
            f(Progress::ReadsProcessed(100_000));
            f(Progress::Finished { reads: 250_000 });
        }

        let events = seen.lock().unwrap();
        assert_eq!(
            *events,
            vec![
                Progress::ReadsProcessed(100_000),
                Progress::Finished { reads: 250_000 },
            ]
        );
    }

    #[test]
    fn absent_callback_is_representable_and_costs_nothing() {
        let progress: ProgressFn = None;
        assert!(progress.is_none(), "silence must be the representable default");
    }
}
