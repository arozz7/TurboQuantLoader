use crate::model::backend::{GenerateEvent, GenerateStream, GenerateSummary};

/// Streaming wrapper around a [`GenerateStream`] channel.
///
/// Provides `next_event` for driving the stream event-by-event (e.g. for
/// printing tokens as they arrive in the REPL) and `collect_full` for
/// accumulating the entire response in one shot.
pub struct TokenStream {
    rx: GenerateStream,
}

impl TokenStream {
    /// Wrap a raw [`GenerateStream`] receiver.
    pub fn new(rx: GenerateStream) -> Self {
        Self { rx }
    }

    /// Receive the next [`GenerateEvent`] from the stream.
    ///
    /// Returns `None` when the underlying channel is closed.
    pub async fn next_event(&mut self) -> Option<GenerateEvent> {
        self.rx.recv().await
    }

    /// Unwrap the underlying [`GenerateStream`] for use with async stream adapters
    /// (e.g. `tokio_stream::wrappers::ReceiverStream` in the HTTP server).
    pub fn into_inner(self) -> GenerateStream {
        self.rx
    }

    /// Drain the stream and return the full text and final [`GenerateSummary`].
    ///
    /// # Errors
    /// Returns `Err(String)` if the backend sent a [`GenerateEvent::Error`] or
    /// the channel closed without a `Done` event.
    pub async fn collect_full(mut self) -> Result<(String, GenerateSummary), String> {
        let mut text = String::new();
        loop {
            match self.rx.recv().await {
                Some(GenerateEvent::Token(tok)) => text.push_str(&tok),
                Some(GenerateEvent::Done(summary)) => return Ok((text, summary)),
                Some(GenerateEvent::Error(e)) => return Err(e),
                None => return Err("stream closed without a Done event".into()),
            }
        }
    }
}
