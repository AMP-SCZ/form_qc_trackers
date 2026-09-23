"""Bounded transport retries for read-only Dropbox tracker operations."""

import time

from requests.exceptions import (
    ChunkedEncodingError, ConnectionError, SSLError, Timeout,
)


_MAX_ATTEMPTS = 5


def _read_with_retry(operation, path):
    # The Dropbox SDK retries API throttling/5xx responses, but does not handle
    # requests transport failures such as RemoteDisconnected. Retry only reads;
    # an exhausted connection must never be interpreted as a missing workbook.
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return operation()
        except SSLError:
            # A certificate/configuration failure needs operator attention.
            raise
        except (ConnectionError, Timeout, ChunkedEncodingError) as exc:
            if attempt == _MAX_ATTEMPTS:
                raise RuntimeError(
                    f"FATAL: Dropbox read failed for {path!r} after "
                    f"{_MAX_ATTEMPTS} attempts. Reviewer reconciliation is "
                    "incomplete; report generation/upload was not started. "
                    "Restore connectivity and retry reconciliation."
                ) from exc
            delay = 2 ** (attempt - 1)
            print(
                f"[dropbox_reads] {type(exc).__name__} reading {path!r} "
                f"(attempt {attempt}/{_MAX_ATTEMPTS}); retrying in {delay}s.",
                flush=True,
            )
            time.sleep(delay)


def get_metadata(dbx, path):
    """Read metadata; preserve genuine not-found and other Dropbox API errors."""
    return _read_with_retry(lambda: dbx.files_get_metadata(path), path)


def download_workbook(dbx, path):
    """Return metadata and complete bytes, retrying interrupted response bodies."""
    def download():
        metadata, response = dbx.files_download(path)
        try:
            # Downloads are streamed: reading content can fail after the SDK
            # call returns. Keep it inside the retry, before workbook parsing.
            return metadata, response.content
        finally:
            close = getattr(response, 'close', None)
            if close is not None:
                close()

    return _read_with_retry(download, path)
