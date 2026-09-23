"""Dropbox transport recovery is bounded and never discards reviewer state.

These tests use synthetic workbooks and clients; no credentials or network calls.
"""
from http.client import RemoteDisconnected
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import pytest
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import GetMetadataError, LookupError
from requests.exceptions import ChunkedEncodingError, ConnectionError, SSLError, Timeout
from urllib3.exceptions import ProtocolError

from generate_reports import dropbox_reads
from generate_reports.calculate_resolved_errors import CalculateResolvedErrors


PATH = '/synthetic/tracker.xlsx'
MAPPING = {
    'subject': 'Participant', 'displayed_timepoint': 'Timepoint',
    'displayed_form': 'Form', 'error_message': 'Flags',
    'manually_resolved': 'Manually Resolved', 'comments': 'Network Comments',
    'site_comments': 'Site Comments',
}


def _disconnected():
    return ConnectionError(ProtocolError(
        'Connection aborted.',
        RemoteDisconnected('Remote end closed connection without response')))


def _not_found():
    return ApiError('synthetic-request',
                    GetMetadataError.path(LookupError.not_found), None, None)


class _Response:
    def __init__(self, content):
        self._content = content
        self.closed = False

    @property
    def content(self):
        if isinstance(self._content, Exception):
            raise self._content
        return self._content

    def close(self):
        self.closed = True


class _Dropbox:
    def __init__(self, metadata=(), downloads=()):
        self.metadata = iter(metadata)
        self.downloads = iter(downloads)
        self.metadata_paths = []
        self.download_paths = []

    @staticmethod
    def _next(outcomes):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    def files_get_metadata(self, path):
        self.metadata_paths.append(path)
        return self._next(self.metadata)

    def files_download(self, path):
        self.download_paths.append(path)
        return self._next(self.downloads)


@pytest.fixture
def delays(monkeypatch):
    result = []
    monkeypatch.setattr(dropbox_reads.time, 'sleep', result.append)
    return result


@pytest.mark.parametrize('error_factory', [
    _disconnected, lambda: Timeout('timeout'),
    lambda: ChunkedEncodingError('incomplete response'),
])
def test_metadata_recovers_transient_transport_failure(error_factory, delays):
    metadata = SimpleNamespace(rev='metadata-revision')
    client = _Dropbox(metadata=[error_factory(), metadata])
    assert dropbox_reads.get_metadata(client, PATH) is metadata
    assert client.metadata_paths == [PATH, PATH]
    assert delays == [1]


def test_metadata_stops_after_five_attempts_and_chains_last_failure(delays):
    failures = [_disconnected() for _ in range(5)]
    client = _Dropbox(metadata=failures)
    with pytest.raises(RuntimeError) as error:
        dropbox_reads.get_metadata(client, PATH)
    assert error.value.__cause__ is failures[-1]
    assert PATH in str(error.value)
    assert '5 attempts' in str(error.value)
    assert client.metadata_paths == [PATH] * 5
    assert delays == [1, 2, 4, 8]


@pytest.mark.parametrize('error_factory', [
    _disconnected, lambda: Timeout('timeout'),
    lambda: ChunkedEncodingError('incomplete response'),
])
def test_download_recovers_request_failure_and_preserves_revision(error_factory, delays):
    metadata = SimpleNamespace(rev='complete-revision')
    response = _Response(b'complete bytes')
    client = _Dropbox(downloads=[error_factory(), (metadata, response)])
    actual_metadata, content = dropbox_reads.download_workbook(client, PATH)
    assert actual_metadata is metadata
    assert content == b'complete bytes'
    assert response.closed
    assert client.download_paths == [PATH, PATH]
    assert delays == [1]


def test_download_retries_interrupted_body_and_closes_both_responses(delays):
    broken = _Response(ChunkedEncodingError('body interrupted'))
    complete = _Response(b'complete workbook')
    old_metadata = SimpleNamespace(rev='incomplete-revision')
    new_metadata = SimpleNamespace(rev='complete-revision')
    client = _Dropbox(downloads=[(old_metadata, broken), (new_metadata, complete)])
    assert dropbox_reads.download_workbook(client, PATH) == (
        new_metadata, b'complete workbook')
    assert broken.closed and complete.closed
    assert client.download_paths == [PATH, PATH]
    assert delays == [1]


def test_download_exhausted_bodies_are_closed_and_error_propagates(delays):
    failures = [ChunkedEncodingError('body interrupted') for _ in range(5)]
    responses = [_Response(failure) for failure in failures]
    client = _Dropbox(downloads=[(None, response) for response in responses])
    with pytest.raises(RuntimeError) as error:
        dropbox_reads.download_workbook(client, PATH)
    assert error.value.__cause__ is failures[-1]
    assert all(response.closed for response in responses)
    assert client.download_paths == [PATH] * 5
    assert delays == [1, 2, 4, 8]


def test_download_accepts_response_without_close_method(delays):
    metadata = SimpleNamespace(rev='revision')
    client = _Dropbox(downloads=[(metadata, SimpleNamespace(content=b'data'))])
    assert dropbox_reads.download_workbook(client, PATH) == (metadata, b'data')
    assert delays == []


@pytest.mark.parametrize('method', ['get_metadata', 'download_workbook'])
@pytest.mark.parametrize('error_factory', [
    lambda: SSLError('certificate verification failed'),
    lambda: AuthError('synthetic-request', None),
    _not_found,
    lambda: ValueError('unexpected client error'),
])
def test_non_transient_errors_propagate_without_retry(method, error_factory, delays):
    failure = error_factory()
    client = _Dropbox(metadata=[failure], downloads=[failure])
    with pytest.raises(type(failure)) as error:
        getattr(dropbox_reads, method)(client, PATH)
    assert error.value is failure
    calls = client.metadata_paths if method == 'get_metadata' else client.download_paths
    assert calls == [PATH]
    assert delays == []


def _flag(**changes):
    result = {
        'network': 'PRESCIENT', 'subject': 'PI001',
        'displayed_timepoint': 'month2', 'displayed_form': 'Synthetic Form',
        'error_message': 'Synthetic flag', 'reports': 'Main Report',
        'manually_resolved': 'saved manual', 'comments': 'saved network',
        'site_comments': 'saved site',
    }
    result.update(changes)
    return result


def _workbook(**changes):
    stream = BytesIO()
    with pd.ExcelWriter(stream, engine='openpyxl') as writer:
        pd.DataFrame([_flag(**changes)]).rename(columns=MAPPING).to_excel(
            writer, sheet_name='Main Report', index=False)
    return stream.getvalue()


def _reader(client, frame=None):
    reader = object.__new__(CalculateResolvedErrors)
    reader.utils = SimpleNamespace(
        pipeline_networks=['PRESCIENT'],
        collect_dropbox_credentials=lambda: client,
        reverse_dictionary=lambda mapping: {value: key for key, value in mapping.items()},
        all_sites={'PRESCIENT': ['PI']},
        site_full_name_translations={'PI': 'Pittsburgh'},
    )
    reader.formatted_column_names = {
        'PRESCIENT': {'combined': MAPPING, 'sites': MAPPING}}
    reader.dropbox_path = '/trackers/'
    reader.melbourne_ras = {}
    reader._corrupt_workbooks = []
    original = pd.DataFrame([_flag()]) if frame is None else frame
    reader._read_current = lambda: original.copy(deep=True)
    writes = []
    reader._write_current = lambda result: writes.append(result.copy(deep=True))
    return reader, original, writes


def test_existence_probe_only_treats_genuine_not_found_as_missing(delays):
    client = _Dropbox(metadata=[_not_found()])
    reader, _, _ = _reader(client)
    assert reader.check_dbx_file_exists(client, PATH) is False
    assert client.metadata_paths == [PATH]
    assert delays == []


def test_existence_probe_recovers_remote_disconnect(delays):
    client = _Dropbox(metadata=[_disconnected(), SimpleNamespace()])
    reader, _, _ = _reader(client)
    assert reader.check_dbx_file_exists(client, PATH) is True
    assert client.metadata_paths == [PATH, PATH]
    assert delays == [1]


def test_missing_workbook_preserves_saved_state_without_download(delays):
    client = _Dropbox(metadata=[_not_found()])
    reader, original, writes = _reader(client)
    result = reader.read_dropbox_data(
        original, MAPPING, ['comments'], PATH, client, 'PRESCIENT', ['Main Report'])
    pd.testing.assert_frame_equal(result, original)
    assert client.download_paths == []
    assert writes == []
    assert reader._corrupt_workbooks == []


def test_reader_recovers_download_and_imports_reviewer_comment(delays):
    response = _Response(_workbook(comments='reviewer edit'))
    client = _Dropbox(metadata=[SimpleNamespace()], downloads=[
        _disconnected(), (SimpleNamespace(rev='revision'), response)])
    reader, original, writes = _reader(client)
    result = reader.read_dropbox_data(
        original, MAPPING, ['comments'], PATH, client, 'PRESCIENT', ['Main Report'])
    assert result.iloc[0]['comments'] == 'reviewer edit'
    assert result.iloc[0]['site_comments'] == 'saved site'
    assert result.iloc[0]['manually_resolved'] == 'saved manual'
    assert writes == []
    assert reader._corrupt_workbooks == []
    assert response.closed
    assert client.download_paths == [PATH, PATH]
    assert delays == [1]


def test_exhausted_download_is_not_quarantined_as_corrupt_workbook(delays):
    failure = _disconnected()
    client = _Dropbox(metadata=[SimpleNamespace()], downloads=[failure] * 5)
    reader, original, writes = _reader(client)
    before = original.copy(deep=True)
    with pytest.raises(RuntimeError) as error:
        reader.read_dropbox_data(
            original, MAPPING, ['comments'], PATH, client, 'PRESCIENT', ['Main Report'])
    assert error.value.__cause__ is failure
    assert reader._corrupt_workbooks == []
    pd.testing.assert_frame_equal(original, before)
    assert writes == []
    assert delays == [1, 2, 4, 8]


@pytest.mark.parametrize('failing_operation', ['metadata', 'download'])
def test_later_site_failure_prevents_partial_reconciliation_write(failing_operation, delays):
    combined_path = '/trackers/PRESCIENT/combined/PRESCIENT_Output_V2.xlsx'
    site_path = '/trackers/PRESCIENT/Pittsburgh/PRESCIENT_PI_Output_V2.xlsx'
    failure = _disconnected()
    response = _Response(_workbook(comments='combined reviewer edit'))
    metadata = [SimpleNamespace()]
    downloads = [(SimpleNamespace(rev='combined-revision'), response)]
    if failing_operation == 'metadata':
        metadata += [failure] * 5
    else:
        metadata.append(SimpleNamespace())
        downloads += [failure] * 5
    client = _Dropbox(metadata=metadata, downloads=downloads)
    reader, original, writes = _reader(client)
    before = original.copy(deep=True)
    with pytest.raises(RuntimeError) as error:
        reader.loop_dropbox_files()
    assert error.value.__cause__ is failure
    assert writes == []
    pd.testing.assert_frame_equal(original, before)
    assert reader._corrupt_workbooks == []
    assert response.closed
    assert client.download_paths[0] == combined_path
    if failing_operation == 'metadata':
        assert client.metadata_paths == [combined_path] + [site_path] * 5
        assert client.download_paths == [combined_path]
    else:
        assert client.metadata_paths == [combined_path, site_path]
        assert client.download_paths == [combined_path] + [site_path] * 5
    assert delays == [1, 2, 4, 8]
