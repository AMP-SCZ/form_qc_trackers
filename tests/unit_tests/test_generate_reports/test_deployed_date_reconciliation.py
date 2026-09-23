"""Date Report reviewer edits round-trip through deployed V2 workbooks.

All workbooks and Dropbox responses are synthetic and kept in memory.
"""
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import pytest

from generate_reports.calculate_resolved_errors import CalculateResolvedErrors


MAPPING = {
    'subject': 'Participant',
    'displayed_timepoint': 'Timepoint',
    'displayed_form': 'Form',
    'error_message': 'Flags',
    'manually_resolved': 'Manually Resolved',
    'comments': 'Network Comments',
    'site_comments': 'Site Comments',
}


def _flag(network, subject, **changes):
    result = {
        'network': network, 'subject': subject,
        'displayed_timepoint': 'month2', 'displayed_form': 'PSYCHS',
        'error_message': 'Visit date precedes the prior visit.',
        'reports': 'Date Report', 'date_gap_days': 10,
        'manually_resolved': 'saved manual', 'comments': 'saved network',
        'site_comments': 'saved site',
    }
    result.update(changes)
    return result


def _workbook(sheets):
    stream = BytesIO()
    with pd.ExcelWriter(stream, engine='openpyxl') as writer:
        for name, rows in sheets.items():
            frame = pd.DataFrame(rows, columns=list(MAPPING))
            frame.rename(columns=MAPPING).to_excel(writer, sheet_name=name, index=False)
    return stream.getvalue()


def _reader(frame, files, networks, sites, ras=None):
    downloads, written = [], []

    class FakeDropbox:
        def files_download(self, path):
            downloads.append(path)
            return None, SimpleNamespace(content=files[path])

    reader = object.__new__(CalculateResolvedErrors)
    reader.utils = SimpleNamespace(
        pipeline_networks=networks,
        collect_dropbox_credentials=lambda: FakeDropbox(),
        reverse_dictionary=lambda mapping: {value: key for key, value in mapping.items()},
        all_sites=sites,
        site_full_name_translations={'KC': 'Kansas City', 'PI': 'Pittsburgh', 'ME': 'Melbourne'},
    )
    reader.formatted_column_names = {
        network: {'combined': MAPPING, 'sites': MAPPING} for network in networks
    }
    reader.dropbox_path = '/trackers/'
    reader.melbourne_ras = ras or {}
    reader.check_dbx_file_exists = lambda dbx, path: path in files
    reader._read_current = lambda: frame.copy(deep=True)
    reader._write_current = lambda result: written.append(result.copy(deep=True))
    return reader, written, downloads


@pytest.mark.parametrize('network', ['PRONET', 'PRESCIENT'])
@pytest.mark.parametrize('comment,manual', [
    ('date network edit', 'True'), ('', ''), ('[CLEAR]', '[clear]'),
])
def test_date_combined_owns_edits_and_ignores_stale_other_tabs(network, comment, manual):
    other_network = 'PRESCIENT' if network == 'PRONET' else 'PRONET'
    date = _flag(network, 'SHARED001')
    main = _flag(network, 'SHARED001', reports='Main Report', error_message='Other main flag')
    date_edit = dict(date, comments=comment, manually_resolved=manual,
                     site_comments='combined cannot edit site')
    stale = dict(date, comments='stale copy', manually_resolved='stale copy')
    main_edit = dict(main, comments='main network edit', manually_resolved='main manual edit')
    # Include a stale non-Date flag copied onto Date to verify other authority.
    stale_main = dict(main, comments='wrong Date edit', manually_resolved='wrong Date edit')
    path = f'/trackers/{network}/combined/{network}_Output_V2.xlsx'
    files = {path: _workbook({
        'Date Report': [date_edit, stale_main],
        'Main Report': [stale, main_edit],
        'Non Team Forms': [stale],
        'Missingness Report': [stale],
        'Conversion Report': [stale],
        'Blood Report': [stale],
        'Cross Checks': [stale],
        'Proposed Checks': [stale],
        'Medication Flags': [stale],
    })}
    original = pd.DataFrame([date, main, dict(date, network=other_network)])
    original['date_gap_days'] = original['date_gap_days'].astype('Int64')
    reader, written, downloads = _reader(original, files, [network], {network: []})
    reader.loop_dropbox_files()
    assert downloads == [path]
    assert len(written) == 1
    result = written[0]
    date_result = result[(result['network'] == network) & (result['reports'] == 'Date Report')].iloc[0]
    expected_comment = 'saved network' if comment == '' else '' if comment == '[CLEAR]' else comment
    expected_manual = 'saved manual' if manual == '' else '' if manual == '[clear]' else manual
    assert date_result['comments'] == expected_comment
    assert date_result['manually_resolved'] == expected_manual
    assert date_result['site_comments'] == 'saved site'
    assert str(result['date_gap_days'].dtype) == 'Int64'
    main_result = result[result['reports'] == 'Main Report'].iloc[0]
    assert main_result['comments'] == 'main network edit'
    assert main_result['manually_resolved'] == 'main manual edit'
    other_result = result[result['network'] == other_network].iloc[0]
    assert other_result['comments'] == 'saved network'
    assert other_result['manually_resolved'] == 'saved manual'


@pytest.mark.parametrize('network,site,folder,legacy_tab', [
    ('PRONET', 'KC', 'Kansas City', 'Main Report'),
    ('PRESCIENT', 'PI', 'Pittsburgh', 'Main Report'),
    ('PRESCIENT', 'ME', 'Melbourne', 'Non Team Forms'),
])
@pytest.mark.parametrize('site_comment', ['site date edit', '', '[CLEAR]'])
def test_date_site_comments_round_trip_for_ordinary_and_melbourne_sites(
        network, site, folder, legacy_tab, site_comment):
    date = _flag(network, site + '001')
    path = f'/trackers/{network}/{folder}/{network}_{site}_Output_V2.xlsx'
    files = {path: _workbook({
        'Date Report': [dict(date, site_comments=site_comment,
                             comments='site cannot edit network', manually_resolved='True')],
        legacy_tab: [dict(date, site_comments='stale site copy')],
    })}
    reader, written, downloads = _reader(
        pd.DataFrame([date]), files, [network], {network: [site]})
    reader.loop_dropbox_files()
    result = written[0].iloc[0]
    expected = 'saved site' if site_comment == '' else '' if site_comment == '[CLEAR]' else site_comment
    assert result['site_comments'] == expected
    assert result['comments'] == 'saved network'
    assert result['manually_resolved'] == 'saved manual'
    assert downloads == [path]


@pytest.mark.parametrize('ra_comment,expected', [
    ('RA date edit', 'RA date edit'), ('', 'Melbourne fallback'), ('[CLEAR]', ''),
])
def test_melbourne_date_ra_comments_override_site_fallback(ra_comment, expected):
    assigned = _flag('PRESCIENT', 'ME001')
    unassigned = _flag('PRESCIENT', 'ME002')
    site_path = '/trackers/PRESCIENT/Melbourne/PRESCIENT_ME_Output_V2.xlsx'
    ra_path = '/trackers/PRESCIENT/Melbourne/RA_One/PRESCIENT_Melbourne_Output_V2.xlsx'
    files = {
        site_path: _workbook({'Date Report': [
            dict(assigned, site_comments='Melbourne fallback'),
            dict(unassigned, site_comments='unassigned site edit'),
        ]}),
        ra_path: _workbook({
            'Date Report': [dict(assigned, site_comments=ra_comment)],
            'Non Team Forms': [dict(assigned, site_comments='stale RA copy')],
        }),
    }
    reader, written, downloads = _reader(
        pd.DataFrame([assigned, unassigned]), files, ['PRESCIENT'],
        {'PRESCIENT': ['ME']}, {'RA One': ['ME001']})
    reader.loop_dropbox_files()
    result = written[0].set_index('subject')
    assert result.loc['ME001', 'site_comments'] == expected
    assert result.loc['ME002', 'site_comments'] == 'unassigned site edit'
    assert downloads == [site_path, ra_path]


def test_header_only_date_workbook_keeps_reviewer_state():
    date = _flag('PRONET', 'KC001')
    files = {'/trackers/PRONET/combined/PRONET_Output_V2.xlsx':
             _workbook({'Date Report': []})}
    reader, written, _ = _reader(pd.DataFrame([date]), files, ['PRONET'], {'PRONET': []})
    reader.loop_dropbox_files()
    result = written[0].iloc[0]
    assert result['comments'] == 'saved network'
    assert result['manually_resolved'] == 'saved manual'
    assert result['site_comments'] == 'saved site'
