import json
import threading

import pandas as pd
import pytest

from discover_errors.llama3_test import (
    LlamaSeverityScreener,
    RANGE_OUTPUT_COLUMNS,
    RANGE_RELEVANT_COLUMNS,
)


def _response(**overrides):
    payload = {
        'is_quantitative': True,
        'minimum': 0,
        'maximum': 300,
        'unit': 'kg',
        'range_basis': 'broad physical limits',
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_range_response_accepts_valid_inclusion_and_exclusion():
    included, error = LlamaSeverityScreener._parse_range_response(
        _response())
    assert error is None
    assert included == {
        'is_quantitative': True,
        'minimum': 0,
        'maximum': 300,
        'unit': 'kg',
        'range_basis': 'broad physical limits',
    }

    excluded, error = LlamaSeverityScreener._parse_range_response(_response(
        is_quantitative=False,
        minimum=0,
        maximum=0,
        unit='',
        range_basis='categorical response codes',
    ))
    assert error is None
    assert excluded['is_quantitative'] is False


@pytest.mark.parametrize('response', [
    'not json',
    _response(minimum=300, maximum=0),
    _response(minimum=10, maximum=10),
    _response(minimum=True),
    _response(is_quantitative=False, minimum=False, maximum=0, unit='',
              range_basis='categorical response codes'),
    _response(minimum=float('nan')),
    _response(is_quantitative=False, maximum=1, unit='',
              range_basis='not numeric'),
    _response(range_basis='none'),
])
def test_range_response_rejects_invalid_or_inconsistent_results(response):
    result, error = LlamaSeverityScreener._parse_range_response(response)
    assert result is None
    assert error


def test_invalid_model_response_is_retried_with_correction():
    screener = object.__new__(LlamaSeverityScreener)
    calls = []
    responses = iter([
        _response(minimum=5, maximum=5),
        _response(minimum=0, maximum=10, unit='points',
                  range_basis='explicit scale definition'),
    ])

    def query(payload):
        calls.append(payload)
        return next(responses)

    screener._query_llm = query
    result, error = screener._classify_numeric_range(
        '{"Variable / Field Name":"score"}')

    assert error is None
    assert result['maximum'] == 10
    assert len(calls) == 2
    assert 'previous response was invalid' in calls[1]['prompt']


def test_batch_response_maps_out_of_order_ids_and_validates_each_result():
    response = json.dumps({'results': [
        {
            'id': 20, 'q': False, 'lo': 0, 'hi': 0, 'u': '',
            'b': 'categorical',
        },
        {
            'id': 10, 'q': True, 'lo': 0, 'hi': 300, 'u': 'kg',
            'b': 'physical_limits',
        },
    ]})
    results, errors = LlamaSeverityScreener._parse_range_batch_response(
        response, [10, 20])

    assert errors == {}
    assert results[10]['maximum'] == 300
    assert results[10]['range_basis'] == 'broad physical limits'
    assert results[20]['is_quantitative'] is False


def test_batch_response_salvages_valid_ids_and_rejects_extra_ids():
    partial = json.dumps({'results': [{
        'id': 10, 'q': True, 'lo': 0, 'hi': 10, 'u': 'points',
        'b': 'defined_scale',
    }]})
    results, errors = LlamaSeverityScreener._parse_range_batch_response(
        partial, [10, 20])
    assert set(results) == {10}
    assert set(errors) == {20}

    extra = json.dumps({'results': [{
        'id': 999, 'q': False, 'lo': 0, 'hi': 0, 'u': '',
        'b': 'other',
    }]})
    results, errors = LlamaSeverityScreener._parse_range_batch_response(
        extra, [10])
    assert results == {}
    assert set(errors) == {10}

    duplicate = json.dumps({'results': [
        {'id': 10, 'q': False, 'lo': 0, 'hi': 0, 'u': '', 'b': 'other'},
        {'id': 10, 'q': False, 'lo': 0, 'hi': 0, 'u': '', 'b': 'other'},
    ]})
    results, errors = LlamaSeverityScreener._parse_range_batch_response(
        duplicate, [10])
    assert results == {}
    assert 'duplicated' in errors[10]


def test_batch_classifier_retries_only_missing_ids():
    screener = object.__new__(LlamaSeverityScreener)
    calls = []
    responses = iter([
        json.dumps({'results': [{
            'id': 10, 'q': True, 'lo': 0, 'hi': 300, 'u': 'kg',
            'b': 'physical_limits',
        }]}),
        json.dumps({'results': [{
            'id': 20, 'q': False, 'lo': 0, 'hi': 0, 'u': '',
            'b': 'categorical',
        }]}),
    ])

    def query(payload):
        calls.append(payload)
        return next(responses)

    screener._query_llm = query
    results, errors, stats = screener._classify_numeric_range_batch([
        (10, '{"Variable / Field Name":"weight"}', 'key-10'),
        (20, '{"Variable / Field Name":"category"}', 'key-20'),
    ])

    assert errors == {}
    assert set(results) == {10, 20}
    assert len(calls) == 2
    assert stats['batch_native'] == 2
    assert stats['single_fallback'] == 0
    assert '"id":20' in calls[1]['prompt']
    assert '"id":10' not in calls[1]['prompt']


@pytest.mark.parametrize(('row', 'reason_fragment'), [
    ({'Variable / Field Name': 'choice', 'Field Type': 'radio'},
     'categorical'),
    ({'Variable / Field Name': 'note', 'Field Type': 'notes'},
     'text/display'),
    ({'Variable / Field Name': 'dob', 'Field Type': 'text',
      'Text Validation Type OR Show Slider Number': 'date_ymd'},
     'validation'),
    ({'Variable / Field Name': 'initials', 'Field Type': 'text',
      'Text Validation Type OR Show Slider Number': 'alpha_only'},
     'validation'),
    ({'Variable / Field Name': 'study_id', 'Field Type': 'text',
      'Identifier?': 'y'}, 'identifier'),
])
def test_deterministic_range_prefilter_only_handles_safe_exclusions(
        row, reason_fragment):
    result = LlamaSeverityScreener._deterministic_range_exclusion(row)
    assert result['is_quantitative'] is False
    assert reason_fragment in result['range_basis']


def test_deterministic_range_prefilter_keeps_numeric_candidates():
    for row in [
        {'Variable / Field Name': 'unvalidated_measure',
         'Field Type': 'text'},
        {'Variable / Field Name': 'calculated_total',
         'Field Type': 'calc'},
        {'Variable / Field Name': 'weight', 'Field Type': 'text',
         'Text Validation Type OR Show Slider Number': 'number'},
    ]:
        assert LlamaSeverityScreener._deterministic_range_exclusion(
            row) is None


def test_numeric_range_mode_writes_only_qualifying_rows_and_caches_negatives(
        tmp_path):
    screener = object.__new__(LlamaSeverityScreener)
    screener.data_dict = pd.DataFrame([
        {
            'Variable / Field Name': 'weight_kg',
            'Form Name': 'health',
            'Field Type': 'text',
            'Field Label': 'Weight in kilograms',
            'Text Validation Type OR Show Slider Number': 'number',
        },
        {
            'Variable / Field Name': 'sex_code',
            'Form Name': 'demographics',
            'Field Type': 'text',
            'Field Label': 'Sex code',
            'Choices, Calculations, OR Slider Labels': '1, A | 2, B',
        },
        {
            'Variable / Field Name': 'failed_field',
            'Form Name': 'health',
            'Field Type': 'text',
            'Field Label': 'Unresolved model call',
        },
    ])
    screener.output_path = str(tmp_path)
    screener._cache_salt = '00' * 32
    screener._cache_lock = threading.Lock()

    def classify(row_json):
        variable = json.loads(row_json)['Variable / Field Name']
        if variable == 'weight_kg':
            return ({
                'is_quantitative': True,
                'minimum': 0,
                'maximum': 300,
                'unit': 'kg',
                'range_basis': 'broad physical limits',
            }, None)
        if variable == 'sex_code':
            return ({
                'is_quantitative': False,
                'minimum': 0,
                'maximum': 0,
                'unit': '',
                'range_basis': 'categorical response codes',
            }, None)
        return None, 'llm_failure'

    screener._classify_numeric_range = classify
    screener.find_numeric_data_dictionary_ranges(batch_size=1)

    output = pd.read_csv(
        tmp_path / 'data_dictionary_numeric_ranges.csv')
    assert list(output.columns) == RANGE_OUTPUT_COLUMNS
    assert output['variable'].tolist() == ['weight_kg']
    assert output.loc[0, 'minimum'] == 0
    assert output.loc[0, 'maximum'] == 300
    assert output.loc[0, 'range'] == '[0, 300] kg'

    unresolved = pd.read_csv(
        tmp_path / 'data_dictionary_numeric_ranges_unresolved.csv')
    assert unresolved.to_dict('records') == [{
        'variable': 'failed_field', 'reason': 'llm_failure'}]

    audit = pd.read_csv(
        tmp_path / 'data_dictionary_numeric_ranges_audit.csv')
    assert audit[['variable', 'status']].to_dict('records') == [
        {'variable': 'weight_kg', 'status': 'included'},
        {'variable': 'sex_code', 'status': 'excluded'},
        {'variable': 'failed_field', 'status': 'unresolved'},
    ]

    cache = json.loads(
        (tmp_path / 'data_dictionary_range_cache.json').read_text())
    assert len(cache) == 2

    # A resumed run must reuse both the positive and negative decisions.
    resumed = object.__new__(LlamaSeverityScreener)
    resumed.data_dict = screener.data_dict.iloc[:2].copy()
    resumed.output_path = str(tmp_path)
    resumed._cache_salt = '00' * 32
    resumed._cache_lock = threading.Lock()

    def must_not_query(_row_json):
        raise AssertionError('cached dictionary rows were queried again')

    resumed._classify_numeric_range = must_not_query
    resumed.find_numeric_data_dictionary_ranges()
    resumed_output = pd.read_csv(
        tmp_path / 'data_dictionary_numeric_ranges.csv')
    assert resumed_output['variable'].tolist() == ['weight_kg']


def test_numeric_range_mode_batches_candidates_instead_of_one_call_per_row(
        tmp_path):
    screener = object.__new__(LlamaSeverityScreener)
    screener.data_dict = pd.DataFrame([{
        'Variable / Field Name': f'numeric_{index}',
        'Field Type': 'text',
        'Text Validation Type OR Show Slider Number': 'number',
    } for index in range(25)])
    screener.output_path = str(tmp_path)
    screener._cache_salt = '00' * 32
    screener._cache_lock = threading.Lock()
    batch_lengths = []
    lock = threading.Lock()

    def classify_batch(batch):
        with lock:
            batch_lengths.append(len(batch))
        return ({
            index: {
                'is_quantitative': False,
                'minimum': 0,
                'maximum': 0,
                'unit': '',
                'range_basis': 'no defensible finite upper bound',
            }
            for index, _, _ in batch
        }, {}, {
            'batch_attempts': 1,
            'batch_native': len(batch),
            'single_fallback': 0,
        })

    screener._classify_numeric_range_batch = classify_batch
    screener.find_numeric_data_dictionary_ranges(
        workers=2, batch_size=12, prefilter=False)

    assert sorted(batch_lengths) == [1, 12, 12]
    audit = pd.read_csv(
        tmp_path / 'data_dictionary_numeric_ranges_audit.csv')
    assert len(audit) == 25
    assert set(audit['status']) == {'excluded'}


def test_numeric_range_mode_disables_bad_batching_after_preflight(tmp_path):
    screener = object.__new__(LlamaSeverityScreener)
    screener.data_dict = pd.DataFrame([{
        'Variable / Field Name': f'numeric_{index}',
        'Field Type': 'text',
    } for index in range(14)])
    screener.output_path = str(tmp_path)
    screener._cache_salt = '00' * 32
    screener._cache_lock = threading.Lock()
    batch_lengths = []

    def classify_batch(batch):
        batch_lengths.append(len(batch))
        results = {
            index: {
                'is_quantitative': False,
                'minimum': 0,
                'maximum': 0,
                'unit': '',
                'range_basis': 'no defensible finite bounds',
            }
            for index, _, _ in batch
        }
        return results, {}, {
            'batch_attempts': 2 if len(batch) > 1 else 0,
            'batch_native': 0,
            'single_fallback': len(batch),
        }

    screener._classify_numeric_range_batch = classify_batch
    screener.find_numeric_data_dictionary_ranges(
        workers=1, batch_size=12, prefilter=False)

    assert batch_lengths == [12, 1, 1]


def test_range_batch_packing_honors_count_and_character_budgets():
    items = [
        (index, '{"Variable / Field Name":"field_' + str(index) + '"}',
         f'key-{index}')
        for index in range(25)
    ]
    batches = LlamaSeverityScreener._pack_range_batches(
        items, batch_size=12, max_prompt_chars=12000)
    assert [len(batch) for batch in batches] == [12, 12, 1]

    large_items = [
        (0, 'x' * 800, 'key-0'),
        (1, 'y' * 800, 'key-1'),
    ]
    batches = LlamaSeverityScreener._pack_range_batches(
        large_items, batch_size=12, max_prompt_chars=1000)
    assert [len(batch) for batch in batches] == [1, 1]

    with pytest.raises(ValueError, match='must not exceed'):
        LlamaSeverityScreener._pack_range_batches(
            items, batch_size=33, max_prompt_chars=12000)


def test_compact_row_is_bounded_and_keeps_variable_identity():
    row = {column: column[0] * 50000
           for column in RANGE_RELEVANT_COLUMNS}
    row['Variable / Field Name'] = 'long_field'
    row_json = LlamaSeverityScreener._compact_range_row(row)
    payload = json.loads(row_json)
    assert payload['Variable / Field Name'] == 'long_field'
    assert len(row_json) <= 16000


def test_compact_row_canonicalizes_short_and_whitespace_header_aliases():
    row_json = LlamaSeverityScreener._compact_range_row({
        'field_name': 'weight_kg',
        'form_name': 'vitals',
        'field_label': 'Weight',
        ' Field Type ': 'text',
        'choices': '0, none | 100, maximum',
        'validation_type': 'number',
        'validation_min': '0',
        'validation_max': '100',
    })
    assert json.loads(row_json) == {
        'Variable / Field Name': 'weight_kg',
        'Form Name': 'vitals',
        'Field Label': 'Weight',
        'Field Type': 'text',
        'Choices, Calculations, OR Slider Labels': (
            '0, none | 100, maximum'),
        'Text Validation Type OR Show Slider Number': 'number',
        'Text Validation Min': '0',
        'Text Validation Max': '100',
    }


def test_numeric_range_mode_accepts_alias_dictionary_headers(tmp_path):
    screener = object.__new__(LlamaSeverityScreener)
    screener.data_dict = pd.DataFrame([{
        'field_name': 'temperature_c',
        'form_name': 'vitals',
        'field_label': 'Body temperature',
    }])
    screener.output_path = str(tmp_path)
    screener._cache_salt = '00' * 32
    screener._cache_lock = threading.Lock()
    screener._classify_numeric_range = lambda _row_json: ({
        'is_quantitative': True,
        'minimum': 0,
        'maximum': 100,
        'unit': 'degrees C',
        'range_basis': 'broad physical limits',
    }, None)

    screener.find_numeric_data_dictionary_ranges(workers=1, batch_size=1)

    output = pd.read_csv(
        tmp_path / 'data_dictionary_numeric_ranges.csv')
    assert output.loc[0, 'variable'] == 'temperature_c'
    assert output.loc[0, 'form_name'] == 'vitals'
    assert output.loc[0, 'field_label'] == 'Body temperature'


def test_numeric_range_mode_rejects_duplicate_variable_names_before_queries(
        tmp_path):
    screener = object.__new__(LlamaSeverityScreener)
    screener.data_dict = pd.DataFrame([
        {'Variable / Field Name': 'weight_kg'},
        {'Variable / Field Name': 'WEIGHT_KG'},
    ])
    screener.output_path = str(tmp_path)

    def must_not_query(_row_json):
        raise AssertionError('duplicate rows reached the model')

    screener._classify_numeric_range = must_not_query
    with pytest.raises(ValueError, match='duplicate variable/field names'):
        screener.find_numeric_data_dictionary_ranges()


def test_range_cache_identity_uses_content_hidden_by_prompt_truncation():
    middle_one = 'a' * 2000 + 'X' + 'a' * 2000
    middle_two = 'a' * 2000 + 'Y' + 'a' * 2000
    row_one = {
        'Variable / Field Name': 'long_calc',
        'Choices, Calculations, OR Slider Labels': middle_one,
    }
    row_two = {
        'Variable / Field Name': 'long_calc',
        'Choices, Calculations, OR Slider Labels': middle_two,
    }
    assert (LlamaSeverityScreener._compact_range_row(row_one)
            == LlamaSeverityScreener._compact_range_row(row_two))

    screener = object.__new__(LlamaSeverityScreener)
    screener._cache_salt = '00' * 32
    full_one = screener._range_row_json(row_one, truncate=False)
    full_two = screener._range_row_json(row_two, truncate=False)
    assert screener._range_cache_key(full_one) != screener._range_cache_key(
        full_two)
