from datetime import datetime
import re

"""
Pure helper for the "date" qc_type (qc_types/date_checks.py).

Standard-library-only on purpose: the cross-timepoint comparison
lives here so it can be unit-tested with plain dicts, WITHOUT
constructing a FormCheck (whose __init__ builds Utils(), which breaks on
Windows). DateChecks just wraps the result in create_row_output.
"""

# Timepoints with no fixed position in the visit calendar. Ordering them
# against numbered visits would manufacture spurious "backward" steps, so
# they never participate as the candidate tp nor as a prior tp. Mirrors
# the exclusion used elsewhere in the date tooling.
NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})

# Stable interview-date fields for forms that must never act as either side of
# a Date Report comparison.  Keeping the variables as a second line of defense
# is important because historical rows and imported dependencies do not always
# retain the same form-name spelling.
DATE_REPORT_EXCLUDED_VARIABLES = frozenset({
    'chrmiss_interview_date',
    'chrpharm_interview_date',
    'chrpsychs_av_interview_date',
    'chrif_interview_date',
    'chrgpc_date',
    'chrcbc_interview_date',
    'chrcbccs_review_date',
})

# Forms whose dates are administrative/non-target for the longitudinal Date
# Report.  They must never be the current flagged form or the earlier
# reference form. Keep this canonical set for callers that need REDCap form
# keys, and use the predicate below whenever a value can be user-facing text.
DATE_REPORT_EXCLUDED_FORMS = frozenset({
    'past_pharmaceutical_treatment',
    'psychs_av_recording_run_sheet',
    'mri_incidental_findings_run_sheet',
    'gcp_current_health_status',
    'cbc_with_differential',
    'gcp_cbc_with_differential',
})

# Labels observed in formatted trackers and supplied for exclusion.  Exact
# normalized aliases are deliberate: fuzzy matching could wrongly suppress the
# distinct, allowed ``current_health_status`` form.
DATE_REPORT_EXCLUDED_FORM_ALIASES = frozenset({
    *DATE_REPORT_EXCLUDED_FORMS,
    'Past pharmaceutical',
    'PSYCHS AV recording run sheet',
    'MRI-Incidental',
    'CurrentHealthStatus_GCP',
    'CBC with differential',
    'GCP-CBC with differential',
})


def normalize_date_report_form(value):
    """Return a case/separator-insensitive exact form-name key."""
    return re.sub(r'[^a-z0-9]+', '', str(value).strip().casefold())


_DATE_REPORT_EXCLUDED_FORM_KEYS = frozenset(
    normalize_date_report_form(value)
    for value in DATE_REPORT_EXCLUDED_FORM_ALIASES)
_DATE_REPORT_EXCLUDED_VARIABLE_KEYS = frozenset(
    str(value).strip().casefold()
    for value in DATE_REPORT_EXCLUDED_VARIABLES)


def is_date_report_excluded_form(value):
    """Whether ``value`` is an exact canonical/display exclusion alias."""
    return normalize_date_report_form(value) in (
        _DATE_REPORT_EXCLUDED_FORM_KEYS)


def is_date_report_excluded_variable(value):
    """Whether ``value`` is a stable excluded interview-date field."""
    return str(value).strip().casefold() in (
        _DATE_REPORT_EXCLUDED_VARIABLE_KEYS)


def _matches_redcap_code(value, codes):
    """Return whether ``value`` is one of the numeric REDCap codes."""
    try:
        return float(value) in {float(code) for code in codes}
    except (TypeError, ValueError):
        return False


def is_form_marked_missing(row, form_info, network=''):
    """Whether a form's explicit missing marker is selected on ``row``.

    Date Report filtering is deliberately narrower than the pipeline's
    general ``check_if_missing`` helper.  That helper can infer an unmarked
    form is effectively missing when it has no missing button or predates the
    button.  Here the requirement is specifically to ignore a populated date
    only when the form was *marked* missing.

    PRESCIENT also represents missing forms with completion status 3/4.  The
    completion field is optional here because an unscheduled form's date and
    missing button can be present without its ``*_complete_rpms`` column.
    """
    if not isinstance(form_info, dict):
        return False
    missing_var = form_info.get('missing_var', '')
    if not missing_var or not hasattr(row, missing_var):
        return False
    if _matches_redcap_code(getattr(row, missing_var), (1,)):
        return True

    if str(network).strip().upper() != 'PRESCIENT':
        return False
    completion_var = form_info.get('completion_var', '')
    if not completion_var:
        return False
    completion_var = (completion_var + '_rpms').replace('_hc', '')
    completion_var = completion_var.replace('onboarding', 'checkin')
    completion_var = completion_var.replace(
        'end_of_12month_study_pe', 'checkin')
    completion_var = completion_var.replace(
        'end_of_12month_study_p', 'checkin')
    return (hasattr(row, completion_var)
            and _matches_redcap_code(
                getattr(row, completion_var), (3, 4)))


def _parse_date_value(raw, missing_set):
    """Parse an interview date, accepting ISO dates with a time suffix."""
    raw_str = str(raw).strip()
    date_str = raw_str.split(' ')[0].split('T')[0]
    if raw_str in missing_set or date_str in missing_set:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d'), date_str
    except ValueError:
        return None


def _visit_date_observations(tp_info, missing_set, network=None,
                             extrema=('earliest', 'latest')):
    """
    Return every usable interview-date observation at one timepoint.

    Generated dependencies provide a compact network-specific
    ``variable_dates`` mapping. A ``dates`` list is also accepted for
    current-row observations and pure tests. The legacy earliest/latest
    fallback keeps older files partly usable when explicit variable metadata
    is present; unknown variables are never guessed.
    """
    if not isinstance(tp_info, dict):
        return []

    network_extrema = tp_info.get('network_extrema')
    if network and isinstance(network_extrema, dict):
        # Once network-specific extrema exist, never fall back to the merged
        # legacy range for a network with no observations at this timepoint.
        tp_info = network_extrema.get(network)
        if not isinstance(tp_info, dict):
            return []

    observations = []
    variable_dates = tp_info.get('variable_dates')
    if isinstance(variable_dates, dict) and variable_dates:
        for variable, item in variable_dates.items():
            if not isinstance(item, dict) or not isinstance(variable, str):
                continue
            form = item.get('form', '')
            if not isinstance(form, str):
                form = ''
            if is_date_report_excluded_form(form):
                continue
            if is_date_report_excluded_variable(variable):
                continue
            parsed = _parse_date_value(item.get('date', ''), missing_set)
            if parsed is None:
                continue
            dt, date_str = parsed
            observations.append({
                'dt': dt,
                'date': date_str,
                'form': form,
                'variable': variable,
                'network': '',
            })
    elif isinstance(tp_info.get('dates'), list):
        for item in tp_info['dates']:
            if not isinstance(item, dict):
                continue
            form = item.get('form', '')
            variable = item.get('variable', '')
            if not isinstance(form, str):
                form = ''
            if not isinstance(variable, str):
                variable = ''
            if is_date_report_excluded_form(form):
                continue
            if is_date_report_excluded_variable(variable):
                continue
            item_network = str(item.get('network', ''))
            if network and item_network and item_network != network:
                continue
            parsed = _parse_date_value(item.get('date', ''), missing_set)
            if parsed is None:
                continue
            dt, date_str = parsed
            observations.append({
                'dt': dt,
                'date': date_str,
                'form': form,
                'variable': variable,
                'network': item_network,
            })
    else:
        # Compact dependency path. Include both extrema when distinct; the
        # current row supplies every form-level date separately.
        for prefix in extrema:
            if is_date_report_excluded_form(
                    tp_info.get(f'{prefix}_form', '')):
                continue
            if is_date_report_excluded_variable(
                    tp_info.get(f'{prefix}_variable', '')):
                continue
            parsed = _parse_date_value(tp_info.get(prefix, ''), missing_set)
            if parsed is None:
                continue
            dt, date_str = parsed
            form = tp_info.get(f'{prefix}_form', '')
            variable = tp_info.get(f'{prefix}_variable', '')
            if not isinstance(form, str):
                form = ''
            if not isinstance(variable, str):
                variable = ''
            observations.append({
                'dt': dt,
                'date': date_str,
                'form': form,
                'variable': variable,
                'network': '',
            })

    # Exact duplicate source rows must not multiply Date Report flags.
    unique = {}
    for observation in observations:
        key = (
            observation['date'], observation['form'],
            observation['variable'], observation['network'])
        unique[key] = observation
    return sorted(
        unique.values(),
        key=lambda item: (
            item['dt'], item['form'], item['variable'], item['network']))


def find_backward_visit_dates(subject_tp_dates, tp_order, current_tp,
                              missing_values=('',), network=None,
                              current_observations=None):
    """
    Find dates at ``current_tp`` that precede the same field at an earlier tp.

    Each current observation is compared only with observations whose REDCap
    variable name is an exact, case-sensitive match. For that variable,
    comparing against its maximum date across all strictly earlier canonical
    timepoints is equivalent to comparing against every earlier value.
    Floating and conversion never participate. One result is returned per
    distinct offending current form/variable/date.
    """
    missing_set = {str(value).strip() for value in missing_values}
    if (current_tp in NON_LONGITUDINAL_TIMEPOINTS
            or current_tp not in tp_order
            or not isinstance(subject_tp_dates, dict)):
        return []

    if current_observations is None:
        current_info = subject_tp_dates.get(current_tp)
    else:
        current_info = {'dates': current_observations}
    current = _visit_date_observations(
        current_info, missing_set, network=network)
    if not current:
        return []

    current_idx = tp_order.index(current_tp)
    prior_max_by_variable = {}
    for prev_idx, prev_tp in enumerate(tp_order[:current_idx]):
        if prev_tp in NON_LONGITUDINAL_TIMEPOINTS:
            continue
        for observation in _visit_date_observations(
                subject_tp_dates.get(prev_tp), missing_set, network=network,
                extrema=('earliest', 'latest')):
            variable = observation['variable']
            # Missing metadata cannot establish that two fields are exactly
            # the same, so old/incomplete dependencies fail closed.
            if not variable:
                continue
            # On equal dates, choose the most recent prior timepoint and then
            # stable source metadata so shuffled input gives identical output.
            candidate_key = (
                observation['dt'], prev_idx,
                observation['form'], observation['variable'])
            existing = prior_max_by_variable.get(variable)
            if existing is None or candidate_key > existing['key']:
                prior_max_by_variable[variable] = {
                    'observation': observation,
                    'timepoint': prev_tp,
                    'key': candidate_key,
                }

    if not prior_max_by_variable:
        return []

    results = []
    for observation in current:
        prior_match = prior_max_by_variable.get(observation['variable'])
        if prior_match is None:
            continue
        prior_max = prior_match['observation']
        if observation['dt'] >= prior_max['dt']:
            continue
        results.append({
            'current_date': observation['date'],
            'current_form': observation['form'],
            'current_variable': observation['variable'],
            'prev_timepoint': prior_match['timepoint'],
            'prev_date': prior_max['date'],
            'prev_form': prior_max['form'],
            'prev_variable': prior_max['variable'],
            'days_apart': (prior_max['dt'] - observation['dt']).days,
        })

    return sorted(
        results,
        key=lambda item: (
            -item['days_apart'], item['current_date'],
            item['current_form'], item['current_variable']))


def find_backward_visit_date(subject_tp_dates, tp_order, current_tp,
                             missing_values=('',), network=None):
    """
    Decide whether ``current_tp`` contains a visit-date field earlier than
    the exact same field at an earlier canonical timepoint.

    Parameters
    ----------
    subject_tp_dates : dict
        One subject's entry from ``earliest_latest_dates_per_tp``.
    tp_order : list[str]
        Canonical ordered timepoints (Utils.create_timepoint_list()).
    current_tp : str
        The timepoint being evaluated (the candidate "later" visit).
    missing_values : iterable[str]
        Values treated as no-date (REDCap missing codes + '').

    Returns
    -------
    dict | None
        None when current_tp is non-longitudinal, not in tp_order, has no
        usable date, or does not precede any earlier timepoint's date.
        Otherwise a dict describing the worst (largest-gap) violation:
        {'current_date','current_form','prev_timepoint','prev_date',
         'prev_form','days_apart'} with days_apart > 0, where prev_* is
        the earlier timepoint holding that variable's latest (running-max)
        prior visit date.
    """
    results = find_backward_visit_dates(
        subject_tp_dates, tp_order, current_tp,
        missing_values=missing_values, network=network)
    return results[0] if results else None
