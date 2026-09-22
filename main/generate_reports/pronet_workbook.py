"""Surgical sheet updater for the live PRONET workbook."""

from copy import deepcopy
from io import BytesIO
import posixpath
import re
from xml.etree import ElementTree as ET
import zipfile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = (
    "http://schemas.openxmlformats.org/package/2006/content-types")
APP_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "extended-properties")
VT_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes")
WORKSHEET_REL_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/worksheet")
WORKSHEET_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml."
    "worksheet+xml")

ET.register_namespace("", MAIN_NS)
ET.register_namespace("r", DOC_REL_NS)


class PronetWorkbookUpdateError(RuntimeError):
    """The workbook cannot be updated without risking unrelated content."""


def _q(namespace, local_name):
    return f"{{{namespace}}}{local_name}"


def _root_start_tag(xml_bytes):
    match = re.search(br"<(?![!?])[^>]+>", xml_bytes)
    if match is None:
        raise PronetWorkbookUpdateError("XML document has no root element")
    return match


def _root_namespaces(xml_bytes):
    tag = _root_start_tag(xml_bytes).group(0)
    return {
        match.group(1).decode("ascii"): match.group(0)
        for match in re.finditer(
            br"\s+xmlns:([A-Za-z_][A-Za-z0-9_.-]*)=(?:\"[^\"]*\"|'[^']*')",
            tag,
        )
    }


def _xml_bytes(root, original_bytes=None):
    original_namespaces = {}
    if original_bytes is not None:
        original_namespaces = _root_namespaces(original_bytes)
        for prefix, declaration in original_namespaces.items():
            uri_match = re.search(br"=(?:\"([^\"]*)\"|'([^']*)')", declaration)
            uri = next(
                value for value in uri_match.groups() if value is not None)
            try:
                ET.register_namespace(prefix, uri.decode("utf-8"))
            except ValueError:
                pass
    serialized = ET.tostring(
        root, encoding="utf-8", xml_declaration=True)
    if not original_namespaces:
        return serialized
    present = _root_namespaces(serialized)
    missing = b"".join(
        declaration
        for prefix, declaration in original_namespaces.items()
        if prefix not in present)
    if not missing:
        return serialized
    match = _root_start_tag(serialized)
    insertion = match.end() - (2 if match.group(0).endswith(b"/>") else 1)
    return serialized[:insertion] + missing + serialized[insertion:]


def _parse_member(members, name, label):
    if name not in members:
        raise PronetWorkbookUpdateError(
            f"{label} is missing required package member {name!r}")
    try:
        return ET.fromstring(members[name])
    except ET.ParseError as exc:
        raise PronetWorkbookUpdateError(
            f"{label} contains invalid XML in {name!r}") from exc


def _relationship_target_to_part(target):
    if target.startswith("/"):
        part = target.lstrip("/")
    else:
        part = posixpath.normpath(posixpath.join("xl", target))
    if part.startswith("../") or not part.startswith("xl/"):
        raise PronetWorkbookUpdateError(
            f"unsafe worksheet relationship target {target!r}")
    return part


def _read_package(data, label):
    try:
        archive = zipfile.ZipFile(BytesIO(data), "r")
    except (zipfile.BadZipFile, TypeError) as exc:
        raise PronetWorkbookUpdateError(
            f"{label} is not a readable XLSX package") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise PronetWorkbookUpdateError(
                f"{label} contains duplicate ZIP members")
        if any(info.flag_bits & 0x1 for info in infos):
            raise PronetWorkbookUpdateError(
                f"{label} contains encrypted ZIP members")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise PronetWorkbookUpdateError(
                f"{label} contains corrupt member {bad_member!r}")
        lowered = {name.casefold() for name in names}
        has_signature = (
            any(name.startswith("_xmlsignatures/") for name in lowered)
            or any(name.endswith("vbaprojectsignature.bin")
                   for name in lowered))
        if has_signature:
            raise PronetWorkbookUpdateError(
                f"{label} is digitally signed")
        members = {name: archive.read(name) for name in names}
        comment = archive.comment
    return infos, members, comment


def _sheet_parts(members, sheet_name, label):
    workbook = _parse_member(members, "xl/workbook.xml", label)
    relationships = _parse_member(
        members, "xl/_rels/workbook.xml.rels", label)
    protection = workbook.find(_q(MAIN_NS, "workbookProtection"))
    if (protection is not None
            and str(protection.get("lockStructure", "")).casefold()
            in {"1", "true", "on"}):
        raise PronetWorkbookUpdateError(
            f"{label} has workbook structure protection enabled")

    sheets = workbook.find(_q(MAIN_NS, "sheets"))
    if sheets is None:
        raise PronetWorkbookUpdateError(f"{label} has no sheets manifest")
    matching_sheets = [
        item for item in sheets
        if str(item.get("name", "")).casefold() == sheet_name.casefold()]
    if len(matching_sheets) > 1:
        raise PronetWorkbookUpdateError(
            f"{label} contains duplicate {sheet_name!r} sheets")
    sheet = matching_sheets[0] if matching_sheets else None
    if sheet is None:
        return workbook, relationships, sheets, None, None

    relationship_id = sheet.get(_q(DOC_REL_NS, "id"))
    relationship = next(
        (item for item in relationships
         if item.get("Id") == relationship_id),
        None)
    if (relationship is None
            or relationship.get("Type") != WORKSHEET_REL_TYPE
            or relationship.get("TargetMode") == "External"):
        raise PronetWorkbookUpdateError(
            f"{label} has an invalid relationship for {sheet_name!r}")
    part = _relationship_target_to_part(relationship.get("Target", ""))
    if part not in members:
        raise PronetWorkbookUpdateError(
            f"{label} is missing worksheet part {part!r}")
    return workbook, relationships, sheets, sheet, part


def _element_key(element):
    return ET.tostring(element, encoding="utf-8")


def _section(root, name, label):
    section = root.find(_q(MAIN_NS, name))
    if section is None:
        raise PronetWorkbookUpdateError(
            f"{label} styles are missing {name!r}")
    return section


def _find_or_append(section, element):
    key = _element_key(element)
    for index, existing in enumerate(section):
        if _element_key(existing) == key:
            return index
    section.append(deepcopy(element))
    section.set("count", str(len(section)))
    return len(section) - 1


def _merge_number_format(source_styles, target_styles, number_format_id):
    if number_format_id < 164:
        return number_format_id
    source_formats = source_styles.find(_q(MAIN_NS, "numFmts"))
    if source_formats is None:
        raise PronetWorkbookUpdateError(
            f"donor style refers to missing numFmtId {number_format_id}")
    source_format = next(
        (item for item in source_formats
         if int(item.get("numFmtId", "-1")) == number_format_id),
        None)
    if source_format is None:
        raise PronetWorkbookUpdateError(
            f"donor style refers to unknown numFmtId {number_format_id}")

    target_formats = target_styles.find(_q(MAIN_NS, "numFmts"))
    if target_formats is None:
        target_formats = ET.Element(_q(MAIN_NS, "numFmts"), {"count": "0"})
        target_styles.insert(0, target_formats)
    format_code = source_format.get("formatCode")
    existing = next(
        (item for item in target_formats
         if item.get("formatCode") == format_code),
        None)
    if existing is not None:
        return int(existing.get("numFmtId"))
    used_ids = {
        int(item.get("numFmtId"))
        for item in target_formats
        if item.get("numFmtId", "").isdigit()
    }
    new_id = max(used_ids | {163}) + 1
    copied = deepcopy(source_format)
    copied.set("numFmtId", str(new_id))
    target_formats.append(copied)
    target_formats.set("count", str(len(target_formats)))
    return new_id


def _merge_sheet_styles(source_members, target_members, sheet_root):
    source_styles = _parse_member(
        source_members, "xl/styles.xml", "generated PRONET workbook")
    target_styles = _parse_member(
        target_members, "xl/styles.xml", "live PRONET workbook")
    source_xfs = _section(
        source_styles, "cellXfs", "generated PRONET workbook")
    target_xfs = _section(
        target_styles, "cellXfs", "live PRONET workbook")

    style_references = [
        (cell, "s")
        for cell in sheet_root.iter(_q(MAIN_NS, "c"))
        if cell.get("s") is not None
    ]
    style_references.extend(
        (row, "s")
        for row in sheet_root.iter(_q(MAIN_NS, "row"))
        if row.get("s") is not None
    )
    style_references.extend(
        (column, "style")
        for column in sheet_root.iter(_q(MAIN_NS, "col"))
        if column.get("style") is not None
    )
    try:
        style_ids = {
            int(element.get(attribute))
            for element, attribute in style_references
        }
    except (TypeError, ValueError) as exc:
        raise PronetWorkbookUpdateError(
            "generated sheet contains an invalid style reference") from exc
    style_map = {}
    component_names = {
        "fontId": "fonts",
        "fillId": "fills",
        "borderId": "borders",
    }
    for style_id in sorted(style_ids):
        if style_id < 0 or style_id >= len(source_xfs):
            raise PronetWorkbookUpdateError(
                f"generated sheet uses invalid style id {style_id}")
        copied_xf = deepcopy(source_xfs[style_id])
        if int(copied_xf.get("xfId", "0")) != 0:
            raise PronetWorkbookUpdateError(
                "generated sheet uses a non-default named cell style")
        for attribute, section_name in component_names.items():
            component_id = int(copied_xf.get(attribute, "0"))
            source_section = _section(
                source_styles, section_name, "generated PRONET workbook")
            target_section = _section(
                target_styles, section_name, "live PRONET workbook")
            if component_id < 0 or component_id >= len(source_section):
                raise PronetWorkbookUpdateError(
                    f"generated style uses invalid {attribute} {component_id}")
            target_id = _find_or_append(
                target_section, source_section[component_id])
            copied_xf.set(attribute, str(target_id))
        number_format_id = int(copied_xf.get("numFmtId", "0"))
        copied_xf.set(
            "numFmtId",
            str(_merge_number_format(
                source_styles, target_styles, number_format_id)))
        style_map[style_id] = _find_or_append(target_xfs, copied_xf)

    for element, attribute in style_references:
        element.set(
            attribute, str(style_map[int(element.get(attribute))]))
    return _xml_bytes(
        target_styles, target_members["xl/styles.xml"]), sheet_root


def _sheet_relationship_part(sheet_part):
    return posixpath.join(
        posixpath.dirname(sheet_part),
        "_rels",
        posixpath.basename(sheet_part) + ".rels")


def _validate_flat_sheet(
        members, sheet_part, sheet_root, label, *,
        allow_shared_strings=False):
    relationship_part = _sheet_relationship_part(sheet_part)
    if relationship_part in members:
        relationships = _parse_member(members, relationship_part, label)
        if len(relationships):
            raise PronetWorkbookUpdateError(
                f"{label} {sheet_part!r} has related package parts")
    disallowed = {
        "conditionalFormatting", "drawing", "extLst", "hyperlinks",
        "legacyDrawing", "legacyDrawingHF", "oleObjects", "picture",
        "tableParts", "controls",
    }
    for element in sheet_root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name in disallowed:
            raise PronetWorkbookUpdateError(
                f"{label} {sheet_part!r} contains {local_name}")
        if (local_name == "c" and element.get("t") == "s"
                and not allow_shared_strings):
            raise PronetWorkbookUpdateError(
                f"{label} {sheet_part!r} uses shared strings")
        if local_name == "f":
            raise PronetWorkbookUpdateError(
                f"{label} {sheet_part!r} contains formulas")


def _append_sheet_manifest(
        members, workbook, relationships, sheets, sheet_name):
    used_sheet_ids = {
        int(item.get("sheetId"))
        for item in sheets
        if item.get("sheetId", "").isdigit()
    }
    new_sheet_id = max(used_sheet_ids | {0}) + 1

    # OPC part names are case-insensitive (ECMA-376 Part 2): a live workbook
    # storing "xl/worksheets/Sheet1.xml" must block lowercase "sheet1.xml",
    # or Excel refuses to open the package with both members present.
    taken_part_names = {name.casefold() for name in members}
    sheet_number = 1
    while f"xl/worksheets/sheet{sheet_number}.xml" in taken_part_names:
        sheet_number += 1
    sheet_part = f"xl/worksheets/sheet{sheet_number}.xml"

    used_relationship_ids = {
        item.get("Id") for item in relationships}
    relationship_number = 1
    while f"rId{relationship_number}" in used_relationship_ids:
        relationship_number += 1
    relationship_id = f"rId{relationship_number}"

    ET.SubElement(
        sheets,
        _q(MAIN_NS, "sheet"),
        {
            "name": sheet_name,
            "sheetId": str(new_sheet_id),
            _q(DOC_REL_NS, "id"): relationship_id,
        },
    )
    ET.SubElement(
        relationships,
        _q(PKG_REL_NS, "Relationship"),
        {
            "Id": relationship_id,
            "Type": WORKSHEET_REL_TYPE,
            "Target": f"worksheets/sheet{sheet_number}.xml",
        },
    )

    content_types = _parse_member(
        members, "[Content_Types].xml", "live PRONET workbook")
    part_name = "/" + sheet_part
    if not any(
            item.get("PartName") == part_name for item in content_types):
        ET.SubElement(
            content_types,
            _q(CONTENT_TYPES_NS, "Override"),
            {
                "PartName": part_name,
                "ContentType": WORKSHEET_CONTENT_TYPE,
            },
        )
    return sheet_part, content_types


def _relationship_for_sheet(sheet, relationships, label):
    relationship_id = sheet.get(_q(DOC_REL_NS, "id"))
    matches = [
        item for item in relationships
        if item.get("Id") == relationship_id]
    if len(matches) != 1:
        raise PronetWorkbookUpdateError(
            f"{label} has an invalid relationship for "
            f"{sheet.get('name', '')!r}")
    return matches[0]


def _worksheet_names(sheets, relationships, label):
    names = []
    for sheet in sheets:
        relationship = _relationship_for_sheet(
            sheet, relationships, label)
        if relationship.get("Type") != WORKSHEET_REL_TYPE:
            continue
        if relationship.get("TargetMode") == "External":
            raise PronetWorkbookUpdateError(
                f"{label} has an external worksheet relationship for "
                f"{sheet.get('name', '')!r}")
        names.append(str(sheet.get("name", "")))
    return names


def _place_sheet_after(
        sheets, relationships, target_sheet, anchor_name, label):
    anchors = [
        item for item in sheets
        if str(item.get("name", "")).casefold()
        == anchor_name.casefold()]
    if len(anchors) != 1:
        raise PronetWorkbookUpdateError(
            f"{label} must contain exactly one {anchor_name!r} worksheet")
    anchor = anchors[0]
    anchor_relationship = _relationship_for_sheet(
        anchor, relationships, label)
    if (anchor_relationship.get("Type") != WORKSHEET_REL_TYPE
            or anchor_relationship.get("TargetMode") == "External"):
        raise PronetWorkbookUpdateError(
            f"{label} {anchor_name!r} anchor is not a worksheet")

    current = list(sheets)
    desired = [item for item in current if item is not target_sheet]
    desired.insert(desired.index(anchor) + 1, target_sheet)
    if current == desired:
        return False
    for item in current:
        sheets.remove(item)
    for item in desired:
        sheets.append(item)
    return True


def _sheet_token(sheet):
    return (
        str(sheet.get("sheetId", "")),
        str(sheet.get(_q(DOC_REL_NS, "id"), "")),
    )


def _remap_sheet_indexes(workbook, original_sheets, final_sheets):
    original_tokens = [_sheet_token(sheet) for sheet in original_sheets]
    final_tokens = [_sheet_token(sheet) for sheet in final_sheets]
    if (len(original_tokens) != len(set(original_tokens))
            or len(final_tokens) != len(set(final_tokens))):
        raise PronetWorkbookUpdateError(
            "live PRONET workbook has duplicate sheet identifiers")
    final_positions = {
        token: index for index, token in enumerate(final_tokens)}
    if any(token not in final_positions for token in original_tokens):
        raise PronetWorkbookUpdateError(
            "sheet placement changed an existing sheet identifier")
    index_map = {
        old_index: final_positions[token]
        for old_index, token in enumerate(original_tokens)}
    if all(old_index == new_index
           for old_index, new_index in index_map.items()):
        return

    for defined_name in workbook.iter(_q(MAIN_NS, "definedName")):
        raw_index = defined_name.get("localSheetId")
        if raw_index is None:
            continue
        try:
            old_index = int(raw_index)
        except ValueError as exc:
            raise PronetWorkbookUpdateError(
                "live PRONET workbook has an invalid localSheetId") from exc
        if old_index not in index_map:
            raise PronetWorkbookUpdateError(
                "live PRONET workbook has an out-of-range localSheetId")
        defined_name.set("localSheetId", str(index_map[old_index]))

    for workbook_view in workbook.iter(_q(MAIN_NS, "workbookView")):
        for attribute in ("activeTab", "firstSheet"):
            raw_index = workbook_view.get(attribute)
            try:
                old_index = int(raw_index) if raw_index is not None else 0
            except ValueError as exc:
                raise PronetWorkbookUpdateError(
                    f"live PRONET workbook has an invalid "
                    f"{attribute}") from exc
            if old_index not in index_map:
                raise PronetWorkbookUpdateError(
                    f"live PRONET workbook has an out-of-range "
                    f"{attribute}")
            new_index = index_map[old_index]
            if raw_index is not None or new_index != 0:
                workbook_view.set(attribute, str(new_index))


def _updated_app_properties(
        members, original_worksheet_names, final_worksheet_names):
    if original_worksheet_names == final_worksheet_names:
        return None
    name = "docProps/app.xml"
    if name not in members:
        return None
    app = _parse_member(members, name, "live PRONET workbook")
    titles = app.find(_q(APP_NS, "TitlesOfParts"))
    if titles is None:
        return None
    title_vector = titles.find(_q(VT_NS, "vector"))
    if title_vector is None:
        return None

    heading_pairs = app.find(_q(APP_NS, "HeadingPairs"))
    if heading_pairs is None:
        return None
    heading_vector = heading_pairs.find(_q(VT_NS, "vector"))
    if heading_vector is None:
        return None
    children = list(heading_vector)
    title_offset = 0
    worksheet_groups = []
    for index in range(0, len(children) - 1, 2):
        label_container = children[index]
        label = (
            label_container[0]
            if (label_container.tag == _q(VT_NS, "variant")
                and len(label_container))
            else label_container)
        count_container = children[index + 1]
        count = (
            count_container[0]
            if (count_container.tag == _q(VT_NS, "variant")
                and len(count_container))
            else count_container)
        if (label.tag != _q(VT_NS, "lpstr")
                or count.tag not in {
                    _q(VT_NS, "i4"), _q(VT_NS, "ui4")}):
            return None
        try:
            group_count = int(count.text or "0")
        except ValueError:
            return None
        if group_count < 0:
            return None
        if (label.text or "") == "Worksheets":
            worksheet_groups.append(
                (title_offset, group_count, count))
        title_offset += group_count
    if not worksheet_groups:
        return None
    if len(worksheet_groups) != 1:
        raise PronetWorkbookUpdateError(
            "live PRONET workbook app properties contain duplicate "
            "Worksheets groups")
    worksheet_offset, worksheet_count, worksheet_count_element = (
        worksheet_groups[0])
    title_items = list(title_vector)
    worksheet_end = worksheet_offset + worksheet_count
    if worksheet_end > len(title_items):
        raise PronetWorkbookUpdateError(
            "live PRONET workbook app properties have an invalid "
            "Worksheets title range")
    worksheet_items = title_items[worksheet_offset:worksheet_end]
    if (any(item.tag != _q(VT_NS, "lpstr")
            for item in worksheet_items)
            or [item.text or "" for item in worksheet_items]
            != original_worksheet_names):
        raise PronetWorkbookUpdateError(
            "live PRONET workbook app properties do not match its "
            "worksheet manifest")

    for item in worksheet_items:
        title_vector.remove(item)
    for offset, worksheet_name in enumerate(final_worksheet_names):
        new_title = ET.Element(_q(VT_NS, "lpstr"))
        new_title.text = worksheet_name
        title_vector.insert(worksheet_offset + offset, new_title)
    title_vector.set("size", str(len(title_vector)))
    worksheet_count_element.text = str(len(final_worksheet_names))
    return _xml_bytes(app, members[name])


def _write_package(infos, members, comment, replacements, removals):
    pending = dict(replacements)
    output = BytesIO()
    with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED,
            allowZip64=True) as archive:
        archive.comment = comment
        for info in infos:
            if info.filename in removals:
                continue
            payload = pending.pop(
                info.filename, members[info.filename])
            archive.writestr(info, payload)
        for name, payload in pending.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    return output.getvalue()


def workbook_has_sheet(workbook_bytes, sheet_name):
    """Return whether an XLSX contains the sheet name, case-insensitively."""
    _, members, _ = _read_package(workbook_bytes, "live PRONET workbook")
    _, _, _, sheet, _ = _sheet_parts(
        members, sheet_name, "live PRONET workbook")
    return sheet is not None


def _sheet_names(members, label):
    workbook = _parse_member(members, "xl/workbook.xml", label)
    sheets = workbook.find(_q(MAIN_NS, "sheets"))
    if sheets is None:
        raise PronetWorkbookUpdateError(f"{label} has no sheets manifest")
    return [str(sheet.get("name", "")) for sheet in sheets]


def _validate_sheet_only_result(
        live_members, updated_bytes, replacements, removals,
        sheet_name, expected_sheet_names):
    """Fail unless every non-target package member stayed byte-identical."""
    _, updated_members, _ = _read_package(
        updated_bytes, "updated PRONET workbook")
    expected_members = (
        (set(live_members) - set(removals)) | set(replacements))
    if set(updated_members) != expected_members:
        raise PronetWorkbookUpdateError(
            "updated workbook package members differ outside the "
            "sheet-only update plan")
    for name, payload in live_members.items():
        if name in replacements or name in removals:
            continue
        if updated_members.get(name) != payload:
            raise PronetWorkbookUpdateError(
                f"sheet-only update changed unrelated package member {name!r}")

    live_workbook = _parse_member(
        live_members, "xl/workbook.xml", "live PRONET workbook")
    live_sheets = live_workbook.find(_q(MAIN_NS, "sheets"))
    updated_workbook = _parse_member(
        updated_members, "xl/workbook.xml", "updated PRONET workbook")
    updated_sheets = updated_workbook.find(_q(MAIN_NS, "sheets"))
    if live_sheets is None or updated_sheets is None:
        raise PronetWorkbookUpdateError(
            "sheet-only update produced an invalid sheets manifest")
    updated_names = [
        str(sheet.get("name", "")) for sheet in updated_sheets]
    if updated_names != expected_sheet_names:
        raise PronetWorkbookUpdateError(
            "sheet-only update changed existing worksheet names or order")
    signature = lambda sheet: tuple(sorted(sheet.attrib.items()))
    live_non_target = [
        signature(sheet) for sheet in live_sheets
        if str(sheet.get("name", "")).casefold()
        != sheet_name.casefold()]
    updated_non_target = [
        signature(sheet) for sheet in updated_sheets
        if str(sheet.get("name", "")).casefold()
        != sheet_name.casefold()]
    if updated_non_target != live_non_target:
        raise PronetWorkbookUpdateError(
            "sheet-only update changed an unrelated sheet manifest entry")
    live_targets = [
        signature(sheet) for sheet in live_sheets
        if str(sheet.get("name", "")).casefold()
        == sheet_name.casefold()]
    updated_targets = [
        signature(sheet) for sheet in updated_sheets
        if str(sheet.get("name", "")).casefold()
        == sheet_name.casefold()]
    if live_targets and updated_targets != live_targets:
        raise PronetWorkbookUpdateError(
            "sheet-only update changed the target sheet identity")


def replace_sheet_from_workbook(
        live_bytes, generated_bytes, sheet_name, *, insert_after=None):
    """Return live workbook bytes with exactly one generated sheet applied."""
    live_infos, live_members, live_comment = _read_package(
        live_bytes, "live PRONET workbook")
    live_sheet_names = _sheet_names(
        live_members, "live PRONET workbook")
    if not any(
            name.casefold() != sheet_name.casefold()
            for name in live_sheet_names):
        raise PronetWorkbookUpdateError(
            "live PRONET workbook is damaged or incomplete: it contains "
            "only Medication Flags and has no other worksheets; "
            "restore the last good Dropbox revision before updating")
    _, generated_members, _ = _read_package(
        generated_bytes, "generated PRONET workbook")

    _, _, _, generated_sheet, generated_part = _sheet_parts(
        generated_members, sheet_name, "generated PRONET workbook")
    if generated_sheet is None:
        raise PronetWorkbookUpdateError(
            f"generated workbook has no {sheet_name!r} sheet")
    generated_root = _parse_member(
        generated_members, generated_part, "generated PRONET workbook")
    _validate_flat_sheet(
        generated_members, generated_part, generated_root,
        "generated PRONET workbook")

    workbook, relationships, sheets, live_sheet, live_part = _sheet_parts(
        live_members, sheet_name, "live PRONET workbook")
    original_sheets = list(sheets)
    original_worksheet_names = _worksheet_names(
        sheets, relationships, "live PRONET workbook")
    replacements = {}
    removals = set()
    target_added = live_sheet is None
    if live_sheet is not None:
        live_root = _parse_member(
            live_members, live_part, "live PRONET workbook")
        # The live copy of the target sheet is wholly replaced by the
        # generated flat sheet, so shared-string cells in it are harmless:
        # every real live workbook is shared-string encoded (Excel saves and
        # the pandas writers both produce them), and the entries a replaced
        # sheet leaves behind in xl/sharedStrings.xml are legal orphans.
        # Formulas, drawings, and related parts still refuse — replacing
        # those could silently destroy reviewer-authored content.
        _validate_flat_sheet(
            live_members, live_part, live_root, "live PRONET workbook",
            allow_shared_strings=True)
        destination_part = live_part
    else:
        destination_part, content_types = _append_sheet_manifest(
            live_members, workbook, relationships, sheets, sheet_name)
        replacements.update({
            "xl/_rels/workbook.xml.rels": _xml_bytes(
                relationships,
                live_members["xl/_rels/workbook.xml.rels"]),
            "[Content_Types].xml": _xml_bytes(
                content_types, live_members["[Content_Types].xml"]),
        })
        live_sheet = next(
            item for item in sheets
            if str(item.get("name", "")).casefold()
            == sheet_name.casefold())

    if insert_after is not None:
        _place_sheet_after(
            sheets,
            relationships,
            live_sheet,
            insert_after,
            "live PRONET workbook",
        )
    final_sheets = list(sheets)
    final_sheet_names = [
        str(sheet.get("name", "")) for sheet in final_sheets]
    manifest_changed = (
        [_sheet_token(sheet) for sheet in original_sheets]
        != [_sheet_token(sheet) for sheet in final_sheets])
    if manifest_changed:
        _remap_sheet_indexes(
            workbook, original_sheets, final_sheets)
        replacements["xl/workbook.xml"] = _xml_bytes(
            workbook, live_members["xl/workbook.xml"])
        final_worksheet_names = _worksheet_names(
            sheets, relationships, "updated PRONET workbook")
        app_properties = _updated_app_properties(
            live_members,
            original_worksheet_names,
            final_worksheet_names,
        )
        if app_properties is not None:
            replacements["docProps/app.xml"] = app_properties

    styles, generated_root = _merge_sheet_styles(
        generated_members, live_members, generated_root)
    replacements["xl/styles.xml"] = styles
    replacements[destination_part] = _xml_bytes(
        generated_root, generated_members[generated_part])

    allowed_replacements = {
        "xl/styles.xml",
        destination_part,
    }
    if target_added:
        allowed_replacements.update({
            "xl/_rels/workbook.xml.rels",
            "[Content_Types].xml",
        })
    if manifest_changed:
        allowed_replacements.update({
            "xl/workbook.xml",
            "docProps/app.xml",
        })
    unexpected_replacements = set(replacements) - allowed_replacements
    if unexpected_replacements:
        raise PronetWorkbookUpdateError(
            "sheet-only update planned unexpected package changes: "
            f"{sorted(unexpected_replacements)!r}")

    updated_bytes = _write_package(
        live_infos, live_members, live_comment, replacements, removals)
    _validate_sheet_only_result(
        live_members,
        updated_bytes,
        replacements,
        removals,
        sheet_name,
        final_sheet_names,
    )
    return updated_bytes
