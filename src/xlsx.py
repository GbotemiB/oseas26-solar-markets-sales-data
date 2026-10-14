"""Minimal reader for .xlsx files, using only the standard library.

An .xlsx file is a zip of XML files. This reads one sheet as a list of rows. We only need cell values
from simple NISR (National Institute of Statistics of Rwanda) tables, so formatting and formulas are ignored.
"""
import re
import zipfile
import xml.etree.ElementTree as ET

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def column_index(ref: str) -> int:
    """'A1' -> 0, 'B7' -> 1, 'AA3' -> 26."""
    index = 0
    for letter in re.match(r"[A-Z]+", ref).group():
        index = index * 26 + ord(letter) - 64
    return index - 1


def read_sheet(path, sheet_name: str) -> list:
    """Rows of one sheet as lists. Column positions are kept: empty cells are None, empty rows are skipped.

    Text cells are str, number cells are float.
    """
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(t.text or "" for t in si.iter(MAIN + "t"))
                      for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        sheets = {s.attrib["name"]: s.attrib[REL + "id"] for s in ET.fromstring(z.read("xl/workbook.xml")).find(MAIN + "sheets")}
        if sheet_name not in sheets:
            raise KeyError(f"No sheet '{sheet_name}' in {path}. Sheets: {sorted(sheets)[:10]} ...")
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        target = targets[sheets[sheet_name]].lstrip("/")
        root = ET.fromstring(z.read(target if target.startswith("xl/") else "xl/" + target))

    rows = []
    for row in root.iter(MAIN + "row"):
        cells = {}
        for c in row.iter(MAIN + "c"):
            kind = c.attrib["t"] if "t" in c.attrib else "n"
            v = c.find(MAIN + "v")
            if kind == "inlineStr":
                cells[column_index(c.attrib["r"])] = "".join(t.text or "" for t in c.iter(MAIN + "t"))
            elif v is None or v.text is None:
                continue
            elif kind == "s":
                cells[column_index(c.attrib["r"])] = shared[int(v.text)]
            elif kind in ("str", "e"):
                cells[column_index(c.attrib["r"])] = v.text
            else:
                cells[column_index(c.attrib["r"])] = float(v.text)
        if cells:
            rows.append([cells[i] if i in cells else None for i in range(max(cells) + 1)])
    return rows
