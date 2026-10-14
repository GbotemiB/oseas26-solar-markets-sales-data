"""Tests for src/xlsx.py and src/labels_open.py. Small synthetic tables in the same layout as the NISR files.

Run from the repo root:  python -m pytest tests/test_labels_open.py -v
"""
import zipfile

import pytest

from src.labels_open import (find_row, norm_name, parse_census_provinces, parse_census_sector_grid, parse_census_sectors,
                             parse_eicv7_districts, parse_eicv7_provinces, split_row)
from src.xlsx import column_index, read_sheet


# ---------------------------------------------------------------- xlsx reader
def make_xlsx(path):
    """A two-column sheet 'T' with a shared string, an inline string, numbers and a gap."""
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rel}><sheets><sheet name="T" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst {ns}><si><t>Gasabo</t></si></sst>')
        z.writestr("xl/worksheets/sheet1.xml",
                   f'<worksheet {ns}><sheetData>'
                   '<row r="1"><c r="B1" t="s"><v>0</v></c><c r="C1"><v>0.5</v></c><c r="E1"><v>7</v></c></row>'
                   '<row r="2"/>'
                   '<row r="3"><c r="A3" t="inlineStr"><is><t>Inline</t></is></c></row>'
                   '</sheetData></worksheet>')


def test_column_index():
    assert [column_index(r) for r in ["A1", "B7", "Z2", "AA3"]] == [0, 1, 25, 26]


def test_read_sheet_keeps_column_positions_and_skips_empty_rows(tmp_path):
    make_xlsx(tmp_path / "t.xlsx")
    rows = read_sheet(tmp_path / "t.xlsx", "T")
    assert rows == [[None, "Gasabo", 0.5, None, 7.0], ["Inline"]]


def test_read_sheet_unknown_name_lists_the_sheets(tmp_path):
    make_xlsx(tmp_path / "t.xlsx")
    with pytest.raises(KeyError, match="No sheet 'X'"):
        read_sheet(tmp_path / "t.xlsx", "X")


# ---------------------------------------------------------------- helpers
def test_split_row_ignores_empty_first_column():
    assert split_row([None, "  Gitega", 0.9, 8239.0]) == ("Gitega", [0.9, 8239.0])
    assert split_row([None, "Nyarugenge"]) == ("Nyarugenge", [])


def test_norm_name_matches_across_spellings():
    assert norm_name("Southern Province") == norm_name("Southern") == "southern"
    assert norm_name("Ruhango ") == norm_name("RUHANGO")
    assert norm_name("N'yanza") == "nyanza"


def test_find_row_raises_when_missing():
    with pytest.raises(KeyError):
        find_row([["a"], ["b"]], "c")


# ---------------------------------------------------------------- table parsers
def test_parse_eicv7_districts_converts_percent_to_share_and_stops_at_source():
    rows = [["Table A5.14"], ["District"],
            ["Nyarugenge", 94.7, 0.9, 92.9, 96.6, 0.98, 0.85],
            ["Gasabo", 88.7, 1.5, 85.6, 91.8, 1.7, 2.8],
            ["Source: NISR, EICV7"], ["Footnote", 1.0]]
    t = parse_eicv7_districts(rows)
    assert t["district"].tolist() == ["Nyarugenge", "Gasabo"]
    assert t.loc[0, "served"] == pytest.approx(0.947) and t.loc[0, "ci_high"] == pytest.approx(0.966)


def test_parse_eicv7_provinces_picks_grid_and_solar_by_header_name():
    rows = [["Table 5.11"],
            ["EICV7", "Main source energy for home lighting", "Total"],
            ["Electricity distributor", "Torch or phone flash", "Solar power", "Firewood", "Total"],
            ["Rwanda", 50.0, 25.0, 22.0, 1.0, 100.0],
            ["Area of residence"],
            ["Urban", 83.0, 9.6, 5.2, 0.2, 100.0],
            ["Province"],
            ["City of Kigali", 90.0, 6.0, 1.0, 0.5, 100.0],
            ["Southern", 34.0, 32.0, 29.0, 0.5, 100.0]]
    t = parse_eicv7_provinces(rows).set_index("province")
    assert list(t.index) == ["Rwanda", "City of Kigali", "Southern"]      # 'Urban' is not a province
    assert t.loc["Southern", "share_grid"] == pytest.approx(0.34)
    assert t.loc["Southern", "share_solar"] == pytest.approx(0.29)
    assert t.loc["Southern", "share_other"] == pytest.approx(0.37)


def test_parse_census_sectors_skips_total_lines_and_assigns_districts():
    rows = [[None, "Table C1.3"],
            [None, "District and Sectors", "Percentage", "Percentage", "Total count"],
            [None, "District and Sectors", "Electricity", "Lantern", "Total"],
            [None, "Rwanda", 0.61, 0.01, 1.0, 3312743.0],
            [None, "Nyarugenge"],
            [None, "  Gitega", 0.98, 0.0, 1.0, 8239.0],
            [None, "  Total", 0.9, 0.0, 1.0, 20000.0],
            [None, "Gasabo"],
            [None, "  Gitega", 0.5, 0.0, 1.0, 5000.0],
            [None, "Source: 5th census"]]
    t = parse_census_sectors(rows)
    assert t["district"].tolist() == ["Nyarugenge", "Gasabo"]
    assert t["sector"].tolist() == ["Gitega", "Gitega"]                    # same sector name in two districts
    assert t["served"].tolist() == [0.98, 0.5] and t["n_households"].tolist() == [8239.0, 5000.0]


def test_parse_census_sectors_stops_if_the_count_column_moves():
    rows = [[None, "x", "Percentage", "Percentage"], [None, "D", "Electricity", "Total"]]
    with pytest.raises(AssertionError):
        parse_census_sectors(rows)


def test_parse_census_provinces_gives_counts_per_area():
    rows = [[None, "Table 8.12"],
            [None, "Province and main source", "Count", "Count", "Count"],
            [None, "Province and main source", "Urban", "Rural", "Total"],
            [None, "Rwanda"],
            [None, "  Electricity from REG or other private source", 80.0, 20.0, 100.0],
            [None, "  Solar power", 3.0, 40.0, 43.0],
            [None, "  Generator", 0.0, 1.0, 1.0],
            [None, "  Lantern", 1.0, 3.0, 4.0],
            [None, "  Total", 100.0, 100.0, 200.0],
            [None, "Southern Province"],
            [None, "  Electricity from REG or other private source", 8.0, 2.0, 10.0],
            [None, "  Solar power", 1.0, 4.0, 5.0],
            [None, "  Lantern", 0.0, 1.0, 1.0],
            [None, "  Total", 10.0, 10.0, 20.0]]
    t = parse_census_provinces(rows)
    assert sorted(t["province"].unique()) == ["Rwanda", "Southern"]
    nat = t[(t["province"] == "Rwanda") & (t["area"] == "total")].iloc[0]
    assert (nat["grid"], nat["solar"], nat["lantern"], nat["total"]) == (100.0, 43.0, 4.0, 200.0)
    south_rural = t[(t["province"] == "Southern") & (t["area"] == "rural")].iloc[0]
    assert south_rural["solar"] == 4.0


def test_parse_census_sector_grid_reads_connected_share_and_count():
    rows = [[None, "Table C1.5"],
            [None, "District and Sectors", "Percentage", "Percentage", "Count", "Count"],
            [None, "District and Sectors", "Grid connected Households", "Not connected Households", "Not stated", "Total",
             "Grid connected Households", "Not connected Households", "Not stated", "Total"],
            [None, "Rwanda", 0.525, 0.475, 0.0, 1.0, 1740316.0, 1572319.0, 108.0, 3312743.0],
            [None, "Nyarugenge"],
            [None, "  Gitega", 0.986, 0.014, 0.0, 1.0, 8121.0, 117.0, 1.0, 8239.0],
            [None, "  Total", 0.9, 0.1, 0.0, 1.0, 9000.0, 1000.0, 0.0, 10000.0],
            [None, "Gasabo"],
            [None, "  Gitega", 0.5, 0.5, 0.0, 1.0, 100.0, 100.0, 0.0, 200.0],
            [None, "Source: 5th census"]]
    t = parse_census_sector_grid(rows)
    assert t["district"].tolist() == ["Nyarugenge", "Gasabo"]
    assert t["connected"].tolist() == [0.986, 0.5] and t["n_households"].tolist() == [8239.0, 200.0]


def test_parse_census_sector_grid_stops_if_the_columns_move():
    rows = [[None, "Table C1.5"], [None, "District and Sectors", "Grid connected Households", "Not connected Households"]]
    with pytest.raises(AssertionError):
        parse_census_sector_grid(rows)
