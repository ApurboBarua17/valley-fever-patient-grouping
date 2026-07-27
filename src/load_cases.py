"""Extract the county case counts from the ADHS Valley Fever annual report PDF.

The counts set how many synthetic patients come from each county, so that the
patient population is distributed the way Arizona's real cases are.
"""

import re

import pandas as pd
import pdfplumber

# A statewide summary row sits inside the same table and would skew the weights.
STATEWIDE_ROW_LABEL = "arizona"

# Counts are printed with thousands separators, as in "7,993".
NUMERIC_NOISE = re.compile(r"[^0-9.]")


def _find_column_indexes(header_row):
    """Locate the county and case columns, by keyword so a renamed header still resolves."""
    indexes = {}
    for position, raw_header in enumerate(header_row):
        header = (raw_header or "").lower()
        if "county" in header:
            indexes["county"] = position
        elif "case" in header:
            indexes["cases"] = position

    missing = {"county", "cases"} - set(indexes)
    if missing:
        raise ValueError(f"Could not locate columns {sorted(missing)} in header {header_row}")
    return indexes


def _looks_like_county_table(table):
    """True if this extracted table is the by-county case table."""
    if not table or len(table) < 5:
        return False
    header = " ".join((cell or "").lower() for cell in table[0])
    return "county" in header and "case" in header


def _to_number(raw_value):
    """Strip separators and footnote markers, then convert to a number."""
    if raw_value is None:
        return None
    cleaned = NUMERIC_NOISE.sub("", raw_value)
    if not cleaned:
        return None
    return float(cleaned)


def load_case_data(pdf_path):
    """Return one row per Arizona county with its 2023 reported case count."""
    county_table = None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if _looks_like_county_table(table):
                    county_table = table
                    break
            if county_table is not None:
                break

    if county_table is None:
        raise ValueError(f"No county case table found in {pdf_path}")

    column_index = _find_column_indexes(county_table[0])

    records = []
    for row in county_table[1:]:
        county_name = (row[column_index["county"]] or "").strip()
        if not county_name:
            continue
        if county_name.lower() == STATEWIDE_ROW_LABEL:
            continue

        records.append(
            {
                "county": county_name,
                "cases": _to_number(row[column_index["cases"]]),
            }
        )

    case_data = pd.DataFrame(records)

    # A county that failed to parse would silently get zero patients.
    if case_data["cases"].isna().any():
        unusable = case_data.loc[case_data["cases"].isna(), "county"].tolist()
        raise ValueError(f"Case counts failed to parse for: {unusable}")

    case_data["cases"] = case_data["cases"].astype(int)
    return case_data.sort_values("county").reset_index(drop=True)


if __name__ == "__main__":
    print(load_case_data("data/valley-fever-2023.pdf").to_string(index=False))
