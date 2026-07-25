"""Extract the county-level Valley Fever case table from the ADHS annual report PDF."""

import re

import pandas as pd
import pdfplumber

# A statewide summary row sits inside the same table and would skew every average.
STATEWIDE_ROW_LABEL = "arizona"

# Strips thousands separators and the asterisks ADHS puts on small count rates.
NUMERIC_NOISE = re.compile(r"[^0-9.]")


def _find_column_indexes(header_row):
    """Locate the four columns we need, by keyword so a renamed header still resolves."""
    indexes = {}
    for position, raw_header in enumerate(header_row):
        header = (raw_header or "").lower()
        if "county" in header:
            indexes["county"] = position
        elif "case" in header:
            indexes["cases"] = position
        elif "avg" in header:
            indexes["rate_5yr_avg"] = position
        elif "rate" in header:
            indexes["rate_2023"] = position

    missing = {"county", "cases", "rate_2023", "rate_5yr_avg"} - set(indexes)
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
    """Strip thousands separators and footnote markers, then convert to float."""
    if raw_value is None:
        return None
    cleaned = NUMERIC_NOISE.sub("", raw_value)
    if not cleaned:
        return None
    return float(cleaned)


def load_case_data(pdf_path):
    """Return one row per Arizona county with case count and rates per 100,000.

    Columns: county, cases, rate_2023, rate_5yr_avg.
    """
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
                "rate_2023": _to_number(row[column_index["rate_2023"]]),
                "rate_5yr_avg": _to_number(row[column_index["rate_5yr_avg"]]),
            }
        )

    case_data = pd.DataFrame(records)
    case_data["cases"] = case_data["cases"].astype(int)

    if case_data[["cases", "rate_2023", "rate_5yr_avg"]].isna().any().any():
        raise ValueError("Some case values failed to parse out of the PDF table")

    return case_data.sort_values("county").reset_index(drop=True)


if __name__ == "__main__":
    print(load_case_data("data/valley-fever-2023.pdf").to_string(index=False))
