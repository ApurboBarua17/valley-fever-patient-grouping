"""Join the Valley Fever case data to the SVI data on county name."""

import pandas as pd

EXPECTED_COUNTY_COUNT = 15


def normalize_county_name(raw_name):
    """Reduce a county label to a form both sources agree on.

    ADHS writes "Santa Cruz" where the SVI file writes "Santa Cruz County".
    """
    name = str(raw_name).strip().lower()
    if name.endswith(" county"):
        name = name[: -len(" county")]
    return " ".join(name.split())


def _to_display_name(normalized_name):
    """Turn a normalized key back into a readable county name."""
    return normalized_name.title()


def join_case_and_svi_data(case_data, svi_data):
    """Return one row per county combining case burden and vulnerability scores.

    Raises if either source contains a county the other does not, since a silent
    partial join would quietly drop counties from the clustering.
    """
    cases = case_data.copy()
    svi = svi_data.copy()

    cases["county_key"] = cases["county"].apply(normalize_county_name)
    svi["county_key"] = svi["county"].apply(normalize_county_name)

    case_keys = set(cases["county_key"])
    svi_keys = set(svi["county_key"])

    only_in_cases = sorted(case_keys - svi_keys)
    only_in_svi = sorted(svi_keys - case_keys)
    if only_in_cases or only_in_svi:
        raise ValueError(
            "County names did not match between sources. "
            f"Only in case data: {only_in_cases}. Only in SVI data: {only_in_svi}."
        )

    combined = pd.merge(
        cases.drop(columns=["county"]),
        svi.drop(columns=["county"]),
        on="county_key",
        how="inner",
        validate="one_to_one",
    )

    combined["county"] = combined["county_key"].apply(_to_display_name)
    combined = combined.drop(columns=["county_key"])

    if len(combined) != EXPECTED_COUNTY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNTY_COUNT} Arizona counties, joined {len(combined)}"
        )

    ordered_columns = [
        "county",
        "population",
        "cases",
        "rate_2023",
        "rate_5yr_avg",
        "svi_overall",
        "svi_socioeconomic",
        "svi_household",
        "svi_minority_language",
        "svi_housing_transport",
    ]
    return combined[ordered_columns].sort_values("county").reset_index(drop=True)
