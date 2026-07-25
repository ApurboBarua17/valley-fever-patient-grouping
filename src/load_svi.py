"""Load Arizona county social determinant rates from the CDC/ATSDR SVI file."""

import pandas as pd

# County prevalences for the five social determinants the brief names. These are
# real published percentages, which is what lets the synthetic patients in
# generate_patients.py carry marginals that match Arizona rather than invented
# ones. Social support has no direct SVI measure, so single parent households
# stands in for it, which is a proxy and the weakest link in this mapping.
SDOH_RATE_COLUMNS = {
    "COUNTY": "county",
    "EP_HBURD": "housing_cost_burden_pct",
    "EP_UNEMP": "unemployment_pct",
    "EP_NOHSDP": "no_highschool_pct",
    "EP_SNGPNT": "single_parent_pct",
    "EP_UNINSUR": "uninsured_pct",
    "EP_NOVEH": "no_vehicle_pct",
}

RATE_COLUMNS = [name for name in SDOH_RATE_COLUMNS.values() if name != "county"]

# CDC codes unavailable values as -999, which a plain read would treat as a rate.
CDC_MISSING_CODE = -999


def normalize_county_name(raw_name):
    """Drop the trailing "County" so names match the ADHS report.

    The SVI file writes "Santa Cruz County" where the health department writes
    "Santa Cruz", and the two have to agree before a patient can be assigned a
    county and then given that county's rates.
    """
    name = str(raw_name).strip()
    if name.endswith(" County"):
        name = name[: -len(" County")]
    return name


def load_sdoh_rates(csv_path):
    """Return each Arizona county's prevalence of the five determinants, as percentages."""
    raw = pd.read_csv(csv_path)

    missing_columns = set(SDOH_RATE_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"SVI file is missing expected columns: {sorted(missing_columns)}")

    rates = raw[list(SDOH_RATE_COLUMNS)].rename(columns=SDOH_RATE_COLUMNS)
    rates["county"] = rates["county"].apply(normalize_county_name)

    for column in RATE_COLUMNS:
        rates.loc[rates[column] == CDC_MISSING_CODE, column] = pd.NA

    # A missing rate would silently become a missing barrier probability, so this
    # fails loudly rather than generating patients from an incomplete county.
    if rates[RATE_COLUMNS].isna().any().any():
        unusable = rates.loc[rates[RATE_COLUMNS].isna().any(axis=1), "county"].tolist()
        raise ValueError(f"SDOH rates unavailable for: {unusable}")

    return rates.sort_values("county").reset_index(drop=True)


if __name__ == "__main__":
    print(load_sdoh_rates("data/Arizona_county.csv").to_string(index=False))
