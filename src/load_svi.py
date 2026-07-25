"""Load the CDC/ATSDR Social Vulnerability Index county file for Arizona."""

import pandas as pd

# RPL_THEMES is the overall vulnerability percentile. THEME1 through THEME4 are
# its four components, in the order CDC documents them.
SVI_COLUMNS = {
    "COUNTY": "county",
    "E_TOTPOP": "population",
    "RPL_THEMES": "svi_overall",
    "RPL_THEME1": "svi_socioeconomic",
    "RPL_THEME2": "svi_household",
    "RPL_THEME3": "svi_minority_language",
    "RPL_THEME4": "svi_housing_transport",
}

SCORE_COLUMNS = [
    "svi_overall",
    "svi_socioeconomic",
    "svi_household",
    "svi_minority_language",
    "svi_housing_transport",
]

# CDC codes unavailable values as -999, which a plain read would treat as a score.
CDC_MISSING_CODE = -999


def load_svi_data(csv_path):
    """Return one row per Arizona county with the overall and four theme scores.

    Scores are percentile ranks from 0 to 1, ranked within Arizona rather than
    nationally because this is the state level file.
    """
    raw = pd.read_csv(csv_path)

    missing_columns = set(SVI_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"SVI file is missing expected columns: {sorted(missing_columns)}")

    svi_data = raw[list(SVI_COLUMNS)].rename(columns=SVI_COLUMNS)

    for column in SCORE_COLUMNS:
        svi_data.loc[svi_data[column] == CDC_MISSING_CODE, column] = pd.NA

    if svi_data[SCORE_COLUMNS].isna().any().any():
        unusable = svi_data.loc[svi_data[SCORE_COLUMNS].isna().any(axis=1), "county"].tolist()
        raise ValueError(f"SVI scores unavailable for: {unusable}")

    return svi_data.sort_values("county").reset_index(drop=True)


if __name__ == "__main__":
    print(load_svi_data("data/Arizona_county.csv").to_string(index=False))
