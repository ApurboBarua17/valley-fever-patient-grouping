"""Build a synthetic patient table grounded in real Arizona statistics.

Individual Valley Fever records are protected and not public, so the patients
here are generated. Two things keep them from being arbitrary. Each patient is
assigned to a county in proportion to that county's real 2023 case count, so the
patient population is distributed the way Arizona's actual cases are. Each
patient's social determinants are then drawn at that county's real published
prevalence, so the marginals match Arizona rather than being invented.

The one thing that is genuinely assumed is the link from access barriers to
delayed presentation and worse disease. That direction is well supported in the
care access literature, but the specific numbers below are mine, and no part of
it is estimated from Valley Fever data. Every clinical pattern the clustering
finds is therefore a pattern I put in.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_PATIENT_COUNT = 1200
RANDOM_SEED = 42

# How strongly a patient's barriers move together. Chosen to produce a realistic
# spread of compounding disadvantage rather than fitted to anything.
SDOH_CORRELATION = 0.55

# The five determinants named in the brief. Access to care is measured twice,
# once as insurance and once as transport, because they are separate barriers
# that the same patient can face independently.
SDOH_FEATURES = [
    "housing_cost_burdened",
    "unemployed",
    "no_highschool_diploma",
    "limited_social_support",
    "uninsured",
    "no_vehicle",
]

SYMPTOM_FEATURES = [
    "days_to_diagnosis",
    "cough_weeks",
    "fatigue_score",
    "fever",
    "chest_pain",
    "weight_loss_kg",
    "disseminated",
]

# Which county prevalence drives which patient level determinant.
SDOH_SOURCE_RATES = {
    "housing_cost_burdened": "housing_cost_burden_pct",
    "unemployed": "unemployment_pct",
    "no_highschool_diploma": "no_highschool_pct",
    "limited_social_support": "single_parent_pct",
    "uninsured": "uninsured_pct",
    "no_vehicle": "no_vehicle_pct",
}

# Cutoffs separating the three latent patient types, so the synthetic data has a
# known answer to check the clustering against. Set at one and two because that
# is where this population actually divides: most patients face no barrier at
# all, and facing two or more compounding barriers is both meaningful and common
# enough to form a group. Cutting at four instead put 2 percent of patients in
# the group that matters most, which is not a group, it is a rounding error.
MODERATE_BARRIER_CUTOFF = 1
HIGH_BARRIER_CUTOFF = 2

BARRIER_GROUP_NAMES = {
    0: "no barriers",
    1: "single barrier",
    2: "multiple barriers",
}

# Symptom parameters by latent group, in order: low, moderate, high barrier.
# The progression encodes the assumption stated at the top of this file, that
# patients facing more barriers present later and sicker.
MEAN_DAYS_TO_DIAGNOSIS = [21.0, 48.0, 82.0]
MEAN_COUGH_WEEKS = [2.5, 5.0, 8.5]
MEAN_FATIGUE_SCORE = [3.0, 5.5, 7.5]
PROBABILITY_FEVER = [0.35, 0.55, 0.72]
PROBABILITY_CHEST_PAIN = [0.20, 0.40, 0.62]
MEAN_WEIGHT_LOSS_KG = [1.0, 3.0, 6.0]
PROBABILITY_DISSEMINATED = [0.01, 0.035, 0.09]


def _assign_counties(case_data, patient_count, generator):
    """Draw a county for each patient in proportion to real 2023 case counts."""
    case_share = case_data["cases"] / case_data["cases"].sum()
    return generator.choice(case_data["county"].values, size=patient_count, p=case_share.values)


def _draw_sdoh_indicators(counties, sdoh_rates, generator):
    """Draw each determinant at the patient's own county prevalence.

    The determinants are drawn correlated rather than independently, through a
    shared per patient disadvantage term. Independent draws give each patient
    about 0.65 barriers on average and make four or more effectively impossible,
    which erases the group that matters most. Real barriers co-occur in the same
    person: someone unemployed is likelier to be uninsured. The correlation
    leaves each county's marginal prevalence unchanged and only changes how the
    barriers stack up within a patient.
    """
    rates_by_county = sdoh_rates.set_index("county")
    patient_count = len(counties)

    # Gaussian copula. A shared latent term moves all of a patient's barrier
    # probabilities together, the independent term keeps them from being
    # identical, and the threshold is set from the county rate so the marginal
    # still comes out at the published prevalence.
    shared_disadvantage = generator.standard_normal(patient_count)
    independent_weight = np.sqrt(1.0 - SDOH_CORRELATION**2)

    indicators = {}
    for feature_name in SDOH_FEATURES:
        rate_column = SDOH_SOURCE_RATES[feature_name]
        probabilities = rates_by_county.loc[counties, rate_column].values / 100.0

        latent = (
            SDOH_CORRELATION * shared_disadvantage
            + independent_weight * generator.standard_normal(patient_count)
        )
        thresholds = norm.ppf(probabilities)
        indicators[feature_name] = (latent < thresholds).astype(int)

    return indicators


def _assign_barrier_groups(barrier_counts):
    """Bucket patients into three latent groups by how many barriers they carry."""
    groups = []
    for count in barrier_counts:
        if count >= HIGH_BARRIER_CUTOFF:
            groups.append(2)
        elif count >= MODERATE_BARRIER_CUTOFF:
            groups.append(1)
        else:
            groups.append(0)
    return np.array(groups)


def _draw_symptoms(barrier_groups, generator):
    """Draw symptom severity conditional on a patient's latent group."""
    patient_count = len(barrier_groups)

    days = generator.normal(
        loc=np.array(MEAN_DAYS_TO_DIAGNOSIS)[barrier_groups], scale=12.0, size=patient_count
    )
    cough = generator.normal(
        loc=np.array(MEAN_COUGH_WEEKS)[barrier_groups], scale=1.8, size=patient_count
    )
    fatigue = generator.normal(
        loc=np.array(MEAN_FATIGUE_SCORE)[barrier_groups], scale=1.5, size=patient_count
    )
    weight_loss = generator.normal(
        loc=np.array(MEAN_WEIGHT_LOSS_KG)[barrier_groups], scale=1.5, size=patient_count
    )

    fever = generator.random(patient_count) < np.array(PROBABILITY_FEVER)[barrier_groups]
    chest_pain = generator.random(patient_count) < np.array(PROBABILITY_CHEST_PAIN)[barrier_groups]
    disseminated = (
        generator.random(patient_count) < np.array(PROBABILITY_DISSEMINATED)[barrier_groups]
    )

    return {
        # Clipped at clinically sensible floors so noise cannot produce a
        # negative duration or a symptom score outside its scale.
        "days_to_diagnosis": np.clip(days, 3, None).round(0),
        "cough_weeks": np.clip(cough, 0, None).round(1),
        "fatigue_score": np.clip(fatigue, 0, 10).round(1),
        "weight_loss_kg": np.clip(weight_loss, 0, None).round(1),
        "fever": fever.astype(int),
        "chest_pain": chest_pain.astype(int),
        "disseminated": disseminated.astype(int),
    }


def generate_patients(case_data, sdoh_rates, patient_count=DEFAULT_PATIENT_COUNT, seed=RANDOM_SEED):
    """Return a synthetic patient table with symptoms and social determinants.

    The true_group column is the latent type each patient was generated from. It
    is deliberately not a clustering feature. It exists so the evaluation can
    measure whether the algorithm recovers a structure that is known to be there,
    which is the one check real patient data could never support.
    """
    generator = np.random.default_rng(seed)

    counties = _assign_counties(case_data, patient_count, generator)
    sdoh_indicators = _draw_sdoh_indicators(counties, sdoh_rates, generator)

    barrier_counts = np.zeros(patient_count, dtype=int)
    for feature_name in SDOH_FEATURES:
        barrier_counts = barrier_counts + sdoh_indicators[feature_name]

    barrier_groups = _assign_barrier_groups(barrier_counts)
    symptoms = _draw_symptoms(barrier_groups, generator)

    patients = pd.DataFrame({"patient_id": np.arange(1, patient_count + 1), "county": counties})
    for feature_name in SDOH_FEATURES:
        patients[feature_name] = sdoh_indicators[feature_name]
    for feature_name in SYMPTOM_FEATURES:
        patients[feature_name] = symptoms[feature_name]

    patients["barrier_count"] = barrier_counts
    patients["true_group"] = barrier_groups
    return patients
