"""Patient grouping: generate synthetic patients, cluster them, evaluate the result."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.cluster_patients import (
    PATIENT_FEATURES,
    SELECTED_K,
    build_patient_matrix,
    compare_algorithms,
    fit_final_model,
    summarize_clusters,
)
from src.generate_patients import (
    BARRIER_GROUP_NAMES,
    DEFAULT_PATIENT_COUNT,
    SDOH_FEATURES,
    SYMPTOM_FEATURES,
    generate_patients,
)
from src.load_cases import load_case_data
from src.load_svi import load_sdoh_rates
from src.visualize_patients import plot_patient_clusters

PROJECT_ROOT = Path(__file__).parent
CASE_PDF = PROJECT_ROOT / "data" / "valley-fever-2023.pdf"
SVI_CSV = PROJECT_ROOT / "data" / "Arizona_county.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def print_heading(text):
    """Print a section heading."""
    print(f"\n--- {text} " + "-" * max(0, 88 - len(text)))


def main():
    """Generate patients, group them, and report how good the grouping is."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    print_heading("STEP 1  Ground the generator in real Arizona data")
    case_data = load_case_data(CASE_PDF)
    sdoh_rates = load_sdoh_rates(SVI_CSV)
    print(f"Case counts for {len(case_data)} counties set how many patients come from each.")
    print("Published prevalences set how likely each patient is to face each barrier.")

    print_heading(f"STEP 2  Generate {DEFAULT_PATIENT_COUNT} synthetic patients")
    patients = generate_patients(case_data, sdoh_rates)
    print(f"Symptoms: {', '.join(SYMPTOM_FEATURES)}")
    print(f"Determinants: {', '.join(SDOH_FEATURES)}")
    print()
    print("Latent groups built into the data (not given to the clustering):")
    for group_number, count in patients["true_group"].value_counts().sort_index().items():
        share = 100 * count / len(patients)
        print(f"  {BARRIER_GROUP_NAMES[group_number]:<20} {count:>5}  ({share:4.1f}%)")

    print_heading("STEP 3  Compare algorithms")
    patient_matrix = build_patient_matrix(patients)
    comparison = compare_algorithms(patient_matrix, patients["true_group"])
    print(comparison.round(3).to_string(index=False))
    print()
    print("Silhouette peaks at k=2, where recovery of the true groups collapses to 0.43.")
    print("K-means at k=3 recovers them at 0.96. Internal fit and correctness disagree.")

    print_heading(f"STEP 4  Cluster assignments (K-means, k={SELECTED_K})")
    labelled_patients, patient_matrix = fit_final_model(patients)
    print(f"Clustered {len(labelled_patients)} patients on {len(PATIENT_FEATURES)} features")
    print()
    print(labelled_patients.head(8)[
        ["patient_id", "county", "days_to_diagnosis", "fatigue_score",
         "barrier_count", "uninsured", "cluster"]
    ].to_string(index=False))
    print("  ... first 8 of", len(labelled_patients))

    print_heading("STEP 5  Cluster profiles")
    for row in summarize_clusters(labelled_patients).itertuples(index=False):
        print(
            f"Cluster {row.cluster}  {row.patients:>4} patients | "
            f"{row.days_to_diagnosis:5.1f} days to diagnosis | "
            f"fatigue {row.fatigue_score:.1f}/10 | cough {row.cough_weeks:.1f}w | "
            f"disseminated {row.disseminated_pct:.1f}%"
        )
        print(
            f"           barriers {row.barriers:.2f} avg | "
            f"uninsured {row.uninsured_pct:.0f}% | unemployed {row.unemployed_pct:.0f}%"
        )

    print_heading("STEP 6  Visualization")
    plot_path = plot_patient_clusters(
        labelled_patients, patient_matrix, RESULTS_DIR / "patient_clusters.png"
    )
    print(f"Saved {plot_path.relative_to(PROJECT_ROOT)}")

    print_heading("STEP 7  Evaluation")
    silhouette = silhouette_score(patient_matrix, labelled_patients["cluster"])
    recovery = adjusted_rand_score(labelled_patients["true_group"], labelled_patients["cluster"])
    print(f"Silhouette (internal fit):        {silhouette:.3f}")
    print(f"Adjusted Rand (recovery of truth): {recovery:.3f}")
    print()
    print(
        "Silhouette says the clusters are loose. Recovery says they are almost exactly\n"
        "the groups the data was built from. Both are true: the groups sit on a gradient\n"
        "rather than in separate lumps, so they overlap at the edges while still being\n"
        "the right groups. On real patient data only the first number is available, which\n"
        "is the argument for not treating silhouette as a verdict.\n\n"
        "Usefulness is the separate question. The grouping is actionable because the\n"
        "clusters differ on things a clinic can act on: the latest presenting cluster\n"
        "waits about three times as long for a diagnosis and carries most of the\n"
        "disseminated disease. Targeting outreach at patients with two or more barriers\n"
        "would reach that group. The caveat is that the link from barriers to delay was\n"
        "an assumption I built in, so this shows the method works, not that Arizona\n"
        "patients behave this way."
    )

    labelled_patients.to_csv(RESULTS_DIR / "patient_clusters.csv", index=False)
    print(f"\nSaved results/patient_clusters.csv\n")


if __name__ == "__main__":
    main()
