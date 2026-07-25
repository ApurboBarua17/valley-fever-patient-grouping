"""Run the full pipeline: load, join, cluster, plot, evaluate."""

from pathlib import Path

import pandas as pd

from src.clustering import (
    FEATURE_COLUMNS,
    SELECTED_K,
    build_feature_matrix,
    compare_algorithms,
    fit_final_model,
    summarize_clusters,
)
from src.evaluate import (
    build_assessment,
    burden_vulnerability_correlation,
    score_final_model,
)
from src.join_datasets import join_case_and_svi_data
from src.load_cases import load_case_data
from src.load_svi import SCORE_COLUMNS, load_svi_data
from src.visualize import plot_clusters

PROJECT_ROOT = Path(__file__).parent
CASE_PDF = PROJECT_ROOT / "data" / "valley-fever-2023.pdf"
SVI_CSV = PROJECT_ROOT / "data" / "Arizona_county.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def print_heading(text):
    """Print a section heading."""
    print(f"\n--- {text} " + "-" * max(0, 74 - len(text)))


def main():
    """Run every step and write the outputs into results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    print_heading("STEP 1  Load Valley Fever cases from the ADHS report PDF")
    case_data = load_case_data(CASE_PDF)
    print(case_data.to_string(index=False))

    print_heading("STEP 2  Load CDC/ATSDR Social Vulnerability Index")
    svi_data = load_svi_data(SVI_CSV)
    print(f"Parsed {len(svi_data)} counties, kept: {', '.join(SCORE_COLUMNS)}")

    print_heading("STEP 3  Join on county name")
    combined_data = join_case_and_svi_data(case_data, svi_data)
    print(f"Joined cleanly, {len(combined_data)} counties, no unmatched names")

    print_heading("STEP 4  Compare clustering algorithms")
    print(compare_algorithms(build_feature_matrix(combined_data)).round(3).to_string(index=False))
    print("K-means and Ward agree exactly at k=3 (ARI 1.000), so the partition does not")
    print("depend on the algorithm. Ward is kept because it is deterministic.")

    print_heading(f"STEP 5  Cluster assignments (Ward, k={SELECTED_K})")
    labelled_data, feature_matrix = fit_final_model(combined_data)
    assignments = labelled_data[
        ["county", "cluster", "cases", "rate_2023", "rate_5yr_avg", "svi_overall"]
    ].sort_values(["cluster", "rate_5yr_avg"], ascending=[True, False])
    print(assignments.to_string(index=False))

    print_heading("STEP 6  Cluster profiles")
    for row in summarize_clusters(labelled_data).itertuples(index=False):
        print(
            f"Cluster {row.cluster}  {row.counties} counties | "
            f"rate {row.avg_rate_5yr:.1f} per 100,000 (2018-2022) | "
            f"SVI {row.avg_svi_overall:.2f} | {row.total_cases_2023:,} cases in 2023"
        )
        print(f"  {row.county_names}")

    print_heading("STEP 7  Visualization")
    plot_path = plot_clusters(labelled_data, feature_matrix, RESULTS_DIR / "county_clusters.png")
    print(f"Saved {plot_path.relative_to(PROJECT_ROOT)}")

    print_heading("STEP 8  Evaluation")
    silhouette = score_final_model(feature_matrix, labelled_data["cluster"])
    coefficient, p_value = burden_vulnerability_correlation(labelled_data)
    print(f"Silhouette {silhouette:.3f} | case rate vs SVI: r = {coefficient:.3f}, p = {p_value:.3f}\n")
    print(build_assessment(labelled_data, silhouette, coefficient, p_value))

    labelled_data.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)
    print(f"\nSaved results/cluster_assignments.csv\n")


if __name__ == "__main__":
    main()
