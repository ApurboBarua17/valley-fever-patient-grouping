"""Group Arizona counties on combined Valley Fever burden and social vulnerability."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Overall SVI is excluded: it is a rank composite of these same four themes, so
# using both would weight vulnerability twice against burden. Burden uses the
# five year average because single year rates on small counts are unstable.
FEATURE_COLUMNS = [
    "svi_socioeconomic",
    "svi_household",
    "svi_minority_language",
    "svi_housing_transport",
    "rate_5yr_avg",
]

CANDIDATE_K_VALUES = [2, 3, 4, 5]

SELECTED_K = 3
RANDOM_SEED = 42


def build_feature_matrix(combined_data):
    """Standardize the features. SVI themes cap at 1, case rates pass 200, so
    without scaling distance would be driven almost entirely by the rate column."""
    raw_features = combined_data[FEATURE_COLUMNS].astype(float).values
    scaler = StandardScaler()
    return scaler.fit_transform(raw_features)


def compare_algorithms(feature_matrix):
    """Score K-means and Ward hierarchical clustering across candidate k values.

    Returns a dataframe with one row per k holding both silhouette scores and
    the adjusted Rand index measuring how far the two partitions agree.
    """
    comparison_rows = []
    for k in CANDIDATE_K_VALUES:
        kmeans_labels = KMeans(
            n_clusters=k, n_init=50, random_state=RANDOM_SEED
        ).fit_predict(feature_matrix)
        ward_labels = AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ).fit_predict(feature_matrix)

        comparison_rows.append(
            {
                "k": k,
                "kmeans_silhouette": silhouette_score(feature_matrix, kmeans_labels),
                "ward_silhouette": silhouette_score(feature_matrix, ward_labels),
                "agreement_ari": adjusted_rand_score(kmeans_labels, ward_labels),
            }
        )

    return pd.DataFrame(comparison_rows)


def _relabel_by_burden(labels, combined_data):
    """Renumber clusters so cluster 1 carries the highest average case rate.

    Both algorithms number clusters arbitrarily; ordering by burden keeps the
    output and legend stable across runs.
    """
    mean_rate_by_label = {}
    for label in set(labels):
        member_rates = combined_data.loc[labels == label, "rate_5yr_avg"]
        mean_rate_by_label[label] = member_rates.mean()

    ordered_labels = sorted(mean_rate_by_label, key=mean_rate_by_label.get, reverse=True)

    new_number_for_label = {}
    for position, label in enumerate(ordered_labels):
        new_number_for_label[label] = position + 1

    renumbered = []
    for label in labels:
        renumbered.append(new_number_for_label[label])
    return renumbered


# Ward at k=3. K-means and Ward return the identical partition here (ARI 1.000),
# so Ward is chosen for being deterministic. Silhouette peaks at k=2, but that
# split collapses the burden axis. See the README for the full comparison.
def fit_final_model(combined_data):
    """Fit the selected clustering model and return the data with cluster labels."""
    feature_matrix = build_feature_matrix(combined_data)
    model = AgglomerativeClustering(n_clusters=SELECTED_K, linkage="ward")
    raw_labels = model.fit_predict(feature_matrix)

    labelled_data = combined_data.copy()
    labelled_data["cluster"] = _relabel_by_burden(raw_labels, combined_data)
    return labelled_data, feature_matrix


def summarize_clusters(labelled_data):
    """Build a per cluster profile of size, average burden and average vulnerability."""
    summary_rows = []
    for cluster_number in sorted(labelled_data["cluster"].unique()):
        members = labelled_data[labelled_data["cluster"] == cluster_number]
        summary_rows.append(
            {
                "cluster": cluster_number,
                "counties": len(members),
                "avg_rate_5yr": members["rate_5yr_avg"].mean(),
                "avg_rate_2023": members["rate_2023"].mean(),
                "avg_svi_overall": members["svi_overall"].mean(),
                "total_cases_2023": int(members["cases"].sum()),
                "county_names": ", ".join(sorted(members["county"])),
            }
        )
    return pd.DataFrame(summary_rows)
