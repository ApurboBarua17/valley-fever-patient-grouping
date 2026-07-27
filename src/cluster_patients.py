"""Group synthetic patients on symptoms and social determinants."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.generate_patients import SDOH_FEATURES, SYMPTOM_FEATURES

# Both halves of what the brief describes. barrier_count and true_group are left
# out on purpose: the first is a sum of features already present, and the second
# is the answer the clustering is supposed to find on its own.
PATIENT_FEATURES = SYMPTOM_FEATURES + SDOH_FEATURES

CANDIDATE_K_VALUES = [2, 3, 4, 5, 6]
RANDOM_SEED = 42


def build_patient_matrix(patients):
    """Standardize patient features so binary and continuous ones weigh evenly.

    Days to diagnosis runs into the hundreds while the determinants are 0 or 1.
    Unscaled, distance would be almost entirely the diagnosis delay.
    """
    return StandardScaler().fit_transform(patients[PATIENT_FEATURES].astype(float).values)


def compare_algorithms(patient_matrix, true_groups):
    """Score K-means and Ward across candidate k.

    Reports silhouette, which is available on real data, alongside recovery of
    the known groups, which is only available because the data is synthetic.
    """
    rows = []
    for k in CANDIDATE_K_VALUES:
        kmeans_labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit_predict(
            patient_matrix
        )
        ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
            patient_matrix
        )

        rows.append(
            {
                "k": k,
                "kmeans_silhouette": silhouette_score(patient_matrix, kmeans_labels),
                "ward_silhouette": silhouette_score(patient_matrix, ward_labels),
                "kmeans_recovery_ari": adjusted_rand_score(true_groups, kmeans_labels),
                "ward_recovery_ari": adjusted_rand_score(true_groups, ward_labels),
            }
        )

    return pd.DataFrame(rows)


def _relabel_by_severity(labels, patients):
    """Renumber clusters so cluster 1 is the most severely affected."""
    mean_delay_by_label = {}
    for label in set(labels):
        mean_delay_by_label[label] = patients.loc[labels == label, "days_to_diagnosis"].mean()

    ordered = sorted(mean_delay_by_label, key=mean_delay_by_label.get, reverse=True)

    new_number = {}
    for position, label in enumerate(ordered):
        new_number[label] = position + 1

    return [new_number[label] for label in labels]


# K-means at k=3, chosen on recovery of the known groups rather than on
# silhouette. Silhouette prefers Ward, and prefers k=2, and is wrong on both
# counts here. compare_algorithms prints the figures on every run.
SELECTED_K = 3


def fit_final_model(patients):
    """Fit the selected model and return the patients with cluster labels."""
    patient_matrix = build_patient_matrix(patients)
    labels = KMeans(n_clusters=SELECTED_K, n_init=20, random_state=RANDOM_SEED).fit_predict(
        patient_matrix
    )

    labelled = patients.copy()
    labelled["cluster"] = _relabel_by_severity(labels, patients)
    return labelled, patient_matrix


def summarize_clusters(labelled_patients):
    """Build a per cluster clinical and social profile."""
    rows = []
    for cluster_number in sorted(labelled_patients["cluster"].unique()):
        members = labelled_patients[labelled_patients["cluster"] == cluster_number]
        rows.append(
            {
                "cluster": cluster_number,
                "patients": len(members),
                "days_to_diagnosis": members["days_to_diagnosis"].mean(),
                "fatigue_score": members["fatigue_score"].mean(),
                "cough_weeks": members["cough_weeks"].mean(),
                "disseminated_pct": 100 * members["disseminated"].mean(),
                "barriers": members["barrier_count"].mean(),
                "uninsured_pct": 100 * members["uninsured"].mean(),
                "unemployed_pct": 100 * members["unemployed"].mean(),
            }
        )
    return pd.DataFrame(rows)
