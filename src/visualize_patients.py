"""Two panel view of the patient clusters, saved as a PNG."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Same colorblind safe hues as the county plot, checked for separation under
# deuteranopia and tritanopia rather than picked by eye.
CLUSTER_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

CLUSTER_LABELS = {
    1: "Cluster 1: latest presentation",
    2: "Cluster 2: intermediate",
    3: "Cluster 3: earliest presentation",
}


def _style_axis(axis, x_label, y_label, title):
    """Apply the shared axis styling."""
    axis.set_xlabel(x_label, fontsize=9)
    axis.set_ylabel(y_label, fontsize=9)
    axis.set_title(title, fontsize=10, pad=10)
    axis.grid(True, color="#e6e6e2", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#c9c9c4")


def plot_patient_clusters(labelled_patients, patient_matrix, output_path):
    """Save a PNG showing the patient clusters in PCA space and by the two axes that matter."""
    projection = PCA(n_components=2, random_state=42).fit(patient_matrix)
    coordinates = projection.transform(patient_matrix)
    explained = projection.explained_variance_ratio_

    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(14, 6))

    for cluster_number in sorted(labelled_patients["cluster"].unique()):
        member_mask = (labelled_patients["cluster"] == cluster_number).values
        colour = CLUSTER_COLORS[cluster_number]

        left_axis.scatter(
            coordinates[member_mask, 0],
            coordinates[member_mask, 1],
            s=14,
            c=colour,
            alpha=0.6,
            linewidths=0,
            label=CLUSTER_LABELS[cluster_number],
            zorder=3,
        )

        members = labelled_patients[member_mask]
        # Jittered because barrier count is a small integer and the points would
        # otherwise stack into six vertical lines and hide the distribution.
        jitter = (members.index.values % 17) / 17.0 - 0.5
        right_axis.scatter(
            members["barrier_count"] + jitter * 0.55,
            members["days_to_diagnosis"],
            s=14,
            c=colour,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )

    _style_axis(
        left_axis,
        f"Component 1 ({explained[0]:.0%} of variance)",
        f"Component 2 ({explained[1]:.0%} of variance)",
        "Patient clusters in PCA space (13 features)",
    )
    _style_axis(
        right_axis,
        "Number of social barriers faced",
        "Days from symptom onset to diagnosis",
        "Clusters by barrier burden and delay to diagnosis",
    )

    handles, labels = left_axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.01), markerscale=2.0,
    )
    figure.suptitle(
        "Synthetic Valley Fever patients grouped by symptoms and social determinants",
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0.07, 1, 0.96])
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)

    return output_path
