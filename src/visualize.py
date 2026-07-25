"""Two panel scatter of the county clusters, saved as a PNG."""

import matplotlib

# Render without a display so the script works the same from a plain terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Checked for separation under deuteranopia and tritanopia rather than picked by
# eye. Every point is also labelled, so identity never depends on color alone.
CLUSTER_COLORS = {
    1: "#2a78d6",
    2: "#eb6834",
    3: "#1baf7a",
}

CLUSTER_LABELS = {
    1: "Cluster 1: high burden",
    2: "Cluster 2: moderate burden",
    3: "Cluster 3: high vulnerability, lower burden",
}

POINT_SIZE = 130
LABEL_OFFSET = 0.045

# Label footprint as a fraction of each axis, used for collision detection.
LABEL_WIDTH_FRACTION = 0.12
LABEL_HEIGHT_FRACTION = 0.06


def _place_labels_without_overlap(x_values, y_values, x_span, y_span):
    """Put each label above its point, or below when that slot is already taken."""
    order = sorted(range(len(x_values)), key=lambda index: x_values[index])

    placements = {}
    occupied_positions = []
    for index in order:
        above_y = y_values[index] + y_span * LABEL_OFFSET
        below_y = y_values[index] - y_span * (LABEL_OFFSET + 0.025)

        collides = False
        for taken_x, taken_y in occupied_positions:
            close_in_x = abs(taken_x - x_values[index]) < x_span * LABEL_WIDTH_FRACTION
            close_in_y = abs(taken_y - above_y) < y_span * LABEL_HEIGHT_FRACTION
            if close_in_x and close_in_y:
                collides = True
                break

        chosen_y = below_y if collides else above_y
        placements[index] = chosen_y
        occupied_positions.append((x_values[index], chosen_y))

    return placements


def _draw_scatter(axis, x_values, y_values, labelled_data, x_label, y_label, title):
    """Plot one panel: points colored by cluster, each labelled with its county."""
    for cluster_number in sorted(labelled_data["cluster"].unique()):
        member_mask = labelled_data["cluster"] == cluster_number
        axis.scatter(
            x_values[member_mask],
            y_values[member_mask],
            s=POINT_SIZE,
            c=CLUSTER_COLORS[cluster_number],
            label=CLUSTER_LABELS[cluster_number],
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )

    x_span = x_values.max() - x_values.min()
    y_span = y_values.max() - y_values.min()
    label_y_positions = _place_labels_without_overlap(x_values, y_values, x_span, y_span)
    for position in range(len(labelled_data)):
        axis.annotate(
            labelled_data["county"].iloc[position],
            (x_values[position], y_values[position]),
            xytext=(x_values[position], label_y_positions[position]),
            fontsize=8,
            ha="center",
            color="#3d3d3a",
            zorder=4,
        )

    axis.set_xlabel(x_label, fontsize=9)
    axis.set_ylabel(y_label, fontsize=9)
    axis.set_title(title, fontsize=10, pad=10)
    axis.grid(True, color="#e6e6e2", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_color("#c9c9c4")

    axis.set_xlim(x_values.min() - x_span * 0.12, x_values.max() + x_span * 0.12)
    axis.set_ylim(y_values.min() - y_span * 0.10, y_values.max() + y_span * 0.14)


def plot_clusters(labelled_data, feature_matrix, output_path):
    """Save a PNG of the clusters in PCA space and in the two source measures.

    PCA shows how the model separated counties across all five features. The
    second panel plots vulnerability against burden directly, which is what a
    public health reader can actually act on.
    """
    projected = PCA(n_components=2, random_state=42).fit(feature_matrix)
    coordinates = projected.transform(feature_matrix)
    explained = projected.explained_variance_ratio_

    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(14, 6.5))

    _draw_scatter(
        left_axis,
        coordinates[:, 0],
        coordinates[:, 1],
        labelled_data,
        f"Component 1 ({explained[0]:.0%} of variance)",
        f"Component 2 ({explained[1]:.0%} of variance)",
        "Clusters in PCA space (all five features)",
    )

    _draw_scatter(
        right_axis,
        labelled_data["svi_overall"].values,
        labelled_data["rate_5yr_avg"].values,
        labelled_data,
        "Overall SVI percentile (higher is more vulnerable)",
        "Valley Fever cases per 100,000, 2018-2022 average",
        "Clusters by vulnerability and burden",
    )

    handles, labels = left_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    figure.suptitle(
        "Arizona counties grouped by Valley Fever burden and social vulnerability",
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0.06, 1, 0.97])
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)

    return output_path
