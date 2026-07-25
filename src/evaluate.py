"""Score the clustering and judge whether the groups mean anything in practice."""

from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

# On continuous social data: above 0.5 is strong, 0.25 to 0.5 is real but loose.
STRONG_STRUCTURE_THRESHOLD = 0.50
WEAK_STRUCTURE_THRESHOLD = 0.25


def score_final_model(feature_matrix, cluster_labels):
    """Silhouette score for the fitted clustering."""
    return silhouette_score(feature_matrix, cluster_labels)


def burden_vulnerability_correlation(labelled_data):
    """Correlate case burden against overall vulnerability across the 15 counties.

    This is the check that matters. If grouping on vulnerability were going to
    find the counties carrying the most disease, the two would move together.
    """
    return pearsonr(labelled_data["rate_5yr_avg"], labelled_data["svi_overall"])


def describe_structure_strength(silhouette):
    """Plain language reading of a silhouette score."""
    if silhouette >= STRONG_STRUCTURE_THRESHOLD:
        return "well separated"
    if silhouette >= WEAK_STRUCTURE_THRESHOLD:
        return "real but loosely separated"
    return "weak enough to treat as arbitrary"


def build_assessment(labelled_data, silhouette, coefficient, p_value):
    """Write the public health reading of the clusters.

    Kept honest on purpose. The result here is a negative one, and reporting it
    as a success would misrepresent what the data shows.
    """
    highest_burden = sorted(labelled_data.loc[labelled_data["cluster"] == 1, "county"])
    most_vulnerable = sorted(labelled_data.nlargest(4, "svi_overall")["county"])
    overlap = sorted(set(highest_burden) & set(most_vulnerable))

    return (
        f"The clusters are {describe_structure_strength(silhouette)} "
        f"(silhouette {silhouette:.3f}), which is about what 15 units of continuously "
        "varying social data should give.\n"
        f"They separate counties on both axes, but not the way the usual framing assumes: "
        f"the highest burden cluster ({', '.join(highest_burden)}) and the four most "
        f"vulnerable counties ({', '.join(most_vulnerable)}) overlap in "
        f"{', '.join(overlap) if overlap else 'no counties at all'}, and across all 15 the "
        f"correlation is {coefficient:.2f} (p = {p_value:.2f}).\n"
        "So the grouping is useful for keeping burden and vulnerability visible as separate "
        "problems, but a program targeting either one alone would miss the other, because "
        "Valley Fever here tracks the dry desert corridor more than it tracks disadvantage."
    )
