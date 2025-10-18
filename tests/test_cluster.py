from __future__ import annotations

import numpy as np

from src import cluster


def test_dbscan_clusters_groups_similar_vectors() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.03, 0.97],
        ]
    )

    labels = cluster.dbscan_clusters(embeddings, eps=0.1, min_samples=2)

    first_group = labels[:2]
    second_group = labels[2:]

    assert first_group[0] == first_group[1] != -1
    assert second_group[0] == second_group[1] != -1
    assert first_group[0] != second_group[0]
