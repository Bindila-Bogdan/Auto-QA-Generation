import scipy.cluster.hierarchy as hierarchical_clustering

import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


class CentroidsClustering:
    """
    This class clusters the centroids of the
    topicsusing the hierarchical clustering.
    """

    def __init__(self, topic_clusters, centroid_columns, new_clusters_no):
        self.linkage = None
        self.topic_clusters = topic_clusters
        self.centroid_columns = centroid_columns
        self.new_clusters_no = new_clusters_no

    def cluster_data(self, method="ward", metric="euclidean"):
        """
        This method applied hierarchial clustering on the clusters' centroids and
        retruns a dataset that has a new column that contains the indices of the nwewly created folds.
        """

        # create a copy of the input data frame
        topic_clusters_copy = self.topic_clusters.copy(deep=True)

        # get centroids of all clusters, except of the outliers
        centroids = topic_clusters_copy[topic_clusters_copy["cluster_id"] != -1][
            ["cluster_id"] + self.centroid_columns
        ].drop_duplicates()

        # compute the linkage matrix
        self.linkage = hierarchical_clustering.linkage(
            centroids[self.centroid_columns], method=method, metric=metric
        )

        # get old cluster ids
        centroid_ids = centroids["cluster_id"].tolist()

        # create clusters and get new cluster ids
        clustered_centroid_ids = hierarchical_clustering.fcluster(
            self.linkage, t=self.new_clusters_no, criterion="maxclust"
        ).tolist()

        # map old cluster ids to the new cluster ids
        cluster_id_mapping = dict(zip(centroid_ids, clustered_centroid_ids))
        cluster_id_mapping[-1] = -1

        topic_clusters_copy["new_cluster_id"] = topic_clusters_copy[
            "cluster_id"
        ].tolist()
        topic_clusters_copy["new_cluster_id"] = topic_clusters_copy[
            "new_cluster_id"
        ].map(lambda x: cluster_id_mapping[x])

        return topic_clusters_copy

    def plot_dendrogram(self):
        """
        This methods plots the dendrogram using the linkage matrix.
        """

        if self.linkage is not None:
            plt.figure(figsize=(16, 6))

            # plot dendrogram
            hierarchical_clustering.dendrogram(self.linkage)

            plt.title("Dendrogram")
            plt.xlabel("Samples")
            plt.ylabel("Distances")
            plt.show()
