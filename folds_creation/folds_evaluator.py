import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt


sns.set()


class ClustersLinkage:
    """
    This class computes various types of
    distances between all pairs of topic clusters.
    """

    def __init__(self, distance_type):
        self.distance_type = distance_type

    def __single_linkage(self, cluster1, cluster2):
        """
        Computes the single linkage between 2 clusters.
        """

        distance = sys.maxsize

        for point1 in cluster1:
            for point2 in cluster2:
                # update the minimum distance between pairs of points
                distance = min(distance, self.distance_type(point1, point2))
        return distance

    def __average_linkage(self, cluster1, cluster2):
        """
        Computes the distance average linkage between 2 clusters.
        """

        # get centroid of each cluster
        centroid1 = cluster1.mean(axis=0)
        centroid2 = cluster2.mean(axis=0)

        # compute the distance between centroids
        distance = self.distance_type(centroid1, centroid2)

        return distance

    def __ward_linkage(self, cluster1, cluster2):
        """
        Computes the ward linkage between 2 clusters.
        """

        # compute centroid of each cluster
        centroid1 = cluster1.mean(axis=0)
        centroid2 = cluster2.mean(axis=0)

        # combine clusters and get combined centroid
        combined_cluster = np.concatenate([cluster1, cluster2])
        centroid_combined = combined_cluster.mean(axis=0)

        wcss1 = 0
        wcss2 = 0
        wcss_combined = 0

        # compute within-cluster sum-of-squares for each cluster
        wcss1 = np.array(
            [self.distance_type(centroid1, point) ** 2 for point in cluster1]
        ).sum()
        wcss2 = np.array(
            [self.distance_type(centroid2, point) ** 2 for point in cluster2]
        ).sum()
        wcss_combined = np.array(
            [
                self.distance_type(centroid_combined, point) ** 2
                for point in combined_cluster
            ]
        ).sum()

        # calculate the ward linkage value
        ward_linkage_value = wcss_combined - (wcss1 + wcss2)

        return ward_linkage_value

    def find_cluster_dist(self, clusters, linkage_type):
        """
        Computes a specific linkage distance between all pairs of clusters.
        """

        # choose linkage type
        if linkage_type == "single":
            linkage_distance = self.__single_linkage
        elif linkage_type == "average":
            linkage_distance = self.__average_linkage
        elif linkage_type == "ward":
            linkage_distance = self.__ward_linkage
        cluster_distances = []

        for i in tqdm(range(len(clusters))):
            for j in range(len(clusters)):
                if i != j:
                    cluster_distances.append(
                        (i, j, linkage_distance(clusters[i], clusters[j]))
                    )
        return cluster_distances


class FoldsEvaluator:
    """
    This class computes all the evaluation
    metrics needed to analyze the created folds.
    """

    def __init__(self, topic_clusters, emebdding_columns, grouped_sections, folds_no=2):
        self.topic_clusters = topic_clusters
        self.emebdding_columns = emebdding_columns
        self.grouped_sections = grouped_sections
        self.folds_no = folds_no

        self.clusters_linkage = None
        self.linkage_type = None

        self.cluster_distances = None
        self.min_dist_df = None
        self.new_cluster_distances = None
        self.new_min_dist_df = None
        self.new_cluster_sizes = None
        self.mean_folds_per_page = None
        self.valid_sections_no = None

    def compute_plot_min_dist(self, distances, plot=False, title="Min"):
        """
        This method extracts and plots for each cluster
        the minimum distance to any other cluster.
        """

        # get min distance to any other cluster
        dist_df = pd.DataFrame(distances).rename(
            {0: "cluster_1", 1: "cluster_2", 2: "dist"}, axis=1
        )
        min_dist_df = dist_df.groupby("cluster_1").agg({"dist": "min"})

        if plot:
            min_dist_df.plot(kind="hist", bins=100)
            plt.title(f"{title} {self.distance_name} distance to any other cluster\n when linkage type is {self.linkage_type}")
            plt.xlabel("distance")

            plt.show()

        return min_dist_df

    def compute_initial_distances(self, plot=False):
        """
        This method computes the distance between all pairs of clusters.
        """

        # compute distances between all pairs of clusters
        clusters_no = len(set(self.topic_clusters["cluster_id"].values)) - 1

        clusters = [
            self.topic_clusters[self.topic_clusters["cluster_id"] == index][
                self.emebdding_columns
            ].to_numpy()
            for index in range(0, clusters_no)
        ]

        self.cluster_distances = self.clusters_linkage.find_cluster_dist(
            clusters, self.linkage_type
        )
        self.min_dist_df = self.compute_plot_min_dist(self.cluster_distances, plot=plot)

    def compute_distance_between_folds(self, plot=False):
        """
        This computes the distance between all pairs of folds.
        """

        # compute distances between all pairs of new clusters
        new_clusters = [
            self.topic_clusters[self.topic_clusters["new_cluster_id"] == index][
                self.emebdding_columns
            ].to_numpy()
            for index in range(1, self.folds_no + 1)
        ]

        self.new_cluster_distances = self.clusters_linkage.find_cluster_dist(
            new_clusters, self.linkage_type
        )
        self.new_min_dist_df = self.compute_plot_min_dist(
            self.new_cluster_distances, plot=plot
        )

    def compute_correlation(self):
        """
        This method computes the correlation between
        cluster size and the minimum distance to any other cluster.
        """

        # get the percentiles of the number of points per cluster
        elements_per_topic = (
            self.topic_clusters[self.topic_clusters["cluster_id"] != -1]
            .groupby("cluster_id")
            .count()
            .iloc[:, 1]
            .tolist()
        )

        corr_cluster_size_min_dist = np.corrcoef(
            self.min_dist_df["dist"].tolist(), elements_per_topic
        )[0][1]
        print(
            f"Correlation between cluster size and minimum distance to any other cluster: {corr_cluster_size_min_dist}"
        )

    def get_cluster_sizes(self):
        # check the new cluster sizes
        self.new_cluster_sizes = (
            self.topic_clusters[self.topic_clusters["new_cluster_id"] != -1]
            .groupby("new_cluster_id")
            .count()
            .iloc[:, 0]
            .tolist()
        )

    def find_sections(self, section):
        """
        It returns the file name and page number that contain the input section.
        Otherwise it returns two emtpy strings.
        Attention: it assumes that the searched sections are unique.
        """

        # iterate over PDFs
        for file_name, data in self.grouped_sections.items():
            # iterate over grouped sections
            for page_number, grouped_sections_ in data.items():
                if section in grouped_sections_:
                    return file_name, page_number
                
        return "", ""
    
    def get_filenames_pages(self):
        """
        This method returns the ids of clusters associated with
        the sections extracted from each page of every flyer.
        """

        filenames_pages = dict()

        # iterate over PDFs
        for file_name, data in self.grouped_sections.items():
            # iterate over grouped sections
            for page_number, _ in data.items():
                filenames_pages[file_name + " --- " + page_number] = list()

        for new_cluster_id in range(1, self.folds_no + 1):
            cluster_sections = self.topic_clusters[
                self.topic_clusters["new_cluster_id"] == new_cluster_id
            ]["section_text"].tolist()

            for section in cluster_sections:
                file_name, page_number = self.find_sections(
                    section
                )
                filenames_pages[file_name + " --- " + page_number].append(new_cluster_id)

        return filenames_pages        

    def compute_homogeneity(self):
        """
        This method computes the average number of folds per pageand the
        number of sections that appear in pages present in only one fold.
        """

        filenames_pages = self.get_filenames_pages()

        # check the mean number of folds that have sections from the same page
        self.mean_folds_per_page = np.array(
            [
                len(set(clusters_no))
                for _, clusters_no in filenames_pages.items()
                if len(set(clusters_no)) != 0
            ]
        ).mean()

        # find how many section appear in pages that don't have sections in multiple folds
        self.valid_sections_no = sum(
            [
                len(clusters_no)
                for _, clusters_no in filenames_pages.items()
                if len(set(clusters_no)) == 1
            ]
        )

    def evaluate(self, distance_types, linkage_types, plot=False):
        """
        Run all metrics and return them.
        """

        self.get_cluster_sizes()
        self.compute_homogeneity()

        cluster_distances = []
        new_cluster_distances = []
        min_dists = []
        new_min_dists = []

        for index in range(len(distance_types)):
            self.distance_name = distance_types[index].__name__ 
            self.clusters_linkage = ClustersLinkage(distance_types[index])
            self.linkage_type = linkage_types[index]

            self.compute_initial_distances(plot)
            self.compute_distance_between_folds()

            cluster_distances.append(self.cluster_distances)
            new_cluster_distances.append(self.new_cluster_distances)
            min_dists.append(min(self.min_dist_df["dist"].tolist()))
            new_min_dists.append(min(self.new_min_dist_df["dist"].tolist()))

        return (
            cluster_distances,
            min_dists,
            new_cluster_distances,
            new_min_dists,
            self.new_cluster_sizes,
            self.mean_folds_per_page,
            self.valid_sections_no,
        )
