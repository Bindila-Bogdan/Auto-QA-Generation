import re
import json
import pandas as pd


class TopicsLoader:
    """
    This class cleans the input data frame of topics
    and embeddings, so that it can be used for further analysis.
    """

    def __init__(self, path_topic_clusters, path_grouped_sections, embeddings_column="section_embedding", centroid_column="centroid"):
        self.topic_clusters = pd.read_csv(path_topic_clusters)
        self.grouped_sections = self.__load_grouped_sections(path_grouped_sections)
        self.embeddings_column = embeddings_column
        self.centroid_column = centroid_column

        # choose if the stored centroids can be used or not
        if "reduced_centroid" in self.topic_clusters.columns:
            self.__clean_topics(True)
        else:
            self.__clean_topics(False)

    def __load_grouped_sections(self, path):
        """
        Loads the sections that are grouped into "pages".
        """

        # load grouped sections
        with open(path) as file:
            grouped_sections = dict(json.load(file))
        return grouped_sections

    def __clean_topics(self, get_centroids):
        """
        It applies data cleaning on the input data frame.
        """

        # convert the string section_embeddings to a list of floats
        self.topic_clusters[self.embeddings_column] = self.topic_clusters[
            self.embeddings_column
        ].map(lambda x: [float(x_) for x_ in x[1:-1].split(", ")])

        self.embeddings_dim = len(self.topic_clusters[self.embeddings_column][0])
        self.emebdding_columns = [
            "dim_" + str(index) for index in range(self.embeddings_dim)
        ]
        self.centroid_columns = [
            "centroid_" + str(index) for index in range(self.embeddings_dim)
        ]

        # expand the columns that contains the newly created lists
        expanded_embeddings = pd.DataFrame(
            self.topic_clusters[self.embeddings_column].to_list(),
            columns=self.emebdding_columns,
        )

        if get_centroids:
            # convert the string centroid to a list of floats
            self.topic_clusters[self.centroid_column] = self.topic_clusters[self.centroid_column].map(
                lambda x: [
                    float(x_)
                    for x_ in re.sub(" +", " ", x[1:-1].strip().replace("\n", " ")).split(
                        " "
                    )
                ]
            )

            # expand centroids
            expanded_centroids = pd.DataFrame(
                self.topic_clusters[self.centroid_column].to_list(),
                columns=self.centroid_columns,
            )

            # remove the centroid column and replace the section_embedding column with the newly extracted columns
            self.topic_clusters = pd.concat(
                [self.topic_clusters, expanded_centroids, expanded_embeddings], axis=1
            )

        else:
            # replace the section_embedding column with the newly extracted columns
            self.topic_clusters = pd.concat(
                [self.topic_clusters, expanded_embeddings], axis=1
            )

            # compute centroids
            name_mapping = dict(zip(self.emebdding_columns, self.centroid_columns))
            expanded_centroids = self.topic_clusters.groupby("cluster_id")[self.emebdding_columns].mean().rename(name_mapping, axis=1).reset_index()

            # add centroids to the topics clusters
            self.topic_clusters = pd.merge(self.topic_clusters, expanded_centroids, on='cluster_id')

        # remove columns that won't be used
        columns_to_remove = [col for col in self.topic_clusters.columns if "embedding" in col] + ["centroid"]
        self.topic_clusters = self.topic_clusters.drop(columns_to_remove, axis=1)

    def get_data(self):
        """
        This methods returns the cleaned data frame and
        other variabled needed during analysis.
        """

        return (
            self.topic_clusters,
            self.grouped_sections,
            self.centroid_columns,
            self.emebdding_columns,
            self.embeddings_dim,
        )
