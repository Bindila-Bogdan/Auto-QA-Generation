import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import math

### Visualization in 2 dimensions

def two_d_visualization(embeddings_list):

    perplexity = [10,20,30,40]
    X = np.array(embeddings_list)
    color = ["blue", "red", "green", "yellow"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Iterar sobre cada subgráfica
    for i, ax in enumerate(axs.flat):

        counter = 1
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random', perplexity=perplexity[i], verbose=3, n_jobs=6).fit_transform(X)
        
        # Let's compute the average distance between points taking around 100 of pairs of them
        added_distance = 0
        print(f"\nCalculating average distance for set number {counter}\n")
        for j in range(2000): 
            random_2 = np.random.choice(range(100), size=2, replace=False)
            added_distance += np.linalg.norm(X_embedded[random_2[0],:] - X_embedded[random_2[1],:])
        av_distance = added_distance/2000

        ax.scatter(X_embedded[:,0], X_embedded[:,1], color=color[i])
        ax.set_title(f'Perplexity: {perplexity[i]}; Distance: {round(av_distance,2)}')
        counter += 1


    plt.suptitle('Representations of embeddings for different perplexities', fontsize=16)
    plt.show(block = False)

    compute_elbow(embeddings_list)

### Compute elbow diagram of k-means

def compute_elbow(embeddings_list):

    # First perform again the dimensionality reduction
    X = np.array(embeddings_list)
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random', perplexity=10, verbose=3, n_jobs=6).fit_transform(X)
    inertia = []
    for k in range(1, 101):  # Prueba con k de 1 a 20
        print("Running clustering number:", k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6), num=2)
    plt.plot(range(1, 101), inertia, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow diagram')
    plt.show()

def reduce_dimensionality(embeddings_list, n_components=50, svd_solver="auto"):

    X = np.array(embeddings_list)
    pca = PCA(n_components=n_components, svd_solver="auto")
    X = pca.fit_transform(X)
    explained_covariance = pca.explained_variance_ratio_
    explained_covariance = np.sum(explained_covariance)
    return X, explained_covariance

def hierarchical_clustering(embeddings_list, distance_threshold=0, n_clusters=None, metric="euclidean", linkage="ward", compute_full_tree=True):

    X = np.array(embeddings_list)
    model = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=n_clusters, 
        metric=metric,
        linkage=linkage,
        compute_distances=True,
        compute_full_tree="auto"
    )
    
    model = model.fit(X)

    return model

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

embeddings_path = "flyers_embeddings_complete.json"

embeddings_id = []
embeddings_list = []
title_id = []

print("Loading embeddings...")
with open(embeddings_path, 'r') as file:
    json_file = json.load(file)
print("Embeddings loaded")

# print("Número de archivos:", len(json_file.keys())) # Check the number of files
# test_file_1 = json_file["(Langdurig) gebroken vliezen"] # Check the content of one document
# print(test_file_1.keys()) # These are the titles of the different sections

title_counter = 1
counter = 1
for doc in json_file.keys():
    for section,embedding in json_file[doc].items(): 
        title_id.append(title_counter)
        embeddings_id.append(counter)
        embeddings_list.append(embedding)
        counter += 1
    title_counter += 1

embeddings_df = pd.DataFrame({
    "doc_id":title_id,
    "emb_id":embeddings_id,
    "emb":embeddings_list
})

print("Number of sections:", len(embeddings_df["doc_id"]))

# Visualize representation 2d of the embeddings
# two_d_visualization(embeddings_list)

'''
# Visualization of the most suitable number of components
number_of_components = np.arange(10, 301, 10)
explained_variances = []
for components in number_of_components:
    print(f"Applying reduction of dimensionality to {components} components")
    reduced_embeddings, explained_variance = reduce_dimensionality(embeddings_list, components)
    explained_variances.append(explained_variance)
    print("Average log-likelihood of all samples:", explained_variance)

plt.plot(number_of_components, explained_variances)

# Mostrar la gráfica
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.grid(True)
plt.show()
'''

# Apply reduction of dimensionality to n number of components
components = 200
print(f"Applying reduction of dimensionality to {components} components")
reduced_embeddings, explained_variance = reduce_dimensionality(embeddings_list, components)
print("Average log-likelihood of all samples:", explained_variance)

'''
# Visualize generated clusters in a tree
print("Applying hierarchical clustering")
model = hierarchical_clustering(embeddings_list, distance_threshold=0, n_clusters=None, metric="euclidean", linkage="ward", 
                                compute_full_tree = "auto")
print("Found_clusters:", model.n_clusters_)
plot_dendrogram(model, truncate_mode="level", p=5, orientation="left")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
'''

'''
# Apply DBSCAN algorithm
print("Number of embeddings:", len(reduced_embeddings))
print("Components of embeddings:", len(reduced_embeddings[0]))
print("\nApplying DBSCAN...")
clustering = DBSCAN(eps=0.2, 
                    min_samples=10, 
                    n_jobs=4).fit(reduced_embeddings)

set_of_clusters = set(list(clustering.labels_))
print("Clusters:", set_of_clusters)
for i,cluster in enumerate(set_of_clusters):
    print(f"Length of cluster {i}:", list(clustering.labels_).count(cluster))
print("Number of outliers:", list(clustering.labels_).count(-1))
'''

# Apply HDBSCAN algorithm
print("Number of embeddings:", len(reduced_embeddings))
print("Components of embeddings:", len(reduced_embeddings[0]))
print("\nApplying HDBSCAN...")
clustering = HDBSCAN(min_cluster_size=10, 
                     n_jobs=4).fit(reduced_embeddings)

set_of_clusters = set(list(clustering.labels_))
print("Clusters:", set_of_clusters)
for i,cluster in enumerate(set_of_clusters):
    print(f"Length of cluster {i}:", list(clustering.labels_).count(cluster))
print("Number of outliers:", list(clustering.labels_).count(-1))