"""
The evaluation module provides several functions to measure clustering quality.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score


def jaccard_index_for_dataframes(df1, df2):
    """
    Jaccard-Index:
    The Jaccard-Index computes the similarity of two sets A and B as follows:
    J(A,B) = Intersection(A,B)/Union(A,B).
    Hint: It is computationally way more efficient to use the jaccard_index_for_labels function
    :param df1: First Pandas Dataframe. It is assumed that df1 does not contain a class-/cluster-label column.
    :param df2: Second Pandas Dataframe having the same column types as df1. It is assumed that df2 does not
    contain a class-/cluster-label column.
    :return: J(df1, df2)
    """
    # Intersection of df1 and df2
    intersection = pd.merge(df1, df2, how='inner')

    # Union of df1 and df2
    union = pd.concat([df1, df2])
    union = union.drop_duplicates()

    # Jaccard_index
    jaccard_index = len(intersection)/len(union)

    return jaccard_index


def jaccard_index_for_labels(labels_true, labels_pred):
    """
    Jaccard-Index:
    The Jaccard-Index computes the similarity of two sets A and B as follows:
    J(A,B) = Intersection(A,B)/Union(A,B).
    :param labels_true: numpy array containing the true labels
    :param labels_pred: numpy array containing the predicted labels (in the same order)
    :return: Jaccard-Index. In case of more then two possible classes, the Jaccard-Indexes for each class
    are returned as a numpy array. Example. array([score_class0, score_class1, score_class2, ..., score_classN])
    """
    return jaccard_score(labels_true, labels_pred)


def sum_of_squares(df, cluster_count):
    """
    Within cluster sum of squares:
    It is the cumulative sum of the squared deviations from each observation of a cluster to the cluster's centroid.
    :param df: Pandas dataframe. It is assumed that df's first column contains the observations cluster-label. It
    is further assumed that the class labels range from 0 to (cluster_count - 1)
    :param cluster_count: Number of clusters.
    :return: Numpy array containing the within cluster sum of squares for all clusters of df.
    """

    result = []

    # Iterate over all clusters
    for i in range(0, cluster_count):

        # sub dataframe containing observations assigned to cluster i
        df_i = df.loc[df[df.columns[0]] == i]

        # sum of squares computation
        ssq = (((df_i - df_i.mean())**2).to_numpy()).sum()

        result = result + [ssq]

    return np.array(result)


def silhouette_coefficient(df, cluster_count):
    """
    Silhouette Coefficient:
    It is an internal measure for the quality of a cluster.
    For each observation i it is defined as:
    s(i) = (Separation(i)-Cohesion(i))/max(Separation(i),Cohesion(i))
    Separation(i) is defined as the smallest mean distance of observation i to all observations of any other cluster
    Cohesion(i) is defined as the mean distance of observation i to all observations in the same cluster as i.
    For a complete cluster the Silhouette Coefficient is defined as the average over all silhouette coefficients of
    its observations.
    :param df: Pandas dataframe. It is assumed that df's first column contains the observations cluster-label. It
    is further assumed that the class labels range from 0 to (cluster_count - 1)
    :param cluster_count: Number of clusters/classes.
    :return: Numpy array containing the silhouette coefficients for all clusters of df.
    """

    result = []
    all_s = []

    # Iterate over all clusters
    for i in range(1, cluster_count+1):
        # sub dataframe containing observations assigned to cluster i
        cluster_i = (df.loc[df[df.columns[0]] == i]).drop([df.columns[0]], axis=1)

        # Convert to numpy array (for computational reasons)
        cluster_i_np = cluster_i.to_numpy()

        # List of silhouette coefficients for all observations in cluster_i
        s = []

        # Iterate over all observations
        for j in range(0, len(cluster_i_np)):

            # First, compute the cohesion of observation j
            # Compute the sum of all distance from point j to points of cluster_i
            dist = np.linalg.norm(cluster_i_np - cluster_i_np[j])

            # Take the average
            avg_dist = dist/(len(cluster_i_np)-1)

            # The cohesion is equal to the average distance
            cohesion_j = avg_dist

            # Second, compute the separation
            # List of all averaged cluster distances
            avg_cluster_dist = []

            # Iterate over all other clusters
            for k in range(1, cluster_count+1):

                # Continue if cluster_i and cluster_k are the same
                if i == k:
                    continue

                # Cluster_k
                cluster_k = (df.loc[df[df.columns[0]] == k]).drop(df.columns[0], axis=1)
                # Convert to numpy array
                cluster_k_np = cluster_k.to_numpy()

                # Compute the sum of all distance from point j of cluster i to  points of cluster_k
                dist = np.linalg.norm(cluster_k_np - cluster_i_np[j])
                # Take the average
                avg_dist = dist / len(cluster_k_np)
                # Add to list
                avg_cluster_dist = avg_cluster_dist + [avg_dist]

            # The separation is given by the min of avg_cluster_dist
            separation_j = min(avg_cluster_dist)

            # Now, we can compute the silhouette coefficient for observation j of cluster i
            s_j = (separation_j - cohesion_j) / max(separation_j, cohesion_j)

            # Adding to list
            s = s + [s_j]
            all_s = all_s + [s_j]

        # Finally, we can compute our cluster's silhouette coefficient
        cluster_sc = sum(s) / len(s)

        # We add the cluster silhouette coefficient to the result list
        result = result + [cluster_sc]

    return result, sum(all_s)/len(all_s)
