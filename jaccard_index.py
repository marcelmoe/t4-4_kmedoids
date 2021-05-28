"""
The Jaccard-Index computes the similarity of two sets A and B as follows:
J(A,B) = Intersection(A,B)/Union(A,B).

Hint: It is computationally way more efficient to use the jaccard_index_for_labels function
"""

import pandas as pd
from sklearn.metrics import jaccard_score

def jaccard_index_for_dataframes(df1,df2):
    """
    :param df1: First Pandas Dataframe
    :param df2: Second Pandas Dataframe haven the same column types as df1
    :return: J(df1, df2)
    """
    # Intersection of df1 and df2
    intersection = pd.merge(df1, df2, how='inner')

    # Union of df1 and df2
    union = pd.concat([df1,df2])
    union = union.drop_duplicates()

    # Jaccard_index
    jaccard_index = len(intersection)/len(union)

    return jaccard_index



def jaccard_index_for_labels(labels_true, labels_pred):
    """
    :param labels_true: numpy array containing the true labels
    :param labels_pred: numpy array containing the predicted labels (in the same order)
    :return: Jaccard-Index. In case of more then two possible classes, the Jaccard-Indexes for each class
    are returned as a numpy array. Example. array([score_class0, score_class1, score_class2, ..., score_classN])
    """
    return jaccard_score(labels_true, labels_pred)