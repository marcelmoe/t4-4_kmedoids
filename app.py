"""
This module deploys KMedoid algorithms and result visualizaation using streamlit.
"""

import pandas as pd
import numpy as np
import evaluation as eval
import streamlit as st
from sklearn_extra.cluster import KMedoids
from pandas import DataFrame
import dataframe_to_pca_plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Cache data to speed up data processing
@st.cache(suppress_st_warning=True)


def upload_data():
    """
    takes raw data and
    :return: three dataframes with actual classes ('..._test'), three without
    """

    # Drag & drop multiple excel files
    data_files = st.file_uploader("Upload a .csv file", type=["csv"],
                                  accept_multiple_files=True)

    # Assign correct data files to data frames
    for file in range(len(data_files)):
        if data_files[file].name.endswith("red.csv"):
            # File was separated by semicolons instead of colons -> define custom delimiter
            df_red_wine_test = pd.read_csv(data_files[file], delimiter=";")
            # Adjust lowest value to 0
            df_red_wine_test["quality"] = df_red_wine_test["quality"] - 3
            # Bring classification column "quality" to front of data frame
            df_red_wine_test = pd.concat([df_red_wine_test["quality"],
                                          df_red_wine_test.drop(["quality"], axis=1)], axis=1)
            # Drop classification column "quality"
            df_red_wine = df_red_wine_test.drop(["quality"], axis=1)

        elif data_files[file].name.endswith("wine.csv"):
            # Set up headers for wine data set and assign when transforming to pandas dataframe
            headers = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                       "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                       "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
            df_wine_test = pd.read_csv(data_files[file], header=None, names=headers)
            df_wine_test["Class"] = df_wine_test["Class"] - 1
            df_wine = df_wine_test.drop(["Class"], axis=1)

        elif data_files[file].name.endswith("iris.csv"):
            headers=['sepal length in cm', 'sepal width in cm', 'petal length in cm',
                     'petal width in cm', 'class']
            df_iris_test = pd.read_csv(data_files[file], header=None, names=headers)
            df_iris_test = pd.concat([df_iris_test["class"], df_iris_test.drop(["class"], axis=1)], axis=1)
            df_iris = df_iris_test.drop(['class'], axis=1)

        else:
            st.error("ERROR: Please submit correct files.")

    return df_red_wine, df_iris, df_wine, df_red_wine_test, df_iris_test, df_wine_test


def kmedoids(selected_dataset, selected_distance, df_wine, df_red_wine, df_iris,
             df_red_wine_test, df_iris_test, df_wine_test):
    """
    Assign metric and data set based on user's choice, then perform kmedoid.
    kmedoid is performed for user's choice and for all other possible combinations of metric and
    metric to gain comparability later on.
    :param selected_dataset: data set selected by user
    :param selected_distance: distance metric selected by user
    :param df_wine: wine quality data set without classification column
    :param df_red_wine: red wine data set without classification column
    :param df_iris: iris data set without classification column
    :param df_red_wine_test: red wine data set with classification column
    :param df_iris_test: iris data set with classification column
    :param df_wine_test: wine quality data set with classification column
    :return: kmedoid results, data frame for PCA visualisation, no. of clusters, etc.
    """

    # Assign respective data frames and no. of clusters based on user's choice
    if selected_dataset == "Red Wine Quality":
        df_medoid = df_red_wine
        df_medoid_test = df_red_wine_test
        n_cluster = 6
    elif selected_dataset == "Wine Classification":
        df_medoid = df_wine
        df_medoid_test = df_wine_test
        n_cluster = 3
    elif selected_dataset == "Iris flower classification":
        df_medoid = df_iris
        df_medoid_test = df_iris_test
        n_cluster = 3
    else:
        st.error("No such data frame selectable")

    # Assign respective metric based on user's choice
    if selected_distance == "Euklid":
        metric = ["euclidean", "chebyshev", "mahalanobis"]
    elif selected_distance == "Chebyshev":
        metric = ["chebyshev", "mahalanobis", "euclidean"]
    elif selected_distance == "Mahalanobis":
        metric = ["mahalanobis", "euclidean", "chebyshev"]
    else:
        st.error("No such distance selectable")

    st.write("Hier: ", df_medoid.shape)

    # Feed kmedoid with no. of cluster, chosen metric and respective data frame
    # Use random initialisation method, but specify random state of 7 for random no. generator
    kmedoid_numpy_0 = KMedoids(n_cluster, metric[0],init='random', max_iter=300, random_state=7).fit(df_medoid)
    kmedoid_numpy_1 = KMedoids(n_cluster, metric[1],init='random', max_iter=300, random_state=7).fit(df_medoid)
    kmedoid_numpy_2 = KMedoids(n_cluster, metric[2],init='random', max_iter=300, random_state=7).fit(df_medoid)

    # Assign predicted cluster for all choices to variables
    kmedoids_result_0 = pd.DataFrame(kmedoid_numpy_0.labels_, columns=["cluster"])
    kmedoids_result_1 = pd.DataFrame(kmedoid_numpy_1.labels_, columns=["cluster"])
    kmedoids_result_2 = pd.DataFrame(kmedoid_numpy_2.labels_, columns=["cluster"])

    # Concatenate predicted cluster data frame and data frame from chosen data file
    df_pca = pd.concat([kmedoids_result_0, df_medoid], axis=1)

    return kmedoids_result_1, kmedoids_result_2, df_pca, n_cluster, df_medoid_test, metric


def dropdown():
    """
    define dropdown menus for choice of metric and data sets
    :return: selected metric and data set
    """

    with st.sidebar.beta_expander("Data Processing"):
        distance = ["Euklid", "Chebyshev", "Mahalanobis"]
        data_sets = ["Red Wine Quality", "Wine Classification", "Iris flower classification"]
        selected_distance = st.selectbox("What distance do you want to use", distance)
        selected_dataset = st.selectbox("What dataset do you want to use", data_sets)

    return selected_dataset, selected_distance


def df_diagram(df_medoid_test, df_pca, n_cluster, kmedoids_result_1, kmedoids_result_2, metric):
    """

    :param df_medoid_test:
    :param df_pca:
    :param n_cluster:
    :param kmedoids_result_1:
    :param kmedoids_result_2:
    :param metric:
    :return:
    """

    with st.sidebar.beta_expander("Validation"):
        validation_choice = st.radio("Select validation method", ["Sum of Squared Errors", "Rand Index",
                                                            "Silhouette Coefficient"])

    # Based on validation chosen by user, perform (eval) and assign validation results
    if validation_choice == "Sum of Squared Errors":
        validation_medoid_0 = round(sum(eval.sum_of_squares(df_pca, n_cluster)))

        validation_medoid_1 = round(sum(eval.sum_of_squares(pd.concat(
            [kmedoids_result_1, df_pca.drop(df_pca.columns[0], axis=1)], axis=1),n_cluster)))

        validation_medoid_2 = round(sum(eval.sum_of_squares(pd.concat(
            [kmedoids_result_2, df_pca.drop(df_pca.columns[0], axis=1)], axis=1),n_cluster)))

        validation_test = round(sum(eval.sum_of_squares(df_medoid_test, n_cluster)))

        fig = make_subplots(rows=1, cols=1, shared_yaxes=True)

        # Sum of squares bar plot; assign axis labels
        fig.add_trace(go.Bar(x=metric + ["Actual"],
                             y=[validation_medoid_0,
                                validation_medoid_1,
                                validation_medoid_2,
                                validation_test],
                                marker=dict(color=["#6a93b0", "#778899", "#008080", "#98b4c8"])))

        fig.update_layout(showlegend=False, title_text="Clustering quality of different distance"
                                                       "measures")
        st.write(fig)

    elif validation_choice == "Rand Index":
        validation_medoid_0 = round(eval.rand_score(df_medoid_test[df_medoid_test.columns[0]],
                                                    df_pca[df_pca.columns[0]]),2)*100
        validation_medoid_1 = round(eval.rand_score(df_medoid_test[df_medoid_test.columns[0]],
                                                    kmedoids_result_1["cluster"]),2)*100
        validation_medoid_2 = round(eval.rand_score(df_medoid_test[df_medoid_test.columns[0]],
                                                    kmedoids_result_2["cluster"]),2)*100

        labels = ["Matching", "Not Matching"]

        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'},
                                                    {'type': 'domain'},
                                                    {'type': 'domain'}]])
        # Rand index pie/ donut subplots
        fig.add_trace(go.Pie(labels=labels,
                             values=[validation_medoid_0, 100-validation_medoid_0],
                             name=metric[0],
                             title=metric[0]), 1, 1)

        fig.add_trace(go.Pie(labels=labels,
                             values=[validation_medoid_1, 100-validation_medoid_1],
                             name=metric[1],
                             title=metric[1]), 1, 2)

        fig.add_trace(go.Pie(labels=labels,
                             values=[validation_medoid_2, 100-validation_medoid_2],
                             name=metric[2],
                             title=metric[2]), 1, 3)

        # Use 'hole' to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")

        fig.update_layout(title_text="Clustering quality measured by Rand Index")
        st.write(fig)

    elif validation_choice == "Silhouette Coefficient":

        validation_medoid_0 = eval.silhouette_coefficient(df_pca, n_cluster)
        df = pd.DataFrame({'Quality': validation_medoid_0,
                           'Cluster': list(range(0,n_cluster,1))
                           })

        # Add column with colors
        df["Color"] = np.where(df["Quality"] < 0, 'red', 'green')

        # Silhouette Coefficient bar subplot
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Quality',
                             x=df['Cluster'],
                             y=df['Quality'],
                             marker_color=df['Color']))

        fig.update_layout(barmode='stack',
                          title_text="Silhouette Coefficient: "+metric[0],
                          xaxis_title="Clusters",
                          yaxis_title="Quality")

        # Avoid sub steps of x-axis by defining tick no.
        fig.update_xaxes(nticks=n_cluster+1)
        st.write(fig)

        validation_medoid_1 = eval.silhouette_coefficient(pd.concat(
            [kmedoids_result_1, df_pca.drop(df_pca.columns[0], axis=1)], axis=1), n_cluster)

        df = pd.DataFrame({'Quality': validation_medoid_1,
                           'Cluster': list(range(0,n_cluster,1))})

        # Add column with colors
        df["Color"] = np.where(df["Quality"] < 0, 'red', 'green')

        # Silhouette Coefficient bar subplot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Quality',
                   x=df['Cluster'],
                   y=df['Quality'],
                   marker_color=df['Color']))

        fig.update_layout(barmode='stack',
                          title_text="Silhouette Coefficient: "+metric[1],
                          xaxis_title="Clusters",
                          yaxis_title="Quality")

        fig.update_xaxes(nticks=n_cluster+1)
        st.write(fig)

        validation_medoid_2 = eval.silhouette_coefficient(pd.concat(
            [kmedoids_result_2, df_pca.drop(df_pca.columns[0], axis=1)], axis=1), n_cluster)

        df = pd.DataFrame({'Quality': validation_medoid_2,
                           'Cluster': list(range(0,n_cluster,1))})

        # Add column with colors
        df["Color"] = np.where(df["Quality"] < 0, 'red', 'green')

        # Silhouette Coefficient bar subplot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Quality',
                   x=df['Cluster'],
                   y=df['Quality'],
                   marker_color=df['Color']))

        fig.update_layout(barmode='stack',
                          title_text="Silhouette Coefficient: "+metric[2],
                          xaxis_title="Clusters",
                          yaxis_title="Quality")

        fig.update_xaxes(nticks=n_cluster+1)
        st.write(fig)

    else:
        st.error("No such validation method")


def main():
    """
    Initialise all functions
    :return:
    """
    st.title("T4-4: Clustering wine and iris flower data using K-Medoids algorithm")

    # Prevent error warning when no data files have been uploaded yet
    try:
        df_red_wine, df_iris, df_wine, df_red_wine_test, df_iris_test, df_wine_test = \
            upload_data()

        selected_dataset, selected_distance = dropdown()

        kmedoids_result_1, kmedoids_result_2, df_pca, n_cluster, df_medoid_test, metric = \
            kmedoids(selected_dataset,
                     selected_distance,
                     df_wine,
                     df_red_wine,
                     df_iris,
                     df_red_wine_test,
                     df_iris_test,
                     df_wine_test)

        fig = dataframe_to_pca_plot.df_to_pca_plot(df_pca.copy())
        st.write(fig)

        df_diagram(df_medoid_test, df_pca, n_cluster, kmedoids_result_1, kmedoids_result_2, metric)

    except UnboundLocalError:
        st.warning("Please upload data files")


if __name__ == "__main__":
    main()