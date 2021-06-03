"""
This module deploys the algorithmns using streamlit
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


@st.cache(suppress_st_warning=True)
def upload_data():
    """
    takes raw data and
    :return: Three dataframes
    """

    # drag & drop multiple excel files
    data_files = st.file_uploader("Upload a .csv file", type=["csv"],
                                  accept_multiple_files=True)

    # assign correct data files to data frames
    for file in range(len(data_files)):
        if data_files[file].name.endswith("red.csv"):
            df_red_wine_test = pd.read_csv(data_files[file], delimiter=";")
            df_red_wine_test["quality"] = df_red_wine_test["quality"] - 3
            df_red_wine_test = pd.concat([df_red_wine_test["quality"], df_red_wine_test.drop(["quality"], axis=1)], axis=1)
            # drop classificaion column "quality"
            df_red_wine = df_red_wine_test.drop(["quality"], axis=1)

        elif data_files[file].name.endswith("wine.csv"):
            # set up headers for wine data set
            headers = ["Class","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                       "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                       "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
            df_wine_test = pd.read_csv(data_files[file], header=None, names=headers)
            df_wine_test["Class"] = df_wine_test["Class"] - 1
            # drop classificaion column "Class"
            df_wine = df_wine_test.drop(["Class"], axis=1)


        elif data_files[file].name.endswith("iris.csv"):
            headers=['sepal length in cm', 'sepal width in cm', 'petal length in cm',
                     'petal width in cm', 'class']
            df_iris_test = pd.read_csv(data_files[file], header=None, names=headers)
            df_iris_test = pd.concat([df_iris_test["class"], df_iris_test.drop(["class"], axis=1)], axis=1)
            df_iris = df_iris_test.drop(['class'], axis=1)


        elif data_files[file].name.endswith("shuffled.csv"):
            df_beer = pd.read_csv(data_files[file])
            df_beer.dropna(axis=0, inplace=True)
            df_beer.reset_index(inplace=True)

            df_unique = df_beer["beer_style"].unique()
            list = df_beer["beer_style"].tolist()

            uniques = pd.Series(df_unique)
            new_list = pd.Series(list)

            dict_map = {}
            keys = uniques
            values = range(0,104)
            for value in values:
                dict_map[value] = keys[value]

            swapped_dict = {value:key for key, value in dict_map.items()}

            final_list = new_list.map(swapped_dict)

            df_class = DataFrame(final_list, columns=["Class"])

            df_beer_test = pd.concat([df_beer, df_class], axis=1)
            df_beer.drop(["beer_style"], axis=1, inplace=True)

        else:
            st.write("ERROR: Please submit correct files.")

    # st.write(df_class.head())

    return df_red_wine, df_iris, df_wine, df_red_wine_test, df_iris_test, df_wine_test


def kmedoids(selected_dataset, selected_distance, df_wine, df_red_wine, df_iris,
             df_red_wine_test, df_iris_test, df_wine_test):
    """"""
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

    if selected_distance == "Euklid":
        metric = ["euclidean", "chebyshev", "mahalanobis"]
    elif selected_distance == "Chebyshev":
        metric = ["chebyshev", "mahalanobis", "euclidean"]
    elif selected_distance == "Mahalanobis":
        metric = ["mahalanobis", "euclidean", "chebyshev"]
    else:
        st.error("No such distance selectable")

    st.write("Hier: ", df_medoid.shape)
    kmedoid_numpy_0 = KMedoids(n_cluster, metric[0],init='random', max_iter=300, random_state=7).fit(df_medoid)
    kmedoid_numpy_1 = KMedoids(n_cluster, metric[1],init='random', max_iter=300, random_state=7).fit(df_medoid)
    kmedoid_numpy_2 = KMedoids(n_cluster, metric[2],init='random', max_iter=300, random_state=7).fit(df_medoid)
    kmedoids_result_0 = pd.DataFrame(kmedoid_numpy_0.labels_, columns=["cluster"])
    kmedoids_result_1 = pd.DataFrame(kmedoid_numpy_1.labels_, columns=["cluster"])
    kmedoids_result_2 = pd.DataFrame(kmedoid_numpy_2.labels_, columns=["cluster"])

    df_pca = pd.concat([kmedoids_result_0, df_medoid], axis=1)

    return kmedoids_result_1, kmedoids_result_2, df_pca, n_cluster, df_medoid_test, metric


def dropdown():
    with st.sidebar.beta_expander("Data Processing"):
        distance = ["Euklid", "Chebyshev", "Mahalanobis"]
        data_sets = ["Red Wine Quality", "Wine Classification", "Iris flower classification"]
        selected_distance = st.selectbox("What distance do you want to use", distance)
        selected_dataset = st.selectbox("What dataset do you want to use", data_sets)

    return selected_dataset, selected_distance


def df_diagram(df_medoid_test, df_pca, n_cluster, kmedoids_result_1, kmedoids_result_2, metric):

    with st.sidebar.beta_expander("Validation"):
        validation_choice = st.radio("Select validation method", ["Sum of Squared Errors", "Rand Index",
                                                            "Silhouette Coefficient"])

    if validation_choice == "Sum of Squared Errors":
        validation_medoid_0 = round(sum(eval.sum_of_squares(df_pca, n_cluster)))
        validation_medoid_1 = round(sum(eval.sum_of_squares(pd.concat([kmedoids_result_1, df_pca.drop(df_pca.columns[0],
                                                                                            axis=1)], axis=1),n_cluster)))
        validation_medoid_2 = round(sum(eval.sum_of_squares(pd.concat([kmedoids_result_2, df_pca.drop(df_pca.columns[0],
                                                                                            axis=1)], axis=1),n_cluster)))
        validation_test = round(sum(eval.sum_of_squares(df_medoid_test, n_cluster)))
        fig = make_subplots(rows=1, cols=1, shared_yaxes=True)


        fig.add_trace(go.Bar(x=metric + ["Actual"], y=[validation_medoid_0, validation_medoid_1, validation_medoid_2,
                                                       validation_test], marker=dict(color=["#6a93b0", "#778899",
                                                                                            "#008080", "#98b4c8"])))
        fig.update_layout(showlegend=False, title_text="Clustering quality of different distance measures")
        st.write(fig)

    elif validation_choice == "Rand Index":
        validation_medoid_0 = round(eval.rand_score(df_medoid_test[df_medoid_test.columns[0]],
                                                    df_pca[df_pca.columns[0]]),2)*100
        validation_medoid_1 = round(eval.rand_score(df_medoid_test[df_medoid_test.columns[0]], kmedoids_result_1["cluster"]),2)*100
        validation_medoid_2 = round(eval.rand_score(df_medoid_test[df_medoid_test.columns[0]], kmedoids_result_2["cluster"]),2)*100

        labels = ["Matching", "Not Matching"]
        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'},{'type': 'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=[validation_medoid_0, 100-validation_medoid_0], name=metric[0], title=metric[0]),
                      1, 1)
        fig.add_trace(go.Pie(labels=labels, values=[validation_medoid_1, 100-validation_medoid_1], name=metric[1], title=metric[1]),
                      1, 2)
        fig.add_trace(go.Pie(labels=labels, values=[validation_medoid_2, 100-validation_medoid_2], name=metric[2], title=metric[2]),
                      1, 3)
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")

        fig.update_layout(title_text="Clustering quality measured by Rand Index")
        st.write(fig)
    elif validation_choice == "Silhouette Coefficient":

        validation_medoid_0 = eval.silhouette_coefficient(df_pca, n_cluster)
        df = pd.DataFrame({
            'Quality': validation_medoid_0,
            'Cluster': list(range(0,n_cluster,1))
        })

        ## here I'm adding a column with colors
        df["Color"] = np.where(df["Quality"] < 0, 'red', 'green')

        # Plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Quality',
                   x=df['Cluster'],
                   y=df['Quality'],
                   marker_color=df['Color']))
        fig.update_layout(barmode='stack', title_text="Silhouette Coefficient: "+metric[0], xaxis_title="Clusters",
                          yaxis_title="Quality")
        fig.update_xaxes(nticks=n_cluster+1)
        st.write(fig)

        validation_medoid_1 = eval.silhouette_coefficient(pd.concat([kmedoids_result_1, df_pca.drop(df_pca.columns[0],
                                                                                            axis=1)], axis=1), n_cluster)
        df = pd.DataFrame({
            'Quality': validation_medoid_1,
            'Cluster': list(range(0,n_cluster,1))
        })

        ## here I'm adding a column with colors
        df["Color"] = np.where(df["Quality"] < 0, 'red', 'green')

        # Plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Quality',
                   x=df['Cluster'],
                   y=df['Quality'],
                   marker_color=df['Color']))
        fig.update_layout(barmode='stack', title_text="Silhouette Coefficient: "+metric[1], xaxis_title="Clusters",
                          yaxis_title="Quality")
        fig.update_xaxes(nticks=n_cluster+1)
        st.write(fig)

        validation_medoid_2 = eval.silhouette_coefficient(pd.concat([kmedoids_result_2, df_pca.drop(df_pca.columns[0],
                                                                                            axis=1)], axis=1), n_cluster)
        df = pd.DataFrame({
            'Quality': validation_medoid_2,
            'Cluster': list(range(0,n_cluster,1))
        })

        ## here I'm adding a column with colors
        df["Color"] = np.where(df["Quality"] < 0, 'red', 'green')

        # Plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Quality',
                   x=df['Cluster'],
                   y=df['Quality'],
                   marker_color=df['Color']))
        fig.update_layout(barmode='stack', title_text="Silhouette Coefficient: "+metric[2], xaxis_title="Clusters",
                          yaxis_title="Quality")
        fig.update_xaxes(nticks=n_cluster+1)
        st.write(fig)

    else:
        st.error("No such validation method")

    col1, col2, col3 = st.beta_columns(3)


def main():
    st.title("T4-4: Clustering wine data using K-Medoids algorithm")

    # Prevent error warning when no data files have been uploaded yet
    try:
        df_red_wine, df_iris, df_wine, df_red_wine_test, df_iris_test, df_wine_test = \
            upload_data()
        selected_dataset, selected_distance = dropdown()
        kmedoids_result_1, kmedoids_result_2, df_pca, n_cluster, df_medoid_test, metric = kmedoids(selected_dataset, selected_distance, df_wine,
                                         df_red_wine, df_iris, df_red_wine_test, df_iris_test, df_wine_test)
        fig = dataframe_to_pca_plot.df_to_pca_plot(df_pca.copy())
        st.write(fig)
        df_diagram(df_medoid_test, df_pca, n_cluster, kmedoids_result_1, kmedoids_result_2, metric)


    except UnboundLocalError:
        st.warning("Upload data files")


if __name__ == "__main__":
    main()