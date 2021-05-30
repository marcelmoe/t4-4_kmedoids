"""
This module deploys the algorithmns using streamlit
"""
import pandas as pd
import streamlit as st
from sklearn_extra.cluster import KMedoids
from pandas import DataFrame

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
            #drop classificaion colum "quality"
            df_red_wine = df_red_wine_test.drop(["quality"], axis=1)

        elif data_files[file].name.endswith("wine.csv"):
            #Set up headers for wine data set
            headers = ["Class","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                       "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                       "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
            df_wine_test = pd.read_csv(data_files[file], header=None, names=headers)
            #drop classificaion colum "Class"
            df_wine = df_wine_test.drop(["Class"], axis=1)

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

    return df_red_wine, df_beer, df_wine, df_red_wine_test, df_beer_test, df_wine_test


def kmedoids():
    kmedoids = KMedoids()
    pass


def dropdown():
    with st.sidebar.beta_expander("Data Processing"):
        distance = ["Euklid", "Jaccard", "Mahalanobis"]
        data_sets = ["Red Wine Quality", "Wine Classification", "Beer reviews"]
        algo_method = ["PAM", "Alternate"]
        st.selectbox("What distance do you want to use", distance)
        st.selectbox("What dataset do you want to use", data_sets)
        st.selectbox("What method do you want to use", algo_method)
    with st.sidebar.beta_expander("Data Visualization"):
        pass


def main():
    st.title("T4-4: Clustering wine data using K-Medoids algorithm")

    # Prevent error warning when no data files have been uploaded yet
    try:
        upload_data()
        dropdown()

    except UnboundLocalError:
        st.warning("Upload data files")


if __name__ == "__main__":
    main()