"""
This module deploys the algorithmns using streamlit
"""
import pandas as pd
import streamlit as st
from sklearn_extra.cluster import KMedoids
import numpy as np


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
            df_red_wine = df_red_wine_test.drop(["quality"], axis=1)
        elif data_files[file].name.endswith("white.csv"):
            df_white_wine_test = pd.read_csv(data_files[file], delimiter=";")
            df_white_wine_test[["alcohol"]].replace(". ","", inplace=True)
            df_white_wine = df_white_wine_test.drop(["quality"], axis=1)
        elif data_files[file].name.endswith("wine.csv"):
            headers = ["Class","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                       "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                       "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
            df_wine_test = pd.read_csv(data_files[file], header=None, names=headers)
            df_wine = df_wine_test.drop(["Class"], axis=1)
        else:
            st.write("ERROR: Please submit correct files.")

    st.write(df_red_wine_test.head())
    return df_red_wine, df_white_wine, df_wine, df_red_wine_test,df_white_wine_test, df_wine_test

def kmedoids():
    kmedoids = KMedoids()
    pass

def dropdown():
    with st.sidebar.beta_expander("Data Processing"):
        distance = ["Euklid", "Jaccard", "Mahalanobis"]
        wine_data = ["Red Wine Quality", "Wine Classification", "White Wine Quality"]
        algo_method = ["PAM", "Alternate"]
        st.selectbox("What distance do you want to use", distance)
        st.selectbox("What dataset do you want to use", wine_data)
        st.selectbox("What method do you want to use", algo_method)
    with st.sidebar.beta_expander("Data Visualization"):
        pass

def main():
    st.title("T4-4: Clustering wine data using K-Medoids algorithm")

    # Prevent error warning when no data files have been uploaded yet
    try:
        df_red_wine, df_white_wine, df_wine, f_red_wine_test, df_white_wine_test, df_wine_test = upload_data()
        dropdown()

    except UnboundLocalError:
        st.warning("Upload data files")

if __name__ == "__main__":
    main()