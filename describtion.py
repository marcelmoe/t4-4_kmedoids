"""
This module is meant to describe the data sets analysed in the deployed app
"""

import pandas as pd

def data_clearance():
    """
    takes raw data and
    :return: modifed data frames used for further analysis
    """
    #Import data
    df_red_wine = pd.read_csv("data_sets\winequality-red.csv", delimiter=";")
    # Adjust lowest value to 0
    df_red_wine["quality"] = df_red_wine["quality"] - 3
    # Bring classification column "quality" to front of data frame
    df_red_wine = pd.concat([df_red_wine["quality"],
                                  df_red_wine.drop(["quality"], axis=1)], axis=1)

    #Introduce column names
    headers = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
               "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
               "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
    #Import data
    df_wine = pd.read_csv("data_sets\wine.csv", header=None, names=headers)
    # Adjust values from column 'Class'
    df_wine["Class"] = df_wine["Class"] - 1

    #Introduce column names
    headers=['sepal length in cm', 'sepal width in cm', 'petal length in cm',
             'petal width in cm', 'class']
    #Import data
    df_iris = pd.read_csv("data_sets\iris.csv", header=None, names=headers)
    # Bring classification column 'class' to front of data frame
    df_iris = pd.concat([df_iris["class"], df_iris.drop(["class"], axis=1)], axis=1)

    return df_red_wine, df_wine, df_iris

def data_description(df_red_wine, df_wine, df_iris):
    """
    prints out respective description of the data frames
    :return: None
    """
    red_describe = df_red_wine.describe()
    with open('mytable.tex','w') as tf:
        tf.write(red_describe.to_latex())
    print(df_wine.describe())
    print(df_iris.describe())

def main():
    """
    Initialise all functions
    :return:
    """
    df_red_wine, df_wine, df_iris = data_clearance()
    data_description(df_red_wine, df_wine, df_iris)


if __name__ == "__main__":
    main()

