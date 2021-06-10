import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def df_to_pca_plot(df):
    """
    :param df: Pandas dataframe
    :return:A figure containing a 3D plot corresponding to the dimension reduced dataframe. The dimensionality reduction
            is done using PCA.
    """
    # Extract the label column and copy it to numpy array containing Strings
    labels = df[df.columns[0]].astype(str)
    # Drop the labels column (which is assumed to be the first one)
    df_copy = df.drop(df.columns[0], axis=1)

    # Prepare dataframe for PCA
    # Centering
    df_copy = df_copy - df_copy.mean()
    # Standardizing
    df_copy = df_copy / df_copy.std()

    # Actual PCA
    # Init PCA to use three PCs
    pca = PCA(n_components=3)
    # Transform data (fit_transform returns numpy array)
    transformed_data = pca.fit_transform(df_copy)

    # Construct the figures
    # Build dataframe for plotly.express.scatter_3d
    features = ["PC1 (" + str(round(pca.explained_variance_ratio_[0] * 100, 2)) + "%)", "PC2 (" +
                str(round(pca.explained_variance_ratio_[1] * 100, 2)) + "%)",
                "PC3 (" + str(round(pca.explained_variance_ratio_[2] * 100, 2)) + "%)"]
    transformed_df = pd.DataFrame({features[0]: transformed_data[:, 0], features[1]: transformed_data[:, 1],
                                   features[2]: transformed_data[:, 2], 'cluster': labels})

    # Pass arguments to scatter_3d
    fig = px.scatter_3d(transformed_df, x=features[0], y=features[1], z=features[2], color='cluster', width=800,
                        height=800)
    fig.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

    return fig
