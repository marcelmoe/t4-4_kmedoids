
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

def dataframe_to_pca_plot(df, labeled=True):
    """

    :param df: Pandas dataframe
    :param labeled: If parameter 'labeled' is not explicitly passed as False, the function assumes that
            the first column contains the correct classes or the assigned clusters.
    :return:A figure containing a 3D plot corresponding to the dimension reduced dataframe. The dimensionality reduction
            is done using PCA.
    """
    # Take care for class (and cluster) labels
    if labeled:
        # Extract the labels column and copy it to numpy array containing Strings
        labels = ((df.iloc[:,0]).copy()).astype(str)
        # Drop the labels column (which is assumed to be the first one)
        df_copy = df.drop(df.columns[0], axis=1)

    else:
        # Just copy the dataframe to avoid distortions
        df_copy = df.copy()

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
    transformed_df = pd.DataFrame({'x': transformed_data[:, 0], 'y': transformed_data[:, 1],
                                   'z': transformed_data[:,2], 'cluster': labels})

    # Pass arguments to scatter_3d
    fig = px.scatter_3d(transformed_df, x='PC1', y='PC2', z='PC3', color='cluster', width=800, height=800)
    fig.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

    return fig