from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read data
# load real data
def load_data(data_path):
    df = pd.read_csv(data_path, sep='\t', header=1)
    df_with_header = pd.read_csv(data_path, sep='\t', header=None)

    x_data, y_data = [], []
    for i in range(len(df.columns) // 2):
        x = df.iloc[:, i * 2].values    # time of m0 curve
        y = df.iloc[:, i * 2 + 1].values    # signal of m0 curve
        x_data.append(x[~np.isnan(x)])
        y_data.append(y[~np.isnan(y)])

    m0vals = np.array([
        float(df_with_header.iloc[0, i * 2].split(': ')[-1])
        for i in range(len(df.columns) // 2)
    ])
    print('m0 values:', m0vals)
    return x_data, y_data, m0vals


if __name__ == "__main__":
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv' 
    # x_data times will not be the same for all m0 curves since i normalized it by hand and eliminated parts of it inconsistently
    # which is why we need to interpolate to get a data matrix 
    x_data, y_data, m0vals = load_data(data_path)
    tend = max(x[-1] for x in x_data)
    t_axis = np.linspace(0, tend, 300)
    data_matrix = [np.interp(t, x, y) for t, x, y in zip([t_axis] * len(x_data), x_data, y_data)]
    data_matrix = np.array(data_matrix)
    pca = PCA()
    pca.fit(data_matrix)
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()
    for i in range(3):
        plt.plot(t_axis, pca.components_[i], label=f'PC{i+1}')
    plt.show()
